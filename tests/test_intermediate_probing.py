"""
Unit tests for Intermediate Layer Probing and leak-free training pipeline.
Runs fast in <2s on CPU using synthetic hidden states and mock CausalLM modules.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.signals.intermediate_probing import (
    compute_geometry_metrics,
    compute_logit_lens_metrics,
    extract_single_intermediate_features,
    get_intermediate_layer_indices,
    get_intermediate_scalar_feature_names,
    resolve_model_components,
)
from src.training.intermediate_models import (
    fit_and_extract_probe_features,
    get_intermediate_saved_model_dir,
    score_with_probe_bundle,
    select_diverse_intermediate_layers,
    sweep_intermediate_layers,
)
from src.utils.model_registry import get_saved_model_dir


class _MockConfig:
    num_hidden_layers = 8


class _MockRMSNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5


class _MockInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = _MockRMSNorm()


class _MockOutput:
    def __init__(self, logits: torch.Tensor, hidden_states: tuple):
        self.logits = logits
        self.hidden_states = hidden_states


class _MockCausalLM(nn.Module):
    def __init__(self, num_layers: int = 8, hidden_dim: int = 16, vocab_size: int = 32):
        super().__init__()
        self.config = _MockConfig()
        self.config.num_hidden_layers = num_layers
        self.model = _MockInnerModel()
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = False):
        b, s = input_ids.shape
        torch.manual_seed(0)
        hidden_states = tuple(
            torch.randn(b, s, self.hidden_dim) + 0.1 * l
            for l in range(self.num_layers + 1)
        )
        logits = self.lm_head(hidden_states[-1])
        return _MockOutput(logits=logits, hidden_states=hidden_states)


class _MockTokenizer:
    bos_token_id = 1
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [5 + (ord(c[0]) % 20) for c in text.split()] or [5]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids


def test_intermediate_layer_index_mapping():
    assert get_intermediate_layer_indices(24) == [6, 12, 16, 20]
    assert get_intermediate_layer_indices(16) == [4, 8, 11, 13]
    assert get_intermediate_layer_indices(8) == [2, 4, 5, 7]


def test_resolve_model_components_and_single_extraction():
    model = _MockCausalLM(num_layers=8, hidden_dim=16, vocab_size=32)
    tok = _MockTokenizer()

    norm, lm_head, n_layers, final_normed = resolve_model_components(model)
    assert norm is model.model.norm
    assert lm_head is model.lm_head
    assert n_layers == 8
    assert final_normed is True

    layers = get_intermediate_layer_indices(n_layers)
    scalars, pooled = extract_single_intermediate_features(
        model=model,
        tokenizer=tok,
        prompt="Question: Capital of France?",
        response="Paris is the capital.",
        layer_indices=layers,
        probe_layers=[4, 6],
        device="cpu",
    )

    expected_names = get_intermediate_scalar_feature_names(layers)
    assert set(scalars.keys()) == set(expected_names)
    for k, v in scalars.items():
        assert isinstance(v, float)
        assert np.isfinite(v), f"Non-finite scalar {k}: {v}"
        if "js_div" in k or "kl_final" in k or "entropy" in k or "hidden_norm" in k:
            assert v >= 0.0

    assert "resp_mean_l4" in pooled and pooled["resp_mean_l4"].shape == (16,)
    assert "resp_last_l6" in pooled and pooled["resp_last_l6"].shape == (16,)
    assert "prompt_last_l4" in pooled and pooled["prompt_last_l4"].shape == (16,)


def test_logit_lens_identical_to_final_has_zero_divergence():
    torch.manual_seed(42)
    logits = torch.randn(4, 10)
    log_p = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)
    targets = [1, 2, 3, 4]
    step = torch.arange(4)
    ll = compute_logit_lens_metrics(
        layer_logits=logits,
        final_probs=p,
        final_log_probs=log_p,
        final_target_log_probs=log_p[step, targets],
        final_target_probs=p[step, targets],
        target_token_ids=targets,
        layer_idx=12,
    )
    assert abs(ll["l12_kl_final_to_layer"]) < 1e-5
    assert abs(ll["l12_js_div_to_final"]) < 1e-5
    assert abs(ll["l12_log_prob_gain"]) < 1e-5
    assert abs(ll["l12_prob_gain"]) < 1e-5


def test_geometry_metrics_bounds():
    h_p = torch.randn(5, 16)
    h_r = torch.randn(4, 16)
    h_f = h_r.mean(dim=0)
    g = compute_geometry_metrics(h_p, h_r, h_f, layer_idx=6)
    assert g["l6_hidden_norm_mean"] > 0.0
    assert -1.0001 <= g["l6_prompt_resp_cos"] <= 1.0001
    assert -1.0001 <= g["l6_tok_cos_smoothness"] <= 1.0001
    assert abs(g["l6_cos_to_final"] - 1.0) < 1e-5


def test_leak_free_layer_sweep_and_probe_bundle():
    rng = np.random.default_rng(42)
    n_total, n_layers, hidden_dim = 120, 8, 16
    y = np.array([0, 1] * (n_total // 2), dtype=np.int64)
    sources = np.array(["halueval", "truthfulqa", "fever"] * (n_total // 3))

    # Inject a truthful signal specifically in layer 5 of resp_mean
    resp_mean = rng.normal(0, 1, size=(n_total, n_layers, hidden_dim)).astype(np.float16)
    resp_mean[:, 4, 0] += (y * 2.5 - 1.25).astype(np.float16)
    resp_last = rng.normal(0, 1, size=(n_total, n_layers, hidden_dim)).astype(np.float16)
    prompt_last = rng.normal(0, 1, size=(n_total, n_layers, hidden_dim)).astype(np.float16)

    pools = {
        "resp_mean": resp_mean,
        "resp_last": resp_last,
        "prompt_last": prompt_last,
    }
    train_idx = np.arange(0, 90)
    test_idx = np.arange(90, 120)

    sweep = sweep_intermediate_layers(
        pools=pools,
        train_indices=train_idx,
        y_all=y,
        sources_all=sources,
        c_grid=(0.05, 0.2),
        pool_modes=("resp_mean", "resp_last"),
        n_splits=3,
        random_state=42,
    )
    assert sweep["best_by_mode"]["resp_mean"]["layer"] == 5
    assert sweep["best_by_mode"]["resp_mean"]["cv_roc_auc"] > 0.80

    selected = select_diverse_intermediate_layers(sweep, pool_mode="resp_mean", top_k=2, min_spacing=2)
    assert len(selected) == 2
    assert any(l == 5 for l, _ in selected)

    specs = [(l, "resp_mean", c) for l, c in selected]
    df_tr, df_te, bundle = fit_and_extract_probe_features(
        pools=pools,
        train_indices=train_idx,
        test_indices=test_idx,
        y_all=y,
        sources_all=sources,
        layer_c_specs=specs,
        n_splits=3,
        random_state=42,
    )
    assert len(df_tr) == 90
    assert len(df_te) == 30
    assert list(df_tr.columns) == list(df_te.columns)

    # Verify live scoring matches test row 0 exactly
    live_vecs = {
        f"resp_mean_l{l}": resp_mean[test_idx[0], l - 1, :].astype(np.float32)
        for l, _ in selected
    }
    scored = score_with_probe_bundle(live_vecs, bundle)
    for col in df_te.columns:
        assert abs(scored[col] - float(df_te.iloc[0][col])) < 1e-4


def test_intermediate_model_dir_isolated_from_legacy():
    legacy_dir = get_saved_model_dir("Qwen/Qwen3.5-0.8B")
    inter_dir = get_intermediate_saved_model_dir("Qwen/Qwen3.5-0.8B")
    assert legacy_dir != inter_dir
    assert legacy_dir.name == "qwen3_5_0_8b"
    assert inter_dir.name == "qwen3_5_0_8b_intermediate"


def test_run_live_inference_with_intermediate_probing_mocked():
    from unittest.mock import MagicMock, patch
    from app.inference import run_live_inference
    from src.features.build_universal_features import UNIVERSAL_CORE_FEATURES
    from src.signals.token_signals import SequenceSignals

    internal = {
        "min_log_prob": -2.5, "mean_log_prob": -0.8, "mean_token_prob": 0.65,
        "token_prob_std": 0.22, "mean_entropy": 1.15, "max_entropy": 2.85,
        "entropy_std": 0.45, "log_perplexity": 0.8, "log_mean_token_rank": 0.55,
        "log_max_token_rank": 1.25, "log_rank_std": 0.40,
    }
    sc = {
        "exact_match_agreement": 0.6, "mean_pairwise_similarity": 0.88,
        "min_pairwise_similarity": 0.72, "max_pairwise_similarity": 0.96,
        "pairwise_similarity_std": 0.08,
    }
    nli = {
        "mean_pairwise_entailment": 0.75, "mean_pairwise_contradiction": 0.05,
        "nli_disagreement": 0.175,
    }
    inter_live = {"l12_mean_log_prob": -0.9, "probe_resp_mean_l12_logit": 0.42}
    probe_bundle = {
        "logit_lens_layers": [12],
        "probe_layers": [12],
        "feature_columns": list(UNIVERSAL_CORE_FEATURES) + ["l12_mean_log_prob", "probe_resp_mean_l12_logit"],
    }

    mock_lr = MagicMock()
    mock_lr.predict_proba.return_value = np.array([[0.3, 0.7]])
    mock_xgb = MagicMock()
    mock_xgb.predict_proba.return_value = np.array([[0.2, 0.8]])

    mock_signals = SequenceSignals(
        num_tokens=4, mean_token_prob=0.8, min_token_prob=0.6,
        mean_log_prob=-0.3, min_log_prob=-0.5, mean_entropy=0.4,
        max_entropy=0.8, entropy_std=0.1, token_prob_std=0.05,
        perplexity=1.35, tokens=[],
    )

    with patch("app.inference.generate_primary_response", return_value=("Paris.", "Prompt: France?")), \
         patch("app.inference.extract_internal_signals", return_value=(internal, mock_signals)), \
         patch("app.inference.run_self_consistency_probe", return_value=(["Paris."] * 5, sc)), \
         patch("app.inference.run_nli_probe", return_value=nli), \
         patch("app.inference.extract_intermediate_live_features", return_value=inter_live):
        res = run_live_inference(
            prompt="What is the capital of France?",
            causal_model=MagicMock(),
            tokenizer=MagicMock(),
            embedding_model=MagicMock(),
            nli_model=MagicMock(),
            nli_label_indices={"contradiction": 0, "entailment": 1, "neutral": 2},
            lr_pipeline=mock_lr,
            xgb_model=mock_xgb,
            use_intermediate_probing=True,
            probe_bundle=probe_bundle,
        )

    assert res["probing_mode"] == "intermediate"
    assert res["intermediate_features"] == inter_live
    assert pytest.approx(res["predicted_risk_xgb"], 1e-4) == 0.8
    assert pytest.approx(res["predicted_risk_lr"], 1e-4) == 0.7


def test_get_cached_models_switches_detector_mode():
    from unittest.mock import MagicMock, patch
    from app.config import (
        DEFAULT_DETECTOR_MODEL_TYPE,
        DETECTOR_MODEL_TYPES,
        get_cached_models,
        get_detector_mode_specs,
    )

    assert DEFAULT_DETECTOR_MODEL_TYPE == "standard"
    assert [m["slug"] for m in DETECTOR_MODEL_TYPES] == ["standard", "intermediate"]
    assert get_detector_mode_specs("intermediate")["short_label"] == "92-FEATURE PROBING"

    std_lr, std_xgb = MagicMock(name="std_lr"), MagicMock(name="std_xgb")
    int_lr, int_xgb = MagicMock(name="int_lr"), MagicMock(name="int_xgb")
    bundle = {"feature_columns": ["f1", "f2"]}
    base_models = {
        "causal_model": None,
        "tokenizer": None,
        "embedding_model": None,
        "nli_model": None,
        "nli_label_indices": {},
        "lr_pipeline": std_lr,
        "xgb_model": std_xgb,
        "device": "cpu",
    }

    with patch("app.config._load_cached_models_resource", return_value=base_models), \
         patch("app.config.has_intermediate_models", return_value=True), \
         patch("app.config.get_intermediate_classifiers", return_value=(int_lr, int_xgb, bundle)):
        res_std = get_cached_models(model_id="Qwen/Qwen3.5-0.8B", target_device="cpu", use_intermediate_probing=False)
        res_int = get_cached_models(model_id="Qwen/Qwen3.5-0.8B", target_device="cpu", use_intermediate_probing=True)

    assert res_std["lr_pipeline"] is std_lr
    assert res_std["xgb_model"] is std_xgb
    assert res_std["probe_bundle"] is None
    assert res_std["probing_mode"] == "standard"

    assert res_int["lr_pipeline"] is int_lr
    assert res_int["xgb_model"] is int_xgb
    assert res_int["probe_bundle"] is bundle
    assert res_int["probing_mode"] == "intermediate"


def test_sidebar_detector_model_toggle_in_app():
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "app" / "app.py"
    at = AppTest.from_file(app_path, default_timeout=20)
    at.run()
    assert not at.exception

    sidebar_keys = [btn.key for btn in at.sidebar.button]
    assert "sidebar_nav_about" in sidebar_keys
    assert "sidebar_detector_standard" in sidebar_keys
    assert "sidebar_detector_intermediate" in sidebar_keys
    assert "sidebar_nav_profile" in sidebar_keys

    # Verify ordering: below option menu (about) and above accounts option (profile)
    idx_about = sidebar_keys.index("sidebar_nav_about")
    idx_std = sidebar_keys.index("sidebar_detector_standard")
    idx_int = sidebar_keys.index("sidebar_detector_intermediate")
    idx_profile = sidebar_keys.index("sidebar_nav_profile")
    assert idx_about < idx_std < idx_int < idx_profile

    assert at.session_state["detector_model_type"] == "standard"

    # Switch to Intermediate Probing
    at.sidebar.button(key="sidebar_detector_intermediate").click().run()
    assert not at.exception
    assert at.session_state["detector_model_type"] == "intermediate"

    # Switch back to Universal Core (19)
    at.sidebar.button(key="sidebar_detector_standard").click().run()
    assert not at.exception
    assert at.session_state["detector_model_type"] == "standard"
