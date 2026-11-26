"""
Intermediate Layer Probing & Logit-Lens Signal Extraction Module.

Extracts internal representations and layer-wise dynamics from intermediate
transformer layers of causal language models (starting with Qwen/Qwen3.5-0.8B):
  1. Pooled intermediate hidden-state activations (response mean, response last,
     prompt last) across all transformer layers for linear truthfulness probing.
  2. Logit-lens projections (projecting intermediate residual-stream states
     through the model's final norm + lm_head) to capture token probability,
     predictive entropy, JS/KL divergence to the final output distribution, and
     layer-to-final probability shifts.
  3. Representation geometry signals (L2 activation norms, token-to-token
     hidden cosine smoothness, prompt-response hidden alignment, layer drift).

All learned linear probes on hidden states are fit strictly on training folds
with out-of-fold (OOF) cross-validation to guarantee zero test leakage.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

logger = logging.getLogger("intermediate_probing")

# Default relative depth fractions for Logit-Lens / geometry probing (mapped to layer indices)
DEFAULT_DEPTH_FRACTIONS: Tuple[float, ...] = (0.25, 0.50, 0.67, 0.83)


def resolve_model_components(model: nn.Module) -> Tuple[Optional[nn.Module], Optional[nn.Module], int, bool]:
    """
    Resolve final normalization module, lm_head projection, layer count, and whether
    hidden_states[-1] already has final normalization applied.

    Supports Qwen3.5, Llama, Gemma, and standard Hugging Face CausalLM wrappers.

    Returns:
        Tuple of (final_norm_module, lm_head_module, num_hidden_layers, final_hs_is_normed).
    """
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) if config is not None else None
    num_layers = int(
        getattr(config, "num_hidden_layers", None)
        or getattr(text_config, "num_hidden_layers", None)
        or 24
    )

    lm_head = None
    if hasattr(model, "get_output_embeddings"):
        try:
            lm_head = model.get_output_embeddings()
        except Exception:
            lm_head = None
    if lm_head is None:
        lm_head = getattr(model, "lm_head", None)

    inner_model = getattr(model, "model", None)
    final_norm = None
    if inner_model is not None:
        final_norm = getattr(inner_model, "norm", None)
        if final_norm is None and hasattr(inner_model, "language_model"):
            final_norm = getattr(inner_model.language_model, "norm", None)

    # Standard HF CausalLM models apply self.norm(hidden_states) inside inner_model.forward(),
    # so hidden_states[-1] is already post-norm while hidden_states[1..-2] are pre-final-norm.
    final_hs_is_normed = final_norm is not None

    return final_norm, lm_head, num_layers, final_hs_is_normed


def get_intermediate_layer_indices(
    num_layers: int,
    depth_fractions: Sequence[float] = DEFAULT_DEPTH_FRACTIONS,
) -> List[int]:
    """
    Map relative depth fractions in (0, 1) to unique sorted 1-based layer indices [1 .. num_layers - 1].
    For num_layers=24 and (0.25, 0.50, 0.67, 0.83), returns [6, 12, 16, 20].
    """
    indices: List[int] = []
    for frac in depth_fractions:
        idx = int(round(frac * num_layers))
        idx = max(1, min(num_layers - 1, idx))
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)


def _deduplicate_leading_bos(prompt_ids: List[int], tokenizer: Any) -> List[int]:
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if (
        bos_id is not None
        and len(prompt_ids) >= 2
        and prompt_ids[0] == bos_id
        and prompt_ids[1] == bos_id
    ):
        return prompt_ids[1:]
    return prompt_ids


def _apply_final_norm_and_lm_head(
    hidden_slice: torch.Tensor,
    final_norm: Optional[nn.Module],
    lm_head: Optional[nn.Module],
    already_normed: bool = False,
) -> Optional[torch.Tensor]:
    """Project a hidden-state slice (R, H) to vocabulary logits (R, V) in float32."""
    if lm_head is None:
        return None
    h = hidden_slice
    if final_norm is not None and not already_normed:
        h = final_norm(h)
    logits = lm_head(h)
    return logits.float()


def compute_logit_lens_metrics(
    layer_logits: torch.Tensor,
    final_probs: torch.Tensor,
    final_log_probs: torch.Tensor,
    final_target_log_probs: torch.Tensor,
    final_target_probs: torch.Tensor,
    target_token_ids: Sequence[int],
    layer_idx: int,
) -> Dict[str, float]:
    """
    Compute Logit-Lens distributional & divergence statistics for one intermediate layer
    against the final output distribution over response tokens.
    """
    r_len = layer_logits.size(0)
    if r_len == 0:
        return {
            f"l{layer_idx}_mean_log_prob": 0.0,
            f"l{layer_idx}_min_log_prob": 0.0,
            f"l{layer_idx}_mean_token_prob": 0.0,
            f"l{layer_idx}_mean_entropy": 0.0,
            f"l{layer_idx}_max_entropy": 0.0,
            f"l{layer_idx}_kl_final_to_layer": 0.0,
            f"l{layer_idx}_js_div_to_final": 0.0,
            f"l{layer_idx}_log_prob_gain": 0.0,
            f"l{layer_idx}_prob_gain": 0.0,
        }

    logits_f32 = layer_logits.float()
    log_probs_l = F.log_softmax(logits_f32, dim=-1)
    probs_l = F.softmax(logits_f32, dim=-1)

    target_idx = torch.tensor(target_token_ids, dtype=torch.long, device=logits_f32.device)
    step_idx = torch.arange(r_len, device=logits_f32.device)

    tok_log_probs_l = log_probs_l[step_idx, target_idx]
    tok_probs_l = probs_l[step_idx, target_idx]

    # Predictive entropy H_l = -sum P_l log P_l
    safe_p_log_p = torch.where(probs_l > 0, probs_l * log_probs_l, torch.zeros_like(probs_l))
    entropies_l = (-safe_p_log_p.sum(dim=-1)).clamp(min=0.0)

    # KL(P_final || P_l) = sum P_final * (log P_final - log P_l)
    kl_terms = torch.where(
        final_probs > 1e-12,
        final_probs * (final_log_probs - log_probs_l),
        torch.zeros_like(final_probs),
    )
    kl_final_to_l = kl_terms.sum(dim=-1).clamp(min=0.0)

    # Symmetric Jensen-Shannon divergence JSD(P_final, P_l)
    m_probs = 0.5 * (final_probs + probs_l)
    m_log_probs = torch.log(m_probs.clamp(min=1e-12))
    kl_f_m = torch.where(final_probs > 1e-12, final_probs * (final_log_probs - m_log_probs), torch.zeros_like(final_probs)).sum(dim=-1)
    kl_l_m = torch.where(probs_l > 1e-12, probs_l * (log_probs_l - m_log_probs), torch.zeros_like(probs_l)).sum(dim=-1)
    js_div = (0.5 * (kl_f_m + kl_l_m)).clamp(min=0.0)

    mean_lp = float(tok_log_probs_l.mean().item())
    min_lp = float(tok_log_probs_l.min().item())
    mean_p = float(tok_probs_l.mean().item())
    mean_ent = float(entropies_l.mean().item())
    max_ent = float(entropies_l.max().item())

    lp_gain = float((final_target_log_probs - tok_log_probs_l).mean().item())
    p_gain = float((final_target_probs - tok_probs_l).mean().item())

    return {
        f"l{layer_idx}_mean_log_prob": mean_lp,
        f"l{layer_idx}_min_log_prob": min_lp,
        f"l{layer_idx}_mean_token_prob": mean_p,
        f"l{layer_idx}_mean_entropy": mean_ent,
        f"l{layer_idx}_max_entropy": max_ent,
        f"l{layer_idx}_kl_final_to_layer": float(kl_final_to_l.mean().item()),
        f"l{layer_idx}_js_div_to_final": float(js_div.mean().item()),
        f"l{layer_idx}_log_prob_gain": lp_gain,
        f"l{layer_idx}_prob_gain": p_gain,
    }


def compute_geometry_metrics(
    layer_hidden_prompt: torch.Tensor,
    layer_hidden_resp: torch.Tensor,
    final_hidden_resp_mean: Optional[torch.Tensor],
    layer_idx: int,
) -> Dict[str, float]:
    """
    Compute hidden-state representation geometry metrics for one intermediate layer:
      - Mean & std L2 norm of response hidden states
      - Prompt-response mean cosine similarity
      - Token-to-token consecutive hidden cosine smoothness across response tokens
      - Cosine similarity between layer l and final layer response mean
    """
    h_resp = layer_hidden_resp.float()
    h_prompt = layer_hidden_prompt.float()

    norms = torch.linalg.vector_norm(h_resp, ord=2, dim=-1)
    norm_mean = float(norms.mean().item()) if norms.numel() > 0 else 0.0
    norm_std = float(norms.std(unbiased=True).item()) if norms.numel() > 1 else 0.0

    resp_mean = h_resp.mean(dim=0)
    prompt_mean = h_prompt.mean(dim=0)

    pr_cos = float(
        F.cosine_similarity(prompt_mean.unsqueeze(0), resp_mean.unsqueeze(0), dim=-1).item()
    )

    if h_resp.size(0) > 1:
        tok_cos = F.cosine_similarity(h_resp[:-1], h_resp[1:], dim=-1)
        tok_cos_mean = float(tok_cos.mean().item())
    else:
        tok_cos_mean = 1.0

    metrics = {
        f"l{layer_idx}_hidden_norm_mean": norm_mean,
        f"l{layer_idx}_hidden_norm_std": norm_std,
        f"l{layer_idx}_prompt_resp_cos": pr_cos,
        f"l{layer_idx}_tok_cos_smoothness": tok_cos_mean,
    }

    if final_hidden_resp_mean is not None:
        final_cos = float(
            F.cosine_similarity(
                resp_mean.unsqueeze(0),
                final_hidden_resp_mean.float().unsqueeze(0),
                dim=-1,
            ).item()
        )
        metrics[f"l{layer_idx}_cos_to_final"] = final_cos

    return metrics


def get_intermediate_scalar_feature_names(layer_indices: Sequence[int]) -> List[str]:
    """Return deterministic ordered list of intermediate logit-lens + geometry scalar feature names."""
    names: List[str] = []
    for l_idx in layer_indices:
        names.extend([
            f"l{l_idx}_mean_log_prob",
            f"l{l_idx}_min_log_prob",
            f"l{l_idx}_mean_token_prob",
            f"l{l_idx}_mean_entropy",
            f"l{l_idx}_max_entropy",
            f"l{l_idx}_kl_final_to_layer",
            f"l{l_idx}_js_div_to_final",
            f"l{l_idx}_log_prob_gain",
            f"l{l_idx}_prob_gain",
            f"l{l_idx}_hidden_norm_mean",
            f"l{l_idx}_hidden_norm_std",
            f"l{l_idx}_prompt_resp_cos",
            f"l{l_idx}_tok_cos_smoothness",
            f"l{l_idx}_cos_to_final",
        ])
    return names


def extract_example_intermediate_signals(
    hidden_states: Sequence[torch.Tensor],
    logits: torch.Tensor,
    p_len: int,
    r_len: int,
    target_token_ids: Sequence[int],
    final_norm: Optional[nn.Module],
    lm_head: Optional[nn.Module],
    layer_indices: Sequence[int],
    batch_idx: int = 0,
    collect_all_layer_pools: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]]]:
    """
    Extract intermediate logit-lens + geometry scalar features (and optionally
    all-layer pooled hidden states) from an already-computed model forward pass.

    Causal alignment:
      - Prompt hidden states: positions [0 .. p_len - 1]
      - Response token hidden states: positions [p_len .. p_len + r_len - 1]
      - Next-token prediction logits/hidden states for response tokens:
        positions [p_len - 1 .. p_len + r_len - 2]
      - Pre-generation transition token: position p_len - 1
    """
    total_len = p_len + r_len
    num_layers = len(hidden_states) - 1

    # Final layer prediction distribution over response tokens
    final_resp_logits = logits[batch_idx, p_len - 1 : total_len - 1, :].float()
    final_log_probs = F.log_softmax(final_resp_logits, dim=-1)
    final_probs = F.softmax(final_resp_logits, dim=-1)

    target_idx = torch.tensor(target_token_ids, dtype=torch.long, device=final_resp_logits.device)
    step_idx = torch.arange(r_len, device=final_resp_logits.device)
    final_target_log_probs = final_log_probs[step_idx, target_idx]
    final_target_probs = final_probs[step_idx, target_idx]

    final_resp_h_mean = hidden_states[-1][batch_idx, p_len:total_len, :].float().mean(dim=0)

    scalar_features: Dict[str, float] = {}
    for l_idx in layer_indices:
        if l_idx < 1 or l_idx >= len(hidden_states):
            raise ValueError(f"Layer index {l_idx} out of bounds for {num_layers} layers.")

        h_layer = hidden_states[l_idx][batch_idx]  # (seq_len, hidden_dim)
        h_pred_slice = h_layer[p_len - 1 : total_len - 1, :]
        h_prompt = h_layer[:p_len, :]
        h_resp = h_layer[p_len:total_len, :]

        layer_logits = _apply_final_norm_and_lm_head(
            h_pred_slice,
            final_norm=final_norm,
            lm_head=lm_head,
            already_normed=(l_idx == num_layers),
        )
        if layer_logits is not None:
            ll_metrics = compute_logit_lens_metrics(
                layer_logits=layer_logits,
                final_probs=final_probs,
                final_log_probs=final_log_probs,
                final_target_log_probs=final_target_log_probs,
                final_target_probs=final_target_probs,
                target_token_ids=target_token_ids,
                layer_idx=l_idx,
            )
            scalar_features.update(ll_metrics)

        geom_metrics = compute_geometry_metrics(
            layer_hidden_prompt=h_prompt,
            layer_hidden_resp=h_resp,
            final_hidden_resp_mean=final_resp_h_mean,
            layer_idx=l_idx,
        )
        scalar_features.update(geom_metrics)

    pooled_dict: Optional[Dict[str, np.ndarray]] = None
    if collect_all_layer_pools:
        # Collect float16 pooled representations across layers [1 .. num_layers]
        # Shapes: (num_layers, hidden_dim)
        resp_means = []
        resp_lasts = []
        prompt_lasts = []
        for l_idx in range(1, num_layers + 1):
            h_l = hidden_states[l_idx][batch_idx]
            resp_means.append(h_l[p_len:total_len, :].float().mean(dim=0).half().cpu().numpy())
            resp_lasts.append(h_l[total_len - 1, :].half().cpu().numpy())
            prompt_lasts.append(h_l[p_len - 1, :].half().cpu().numpy())

        pooled_dict = {
            "resp_mean": np.stack(resp_means, axis=0),
            "resp_last": np.stack(resp_lasts, axis=0),
            "prompt_last": np.stack(prompt_lasts, axis=0),
        }

    return scalar_features, pooled_dict


def extract_single_intermediate_features(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    layer_indices: Optional[Sequence[int]] = None,
    probe_layers: Optional[Sequence[int]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Run a single forward pass with output_hidden_states=True for live inference,
    returning both intermediate logit-lens/geometry scalar features and pooled
    hidden vectors for the specified probe layers.
    """
    try:
        model_dev = next(model.parameters()).device
        if device is None or torch.device(device) != model_dev:
            device = model_dev
    except (StopIteration, AttributeError):
        if device is None:
            device = torch.device("cpu")

    prompt_ids = _deduplicate_leading_bos(
        tokenizer.encode(prompt, add_special_tokens=True),
        tokenizer,
    )
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    p_len = len(prompt_ids)
    r_len = len(response_ids)
    if p_len == 0:
        raise ValueError("Prompt must contain at least one token.")

    final_norm, lm_head, num_layers, _ = resolve_model_components(model)
    if layer_indices is None:
        layer_indices = get_intermediate_layer_indices(num_layers)
    if probe_layers is None:
        probe_layers = list(layer_indices)

    if r_len == 0:
        scalar_defaults = {k: 0.0 for k in get_intermediate_scalar_feature_names(layer_indices)}
        return scalar_defaults, {}

    full_input_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(full_input_ids, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    logits = outputs.logits

    scalar_features, _ = extract_example_intermediate_signals(
        hidden_states=hidden_states,
        logits=logits,
        p_len=p_len,
        r_len=r_len,
        target_token_ids=response_ids,
        final_norm=final_norm,
        lm_head=lm_head,
        layer_indices=layer_indices,
        batch_idx=0,
        collect_all_layer_pools=False,
    )

    total_len = p_len + r_len
    pooled_vectors: Dict[str, np.ndarray] = {}
    for l_idx in probe_layers:
        h_l = hidden_states[l_idx][0]
        pooled_vectors[f"resp_mean_l{l_idx}"] = h_l[p_len:total_len, :].float().mean(dim=0).cpu().numpy()
        pooled_vectors[f"resp_last_l{l_idx}"] = h_l[total_len - 1, :].float().cpu().numpy()
        pooled_vectors[f"prompt_last_l{l_idx}"] = h_l[p_len - 1, :].float().cpu().numpy()

    return scalar_features, pooled_vectors


def extract_dataset_intermediate_representations(
    input_signals_path: str = "experiments/baselines/supervised_signals_combined.parquet",
    output_dir: str = "experiments/models/qwen3_5_0_8b/intermediate",
    model_id: str = "Qwen/Qwen3.5-0.8B",
    device: str = "cuda:0",
    batch_size: int = 16,
    depth_fractions: Sequence[float] = DEFAULT_DEPTH_FRACTIONS,
    limit: Optional[int] = None,
    seed: int = 42,
    checkpoint_interval: int = 400,
    resume: bool = True,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    token: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[int]]:
    """
    Batched extraction of intermediate logit-lens/geometry scalar signals and all-layer
    pooled hidden states across the benchmark dataset with length-sorted batching and
    checkpoint/resume support.

    Saves:
      - intermediate_signals_combined.parquet / .csv
      - hidden_states_pooled.npz  (ids, resp_mean, resp_last, prompt_last, layer_indices)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_parquet(input_signals_path)
    if limit is not None and limit < len(df_in):
        df_in = df_in.sample(n=limit, random_state=seed).reset_index(drop=True)
    else:
        df_in = df_in.reset_index(drop=True)

    final_parquet = out_path / "intermediate_signals_combined.parquet"
    final_csv = out_path / "intermediate_signals_combined.csv"
    final_npz = out_path / "hidden_states_pooled.npz"
    ckpt_parquet = out_path / "intermediate_signals_checkpoint.parquet"
    ckpt_npz = out_path / "hidden_states_checkpoint.npz"

    if model is None or tokenizer is None:
        from src.generation.generate_signals import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(model_id=model_id, device=device, token=token)

    final_norm, lm_head, num_layers, _ = resolve_model_components(model)
    layer_indices = get_intermediate_layer_indices(num_layers, depth_fractions)
    logger.info(
        f"Extracting intermediate representations for '{model_id}' "
        f"(num_layers={num_layers}, logit_lens_layers={layer_indices}, batch_size={batch_size})..."
    )

    # Check if already complete
    if resume and final_parquet.exists() and final_npz.exists():
        df_existing = pd.read_parquet(final_parquet)
        if len(df_existing) == len(df_in) and set(df_existing["id"].astype(str)) == set(df_in["id"].astype(str)):
            logger.info(f"Reusing complete intermediate outputs from '{out_path}'.")
            npz_data = np.load(final_npz, allow_pickle=False)
            pools = {k: npz_data[k] for k in npz_data.files}
            return df_existing, pools, layer_indices

    # Resume partial checkpoint if present
    completed_records: List[Dict[str, Any]] = []
    completed_ids: Set[str] = set()
    pool_resp_mean: List[np.ndarray] = []
    pool_resp_last: List[np.ndarray] = []
    pool_prompt_last: List[np.ndarray] = []

    if resume and ckpt_parquet.exists() and ckpt_npz.exists():
        try:
            df_ck = pd.read_parquet(ckpt_parquet)
            npz_ck = np.load(ckpt_npz, allow_pickle=False)
            if len(df_ck) == len(npz_ck["ids"]):
                completed_records = df_ck.to_dict(orient="records")
                completed_ids = set(df_ck["id"].astype(str))
                pool_resp_mean = [npz_ck["resp_mean"][i] for i in range(len(df_ck))]
                pool_resp_last = [npz_ck["resp_last"][i] for i in range(len(df_ck))]
                pool_prompt_last = [npz_ck["prompt_last"][i] for i in range(len(df_ck))]
                logger.info(f"Resumed {len(completed_ids)}/{len(df_in)} examples from checkpoint.")
        except Exception as exc:
            logger.warning(f"Ignoring corrupt checkpoint ({exc}); starting fresh.")
            completed_records, completed_ids = [], set()
            pool_resp_mean, pool_resp_last, pool_prompt_last = [], [], []

    # Prepare remaining items and sort by sequence length for minimal padding waste
    pending_items: List[Tuple[int, str, str, int, str, List[int], List[int], List[int]]] = []
    for idx, row in df_in.iterrows():
        ex_id = str(row["id"])
        if ex_id in completed_ids:
            continue
        prompt_text = str(row["model_input"]) if "model_input" in row and pd.notna(row["model_input"]) else str(row["prompt"])
        resp_text = str(row["response"]).strip()
        p_ids = _deduplicate_leading_bos(tokenizer.encode(prompt_text, add_special_tokens=True), tokenizer)
        r_ids = tokenizer.encode(resp_text, add_special_tokens=False)
        pending_items.append((
            idx,
            ex_id,
            str(row["source_dataset"]),
            int(row["label"]),
            resp_text,
            p_ids,
            r_ids,
            p_ids + r_ids,
        ))

    pending_items.sort(key=lambda x: len(x[7]))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    target_dev = next(model.parameters()).device

    model.eval()
    t0 = time.perf_counter()
    processed_since_ckpt = 0
    newly_processed = 0
    max_tokens_per_batch = max(512, batch_size * 160)

    cursor = 0
    while cursor < len(pending_items):
        # Dynamic token-budget batching to prevent VRAM spikes on longer sequences
        chunk = [pending_items[cursor]]
        cursor += 1
        max_len = len(chunk[0][7])
        while cursor < len(pending_items) and len(chunk) < batch_size:
            cand_len = len(pending_items[cursor][7])
            new_max = max(max_len, cand_len)
            if new_max * (len(chunk) + 1) > max_tokens_per_batch:
                break
            chunk.append(pending_items[cursor])
            max_len = new_max
            cursor += 1

        b_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long, device=target_dev)
        b_mask = torch.zeros((len(chunk), max_len), dtype=torch.long, device=target_dev)
        for bi, item in enumerate(chunk):
            full_ids = item[7]
            b_ids[bi, : len(full_ids)] = torch.tensor(full_ids, dtype=torch.long, device=target_dev)
            b_mask[bi, : len(full_ids)] = 1

        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        logits = outputs.logits

        for bi, (_, ex_id, source, label, _, p_ids, r_ids, _) in enumerate(chunk):
            p_len = len(p_ids)
            r_len = len(r_ids)
            scalar_feats, pooled = extract_example_intermediate_signals(
                hidden_states=hidden_states,
                logits=logits,
                p_len=p_len,
                r_len=r_len,
                target_token_ids=r_ids,
                final_norm=final_norm,
                lm_head=lm_head,
                layer_indices=layer_indices,
                batch_idx=bi,
                collect_all_layer_pools=True,
            )
            completed_records.append({
                "id": ex_id,
                "source_dataset": source,
                "label": label,
                **scalar_feats,
            })
            completed_ids.add(ex_id)
            assert pooled is not None
            pool_resp_mean.append(pooled["resp_mean"])
            pool_resp_last.append(pooled["resp_last"])
            pool_prompt_last.append(pooled["prompt_last"])
            processed_since_ckpt += 1
            newly_processed += 1

        del outputs, hidden_states, logits, b_ids, b_mask

        done = len(completed_records)
        if done % 200 < len(chunk) or done == len(df_in):
            elapsed = time.perf_counter() - t0
            rate = newly_processed / max(elapsed, 1e-6)
            eta = (len(df_in) - done) / max(rate, 1e-6)
            logger.info(f"Extracted [{done:04d}/{len(df_in):04d}] ({rate:.1f} ex/s, ETA {eta:.0f}s)")

        if processed_since_ckpt >= checkpoint_interval and done < len(df_in):
            df_ck = pd.DataFrame(completed_records)
            df_ck.to_parquet(ckpt_parquet, index=False)
            np.savez(
                ckpt_npz,
                ids=np.array([r["id"] for r in completed_records]),
                resp_mean=np.stack(pool_resp_mean, axis=0),
                resp_last=np.stack(pool_resp_last, axis=0),
                prompt_last=np.stack(pool_prompt_last, axis=0),
            )
            processed_since_ckpt = 0
            logger.info(f"Checkpoint saved at {done} examples.")

    # Align final row order back to df_in order
    id_to_pos = {str(r["id"]): i for i, r in enumerate(completed_records)}
    ordered_indices = [id_to_pos[str(ex_id)] for ex_id in df_in["id"].astype(str)]

    df_out = pd.DataFrame([completed_records[i] for i in ordered_indices])
    pools_out = {
        "ids": np.array(df_out["id"].astype(str).tolist()),
        "labels": df_out["label"].to_numpy(dtype=np.int64),
        "sources": np.array(df_out["source_dataset"].astype(str).tolist()),
        "resp_mean": np.stack([pool_resp_mean[i] for i in ordered_indices], axis=0),
        "resp_last": np.stack([pool_resp_last[i] for i in ordered_indices], axis=0),
        "prompt_last": np.stack([pool_prompt_last[i] for i in ordered_indices], axis=0),
        "logit_lens_layers": np.array(layer_indices, dtype=np.int64),
    }

    df_out.to_parquet(final_parquet, index=False)
    df_out.to_csv(final_csv, index=False)
    np.savez(final_npz, **pools_out)
    logger.info(f"Saved intermediate scalar signals to '{final_parquet}' and pooled hidden states to '{final_npz}'.")

    for p in (ckpt_parquet, ckpt_npz):
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    return df_out, pools_out, layer_indices
