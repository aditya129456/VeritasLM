"""
Intermediate Layer Probing Training & Evaluation Pipeline.

Trains leak-free hidden-state linear probes across intermediate transformer layers
and combines them with intermediate logit-lens/geometry dynamics and the 19
Universal Core features to retrain Logistic Regression and XGBoost classifiers.

Strictly preserves existing baseline/Universal Core models in models/{slug}/ by
saving new intermediate-layer models to models/{slug}_intermediate/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features.build_universal_features import (
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    UNIVERSAL_CORE_FEATURES,
)
from src.training.combined_models import (
    compute_baseline_comparison,
    evaluate_by_source_dataset,
    evaluate_predictions,
    split_dataset,
)
from src.utils.model_registry import (
    REPO_ROOT,
    get_model_experiment_dir,
    get_model_slug,
    get_universal_features_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("intermediate_models")


def get_intermediate_saved_model_dir(model_id: str) -> Path:
    """
    Resolve isolated directory for serialized intermediate-layer classifiers.
    Keeps original models/{slug}/ untouched.
    """
    slug = get_model_slug(model_id)
    save_dir = REPO_ROOT / "models" / f"{slug}_intermediate"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_intermediate_experiment_dir(model_id: str) -> Path:
    """Resolve experiment output directory for intermediate-layer probing."""
    exp_dir = get_model_experiment_dir(model_id) / "intermediate"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _make_stratified_folds(
    y: np.ndarray,
    sources: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create deterministic stratified CV folds preserving (source_dataset, label)."""
    strat_key = np.array([f"{s}_{lbl}" for s, lbl in zip(sources, y)])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), strat_key))


def _build_probe_pipeline(C: float = 0.05, random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=random_state)),
    ])


def sweep_intermediate_layers(
    pools: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    y_all: np.ndarray,
    sources_all: np.ndarray,
    c_grid: Sequence[float] = (0.005, 0.02, 0.1, 0.5),
    pool_modes: Sequence[str] = ("resp_mean", "resp_last", "prompt_last"),
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate linear probing accuracy and ROC-AUC for every transformer layer [1..L]
    strictly using cross-validation on the training split.
    """
    y_train = y_all[train_indices]
    src_train = sources_all[train_indices]
    folds = _make_stratified_folds(y_train, src_train, n_splits=n_splits, random_state=random_state)

    num_layers = int(pools["resp_mean"].shape[1])
    sweep_records: List[Dict[str, Any]] = []

    for mode in pool_modes:
        if mode not in pools:
            continue
        X_mode = pools[mode][train_indices].astype(np.float32)  # (N_train, L, D)

        for layer_idx in range(1, num_layers + 1):
            X_layer = X_mode[:, layer_idx - 1, :]
            best_c = float(c_grid[0])
            best_auc = -1.0
            best_metrics: Dict[str, float] = {}

            for c_val in c_grid:
                oof_probs = np.zeros(len(y_train), dtype=np.float64)
                for tr_i, val_i in folds:
                    pipe = _build_probe_pipeline(C=float(c_val), random_state=random_state)
                    pipe.fit(X_layer[tr_i], y_train[tr_i])
                    oof_probs[val_i] = pipe.predict_proba(X_layer[val_i])[:, 1]

                oof_preds = (oof_probs >= 0.5).astype(int)
                roc = float(roc_auc_score(y_train, oof_probs))
                acc = float(accuracy_score(y_train, oof_preds))
                f1 = float(f1_score(y_train, oof_preds, zero_division=0))
                pr = float(average_precision_score(y_train, oof_probs))

                if roc > best_auc:
                    best_auc = roc
                    best_c = float(c_val)
                    best_metrics = {
                        "cv_accuracy": acc,
                        "cv_f1": f1,
                        "cv_roc_auc": roc,
                        "cv_pr_auc": pr,
                    }

            sweep_records.append({
                "pool_mode": mode,
                "layer": layer_idx,
                "best_C": best_c,
                **best_metrics,
            })

    df_sweep = pd.DataFrame(sweep_records)
    best_by_mode: Dict[str, Dict[str, Any]] = {}
    for mode in pool_modes:
        sub = df_sweep[df_sweep["pool_mode"] == mode]
        if not sub.empty:
            top_row = sub.sort_values(by=["cv_roc_auc", "cv_accuracy"], ascending=False).iloc[0]
            best_by_mode[mode] = {
                "layer": int(top_row["layer"]),
                "best_C": float(top_row["best_C"]),
                "cv_accuracy": float(top_row["cv_accuracy"]),
                "cv_roc_auc": float(top_row["cv_roc_auc"]),
                "cv_f1": float(top_row["cv_f1"]),
            }

    return {
        "num_layers": num_layers,
        "train_samples": int(len(train_indices)),
        "n_splits": n_splits,
        "best_by_mode": best_by_mode,
        "per_layer_results": sweep_records,
    }


def select_diverse_intermediate_layers(
    sweep_results: Dict[str, Any],
    pool_mode: str = "resp_mean",
    top_k: int = 3,
    min_spacing: int = 3,
    exclude_final_layer: bool = True,
) -> List[Tuple[int, float]]:
    """
    Select top-k well-spaced intermediate layers and their tuned regularization C
    from training CV sweep results.
    """
    num_layers = int(sweep_results["num_layers"])
    rows = [
        r for r in sweep_results["per_layer_results"]
        if r["pool_mode"] == pool_mode and (not exclude_final_layer or r["layer"] < num_layers)
    ]
    rows.sort(key=lambda r: (r["cv_roc_auc"], r["cv_accuracy"]), reverse=True)

    selected: List[Tuple[int, float]] = []
    for r in rows:
        l_idx = int(r["layer"])
        if all(abs(l_idx - s_l) >= min_spacing for s_l, _ in selected):
            selected.append((l_idx, float(r["best_C"])))
            if len(selected) >= top_k:
                break

    selected.sort(key=lambda x: x[0])
    return selected


def fit_and_extract_probe_features(
    pools: Dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    y_all: np.ndarray,
    sources_all: np.ndarray,
    layer_c_specs: Sequence[Tuple[int, str, float]],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fit leak-free intermediate hidden-state linear probes.

    On the training split, produces 5-fold out-of-fold (OOF) decision-function logits
    and probabilities so downstream classifiers never see in-sample probe overfitting.
    Fits final probes on the full training split to score the test split and future queries.

    Parameters:
        layer_c_specs: Sequence of (layer_idx, pool_mode, C).

    Returns:
        Tuple of (train_probe_df, test_probe_df, probe_bundle_dict).
    """
    y_train = y_all[train_indices]
    src_train = sources_all[train_indices]
    folds = _make_stratified_folds(y_train, src_train, n_splits=n_splits, random_state=random_state)

    train_cols: Dict[str, np.ndarray] = {}
    test_cols: Dict[str, np.ndarray] = {}
    fitted_probes: List[Dict[str, Any]] = []

    train_logit_matrix: List[np.ndarray] = []
    test_logit_matrix: List[np.ndarray] = []

    for layer_idx, pool_mode, c_val in layer_c_specs:
        X_full = pools[pool_mode][:, layer_idx - 1, :].astype(np.float32)
        X_tr = X_full[train_indices]
        X_te = X_full[test_indices]

        oof_logits = np.zeros(len(train_indices), dtype=np.float64)
        for tr_i, val_i in folds:
            fold_pipe = _build_probe_pipeline(C=c_val, random_state=random_state)
            fold_pipe.fit(X_tr[tr_i], y_train[tr_i])
            oof_logits[val_i] = fold_pipe.decision_function(X_tr[val_i])

        full_pipe = _build_probe_pipeline(C=c_val, random_state=random_state)
        full_pipe.fit(X_tr, y_train)
        test_logits = full_pipe.decision_function(X_te)

        feat_logit = f"probe_{pool_mode}_l{layer_idx}_logit"
        feat_prob = f"probe_{pool_mode}_l{layer_idx}_prob"

        oof_probs = 1.0 / (1.0 + np.exp(-np.clip(oof_logits, -30.0, 30.0)))
        test_probs = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -30.0, 30.0)))

        train_cols[feat_logit] = oof_logits
        train_cols[feat_prob] = oof_probs
        test_cols[feat_logit] = test_logits
        test_cols[feat_prob] = test_probs

        train_logit_matrix.append(oof_logits)
        test_logit_matrix.append(test_logits)

        fitted_probes.append({
            "layer": int(layer_idx),
            "pool_mode": str(pool_mode),
            "C": float(c_val),
            "logit_feature": feat_logit,
            "prob_feature": feat_prob,
            "pipeline": full_pipe,
        })

    # Add cross-layer probe summary features
    tr_mat = np.stack(train_logit_matrix, axis=1)
    te_mat = np.stack(test_logit_matrix, axis=1)
    train_cols["probe_mean_logit"] = tr_mat.mean(axis=1)
    train_cols["probe_max_logit"] = tr_mat.max(axis=1)
    train_cols["probe_std_logit"] = tr_mat.std(axis=1)
    test_cols["probe_mean_logit"] = te_mat.mean(axis=1)
    test_cols["probe_max_logit"] = te_mat.max(axis=1)
    test_cols["probe_std_logit"] = te_mat.std(axis=1)

    probe_bundle = {
        "probes": fitted_probes,
        "probe_layers": sorted(set(int(p["layer"]) for p in fitted_probes)),
        "pool_modes": sorted(set(str(p["pool_mode"]) for p in fitted_probes)),
        "probe_feature_names": list(train_cols.keys()),
    }

    return pd.DataFrame(train_cols), pd.DataFrame(test_cols), probe_bundle


def score_with_probe_bundle(
    pooled_vectors: Dict[str, np.ndarray],
    probe_bundle: Dict[str, Any],
) -> Dict[str, float]:
    """
    Score a single live inference example's pooled intermediate hidden vectors
    using the serialized layer_probes.joblib bundle.
    """
    out: Dict[str, float] = {}
    logits_list: List[float] = []

    for spec in probe_bundle["probes"]:
        key = f"{spec['pool_mode']}_l{spec['layer']}"
        vec = pooled_vectors.get(key)
        if vec is None:
            raise KeyError(f"Missing pooled hidden vector '{key}' for intermediate probe scoring.")
        x_arr = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        logit = float(spec["pipeline"].decision_function(x_arr)[0])
        prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
        out[spec["logit_feature"]] = logit
        out[spec["prob_feature"]] = prob
        logits_list.append(logit)

    arr = np.asarray(logits_list, dtype=np.float64)
    out["probe_mean_logit"] = float(arr.mean())
    out["probe_max_logit"] = float(arr.max())
    out["probe_std_logit"] = float(arr.std())
    return out


def _tune_meta_models_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    meta_train: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Pipeline, XGBClassifier, Dict[str, Any]]:
    """
    Select regularized Logistic Regression C and XGBoost hyperparameters strictly
    via 5-fold stratified CV on the training split, then refit on full X_train.
    """
    y_arr = y_train.to_numpy(dtype=int)
    src_arr = meta_train["source_dataset"].astype(str).to_numpy()
    folds = _make_stratified_folds(y_arr, src_arr, n_splits=5, random_state=random_state)

    # 1. Tune Logistic Regression C
    lr_candidates = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    best_lr_c = 0.1
    best_lr_score = -1.0
    for c_val in lr_candidates:
        oof = np.zeros(len(X_train), dtype=np.float64)
        for tr_i, val_i in folds:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs", random_state=random_state)),
            ])
            pipe.fit(X_train.iloc[tr_i], y_arr[tr_i])
            oof[val_i] = pipe.predict_proba(X_train.iloc[val_i])[:, 1]
        score = 0.5 * roc_auc_score(y_arr, oof) + 0.5 * accuracy_score(y_arr, (oof >= 0.5).astype(int))
        if score > best_lr_score:
            best_lr_score = score
            best_lr_c = c_val

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=best_lr_c, max_iter=1000, solver="lbfgs", random_state=random_state)),
    ])
    lr_pipeline.fit(X_train, y_arr)

    # 2. Tune XGBoost configuration
    xgb_candidates = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "gamma": 0.1, "reg_lambda": 1.0},
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6, "gamma": 0.2, "reg_lambda": 2.0},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.5, "gamma": 0.2, "reg_lambda": 3.0},
        {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.5, "gamma": 0.3, "reg_lambda": 5.0},
    ]
    best_xgb_params = xgb_candidates[0]
    best_xgb_score = -1.0
    for params in xgb_candidates:
        oof = np.zeros(len(X_train), dtype=np.float64)
        for tr_i, val_i in folds:
            mdl = XGBClassifier(
                **params,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
            mdl.fit(X_train.iloc[tr_i], y_arr[tr_i])
            oof[val_i] = mdl.predict_proba(X_train.iloc[val_i])[:, 1]
        score = 0.5 * roc_auc_score(y_arr, oof) + 0.5 * accuracy_score(y_arr, (oof >= 0.5).astype(int))
        if score > best_xgb_score:
            best_xgb_score = score
            best_xgb_params = params

    xgb_model = XGBClassifier(
        **best_xgb_params,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_arr)

    tuning_info = {
        "best_lr_C": best_lr_c,
        "best_lr_cv_score": float(best_lr_score),
        "best_xgb_params": best_xgb_params,
        "best_xgb_cv_score": float(best_xgb_score),
    }
    return lr_pipeline, xgb_model, tuning_info


def train_intermediate_models_for_model(
    model_id: str = "Qwen/Qwen3.5-0.8B",
    universal_features_path: Optional[Union[str, Path]] = None,
    intermediate_dir: Optional[Union[str, Path]] = None,
    old_results_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    top_k_probe_layers: int = 3,
) -> Dict[str, Any]:
    """
    Run layer-sweep CV on the training split, build leak-free intermediate probe
    features combined with logit-lens/geometry and 19 Universal Core features,
    retrain Logistic Regression and XGBoost, evaluate on the identical 800-sample
    holdout split, and serialize new models to models/{slug}_intermediate/.
    """
    t_start = time.perf_counter()
    slug = get_model_slug(model_id)
    exp_dir = Path(intermediate_dir) if intermediate_dir is not None else get_intermediate_experiment_dir(model_id)
    save_dir = get_intermediate_saved_model_dir(model_id)

    univ_path = Path(universal_features_path) if universal_features_path is not None else get_universal_features_path(model_id)
    inter_parquet = exp_dir / "intermediate_signals_combined.parquet"
    pools_npz = exp_dir / "hidden_states_pooled.npz"

    if not univ_path.exists():
        raise FileNotFoundError(f"Universal features not found: '{univ_path}'")
    if not inter_parquet.exists() or not pools_npz.exists():
        raise FileNotFoundError(
            f"Intermediate representations not found in '{exp_dir}'. "
            "Run src.signals.intermediate_probing.extract_dataset_intermediate_representations first."
        )

    df_univ = pd.read_parquet(univ_path)
    df_inter = pd.read_parquet(inter_parquet)
    npz_data = np.load(pools_npz, allow_pickle=False)
    pools = {k: npz_data[k] for k in npz_data.files}

    # Align everything strictly to df_univ ID order so split_dataset matches baseline identically
    df_univ["id"] = df_univ["id"].astype(str)
    df_inter["id"] = df_inter["id"].astype(str)
    id_order = df_univ["id"].tolist()

    df_inter = df_inter.set_index("id").loc[id_order].reset_index()

    npz_id_map = {str(eid): idx for idx, eid in enumerate(pools["ids"])}
    npz_order = np.array([npz_id_map[eid] for eid in id_order], dtype=np.int64)
    for k in ("ids", "labels", "sources", "resp_mean", "resp_last", "prompt_last"):
        if k in pools:
            pools[k] = pools[k][npz_order]

    # Exact baseline 80/20 stratified split
    X_univ = df_univ[UNIVERSAL_CORE_FEATURES].copy()
    y_all = df_univ["label"].astype(int).copy()
    meta_all = df_univ[METADATA_COLUMNS].copy()

    # Compute deterministic train/test row indices using the exact same split_dataset key
    strat_key = meta_all["source_dataset"].astype(str) + "_" + y_all.astype(str)
    from sklearn.model_selection import train_test_split
    all_idx = np.arange(len(df_univ))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    # Verify identical split IDs vs split_dataset()
    _, _, _, _, meta_tr_ref, meta_te_ref = split_dataset(X_univ, y_all, meta_all, test_size=test_size, random_state=random_state)
    assert meta_all.iloc[train_idx]["id"].tolist() == meta_tr_ref["id"].tolist(), "Train split ID mismatch!"
    assert meta_all.iloc[test_idx]["id"].tolist() == meta_te_ref["id"].tolist(), "Test split ID mismatch!"

    y_np = y_all.to_numpy(dtype=int)
    src_np = meta_all["source_dataset"].astype(str).to_numpy()

    # Step A: Layer-by-layer CV sweep on train_idx ONLY
    logger.info("Running layer-by-layer CV sweep on training split (3,200 rows)...")
    sweep_results = sweep_intermediate_layers(
        pools=pools,
        train_indices=train_idx,
        y_all=y_np,
        sources_all=src_np,
        random_state=random_state,
    )
    with open(exp_dir / "layer_sweep_results.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    for mode, info in sweep_results["best_by_mode"].items():
        logger.info(
            f"  Best [{mode}] -> Layer {info['layer']} (C={info['best_C']}): "
            f"CV Acc={info['cv_accuracy']:.4f}, ROC-AUC={info['cv_roc_auc']:.4f}"
        )

    # Select top diverse intermediate layers for resp_mean, resp_last, and prompt_last
    sel_resp_mean = select_diverse_intermediate_layers(
        sweep_results, pool_mode="resp_mean", top_k=top_k_probe_layers, min_spacing=3, exclude_final_layer=True
    )
    sel_resp_last = select_diverse_intermediate_layers(
        sweep_results, pool_mode="resp_last", top_k=2, min_spacing=4, exclude_final_layer=True
    )
    sel_prompt_last = select_diverse_intermediate_layers(
        sweep_results, pool_mode="prompt_last", top_k=2, min_spacing=4, exclude_final_layer=True
    )

    layer_c_specs: List[Tuple[int, str, float]] = (
        [(l, "resp_mean", c) for l, c in sel_resp_mean]
        + [(l, "resp_last", c) for l, c in sel_resp_last]
        + [(l, "prompt_last", c) for l, c in sel_prompt_last]
    )
    logger.info(f"Selected intermediate layer probe specs: {layer_c_specs}")

    # Step B: Leak-free OOF probe features for train, full-train probe features for test
    df_probe_train, df_probe_test, probe_bundle = fit_and_extract_probe_features(
        pools=pools,
        train_indices=train_idx,
        test_indices=test_idx,
        y_all=y_np,
        sources_all=src_np,
        layer_c_specs=layer_c_specs,
        random_state=random_state,
    )

    # Add logit-lens layer list to probe bundle for live inference alignment
    logit_lens_layers = pools["logit_lens_layers"].tolist() if "logit_lens_layers" in pools else []
    probe_bundle["logit_lens_layers"] = [int(x) for x in logit_lens_layers]
    probe_bundle["model_id"] = model_id

    # Step C: Assemble combined feature matrix (Universal 19 + Intermediate Scalar + Layer Probes)
    inter_scalar_cols = [c for c in df_inter.columns if c not in METADATA_COLUMNS]
    forbidden_overlap = (set(inter_scalar_cols) | set(UNIVERSAL_CORE_FEATURES) | set(df_probe_train.columns)).intersection(
        FORBIDDEN_FEATURE_COLUMNS
    )
    if forbidden_overlap:
        raise ValueError(f"Forbidden columns detected in intermediate feature set: {forbidden_overlap}")

    X_base_train = pd.concat(
        [
            X_univ.iloc[train_idx].reset_index(drop=True),
            df_inter[inter_scalar_cols].iloc[train_idx].reset_index(drop=True),
            df_probe_train.reset_index(drop=True),
        ],
        axis=1,
    )
    X_base_test = pd.concat(
        [
            X_univ.iloc[test_idx].reset_index(drop=True),
            df_inter[inter_scalar_cols].iloc[test_idx].reset_index(drop=True),
            df_probe_test.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_columns = list(X_base_train.columns)
    probe_bundle["feature_columns"] = feature_columns
    probe_bundle["intermediate_scalar_columns"] = inter_scalar_cols

    if not np.isfinite(X_base_train.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in X_base_train.")
    if not np.isfinite(X_base_test.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values in X_base_test.")

    y_train = y_all.iloc[train_idx].reset_index(drop=True)
    y_test = y_all.iloc[test_idx].reset_index(drop=True)
    meta_train = meta_all.iloc[train_idx].reset_index(drop=True)
    meta_test = meta_all.iloc[test_idx].reset_index(drop=True)

    # Step D: Train and tune LR & XGBoost strictly on X_base_train
    logger.info(f"Training new LR & XGBoost models on {len(feature_columns)} features ({len(X_base_train)} train rows)...")
    lr_pipeline, xgb_model, tuning_info = _tune_meta_models_cv(
        X_train=X_base_train,
        y_train=y_train,
        meta_train=meta_train,
        random_state=random_state,
    )

    lr_pred = lr_pipeline.predict(X_base_test)
    lr_prob = lr_pipeline.predict_proba(X_base_test)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)
    lr_subgroups = evaluate_by_source_dataset(y_test, lr_pred, lr_prob, meta_test)

    xgb_pred = xgb_model.predict(X_base_test)
    xgb_prob = xgb_model.predict_proba(X_base_test)[:, 1]
    xgb_metrics = evaluate_predictions(y_test, xgb_pred, xgb_prob)
    xgb_subgroups = evaluate_by_source_dataset(y_test, xgb_pred, xgb_prob, meta_test)

    # Also evaluate pure intermediate-layer linear probe (best single layer on train CV) for diagnostic transparency
    best_single = sweep_results["best_by_mode"].get("resp_mean", {})
    single_probe_metrics: Dict[str, Any] = {}
    if best_single:
        bl = best_single["layer"]
        bc = best_single["best_C"]
        sp_pipe = _build_probe_pipeline(C=bc, random_state=random_state)
        X_sp = pools["resp_mean"][:, bl - 1, :].astype(np.float32)
        sp_pipe.fit(X_sp[train_idx], y_np[train_idx])
        sp_prob = sp_pipe.predict_proba(X_sp[test_idx])[:, 1]
        sp_pred = (sp_prob >= 0.5).astype(int)
        single_probe_metrics = {
            "layer": bl,
            "pool_mode": "resp_mean",
            "C": bc,
            "overall": evaluate_predictions(y_test, sp_pred, sp_prob),
            "subgroups": evaluate_by_source_dataset(y_test, sp_pred, sp_prob, meta_test),
        }

    # Compare against old Universal Core models for this model
    if old_results_path is None:
        model_old_path = REPO_ROOT / "experiments" / "models" / slug / "combined" / "combined_model_results.json"
        if model_old_path.exists():
            old_results_path = model_old_path
        elif "qwen" in slug:
            old_results_path = REPO_ROOT / "experiments" / "baselines" / "combined" / "combined_model_results.json"

    lr_deltas: Dict[str, float] = {}
    xgb_deltas: Dict[str, float] = {}
    old_data: Optional[Dict[str, Any]] = None
    if old_results_path is not None and Path(old_results_path).exists():
        with open(old_results_path, "r", encoding="utf-8") as f:
            old_data = json.load(f)
        old_overall = old_data.get("overall_metrics", old_data)
        if "logistic_regression" in old_overall:
            lr_deltas = compute_baseline_comparison(lr_metrics, old_overall["logistic_regression"])
        if "xgboost" in old_overall:
            xgb_deltas = compute_baseline_comparison(xgb_metrics, old_overall["xgboost"])

    # Save new model artifacts (preserving old models/{slug}/)
    joblib.dump(lr_pipeline, save_dir / "lr_pipeline.joblib")
    joblib.dump(xgb_model, save_dir / "xgb_model.joblib")
    joblib.dump(probe_bundle, save_dir / "layer_probes.joblib")

    # Save combined feature table (train OOF + test)
    df_train_full = pd.concat([meta_train, X_base_train], axis=1)
    df_train_full["split"] = "train"
    df_test_full = pd.concat([meta_test, X_base_test], axis=1)
    df_test_full["split"] = "test"
    df_features_all = pd.concat([df_train_full, df_test_full], axis=0, ignore_index=True)
    df_features_all.to_parquet(exp_dir / "intermediate_features.parquet", index=False)

    # Save test predictions
    preds_df = pd.DataFrame({
        "id": meta_test["id"].to_numpy(),
        "source_dataset": meta_test["source_dataset"].to_numpy(),
        "label": y_test.to_numpy(),
        "lr_probability": lr_prob,
        "lr_prediction": lr_pred,
        "xgb_probability": xgb_prob,
        "xgb_prediction": xgb_pred,
    })
    preds_df.to_parquet(exp_dir / "intermediate_test_predictions.parquet", index=False)
    preds_df.to_csv(exp_dir / "intermediate_test_predictions.csv", index=False)

    elapsed_s = time.perf_counter() - t_start
    results = {
        "model_id": model_id,
        "model_slug": slug,
        "training_time_s": round(elapsed_s, 2),
        "train_samples": int(len(X_base_train)),
        "test_samples": int(len(X_base_test)),
        "total_features": len(feature_columns),
        "feature_breakdown": {
            "universal_core": len(UNIVERSAL_CORE_FEATURES),
            "intermediate_logit_lens_and_geometry": len(inter_scalar_cols),
            "intermediate_hidden_probes": len(df_probe_train.columns),
        },
        "selected_probe_specs": [
            {"layer": l, "pool_mode": m, "C": c} for l, m, c in layer_c_specs
        ],
        "logit_lens_layers": probe_bundle["logit_lens_layers"],
        "tuning_info": tuning_info,
        "best_single_layer_probe": single_probe_metrics,
        "overall_metrics": {
            "logistic_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
        "subgroup_breakdown_by_source": {
            "logistic_regression": lr_subgroups,
            "xgboost": xgb_subgroups,
        },
        "delta_vs_old_universal_core": {
            "logistic_regression": lr_deltas,
            "xgboost": xgb_deltas,
        },
        "artifacts": {
            "saved_model_dir": str(save_dir),
            "lr_pipeline": str(save_dir / "lr_pipeline.joblib"),
            "xgb_model": str(save_dir / "xgb_model.joblib"),
            "layer_probes": str(save_dir / "layer_probes.joblib"),
            "experiment_dir": str(exp_dir),
        },
    }

    with open(exp_dir / "intermediate_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save CSV summary
    summary_rows = []
    for m_name, m_dict, d_dict in [
        ("logistic_regression", lr_metrics, lr_deltas),
        ("xgboost", xgb_metrics, xgb_deltas),
    ]:
        summary_rows.append({
            "model": m_name,
            "subset": "overall",
            "samples": len(y_test),
            "accuracy": m_dict["accuracy"],
            "precision": m_dict["precision"],
            "recall": m_dict["recall"],
            "f1": m_dict["f1"],
            "roc_auc": m_dict["roc_auc"],
            "pr_auc": m_dict["pr_auc"],
            "brier_score": m_dict["brier_score"],
            "ece": m_dict["ece"],
            "accuracy_delta_vs_old": d_dict.get("accuracy_delta", np.nan),
            "roc_auc_delta_vs_old": d_dict.get("roc_auc_delta", np.nan),
        })
    for m_name, sub_dict in [("logistic_regression", lr_subgroups), ("xgboost", xgb_subgroups)]:
        for ds, m_dict in sub_dict.items():
            summary_rows.append({
                "model": m_name,
                "subset": ds,
                "samples": m_dict["test_samples"],
                "accuracy": m_dict["accuracy"],
                "precision": m_dict["precision"],
                "recall": m_dict["recall"],
                "f1": m_dict["f1"],
                "roc_auc": m_dict["roc_auc"],
                "pr_auc": m_dict["pr_auc"],
                "brier_score": m_dict["brier_score"],
                "ece": m_dict["ece"],
                "accuracy_delta_vs_old": np.nan,
                "roc_auc_delta_vs_old": np.nan,
            })
    pd.DataFrame(summary_rows).to_csv(exp_dir / "intermediate_model_results.csv", index=False)

    logger.info("=" * 72)
    logger.info(f"INTERMEDIATE PROBING RETRAINING COMPLETE ({model_id})")
    logger.info(
        f"New LR  | Acc: {lr_metrics['accuracy']:.4f} ({lr_deltas.get('accuracy_delta', 0):+.4f}) | "
        f"F1: {lr_metrics['f1']:.4f} | ROC-AUC: {lr_metrics['roc_auc']:.4f} ({lr_deltas.get('roc_auc_delta', 0):+.4f})"
    )
    logger.info(
        f"New XGB | Acc: {xgb_metrics['accuracy']:.4f} ({xgb_deltas.get('accuracy_delta', 0):+.4f}) | "
        f"F1: {xgb_metrics['f1']:.4f} | ROC-AUC: {xgb_metrics['roc_auc']:.4f} ({xgb_deltas.get('roc_auc_delta', 0):+.4f})"
    )
    logger.info(f"Saved new models to '{save_dir}' (old models preserved).")
    logger.info("=" * 72)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train LR & XGBoost with Intermediate Layer Probing.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B", help="Target Hugging Face model ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction.")
    parser.add_argument("--top-k-layers", type=int, default=3, help="Number of diverse resp_mean probe layers.")
    args = parser.parse_args()

    train_intermediate_models_for_model(
        model_id=args.model_id,
        random_state=args.seed,
        test_size=args.test_size,
        top_k_probe_layers=args.top_k_layers,
    )


if __name__ == "__main__":
    main()
