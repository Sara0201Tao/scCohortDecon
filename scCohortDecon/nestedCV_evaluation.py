# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

# Shared experiment defaults.
RANDOM_SEED = 20260513
NESTED_CV_REPEATS = 5
N_PERMUTATIONS = 1000
FIGURE_FORMATS = ("png", "pdf")

# Shared SVM pipeline and hyperparameter grid.
SVM_PIPELINE = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(class_weight="balanced", random_state=RANDOM_SEED)),
    ]
)
_SVM_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
_SVM_GAMMA_GRID = ["scale", "auto", 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]

SVM_PARAM_GRID = [
    {"svc__kernel": ["linear"], "svc__C": _SVM_C_GRID},
    {"svc__kernel": ["rbf"], "svc__C": _SVM_C_GRID, "svc__gamma": _SVM_GAMMA_GRID},
]


# ──────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def make_result_dirs(base_dir: str) -> dict[str, str]:
    dirs = {
        "root": base_dir,
        "svm_tables": base_dir,
        "svm_figures": base_dir,
        "svm_arrays": base_dir,
    }
    ensure_dirs(*dirs.values())
    return dirs


# ──────────────────────────────────────────────────────────────────────────────
# SVM GridSearch helper
# ──────────────────────────────────────────────────────────────────────────────

def make_svm_search(inner_splits: int, cv_seed: int, n_jobs: int = -1) -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=cv_seed)
    return GridSearchCV(
        estimator=SVM_PIPELINE,
        param_grid=SVM_PARAM_GRID,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Primary nested CV unit used by repeated nested CV and permutation tests
# ──────────────────────────────────────────────────────────────────────────────

def run_primary_nested_cv(
    X_df: pd.DataFrame,
    y: pd.Series,
    sample_ids: pd.Series,
    label_order: list[str],
    outer_splits: int,
    cv_seed: int,
    run_label: str,
    n_jobs: int = -1,
) -> dict:
    """
    Run one complete nested-CV pass.

    Outer loop: out-of-fold evaluation.
    Inner loop: SVM hyperparameter selection by GridSearchCV using macro-F1.

    Returns pooled outer-test predictions and metrics across all outer folds.
    """
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=cv_seed)
    y_true, y_pred, records = [], [], []

    for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(X_df, y), start=1):
        X_train = X_df.iloc[train_idx]
        X_test = X_df.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        inner_splits = min(5, int(y_train.value_counts().min()))
        if inner_splits < 2:
            raise ValueError(f"Inner CV cannot be built in {run_label}, fold {fold_id}.")

        search = make_svm_search(inner_splits, cv_seed + fold_id, n_jobs=n_jobs)
        search.fit(X_train, y_train)
        fold_pred = search.best_estimator_.predict(X_test)

        y_true.extend(y_test)
        y_pred.extend(fold_pred)

        for local_i, sample_pos in enumerate(test_idx):
            records.append(
                {
                    "run_label": run_label,
                    "fold": fold_id,
                    "sample_id": sample_ids.iloc[sample_pos],
                    "true_label": y_test.iloc[local_i],
                    "pred_label": fold_pred[local_i],
                    "correct": y_test.iloc[local_i] == fold_pred[local_i],
                    "inner_cv_splits": inner_splits,
                    "inner_cv_best_f1_macro": search.best_score_,
                    "best_params": str(search.best_params_),
                }
            )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "run_label": run_label,
        "cv_seed": cv_seed,
        "y_true": y_true,
        "y_pred": y_pred,
        "records": pd.DataFrame(records),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=label_order,
            average="macro",
            zero_division=0,
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# nested-CV macro-F1 permutation test
# ──────────────────────────────────────────────────────────────────────────────

def run_nested_cv_macro_f1_permutation_test(
    X_df: pd.DataFrame,
    y: pd.Series,
    label_order: list[str],
    observed_macro_f1: float,
    outer_splits: int,
    outer_repeats: int,
    n_permutations: int,
    random_state: int,
    n_jobs: int = -1,
) -> tuple[float, np.ndarray, float]:
    """
    Permutation test for the repeated nested-CV mean macro-F1.

    For each permutation:
        1. Shuffle the labels once; every repeat of this permutation reuses
           that single shuffled label vector.
        2. Run outer_repeats complete nested CVs under those labels, each with
           the same cv_seed schedule as the real-label run (random_state +
           repeat_index, repeat_index from 0).
        3. Record the mean macro-F1 across the outer_repeats repeats.

    Because the observed statistic is also the mean macro-F1 across the
    repeats of the real-label run, both sides of the comparison come from the
    same repeated nested-CV procedure.

    The p-value is computed by comparing permuted mean macro-F1 scores
    against the observed repeated nested-CV mean macro-F1.
    """
    rng = np.random.default_rng(random_state)
    y_values = y.to_numpy()
    perm_seeds = rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=n_permutations,
        endpoint=False,
    )

    def score_one_permutation(seed: int) -> float:
        local_rng = np.random.default_rng(int(seed))
        perm_y = pd.Series(
            local_rng.permutation(y_values),
            index=y.index,
        )

        perm_rep_f1s = []
        for repeat_index in range(outer_repeats):
            result = run_primary_nested_cv(
                X_df=X_df,
                y=perm_y,
                sample_ids=pd.Series(perm_y.index.astype(str), index=perm_y.index),
                outer_splits=outer_splits,
                cv_seed=random_state + repeat_index,
                run_label=f"perm_repeat_{repeat_index + 1:02d}",
                label_order=label_order,
                n_jobs=1,
            )
            perm_rep_f1s.append(result["macro_f1"])

        return float(np.mean(perm_rep_f1s))

    perm_scores = Parallel(n_jobs=n_jobs)(
        delayed(score_one_permutation)(seed)
        for seed in perm_seeds
    )

    perm_scores = np.asarray(perm_scores, dtype=float)
    p_value = (1 + np.sum(perm_scores >= observed_macro_f1)) / (len(perm_scores) + 1)
    return observed_macro_f1, perm_scores, float(p_value) 

# ──────────────────────────────────────────────────────────────────────────────
# Confusion matrix plotting helper
# ──────────────────────────────────────────────────────────────────────────────

def draw_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_order: list[str],
    save_stem: str,
    title: str,
    formats: tuple[str, ...] = FIGURE_FORMATS,
    is_main: bool = False,
) -> dict[str, float]:
    """
    Draw and save one confusion-matrix heatmap.

    is_main=False: the panel title is the given title plus the Accuracy and
        Macro-F1 computed from this single y_true/y_pred pair.
    is_main=True: the given title is used as-is, because the caller already
        built the repeated nested-CV summary title. The returned metrics are
        still this single y_true/y_pred pair's, they never replace the
        caller's summary statistics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    unique_labels = set(pd.Series(y_true).astype(str).unique().tolist())
    labels = [lab for lab in label_order if lab in unique_labels]
    labels += [lab for lab in sorted(unique_labels, key=str) if lab not in set(labels)]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_labels = len(labels)
    fig, ax = plt.subplots(figsize=(max(4, n_labels * 1.9), max(3.6, n_labels * 1.7)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.6,
        linecolor="gray",
        annot_kws={"size": 14, "fontweight": "bold"},
        ax=ax,
        cbar=False,
    )

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    if is_main:
        ax.set_title(f"{title}", fontsize=9, fontweight="bold")
    else:
        ax.set_title(f"{title}\nAccuracy={acc:.3f}; Macro-F1={macro_f1:.3f}", fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()

    for fmt in formats:
        path = f"{save_stem}.{fmt}"
        plt.savefig(path, dpi=300, bbox_inches="tight", format=fmt)
        print(f"Saved {path}")
    plt.close(fig)

    return {"accuracy": float(acc), "macro_f1": float(macro_f1)}


# ──────────────────────────────────────────────────────────────────────────────
# Parameter selection stability summary
# ──────────────────────────────────────────────────────────────────────────────

def _param_stability_summary(
    all_records: pd.DataFrame,
    output_dir: str,
    prefix: str,
    label_order: list[str] | None = None,
):
    """
    Parameter selection stability module.

    Outputs:
    - Table 1: per outer-fold detail across repeated nested CV.
    - Table 2: parameter selection frequency summary.

    Both tables are descriptive stability statistics: they report how often an
    outer fold picked a parameter set, and no global fixed parameter set is
    selected or re-evaluated here.
    """

    def _parse_params(s: str) -> dict:
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return {}

    if label_order is None:
        label_order = sorted(
            pd.Series(all_records["true_label"]).astype(str).unique().tolist()
        )

    # ── Table 1: per-fold detail ──────────────────────────────────────────────
    print(f"Running Table1")
    rows = []
    for (rl, fid), g in all_records.groupby(["run_label", "fold"], sort=True):
        yt = g["true_label"].values
        yp = g["pred_label"].values
        rows.append(
            {
                "repeat": rl,
                "fold": int(fid),
                "n_test": len(g),
                "fold_accuracy": round(accuracy_score(yt, yp), 4),
                "fold_macro_f1": round(
                    f1_score(yt, yp, labels=label_order, average="macro", zero_division=0),
                    4,
                ),
                "inner_cv_best_f1_macro": round(float(g["inner_cv_best_f1_macro"].iloc[0]), 4),
                "best_params": g["best_params"].iloc[0],
            }
        )
    t1 = pd.DataFrame(rows)
    t1_path = os.path.join(output_dir, f"{prefix}_Table1_PerFoldDetail.csv")
    t1.to_csv(t1_path, index=False)

    # ── Table 2: parameter stability ─────────────────────────────────────────
    print(f"Running Table2")
    t1["_pk"] = t1["best_params"].apply(
        lambda s: json.dumps(_parse_params(s), sort_keys=True, default=str)
    )
    t2 = (
        t1.groupby("_pk")
        .agg(
            selection_count=("_pk", "count"),
            mean_fold_macro_f1=("fold_macro_f1", "mean"),
            mean_inner_cv_f1=("inner_cv_best_f1_macro", "mean"),
        )
        .reset_index()
        .rename(columns={"_pk": "params"})
    )
    t2["selection_frequency"] = (t2["selection_count"] / len(t1)).round(4)
    t2["mean_fold_macro_f1"] = t2["mean_fold_macro_f1"].round(4)
    t2["mean_inner_cv_f1"] = t2["mean_inner_cv_f1"].round(4)
    t2 = t2.sort_values(["selection_count"], ascending=False).reset_index(drop=True)
    t2_path = os.path.join(output_dir, f"{prefix}_Table2_ParamStability.csv")
    t2.to_csv(t2_path, index=False)

    print("─" * 70)
    print(f"[Param Stability] {prefix}  ({len(t1)} total CV folds across all repeats)")
    print(f"  Table 1 (per-fold detail)            → {t1_path}")
    print(f"  Table 2 (param selection frequency)  → {t2_path}")
    print("\n  Parameter selection frequency (descriptive stability only):")
    for _, r in t2.head(5).iterrows():
        print(
            f"    {int(r.selection_count):3d}/{len(t1)} ({r.selection_frequency:.0%})"
            f"  fold_F1={r.mean_fold_macro_f1:.3f}"
            f"  inner_F1={r.mean_inner_cv_f1:.3f}"
            f"  → {r.params}"
        )

    return t1, t2


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_svm_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    order: list[str],
    analysis_name: str,
    output_dir: str,
    output_prefix: str,
    outer_repeats: int = NESTED_CV_REPEATS,
    n_permutations: int = N_PERMUTATIONS,
    note: str = "",
) -> dict:
    """
    Run the refined SVM evaluation pipeline.

    Main outputs:
    - repeated nested-CV stability table and summary;
    - repeated nested-CV permutation null distribution;
    - per-sample out-of-fold prediction table across all repeats;
    - one confusion matrix per repeat (PNG);
    - representative main confusion matrix (PNG + SVG) with the repeated
      nested-CV Accuracy / Macro-F1 mean ± SD and the permutation P value;
    - representative nested-CV classification report;
    - parameter selection stability tables (Table 1, Table 2).
    """
    X_df = df[feature_cols].copy()
    y = df[label_col].astype(str).copy()
    sample_ids = df["sample_id"].astype(str).copy()
    label_order = [str(x) for x in order]

    min_class_n = int(y.value_counts().min())
    if min_class_n < 2:
        raise ValueError(f"{analysis_name}: each class needs at least two samples.")
    outer_splits = min(5, min_class_n)
    result_dirs = make_result_dirs(output_dir)

    print("\n" + "=" * 80)
    print(f"Running repeated nested-CV SVM: {analysis_name}")
    print("=" * 80)
    print(f"Class counts:  {y.value_counts().reindex(label_order).to_dict()}")
    print(f"Features used: {len(feature_cols)} ({', '.join(feature_cols)})")
    print(f"Outer CV:      {outer_splits}-fold × {outer_repeats} repeated nested CV")

    # ── Repeated nested CV ───────────────────────────────────────────────────
    nested_results = []
    for repeat_id in range(1, outer_repeats + 1):
        nested_results.append(
            run_primary_nested_cv(
                X_df,
                y,
                sample_ids=sample_ids,
                label_order=label_order,
                outer_splits=outer_splits,
                cv_seed=RANDOM_SEED + repeat_id - 1,
                run_label=f"repeat_{repeat_id:02d}",
                n_jobs=-1,
            )
        )

    stability_df = pd.DataFrame(
        [
            {
                "run_label": item["run_label"],
                "cv_seed": item["cv_seed"],
                "accuracy": item["accuracy"],
                "balanced_accuracy": item["balanced_accuracy"],
                "macro_f1": item["macro_f1"],
            }
            for item in nested_results
        ]
    )
    stability_out = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepeatedNestedCV_stability.csv")
    stability_df.to_csv(stability_out, index=False)

    # Unrounded per-repeat metrics drive the summary statistics and the
    # representative-repeat choice; rounding happens only when displayed.
    rep_accs = np.asarray([item["accuracy"] for item in nested_results], dtype=float)
    rep_f1s = np.asarray([item["macro_f1"] for item in nested_results], dtype=float)
    mean_acc = float(rep_accs.mean())
    std_acc = float(rep_accs.std(ddof=1) if len(rep_accs) > 1 else 0.0)
    mean_f1 = float(rep_f1s.mean())
    std_f1 = float(rep_f1s.std(ddof=1) if len(rep_f1s) > 1 else 0.0)
    rep_idx = int(np.argmin(np.abs(rep_f1s - mean_f1)))
    rep_number = rep_idx + 1
    rep_result = nested_results[rep_idx]
    observed_mean_macro_f1 = mean_f1

    print(
        f"Representative repeat (macro-F1 closest to the repeated nested-CV mean): "
        f"repeat_{rep_number:02d}"
    )

    # ── Repeated nested-CV permutation test ──────────────────────────────────
    print(f"Running nested-CV macro-F1 permutation test (n={n_permutations}, repeats={outer_repeats})...")
    perm_score, perm_scores, perm_p = run_nested_cv_macro_f1_permutation_test(
        X_df=X_df,
        y=y,
        label_order=label_order,
        observed_macro_f1=observed_mean_macro_f1,
        outer_splits=outer_splits,
        outer_repeats=outer_repeats,
        n_permutations=n_permutations,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    perm_null_out = os.path.join(result_dirs["svm_arrays"], f"{output_prefix}_RepeatedNestedCV_permutation_null.npy")
    np.save(perm_null_out, perm_scores)
    print("P value of nested-CV macro-F1 permutation test: "+ str(perm_p))

    # ── Per-sample out-of-fold predictions across all repeats ────────────────
    all_records = pd.concat([r["records"] for r in nested_results], ignore_index=True)
    repeat_ids = []
    for repeat_id, item in enumerate(nested_results, start=1):
        repeat_ids.extend([repeat_id] * len(item["records"]))
    all_records["repeat_id"] = repeat_ids
    all_records["is_representative_repeat"] = all_records["repeat_id"] == rep_number
    prediction_cols = [
        "repeat_id",
        "run_label",
        "fold",
        "sample_id",
        "true_label",
        "pred_label",
        "correct",
        "is_representative_repeat",
        "inner_cv_splits",
        "inner_cv_best_f1_macro",
        "best_params",
    ]
    predictions_out = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepeatedNestedCV_predictions.csv")
    all_records[prediction_cols].to_csv(predictions_out, index=False)
    print(f"[{output_prefix}] Per-sample outer-test predictions → {predictions_out}")

    # ── Main repeated nested-CV summary ──────────────────────────────────────
    summary = {
                "analysis": analysis_name,
                "n_features": len(feature_cols),
                "features": ", ".join(feature_cols),
                "repeated_nested_cv": f"{outer_splits}-fold × {outer_repeats} repeated nested CV",
                "stability_accuracy_mean": mean_acc,
                "stability_accuracy_std": std_acc,
                "stability_balanced_accuracy_mean": stability_df["balanced_accuracy"].mean(),
                "stability_balanced_accuracy_std": stability_df["balanced_accuracy"].std(ddof=1),
                "stability_macro_f1_mean": mean_f1,
                "stability_macro_f1_std": std_f1,
                "permutation_score_mean_macro_f1": perm_score,
                "permutation_p_value": perm_p,
                "n_permutations": n_permutations,
                "permutation_statistic": "mean macro-F1 across repeated nested CV",
                "representative_nested_cv_repeat": f"repeat_{rep_number:02d}",
                "representative_nested_cv_accuracy": rep_result["accuracy"],
                "representative_nested_cv_macro_f1": rep_result["macro_f1"],
                "note": note,
            }
    summary_out = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepeatedNestedCV_summary.txt")
    with open(summary_out, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    # ── Confusion matrices: one per repeat, plus the representative main figure ──
    print("\n" + "=" * 80)
    print(f"Drawing confusion matrices")
    print("=" * 80)
    for repeat_id, item in enumerate(nested_results, start=1):
        repeat_cm_stem = os.path.join(
            result_dirs["svm_figures"],
            f"{output_prefix}_Repeat{repeat_id:02d}_ConfusionMatrix",
        )
        draw_confusion_matrix(
            item["y_true"],
            item["y_pred"],
            label_order,
            repeat_cm_stem,
            title=f"{output_prefix} · nested-CV repeat {repeat_id:02d}",
            formats=("png",),
            is_main=False,
        )

    main_cm_stem = os.path.join(
        result_dirs["svm_figures"],
        f"{output_prefix}_RepresentativeNestedCV_ConfusionMatrix",
    )
    main_cm_title = (
        f"{output_prefix} · repeated nested CV ({outer_splits}-fold × {outer_repeats} repeats)\n"
        f"matrix: representative repeat {rep_number:02d}; "
        f"all repeats: Accuracy={mean_acc:.3f}±{std_acc:.3f}; "
        f"Macro-F1={mean_f1:.3f}±{std_f1:.3f}; permutation P={perm_p:.4f}"
    )
    draw_confusion_matrix(
        rep_result["y_true"],
        rep_result["y_pred"],
        label_order,
        main_cm_stem,
        main_cm_title,
        formats=("png", "svg"),
        is_main=True,
    )

    # ── Representative nested-CV classification report ───────────────────────
    report = classification_report(
        rep_result["y_true"],
        rep_result["y_pred"],
        labels=label_order,
        target_names=label_order,
        zero_division=0,
        output_dict=True,
    )
    report_path = os.path.join(result_dirs["svm_tables"], f"{output_prefix}_RepresentativeNestedCV_ClassificationReport.csv")
    pd.DataFrame(report).T.to_csv(report_path)
    print(f"[{output_prefix}] Representative nested-CV classification report → {report_path}")

    # ── Parameter selection stability summary ────────────────────────────────
    print("\n" + "=" * 80)
    print(f"Running Parameter stability testing")
    print("=" * 80)

    _param_stability_summary(
        all_records,
        output_dir,
        output_prefix,
        label_order=label_order,
    )
