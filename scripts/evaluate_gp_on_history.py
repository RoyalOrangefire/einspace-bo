#!/usr/bin/env python3
"""
Evaluate PrecomputedKernelGP on *real* validation scores from a saved NAS run.

Loads ``results/<name>.pkl`` (same format as RandomSearch / RE / TreeKernelBO: Individuals
with ``.arch`` and ``.accuracy`` = val_score).

Split strategies:
  * random — shuffle then hold out a fraction (OK for random search--like histories).
  * chronological — first (1 - test_frac) points train, last points test (simulates
    predicting *upcoming* trials from past ones; use for BO/RE order).
  * kfold — K random folds; reports mean ± std of metrics.

Metrics on the test set (or per fold): RMSE, MAE, Pearson r, Spearman rho (rank
correlation; often more meaningful for NAS than raw RMSE).

Example:
  conda activate einspace-bo
  cd einspace-bo
  python scripts/evaluate_gp_on_history.py \\
    --pkl results/search_strategy=rs_dataset=language_....pkl \\
    --split random --test-frac 0.25 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from pickle import load
from typing import Any, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from einspace.tree_kernel_bo import PrecomputedKernelGP, TreeKernelCache

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None


def load_individuals(pkl_path: Path) -> List[Any]:
    with open(pkl_path, "rb") as f:
        obj = load(f)
    if isinstance(obj, list):
        return obj
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "individuals"):
        return list(obj.individuals)
    raise TypeError(f"Unexpected pickle root type: {type(obj)!r}")


def extract_xy(
    individuals: Sequence[Any],
) -> Tuple[List[Any], np.ndarray, np.ndarray]:
    """Return trees, y (accuracy), indices kept (into original list)."""
    trees: List[Any] = []
    ys: List[float] = []
    kept_idx: List[int] = []
    for i, ind in enumerate(individuals):
        acc = getattr(ind, "accuracy", None)
        arch = getattr(ind, "arch", None)
        if arch is None or acc is None:
            continue
        try:
            y = float(acc)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y):
            continue
        trees.append(arch)
        ys.append(y)
        kept_idx.append(i)
    return trees, np.asarray(ys, dtype=float), np.asarray(kept_idx, dtype=int)


def metrics_block(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray | None = None
) -> dict:
    out: dict = {}
    mask = np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    err = yt - yp
    out["rmse"] = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    out["mae"] = float(np.mean(np.abs(err))) if len(err) else float("nan")
    if len(yt) > 2 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        out["pearson_r"] = float(np.corrcoef(yt, yp)[0, 1])
    else:
        out["pearson_r"] = float("nan")
    if spearmanr is not None and len(yt) > 2:
        rho, p = spearmanr(yt, yp)
        out["spearman_rho"] = float(rho) if np.isfinite(rho) else float("nan")
        out["spearman_pvalue"] = float(p) if np.isfinite(p) else float("nan")
    else:
        out["spearman_rho"] = float("nan")
        out["spearman_pvalue"] = float("nan")
    if sigma is not None and np.any(np.isfinite(sigma)):
        sig = sigma[mask]
        sig = np.maximum(sig, 1e-12)
        nll = 0.5 * np.log(2 * np.pi * sig**2) + 0.5 * ((err) / sig) ** 2
        out["mean_nll"] = float(np.mean(nll))
    return out


def _fit_predict(
    train_trees: List[Any],
    train_y: np.ndarray,
    test_trees: List[Any],
    gp_noise: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cache = TreeKernelCache()
    gp = PrecomputedKernelGP(cache, noise=gp_noise)
    gp.fit(train_trees, train_y)
    mu, sig = gp.predict(test_trees)
    return mu, sig


def run_holdout(
    trees: List[Any],
    y: np.ndarray,
    split: str,
    test_frac: float,
    seed: int,
    gp_noise: float,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(trees)
    if n < 4:
        raise ValueError(f"Need at least 4 evaluated architectures; got {n}.")

    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test
    if n_train < 2:
        raise ValueError(f"Too few training points after split: n_train={n_train}.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if split == "random":
        rng.shuffle(idx)
    elif split == "chronological":
        pass  # 0..n-1 order
    else:
        raise ValueError(split)

    test_idx = idx[-n_test:]
    train_idx = idx[:-n_test]

    tr_trees = [trees[i] for i in train_idx]
    tr_y = y[train_idx]
    te_trees = [trees[i] for i in test_idx]
    te_y = y[test_idx]

    try:
        mu, sig = _fit_predict(tr_trees, tr_y, te_trees, gp_noise)
    except np.linalg.LinAlgError as e:
        warnings.warn(f"GP fit failed: {e!r}", stacklevel=2)
        mu = np.full(len(te_y), np.nan)
        sig = np.full(len(te_y), np.nan)

    m = metrics_block(te_y, mu, sig)
    m["n_train"] = int(n_train)
    m["n_test"] = int(n_test)
    m["split"] = split
    return m, te_y, mu, sig, test_idx


def run_kfold(
    trees: List[Any],
    y: np.ndarray,
    k: int,
    seed: int,
    gp_noise: float,
) -> Tuple[dict, List[dict], np.ndarray, np.ndarray]:
    n = len(trees)
    if n < k + 2:
        raise ValueError(f"Need at least k+2={k+2} points for {k}-fold CV; got {n}.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    fold_metrics: List[dict] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for fi in range(k):
        test_idx = folds[fi]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fi])
        tr_trees = [trees[i] for i in train_idx]
        tr_y = y[train_idx]
        te_trees = [trees[i] for i in test_idx]
        te_y = y[test_idx]
        try:
            mu, sig = _fit_predict(tr_trees, tr_y, te_trees, gp_noise)
        except np.linalg.LinAlgError as e:
            warnings.warn(f"Fold {fi} GP failed: {e!r}", stacklevel=2)
            mu = np.full(len(te_y), np.nan)
            sig = np.full(len(te_y), np.nan)
        fold_metrics.append(metrics_block(te_y, mu, sig))
        all_true.append(te_y)
        all_pred.append(mu)

    # Aggregate
    keys = ["rmse", "mae", "pearson_r", "spearman_rho"]
    agg: dict = {"kfold_k": k, "n_total": n}
    for key in keys:
        vals = [float(fm[key]) for fm in fold_metrics if np.isfinite(fm[key])]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        else:
            agg[f"{key}_mean"] = float("nan")
            agg[f"{key}_std"] = float("nan")

    stacked_y = np.concatenate(all_true)
    stacked_p = np.concatenate(all_pred)
    agg["pooled"] = metrics_block(stacked_y, stacked_p, None)
    return agg, fold_metrics, stacked_y, stacked_p


def plot_scatter(out_path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.scatter(yt, yp, s=16, alpha=0.75)
    ax.set_xlabel("True val_score")
    ax.set_ylabel("GP predicted mean")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="GP quality on real val_score history.")
    p.add_argument(
        "--pkl",
        type=str,
        required=True,
        help="Path to results .pkl (list of Individuals)",
    )
    p.add_argument(
        "--split",
        choices=("random", "chronological", "kfold"),
        default="random",
        help="random: shuffled holdout; chronological: last test_frac as test; "
        "kfold: K-fold CV (--k)",
    )
    p.add_argument("--test-frac", type=float, default=0.25, help="Holdout fraction")
    p.add_argument("--k", type=int, default=5, help="Folds for kfold split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gp-noise", type=float, default=1e-4)
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output dir (default: results/gp_real_eval/<timestamp>)",
    )
    args = p.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.is_file():
        alt = _REPO_ROOT / args.pkl
        if alt.is_file():
            pkl_path = alt
        else:
            raise FileNotFoundError(f"Not found: {args.pkl}")

    individuals = load_individuals(pkl_path)
    trees, y, _ = extract_xy(individuals)
    if len(trees) < 4:
        raise SystemExit(
            f"Need >= 4 valid (arch, accuracy) pairs after filtering; got {len(trees)}. "
            "Run random search or RE first to populate results/*.pkl."
        )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out) if args.out else _REPO_ROOT / "results" / "gp_real_eval" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.split == "kfold":
        agg, fold_metrics, pooled_y, pooled_p = run_kfold(
            trees, y, args.k, args.seed, args.gp_noise
        )
        report = {
            "pkl": str(pkl_path),
            "n_used": len(trees),
            "aggregate": agg,
            "per_fold": fold_metrics,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(agg, indent=2))
        plot_scatter(
            out_dir / "scatter_kfold_pooled.png",
            pooled_y,
            pooled_p,
            f"K={args.k}-fold CV pooled predictions vs true",
        )
    else:
        m, te_y, mu, sig, _ = run_holdout(
            trees, y, args.split, args.test_frac, args.seed, args.gp_noise
        )
        report = {
            "pkl": str(pkl_path),
            "n_used": len(trees),
            "metrics": m,
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(report, f, indent=2)
        np.savetxt(
            out_dir / "predictions.csv",
            np.column_stack([te_y, mu, sig]),
            delimiter=",",
            header="y_true,y_pred,sigma",
            comments="",
        )
        plot_scatter(
            out_dir / "scatter_test.png",
            te_y,
            mu,
            f"Test set ({args.split}, frac={args.test_frac})",
        )
        print(json.dumps(m, indent=2))

    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
