#!/usr/bin/env python3
"""
Validate PrecomputedKernelGP + tree kernel via leave-one-out CV.

Mode ``synthetic`` (default): random nested dicts + oracle targets k_norm(t_i, t0) + noise.
  Fast, no GPU/datasets; checks GP math and kernel code.

Mode ``einspace``: sample architectures from EinSpace + same oracle targets on real trees.
  Validates kernel/GP on grammar-shaped dicts without training networks.

Outputs under results/gp_validation/<run_id>/: metrics.json, predictions.csv, PNG figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np

# Repo root on path when run as `python scripts/validate_tree_gp.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from einspace.tree_kernel_bo import PrecomputedKernelGP, TreeKernelCache


def _synthetic_tree(rng: np.random.Generator, max_depth: int = 5) -> dict:
    """Nested dicts/lists compatible with tree_kernel_bo subtree traversal."""

    def inner(depth: int) -> Any:
        if depth >= max_depth or rng.random() < 0.25:
            return {
                "node_type": "terminal",
                "fn": f"leaf_{int(rng.integers(0, 40))}",
                "tag": int(rng.integers(0, 10000)),
            }
        n = int(rng.integers(1, 4))
        kids = [inner(depth + 1) for _ in range(n)]
        return {
            "node_type": "nonterminal",
            "fn": f"op_{int(rng.integers(0, 12))}",
            "branch_children": kids,
        }

    return inner(0)


def sample_synthetic_trees(n: int, seed: int, max_depth: int) -> List[dict]:
    rng = np.random.default_rng(seed)
    trees: List[dict] = []
    for _ in range(n):
        trees.append(_synthetic_tree(rng, max_depth=max_depth))
    return trees


def sample_einspace_trees(
    n: int,
    seed: int,
    config: dict,
    device_override: str | None = None,
    max_attempts_factor: int = 50,
) -> List[dict]:
    """Sample n valid architectures from EinSpace (same construction as main.py)."""
    import torch

    from einspace.search_spaces import EinSpace
    from einspace.utils import set_seed

    set_seed(int(config["seed"]))

    if device_override is not None:
        device = device_override
    else:
        device = config["device"] if torch.cuda.is_available() else "cpu"

    einspace = EinSpace(
        input_shape=(
            config["batch_size"],
            config["channels"],
            *config["image_size"],
        ),
        input_mode=config["input_mode"],
        num_repeated_cells=config["search_space_num_repeated_cells"],
        device=device,
        computation_module_prob=config["search_space_computation_module_prob"],
        min_module_depth=config["search_space_min_module_depth"],
        max_module_depth=config["search_space_max_module_depth"],
    )

    trees: List[dict] = []
    attempts = 0
    sample_failures = 0
    max_attempts = max(n * max_attempts_factor, 100)
    while len(trees) < n and attempts < max_attempts:
        attempts += 1
        try:
            arch = einspace.sample()
            trees.append(arch)
        except Exception:
            sample_failures += 1
            continue

    if sample_failures:
        print(
            f"[validate_tree_gp] EinSpace.sample: {sample_failures} failed attempts, "
            f"{len(trees)} architectures collected in {attempts} tries.",
            flush=True,
        )

    if len(trees) < n:
        raise RuntimeError(
            f"Only collected {len(trees)}/{n} EinSpace samples after {attempts} attempts."
        )
    return trees


def oracle_targets(
    trees: Sequence[Any],
    cache: TreeKernelCache,
    rng: np.random.Generator,
    noise_std: float,
) -> Tuple[np.ndarray, Any]:
    """y_i = k_norm(t_i, t_oracle) + Normal(0, noise_std^2); t_oracle is first tree."""
    t0 = trees[0]
    y = np.zeros(len(trees), dtype=float)
    for i, t in enumerate(trees):
        y[i] = cache.normalized_kernel(t, t0)
    y = y + rng.normal(0.0, noise_std, size=len(trees))
    return y, t0


def loocv(
    trees: Sequence[Any],
    y: np.ndarray,
    gp_noise: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-out predictions.

    Returns y_true, y_pred, sigma (all same shape [n]).
    """
    n = len(trees)
    y_true = y.astype(float).copy()
    y_pred = np.zeros(n, dtype=float)
    sigma = np.zeros(n, dtype=float)

    for i in range(n):
        idx = [j for j in range(n) if j != i]
        tr = [trees[j] for j in idx]
        yy = y[idx]

        cache = TreeKernelCache()
        gp = PrecomputedKernelGP(cache, noise=gp_noise)
        try:
            gp.fit(tr, yy)
            mu, sig = gp.predict([trees[i]])
        except np.linalg.LinAlgError as e:
            warnings.warn(f"LOOCV fold {i} failed: {e!r}; using nan placeholders.", stacklevel=2)
            y_pred[i] = np.nan
            sigma[i] = np.nan
            continue

        y_pred[i] = float(mu[0])
        sigma[i] = float(sig[0])

    return y_true, y_pred, sigma


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray
) -> dict:
    mask = np.isfinite(y_pred) & np.isfinite(sigma)
    yt = y_true[mask]
    yp = y_pred[mask]
    sig = sigma[mask]

    err = yt - yp
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    if len(yt) > 1 and np.std(yt) > 1e-12 and np.std(yp) > 1e-12:
        r = float(np.corrcoef(yt, yp)[0, 1])
    else:
        r = float("nan")

    # Gaussian NLL in observation space (diagnostic; assumes predicted sigma is calibrated)
    nll_terms = []
    for a, m, s in zip(yt, yp, sig):
        s = max(s, 1e-12)
        nll_terms.append(
            0.5 * np.log(2 * np.pi * s**2) + 0.5 * ((a - m) / s) ** 2
        )
    mean_nll = float(np.mean(nll_terms)) if nll_terms else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": r,
        "mean_nll": mean_nll,
        "n_folds_ok": int(mask.sum()),
        "n_folds_total": int(len(y_true)),
    }


def save_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "y_true", "y_pred", "sigma"])
        for i in range(len(y_true)):
            w.writerow([i, y_true[i], y_pred[i], sigma[i]])


def plot_figures(
    out_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = np.isfinite(y_pred) & np.isfinite(sigma)
    yt = y_true[mask]
    yp = y_pred[mask]
    sig = sigma[mask]
    err = yt - yp

    fig, ax = plt.subplots(figsize=(5, 5))
    lo = float(min(yt.min(), yp.min()))
    hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x")
    ax.scatter(yt, yp, s=12, alpha=0.7)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.set_title("LOOCV: predicted vs true")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_true_vs_pred.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.4)
    ax.scatter(np.arange(len(err)), err, s=10, alpha=0.7)
    ax.set_xlabel("fold / index")
    ax.set_ylabel("y_true - y_pred")
    ax.set_title("Residuals")
    fig.tight_layout()
    fig.savefig(out_dir / "residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sig, np.abs(err), s=12, alpha=0.7)
    ax.set_xlabel("predicted sigma")
    ax.set_ylabel("|error|")
    ax.set_title("Calibration (crude)")
    fig.tight_layout()
    fig.savefig(out_dir / "abs_error_vs_sigma.png", dpi=150)
    plt.close(fig)


def load_config(path: str) -> dict:
    import yaml

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    for key, value in list(config.items()):
        if value == "None":
            config[key] = None
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tree-kernel GP (LOOCV).")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "einspace"),
        default="synthetic",
        help="synthetic: fast random trees; einspace: sample from EinSpace YAML config",
    )
    parser.add_argument("--n", type=int, default=40, help="number of trees")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="max recursion depth for synthetic trees",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/language/tree_bo_language.yaml",
        help="YAML config for einspace mode (EinSpace kwargs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override config device for einspace mode (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--oracle-noise",
        type=float,
        default=0.02,
        help="Gaussian noise std on oracle targets",
    )
    parser.add_argument(
        "--gp-noise",
        type=float,
        default=1e-4,
        help="GP observation noise (diagonal of K)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="output directory (default: results/gp_validation/<timestamp>)",
    )
    args = parser.parse_args()
    if args.n < 2:
        parser.error("LOOCV requires --n >= 2")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out)
        if args.out
        else _REPO_ROOT / "results" / "gp_validation" / run_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    if args.mode == "synthetic":
        trees = sample_synthetic_trees(args.n, args.seed, args.max_depth)
    else:
        cfg_path = _REPO_ROOT / args.config
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        config = load_config(str(cfg_path))
        trees = sample_einspace_trees(
            args.n, args.seed, config, device_override=args.device
        )

    cache = TreeKernelCache()
    y, _oracle = oracle_targets(trees, cache, rng, noise_std=args.oracle_noise)

    y_true, y_pred, sigma = loocv(trees, y, gp_noise=args.gp_noise)
    metrics = compute_metrics(y_true, y_pred, sigma)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                **metrics,
                "mode": args.mode,
                "n": args.n,
                "seed": args.seed,
                "oracle_noise": args.oracle_noise,
                "gp_noise": args.gp_noise,
            },
            f,
            indent=2,
        )

    save_csv(out_dir / "predictions.csv", y_true, y_pred, sigma)
    plot_figures(out_dir, y_true, y_pred, sigma)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
