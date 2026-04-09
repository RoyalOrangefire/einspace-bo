"""
Tree-kernel Bayesian optimization for einspace.

This module is intentionally self-contained so you can drop it into the
existing repo with as few moving pieces as possible.

Design goals:
- Reuse einspace's existing search-space sampler, mutator, compiler, and trainer.
- Keep architectures in their original tree/dict form instead of flattening them
  into manual feature vectors.
- Build a simple Gaussian Process surrogate directly from a tree kernel matrix.
- Score a discrete candidate pool with Expected Improvement (EI) or UCB.

This version is more robust than the first draft:
- It retries when sampled architectures are invalid.
- It retries when mutated architectures are invalid.
- It retries when compilation/evaluation fails.
- It is designed to survive the kinds of invalid intermediate samples that
  often happen in flexible architecture search spaces like einspace.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from math import erf, pi, sqrt
from os import makedirs
from os.path import exists, join
from pickle import dump, load
from random import choice
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_individual_population = None


def _get_individual_population():
    """Lazy import so PrecomputedKernelGP / tree kernel can be used without torch."""
    global _individual_population
    if _individual_population is None:
        from einspace.search_strategies import Individual, Population

        _individual_population = (Individual, Population)
    return _individual_population


# -----------------------------------------------------------------------------
# Tree utilities
# -----------------------------------------------------------------------------

def _is_tree_node(x: Any) -> bool:
    """Return True if x is a nested structure we want to traverse."""
    return isinstance(x, (dict, list, tuple))


KEY_PRIORITY = [
    "node_type",
    "fn",
    "op",
    "name",
    "type",
    "module",
    "computation_fn",
    "aggregation_fn",
    "branching_fn",
    "prerouting_fn",
    "postrouting_fn",
]

# These keys usually contain bookkeeping / metadata rather than structure.
STRUCTURAL_BLACKLIST = {
    "input_shape",
    "output_shape",
    "input_mode",
    "output_mode",
    "depth",
    "node_id",
    "id",
    "parent_id",
    "num_samples",
}


def _label_from_dict(d: Dict[str, Any]) -> str:
    """
    Extract a readable label from an architecture dict node.

    We prefer semantically meaningful keys first. If none are present,
    we fall back to a sorted list of keys.
    """
    for key in KEY_PRIORITY:
        if key in d:
            value = d[key]
            if hasattr(value, "__name__"):
                return f"{key}:{value.__name__}"
            return f"{key}:{value}"
    if "type" in d:
        return f"type:{d['type']}"
    return "dict{" + ",".join(sorted(map(str, d.keys()))) + "}"


def _node_label(node: Any) -> str:
    """Return a label for a tree node."""
    if isinstance(node, dict):
        return _label_from_dict(node)
    if isinstance(node, list):
        return "list"
    if isinstance(node, tuple):
        return "tuple"
    if hasattr(node, "__name__"):
        return node.__name__
    return str(node)


def _children(node: Any) -> List[Any]:
    """
    Return the structural children of a node.

    For dicts, we traverse nested dict/list/tuple values that are not blacklisted.
    For lists/tuples, we traverse nested structural items.
    """
    children: List[Any] = []

    if isinstance(node, dict):
        for key in sorted(node.keys()):
            if key in STRUCTURAL_BLACKLIST:
                continue
            value = node[key]
            if isinstance(value, dict):
                children.append(value)
            elif isinstance(value, (list, tuple)):
                children.append(value)

    elif isinstance(node, (list, tuple)):
        for item in node:
            if _is_tree_node(item):
                children.append(item)

    return children


def _subtree_signature(node: Any) -> Tuple[Any, ...]:
    """
    Build a canonical recursive signature for a subtree.

    Child signatures are sorted so that structurally equivalent subtrees
    map to the same signature even if the child ordering is not meaningful.
    """
    child_sigs = [_subtree_signature(child) for child in _children(node)]
    child_sigs = tuple(sorted(child_sigs, key=repr))
    return (_node_label(node),) + child_sigs


def _collect_subtrees(node: Any, bag: Counter) -> None:
    """
    Recursively add every subtree signature rooted in this tree to a bag-of-subtrees.
    """
    sig = _subtree_signature(node)
    bag[sig] += 1
    for child in _children(node):
        _collect_subtrees(child, bag)


def subtree_bag(tree: Any) -> Counter:
    """Convert a tree into a multiset (Counter) of subtree signatures."""
    bag: Counter = Counter()
    _collect_subtrees(tree, bag)
    return bag


def tree_signature(tree: Any) -> str:
    """Return a stable, comparable string signature for an architecture tree."""
    return repr(_subtree_signature(tree))


def tree_kernel_from_bags(bag1: Counter, bag2: Counter) -> float:
    """
    A simple subtree-count kernel:
    dot product between subtree count vectors represented as Counters.
    """
    if len(bag1) > len(bag2):
        bag1, bag2 = bag2, bag1

    score = 0.0
    for sig, c1 in bag1.items():
        c2 = bag2.get(sig, 0)
        score += float(c1 * c2)
    return score


class TreeKernelCache:
    """
    Cache bags and self-kernels so repeated GP calls are cheaper.
    """

    def __init__(self) -> None:
        self._bags: Dict[int, Counter] = {}
        self._self_k: Dict[int, float] = {}

    def bag(self, tree: Any) -> Counter:
        key = id(tree)
        if key not in self._bags:
            self._bags[key] = subtree_bag(tree)
        return self._bags[key]

    def raw_kernel(self, t1: Any, t2: Any) -> float:
        return tree_kernel_from_bags(self.bag(t1), self.bag(t2))

    def normalized_kernel(self, t1: Any, t2: Any, eps: float = 1e-12) -> float:
        """
        Normalized kernel:
            k(t1,t2) / sqrt(k(t1,t1) * k(t2,t2))
        """
        k12 = self.raw_kernel(t1, t2)

        key1, key2 = id(t1), id(t2)
        if key1 not in self._self_k:
            self._self_k[key1] = self.raw_kernel(t1, t1)
        if key2 not in self._self_k:
            self._self_k[key2] = self.raw_kernel(t2, t2)

        k11 = self._self_k[key1]
        k22 = self._self_k[key2]
        return float(k12 / sqrt((k11 + eps) * (k22 + eps)))


# -----------------------------------------------------------------------------
# Acquisition utilities
# -----------------------------------------------------------------------------

def _standard_normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / sqrt(2.0 * pi)


def _standard_normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def expected_improvement(
    mu: np.ndarray,
    sigma: np.ndarray,
    best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement for maximization.
    """
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - xi
    z = improvement / sigma
    return improvement * _standard_normal_cdf(z) + sigma * _standard_normal_pdf(z)


def upper_confidence_bound(
    mu: np.ndarray,
    sigma: np.ndarray,
    beta: float = 2.0,
) -> np.ndarray:
    """
    UCB for maximization.
    """
    return mu + beta * sigma


# -----------------------------------------------------------------------------
# GP on precomputed tree kernel
# -----------------------------------------------------------------------------

class PrecomputedKernelGP:
    """
    A tiny GP implementation that works directly from a precomputed kernel.

    We standardize targets, build K, use Cholesky for stability, and
    predict mean / std for candidate trees.
    """

    def __init__(self, kernel_cache: TreeKernelCache, noise: float = 1e-4) -> None:
        self.kernel_cache = kernel_cache
        self.noise = noise
        self.is_fit = False

    def _kernel(self, t1: Any, t2: Any) -> float:
        return self.kernel_cache.normalized_kernel(t1, t2)

    def fit(self, trees: Sequence[Any], y: Sequence[float]) -> "PrecomputedKernelGP":
        self.trees = list(trees)

        y = np.asarray(y, dtype=float)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) if float(y.std()) > 1e-12 else 1.0
        self.y_train = (y - self.y_mean) / self.y_std

        n = len(self.trees)
        K = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i, n):
                kij = self._kernel(self.trees[i], self.trees[j])
                K[i, j] = kij
                K[j, i] = kij

        K = K + self.noise * np.eye(n, dtype=float)

        # Add jitter if needed.
        jitter = 1e-8
        for _ in range(6):
            try:
                self.L = np.linalg.cholesky(K + jitter * np.eye(n, dtype=float))
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            raise np.linalg.LinAlgError(
                "Failed to stabilize kernel matrix for TreeKernel GP."
            )

        v = np.linalg.solve(self.L, self.y_train)
        self.alpha = np.linalg.solve(self.L.T, v)
        self.K = K
        self.is_fit = True
        return self

    def predict(self, test_trees: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fit:
            raise RuntimeError("GP must be fit before predict().")

        mus: List[float] = []
        sigmas: List[float] = []

        for t in test_trees:
            k_star = np.asarray([self._kernel(t, tr) for tr in self.trees], dtype=float)
            mu_std = float(k_star @ self.alpha)

            v = np.linalg.solve(self.L, k_star)
            # Posterior variance of latent f(x*); observation noise is only in K, not in k(x*,x*).
            k_tt = float(self._kernel(t, t))
            var_std = max(k_tt - float(v @ v), 1e-12)

            mus.append(mu_std * self.y_std + self.y_mean)
            sigmas.append(sqrt(var_std) * self.y_std)

        return np.asarray(mus), np.asarray(sigmas)


# -----------------------------------------------------------------------------
# Tree-kernel BO search strategy
# -----------------------------------------------------------------------------

class TreeKernelBO:
    """
    Bayesian optimization over einspace architectures using a tree kernel GP.

    High-level loop:
    1. Warm start with valid evaluated architectures.
    2. Fit a GP surrogate on past (architecture tree, score) pairs.
    3. Generate a candidate pool via random sampling / mutation.
    4. Score candidates with EI or UCB.
    5. Evaluate the chosen candidate with the real objective.
    6. Repeat.

    This class includes retry logic because einspace can sometimes sample
    architectures that are invalid under shape inference / compilation.
    """

    def __init__(
        self,
        search_space,
        compiler,
        evaluation_fn,
        num_samples: int,
        init_num_samples: int,
        candidate_pool_size: int,
        acquisition: str,
        save_name: str,
        continue_search: bool = False,
        mutation_prob: float = 0.7,
        local_tournament_size: int = 5,
        gp_noise: float = 1e-4,
        acq_xi: float = 0.01,
        acq_beta: float = 2.0,
        architecture_seed=None,
    ) -> None:
        self.search_space = search_space
        self.compiler = compiler
        self.evaluation_fn = evaluation_fn

        self.num_samples = int(num_samples)
        self.init_num_samples = int(init_num_samples)
        self.candidate_pool_size = int(candidate_pool_size)

        self.acquisition = acquisition.lower()
        self.save_name = save_name
        self.continue_search = continue_search

        self.mutation_prob = float(mutation_prob)
        self.local_tournament_size = int(local_tournament_size)

        self.gp_noise = float(gp_noise)
        self.acq_xi = float(acq_xi)
        self.acq_beta = float(acq_beta)

        self.architecture_seed = architecture_seed or []
        self.kernel_cache = TreeKernelCache()

        if self.acquisition not in {"ei", "ucb"}:
            raise ValueError("acquisition must be one of {'ei', 'ucb'}")

        _, Population = _get_individual_population()
        results_pkl = join("results", self.save_name + ".pkl")
        if continue_search and exists(results_pkl):
            with open(results_pkl, "rb") as f:
                self.history = self._normalize_history(load(f))
        else:
            self.history = Population([])

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _normalize_history(self, history):
        """
        Normalize persisted history to the Population wrapper used by the
        existing search strategies.
        """
        _, Population = _get_individual_population()
        if isinstance(history, Population):
            return history
        if isinstance(history, list):
            return Population(history)
        if hasattr(history, "tolist"):
            return Population(history.tolist())
        raise TypeError(f"Unsupported persisted history type: {type(history)!r}")

    def _save_history(self) -> None:
        """
        Save search history to results/<save_name>.pkl (same layout as RandomSearch / RE).
        """
        makedirs("results", exist_ok=True)
        pkl_path = join("results", self.save_name + ".pkl")
        parent = "/".join(pkl_path.split("/")[:-1])
        if parent:
            makedirs(parent, exist_ok=True)

        with open(pkl_path, "wb") as f:
            dump(self.history.tolist(), f)

    def _seen_signatures(self, limit: Optional[int] = None) -> set[str]:
        """
        Track recent architecture signatures to cheaply avoid obvious duplicates.
        """
        history = self.history if limit is None else self.history[-limit:]
        return {tree_signature(ind.arch) for ind in history}

    # -------------------------------------------------------------------------
    # Individual creation and robustness
    # -------------------------------------------------------------------------

    def _evaluate_architecture(self, architecture, individual_id: int, parent_id=None):
        """
        Compile and evaluate a single architecture.

        This assumes the architecture is valid. If you want robustness against
        invalid architectures, use _try_create_individual instead.
        """
        modules = self.compiler.compile(architecture)
        best_model = self.evaluation_fn(architecture, modules)

        Individual, _ = _get_individual_population()
        individual = Individual(individual_id, parent_id, architecture, modules)
        individual.accuracy = best_model["val_score"]
        individual.duration = best_model.get("duration", None)
        individual.hpo_dict = {
            key: best_model[key]
            for key in ["lr", "momentum", "weight_decay", "epoch"]
            if key in best_model
        }
        return individual

    def _try_create_individual(
        self,
        individual_id: int,
        parent_id=None,
        architecture=None,
        max_attempts: int = 50,
    ):
        """
        Keep trying until we get a valid evaluated individual.

        Why this exists:
        - einspace can sample invalid architectures
        - shape inference can fail during search_space.sample()
        - mutation can produce invalid architectures
        - compile/evaluation can fail

        If an explicit architecture is passed and it fails, we discard it
        and fall back to fresh resampling on subsequent attempts.
        """
        last_error = None
        candidate_arch = architecture

        for attempt in range(max_attempts):
            try:
                if candidate_arch is None:
                    candidate_arch = self.search_space.sample()

                modules = self.compiler.compile(candidate_arch)
                best_model = self.evaluation_fn(candidate_arch, modules)

                Individual, _ = _get_individual_population()
                individual = Individual(individual_id, parent_id, candidate_arch, modules)
                individual.accuracy = best_model["val_score"]
                individual.duration = best_model.get("duration", None)
                individual.hpo_dict = {
                    key: best_model[key]
                    for key in ["lr", "momentum", "weight_decay", "epoch"]
                    if key in best_model
                }
                return individual

            except Exception as e:
                last_error = e
                print(
                    f"[TreeBO] Invalid architecture attempt "
                    f"{attempt + 1}/{max_attempts}: {repr(e)}"
                )

                # Discard failed candidate and resample next iteration.
                candidate_arch = None
                continue

        raise RuntimeError(
            "Failed to create a valid individual after "
            f"{max_attempts} attempts. Last error: {last_error}"
        )

    # -------------------------------------------------------------------------
    # Warm start and candidate generation
    # -------------------------------------------------------------------------

    def _warm_start(self) -> None:
        """
        Fill initial history with valid evaluated architectures.
        """
        seed_archs = [deepcopy(a) for a in self.architecture_seed]

        while len(self.history) < min(self.init_num_samples, self.num_samples):
            if seed_archs:
                candidate_arch = seed_archs.pop(0)
            else:
                candidate_arch = None

            individual = self._try_create_individual(
                individual_id=len(self.history),
                parent_id=None,
                architecture=candidate_arch,
                max_attempts=100,
            )

            self.history.append(individual)
            self._save_history()

            print(
                f"[TreeBO] Warm-started {len(self.history)}/{self.init_num_samples}: "
                f"val_score={individual.accuracy}"
            )

    def _choose_parent(self):
        """
        Choose a parent from history using a small tournament among past individuals.
        """
        if len(self.history) == 0:
            return None

        k = min(self.local_tournament_size, len(self.history))
        candidates = [choice(self.history) for _ in range(k)]
        return max(candidates, key=lambda ind: ind.accuracy)

    def _propose_candidates(self):
        """
        Propose a candidate pool.

        Some are sampled randomly, others are mutations of good parents.
        Invalid candidate generation is skipped and retried.
        """
        candidates = []
        parent_ids = []

        while len(candidates) < self.candidate_pool_size:
            try:
                do_mutate = (len(self.history) > 0) and (
                    np.random.rand() < self.mutation_prob
                )

                if do_mutate:
                    parent = self._choose_parent()
                    arch = self.search_space.mutate(deepcopy(parent.arch))
                    proposed_parent_id = parent.id
                else:
                    arch = self.search_space.sample()
                    proposed_parent_id = None

                recent_signatures = self._seen_signatures(limit=256)
                candidate_signatures = {tree_signature(candidate) for candidate in candidates}
                arch_signature = tree_signature(arch)
                if arch_signature in recent_signatures or arch_signature in candidate_signatures:
                    continue

                candidates.append(arch)
                parent_ids.append(proposed_parent_id)

            except Exception as e:
                print(f"[TreeBO] Candidate generation failed: {repr(e)}")
                continue

        return candidates, parent_ids

    # -------------------------------------------------------------------------
    # Acquisition scoring
    # -------------------------------------------------------------------------

    def _score_candidates(self, gp, candidates, y_best: float) -> np.ndarray:
        mu, sigma = gp.predict(candidates)

        if self.acquisition == "ei":
            return expected_improvement(
                mu,
                sigma,
                best=y_best,
                xi=self.acq_xi,
            )

        return upper_confidence_bound(
            mu,
            sigma,
            beta=self.acq_beta,
        )

    # -------------------------------------------------------------------------
    # Main search loop
    # -------------------------------------------------------------------------

    def search(self):
        """
        Run Tree-kernel Bayesian optimization.
        """
        if len(self.history) < self.init_num_samples:
            self._warm_start()

        while len(self.history) < self.num_samples:
            # Fit surrogate on evaluated architectures.
            trees = [ind.arch for ind in self.history]
            y = np.asarray([ind.accuracy for ind in self.history], dtype=float)

            gp = PrecomputedKernelGP(
                self.kernel_cache,
                noise=self.gp_noise,
            ).fit(trees, y)

            # Generate and score a candidate pool.
            candidates, parent_ids = self._propose_candidates()
            acq_scores = self._score_candidates(gp, candidates, y_best=float(np.max(y)))

            best_idx = int(np.argmax(acq_scores))
            chosen_arch = candidates[best_idx]
            chosen_parent_id = parent_ids[best_idx]

            # Try to evaluate the chosen candidate. If it fails, fall back to
            # fresh resampling until we get a valid one.
            try:
                individual = self._try_create_individual(
                    individual_id=len(self.history),
                    parent_id=chosen_parent_id,
                    architecture=chosen_arch,
                    max_attempts=20,
                )
            except Exception as e:
                print(f"[TreeBO] Chosen candidate failed: {repr(e)}")
                print("[TreeBO] Falling back to fresh resampling.")

                individual = self._try_create_individual(
                    individual_id=len(self.history),
                    parent_id=None,
                    architecture=None,
                    max_attempts=100,
                )

            self.history.append(individual)
            self._save_history()

            print(
                f"[TreeBO] Sample {len(self.history)}/{self.num_samples}: "
                f"val_score={individual.accuracy} "
                f"parent_id={chosen_parent_id} "
                f"acq_score={float(acq_scores[best_idx]):.6f}"
            )

        return self.history
