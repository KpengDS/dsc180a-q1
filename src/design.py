"""Pure, reproducible crossed demonstration-set/permutation design and metrics."""
from dataclasses import asdict, dataclass
import hashlib
import math
import random
import statistics

from .ag_news import LABELS
from .inference import MODEL, MODEL_REVISION, DATASET_REVISION


@dataclass(frozen=True)
class ExperimentConfig:
    num_sets: int = 2
    num_permutations: int = 3
    num_test: int = 10
    per_class: int = 2
    seed: int = 180
    selection: str = "balanced"
    ordering: str = "random"
    test_sampling: str = "near_balanced"
    max_chars: int = 1000
    std_ddof: int = 0
    device: str = "auto"
    model_id: str = MODEL
    prompt_mode: str = "chat"
    model_revision: str = MODEL_REVISION
    dataset_revision: str = DATASET_REVISION

    def validate(self):
        for name in ("num_sets", "num_permutations", "num_test", "per_class", "max_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.std_ddof not in (0, 1):
            raise ValueError("std_ddof must be 0 or 1")
        if self.num_sets <= self.std_ddof or self.num_permutations <= self.std_ddof:
            raise ValueError("std_ddof must be smaller than both matrix dimensions")
        for name, choices in dict(selection=("balanced", "uniform"), ordering=("random", "canonical"),
                                  test_sampling=("near_balanced", "uniform"), device=("auto", "mps", "cpu")).items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")
        if self.prompt_mode not in ("chat", "completion"):
            raise ValueError("prompt_mode must be chat or completion")
        if self.prompt_mode == "completion" and (self.per_class != 2 or self.selection != "balanced"):
            raise ValueError("Base completion protocol requires balanced eight-shot selection")
        if self.ordering == "canonical" and self.num_permutations != 1:
            raise ValueError("canonical ordering requires num_permutations=1")
        if self.num_permutations > math.factorial(4 * self.per_class):
            raise ValueError("More unique permutations requested than exist")
        return self


def stream_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def canonical_indices(train, indices):
    # Full original strings, case-sensitive Unicode sort; index breaks exact ties.
    return sorted(indices, key=lambda i: (LABELS[train[i]["label"]], train[i]["text"], i))


def make_plan(dataset, config):
    config.validate()
    train, test = dataset["train"], dataset["test"]
    pools = [[i for i, label in enumerate(train["label"]) if label == c] for c in range(4)]
    size = 4 * config.per_class
    if config.selection == "balanced":
        if any(len(pool) < config.per_class for pool in pools):
            raise ValueError("Insufficient training examples per class")
        capacity = math.prod(math.comb(len(pool), config.per_class) for pool in pools)
    else:
        capacity = math.comb(len(train), size) if len(train) >= size else 0
    if config.num_sets > capacity:
        raise ValueError("More distinct demonstration sets requested than available")
    if config.num_test > len(test):
        raise ValueError("Requested test count exceeds test split")
    seeds = {name: stream_seed(config.seed, name) for name in ("selection", "ordering", "test")}
    rng = random.Random(seeds["selection"])
    sets, seen = [], set()
    for _ in range(max(10000, config.num_sets * 100)):
        indices = ([i for pool in pools for i in rng.sample(pool, config.per_class)]
                   if config.selection == "balanced" else rng.sample(range(len(train)), size))
        key = tuple(sorted(indices))
        if key not in seen:
            seen.add(key)
            sets.append(canonical_indices(train, indices))
        if len(sets) == config.num_sets:
            break
    if len(sets) != config.num_sets:
        raise ValueError("Could not sample enough distinct sets; reduce num_sets")
    rng = random.Random(seeds["ordering"])
    if config.ordering == "canonical":
        permutations = [list(range(size))]
    else:
        # Rejection sampling yields unique uniform permutations without enumerating k!.
        permutations, seen = [], set()
        for _ in range(max(10000, config.num_permutations * 100)):
            permutation = tuple(rng.sample(range(size), size))
            if permutation not in seen:
                seen.add(permutation)
                permutations.append(list(permutation))
            if len(permutations) == config.num_permutations:
                break
        if len(permutations) != config.num_permutations:
            raise ValueError("Could not sample enough unique permutations; reduce num_permutations")
    rng = random.Random(seeds["test"])
    if config.test_sampling == "uniform":
        test_ids = rng.sample(range(len(test)), config.num_test)
    else:
        # Remainder classes chosen without label-name bias; no replacement or duplicates.
        extras = set(rng.sample(range(4), config.num_test % 4))
        test_pools = [[i for i, label in enumerate(test["label"]) if label == c] for c in range(4)]
        counts = [config.num_test // 4 + int(c in extras) for c in range(4)]
        if any(len(pool) < count for pool, count in zip(test_pools, counts)):
            raise ValueError("Insufficient test examples for near-balanced sampling")
        test_ids = [i for pool, count in zip(test_pools, counts) for i in rng.sample(pool, count)]
        rng.shuffle(test_ids)
    configurations = [dict(set_id=i, permutation_id=j, canonical_demo_indices=indices,
                           permutation=permutation, ordered_demo_indices=[indices[p] for p in permutation])
                      for i, indices in enumerate(sets) for j, permutation in enumerate(permutations)]
    return dict(config=asdict(config), stream_seeds=seeds, demonstration_sets=sets,
                permutations=permutations, test_indices=test_ids,
                test_label_counts={name: sum(test[i]["label"] == c for i in test_ids)
                                   for c, name in enumerate(LABELS)}, configurations=configurations)


def sensitivities(matrix, ddof=0):
    if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("Accuracy matrix must be nonempty and rectangular")
    if ddof not in (0, 1) or min(len(matrix), len(matrix[0])) <= ddof:
        raise ValueError("Invalid standard deviation degrees of freedom")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for row in matrix for v in row):
        raise ValueError("Accuracies must be finite values in [0, 1]")
    std = statistics.pstdev if ddof == 0 else statistics.stdev
    rows = [std(row) for row in matrix]
    cols = [std(col) for col in zip(*matrix)]
    order, selection = statistics.mean(rows), statistics.mean(cols)
    return dict(accuracy_matrix=matrix, std_ddof=ddof, order_std_by_set=rows,
                selection_std_by_permutation=cols, order_sensitivity=order,
                selection_sensitivity=selection,
                selection_over_order=selection / order if order else None,
                ratio_status="defined" if order else "undefined: zero order sensitivity",
                mean_accuracy=statistics.mean(v for row in matrix for v in row))
