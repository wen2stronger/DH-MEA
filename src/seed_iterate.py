from typing import List, Tuple, Dict, Any
import numpy as np
import torch


class SeedIterator:
    def __init__(self,
                 total_epochs: int = 40,
                 initial_top_percent: float = 0.4,
                 final_top_percent: float = 0.8,
                 base_threshold: float = 0.9,
                 Scheduling: float = 0.1,
                 verbose: bool = True):
        self.total_epochs = total_epochs
        self.initial_top_percent = initial_top_percent
        self.final_top_percent = final_top_percent
        self.base_threshold = base_threshold
        self.Scheduling = Scheduling
        self.verbose = verbose

    # ----------------------------- helpers -----------------------------

    @staticmethod
    def _to_cpu_tensor(x: Any) -> torch.Tensor:
        """Accept numpy/torch (any device) -> torch.FloatTensor on CPU."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        raise TypeError(f"Unsupported type for tensor conversion: {type(x)}")

    @staticmethod
    def _as_tuple_set(pairs: Any) -> set:
        """Convert shape (N,2) array/list into { (a,b), ... } set."""
        if isinstance(pairs, np.ndarray):
            return set(map(tuple, pairs.tolist()))
        return set(map(tuple, pairs))

    # ------------------------- core primitives -------------------------

    def _select_top_pairs(self,
                          cosine: torch.Tensor,
                          entityid1: List[int],
                          entityid2: List[int],
                          epoch: int) -> List[List[int]]:
        """Row-wise argmax match with linear schedule on top_percent."""
        top_percent = self.initial_top_percent + \
                      (self.final_top_percent - self.initial_top_percent) * (epoch / self.total_epochs)
        max_values, max_indices = torch.max(cosine, dim=1)
        seed_pairs = [[int(entityid1[i]), int(entityid2[int(max_indices[i])])]
                      for i in range(len(max_values))]
        # score-sort desc, take top k%
        scored = sorted(((float(max_values[i]), seed_pairs[i]) for i in range(len(max_values))),
                        key=lambda x: x[0], reverse=True)
        k = int(len(scored) * top_percent)
        return [pair for _, pair in scored[:k]]

    def _get_newpairs_with_dynamic_threshold(self,
                                             cosine: torch.Tensor,
                                             entityid1: List[int],
                                             entityid2: List[int],
                                             epoch: int) -> List[List[int]]:
        """Dynamic threshold with gentle schedule; keep best per row > threshold."""
        max_values, max_indices = torch.max(cosine, dim=1)
        th = self.base_threshold - self.Scheduling * (epoch / self.total_epochs)
        th = max(th, torch.mean(max_values).item() - torch.std(max_values).item())
        new_pairs = []
        for i in range(len(entityid1)):
            if float(max_values[i]) > th:
                new_pairs.append([int(entityid1[i]), int(entityid2[int(max_indices[i])])])
        return new_pairs

    @staticmethod
    def _mutual_best(pairs_ab: List[List[int]], pairs_ba: List[List[int]]) -> List[List[int]]:
        """Keep pairs where A->B and B->A agree (mutual best)."""
        set_ba = set((b, a) for a, b in pairs_ba)
        return [p for p in pairs_ab if (p[0], p[1]) in set_ba]

    @staticmethod
    def _intersection(list1: List[List[int]], list2: List[List[int]]) -> List[List[int]]:
        """Intersection of pair lists."""
        s2 = set(map(tuple, list2))
        return [p for p in list1 if tuple(p) in s2]

    # -------------------------- public pipeline --------------------------

    def update(self,
               train_pair: np.ndarray,
               summed_matrix: Any,
               entity1: List[int],
               entity2: List[int],
               all_pairs: np.ndarray,
               epoch: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        One iteration of seed augmentation.

        Args:
          train_pair   : np.ndarray shape (N,2)
          summed_matrix: similarity matrix (torch/numpy), shape (|E1|, |E2|)
          entity1      : list of entity ids in KG1 (row order of summed_matrix)
          entity2      : list of entity ids in KG2 (col order of summed_matrix)
          all_pairs    : np.ndarray of gold pairs (for quick quality check)
          epoch        : current epoch index (0-based)

        Returns:
          updated_train_pair: np.ndarray (existing + new)
          stats: {'added': int, 'pair_precision': float}
        """
        sim = self._to_cpu_tensor(summed_matrix).float()

        # threshold-based mutual best
        th_ab = self._get_newpairs_with_dynamic_threshold(sim, entity1, entity2, epoch)
        th_ba = self._get_newpairs_with_dynamic_threshold(sim.t(), entity2, entity1, epoch)
        th_mutual = self._mutual_best(th_ab, th_ba)

        # rank-based mutual best
        rk_ab = self._select_top_pairs(sim, entity1, entity2, epoch)
        rk_ba = self._select_top_pairs(sim.t(), entity2, entity1, epoch)
        rk_mutual = self._mutual_best(rk_ab, rk_ba)

        # final candidates = intersection of two criteria
        candidates = self._intersection(th_mutual, rk_mutual)

        # remove those already in train pairs
        train_set = self._as_tuple_set(train_pair)
        new_seeds = [p for p in candidates if tuple(p) not in train_set]

        # concat
        if len(new_seeds) > 0:
            updated = np.concatenate([train_pair, np.array(new_seeds, dtype=train_pair.dtype)], axis=0)
        else:
            updated = train_pair

        # quick score against reference
        added = len(new_seeds)
        if added > 0 and all_pairs is not None and len(all_pairs) > 0:
            gold = set(map(tuple, all_pairs.tolist())) if isinstance(all_pairs, np.ndarray) else set(map(tuple, all_pairs))
            hit = sum(1 for p in new_seeds if tuple(p) in gold)
            pair_precision = hit / added
        else:
            pair_precision = 0.0

        if self.verbose:
            print(f"Added {added} new pairs.")
            print(pair_precision)

        return updated, {'added': float(added), 'pair_precision': float(pair_precision)}


# -----------------------------------------------------------------------------
# Backward-compatible wrapper (same signature as your original get_pair)
# -----------------------------------------------------------------------------
def get_pair(train_pair, summed_matrix, entity1, entity2, all_pairs, epoch):
    """
    Drop-in replacement for your original get_pair(...).
    Uses default SeedIterator hyperparameters identical to your codepaths.
    """
    it = SeedIterator(
        total_epochs=40,
        initial_top_percent=0.4,
        final_top_percent=0.8,
        base_threshold=0.9,
        Scheduling=0.1,
        verbose=True
    )
    updated, stats = it.update(train_pair, summed_matrix, entity1, entity2, all_pairs, epoch)
    return updated
