import os

from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional, Dict
import numpy as np
import random
import pickle
import hashlib
import tempfile
import errno
from multiprocessing.pool import Pool
from tqdm import tqdm
from math import exp, fsum

Hyperedge = Tuple[int, ...]

class IndexSet:
    __slots__ = ("arr", "pos")
    def __init__(self):
        self.arr: List[int] = []
        self.pos: Dict[int, int] = {}

    def __len__(self):
        return len(self.arr)

    def add(self, x: int):
        if x in self.pos:
            return
        self.pos[x] = len(self.arr)
        self.arr.append(x)

    def remove(self, x: int):
        i = self.pos.pop(x)
        last = self.arr[-1]
        if i != len(self.arr) - 1:
            self.arr[i] = last
            self.pos[last] = i
        self.arr.pop()

    def sample_uniform(self, rng: random.Random) -> int:
        i = rng.randrange(len(self.arr))
        return self.arr[i]

@dataclass
class FullSplitParams:
    beta: float
    v: float
    mu: float
    r: float
    h: float
    lam: float = 1.0
    split_rule: str = "split-1"

    seed: Optional[int] = None

@dataclass
class SimResult:
    Y_count: List[int]

def make_pow_table(max_m: int, exp_: float) -> np.ndarray:

    p = np.zeros(max_m + 1, dtype=float)
    if max_m >= 1:
        p[1:] = np.power(np.arange(1, max_m + 1, dtype=float), exp_)
    return p

class FullSplitHOSIS_Gillespie:

    def __init__(
        self,
        N: int,
        E: Iterable[Iterable[int]],
        params: FullSplitParams,
        X0: Optional[Iterable[int]] = None,
        Y0: Optional[Iterable[int]] = None,
    ):
        self.N = int(N)
        self.params = params
        self.rng = random.Random(params.seed)
        self.np_rng = np.random.default_rng(params.seed)

        self.beta = float(params.beta)
        self.v = float(params.v)
        self.mu = float(params.mu)
        self.r = float(params.r)
        self.h = float(params.h)
        self.lambda_bias = float(params.lam)
        self.split_rule = str(params.split_rule)

        if self.split_rule not in ("split-1", "split-2"):
            raise ValueError(f"Unknown split_rule={self.split_rule!r}. Use 'split-1' or 'split-2'.")

        self.state = np.zeros(self.N, dtype=np.int8)
        if Y0 is not None:
            for u in Y0:
                self.state[int(u)] = 1
        if X0 is not None:
            for u in X0:
                self.state[int(u)] = 0

        self.hyperedges: List[Hyperedge] = []
        for raw_edge in E:
            edge = tuple(sorted(map(int, raw_edge)))
            if len(edge) != len(set(edge)):
                raise ValueError(f"Hyperedges must not contain duplicate nodes: {edge!r}")
            if any(u < 0 or u >= self.N for u in edge):
                raise ValueError(f"Hyperedge contains a node outside [0, {self.N}): {edge!r}")
            if len(edge) > self.N:
                raise ValueError(f"Hyperedge size {len(edge)} exceeds N={self.N}")
            self.hyperedges.append(edge)
        self.M = len(self.hyperedges)
        self.node2edges: List[IndexSet] = [IndexSet() for _ in range(self.N)]
        self.edge_size = np.zeros(self.M, dtype=np.int32)

        for ei, e in enumerate(self.hyperedges):
            self.edge_size[ei] = len(e)
            for u in e:
                self.node2edges[u].add(ei)

        self.max_m = int(self.edge_size.max()) if self.M > 0 else 0

        self.size_to_edges: Dict[int, IndexSet] = {}
        for ei in range(self.M):
            m = int(self.edge_size[ei])
            if m not in self.size_to_edges:
                self.size_to_edges[m] = IndexSet()
            self.size_to_edges[m].add(ei)

        self.j_pow = make_pow_table(self.max_m, self.v)

        self.split_factor_tbl = np.zeros(self.max_m + 1, dtype=float)
        self._rebuild_split_factor_table(old_max=0, new_max=self.max_m)

        self.j_count = np.zeros(self.M, dtype=np.int32)

        for ei, e in enumerate(self.hyperedges):
            self.j_count[ei] = sum(int(self.state[u]) for u in e)

        self.X_nodes = IndexSet()
        self.Y_nodes = IndexSet()
        for u in range(self.N):
            (self.Y_nodes if self.state[u] == 1 else self.X_nodes).add(u)

        self.edge_type_buckets: Dict[Tuple[int, int], IndexSet] = {}
        self.type_keys: List[Tuple[int, int]] = []
        self.type_inf_rates: Dict[Tuple[int, int], float] = {}
        self.type_split_rates: Dict[Tuple[int, int], float] = {}
        for m in sorted(self.size_to_edges):
            self._ensure_type_keys_for_size(m)

        for ei in range(self.M):
            key = (int(self.edge_size[ei]), int(self.j_count[ei]))
            self.edge_type_buckets[key].add(ei)

        for key in self.type_keys:
            self._sync_type_rates(key)
        self._refresh_total_rates()
        self.sum_rec = self.mu * len(self.Y_nodes)

    @staticmethod
    def _safe_exp(x: float) -> float:

        if x > 700.0:
            x = 700.0
        return float(exp(x))

    def _split_factor(self, j: int) -> float:

        if j <= 0:
            return 0.0
        if j <= self.max_m:
            return float(self.split_factor_tbl[j])

        if self.split_rule == "split-1":
            return float(self.h ** (j - 1))
        else:
            return self._safe_exp(self.h * (j - 1))

    def _rebuild_split_factor_table(self, old_max: int, new_max: int):

        if new_max < 0:
            return
        if self.split_factor_tbl.shape[0] != new_max + 1:
            new_tbl = np.zeros(new_max + 1, dtype=float)
            keep = min(old_max, new_max)
            if keep >= 0 and self.split_factor_tbl.size > 0:
                new_tbl[:keep + 1] = self.split_factor_tbl[:keep + 1]
            self.split_factor_tbl = new_tbl

        start = max(1, old_max + 1)
        for j in range(start, new_max + 1):
            if self.split_rule == "split-1":
                self.split_factor_tbl[j] = float(self.h ** (j - 1))
            else:
                self.split_factor_tbl[j] = self._safe_exp(self.h * (j - 1))

        self.split_factor_tbl[0] = 0.0

    def _ensure_type_keys_for_size(self, m: int):

        m = int(m)
        for j in range(m + 1):
            key = (m, j)
            if key not in self.edge_type_buckets:
                self.edge_type_buckets[key] = IndexSet()
                self.type_inf_rates[key] = 0.0
                self.type_split_rates[key] = 0.0
                self.type_keys.append(key)
        self.type_keys.sort()

    def _type_unit_weights(self, key: Tuple[int, int]) -> Tuple[float, float]:

        m, j = key
        if j < 0 or j > m:
            raise ValueError(f"Invalid hyperedge type {key!r}")

        jp = self.j_pow[j] if j <= self.max_m else (j ** self.v)
        wi = self.beta * jp * (m - j)
        ws = self.r * self._split_factor(j) if j >= 1 else 0.0
        return max(0.0, wi), max(0.0, ws)

    def _sync_type_rates(self, key: Tuple[int, int]):

        n_edges = len(self.edge_type_buckets[key])
        wi, ws = self._type_unit_weights(key)
        self.type_inf_rates[key] = float(n_edges) * wi
        self.type_split_rates[key] = float(n_edges) * ws

    def _refresh_total_rates(self):

        self.sum_inf = float(fsum(self.type_inf_rates.values()))
        self.sum_split = float(fsum(self.type_split_rates.values()))

    def _move_edge_type(
        self,
        ei: int,
        old_key: Tuple[int, int],
        new_key: Tuple[int, int],
    ):

        if new_key not in self.edge_type_buckets:
            self._ensure_type_keys_for_size(new_key[0])

        if old_key == new_key:
            bucket = self.edge_type_buckets.get(old_key)
            if bucket is None or ei not in bucket.pos:
                raise RuntimeError(f"Edge {ei} is missing from type bucket {old_key!r}")
            return

        old_bucket = self.edge_type_buckets.get(old_key)
        if old_bucket is None or ei not in old_bucket.pos:
            raise RuntimeError(f"Edge {ei} is missing from type bucket {old_key!r}")

        old_bucket.remove(ei)
        self.edge_type_buckets[new_key].add(ei)
        self._sync_type_rates(old_key)
        self._sync_type_rates(new_key)

    def _pick_edge_from_type_buckets(self, event: str) -> Optional[int]:

        if event == "infection":
            rates = self.type_inf_rates
            total = self.sum_inf
        elif event == "split":
            rates = self.type_split_rates
            total = self.sum_split
        else:
            raise ValueError(f"Unknown event type: {event!r}")

        if total <= 0.0:
            return None

        target = self.rng.random() * total
        last_positive_edge: Optional[int] = None
        for key in self.type_keys:
            block_rate = rates[key]
            bucket = self.edge_type_buckets[key]
            n_edges = len(bucket)
            if block_rate <= 0.0 or n_edges == 0:
                continue

            last_positive_edge = bucket.arr[-1]
            if target < block_rate:
                unit_rate = block_rate / n_edges
                rank = min(int(target / unit_rate), n_edges - 1)
                return bucket.arr[rank]
            target -= block_rate

        tolerance = (
            16.0 * np.finfo(float).eps * max(1, len(self.type_keys)) * abs(total)
        )
        if last_positive_edge is not None and target <= tolerance:
            return last_positive_edge
        raise RuntimeError(
            f"Type-bucket rates do not sum to the cached {event} rate; residual={target}"
        )

    def _update_edge_type(self, ei: int, old_key: Tuple[int, int]):

        new_key = (int(self.edge_size[ei]), int(self.j_count[ei]))
        self._move_edge_type(ei, old_key, new_key)

    def _set_edge_nodes(self, ei: int, new_nodes: Iterable[int]):

        raw_nodes = tuple(int(u) for u in new_nodes)
        if len(raw_nodes) != len(set(raw_nodes)):
            raise ValueError(f"Rewired edge contains duplicate nodes: {raw_nodes!r}")
        if any(u < 0 or u >= self.N for u in raw_nodes):
            raise ValueError(f"Rewired edge contains a node outside [0, {self.N}): {raw_nodes!r}")
        self._replace_edge_nodes(ei, tuple(sorted(raw_nodes)))

    def _replace_edge_nodes(self, ei: int, new_nodes: Hyperedge):

        old_nodes = self.hyperedges[ei]
        old_m = int(self.edge_size[ei])
        new_m = len(new_nodes)

        if new_m > self.N:
            raise ValueError(f"Rewired edge size {new_m} exceeds N={self.N}")

        if old_m == new_m and new_m in (2, 3):
            self._rewire_small_edge(ei, new_nodes)
            return

        old_key = (old_m, int(self.j_count[ei]))
        old_set = set(old_nodes)
        new_set = set(new_nodes)
        for u in old_set - new_set:
            self.node2edges[u].remove(ei)
        for u in new_set - old_set:
            self.node2edges[u].add(ei)

        self.hyperedges[ei] = new_nodes
        self.edge_size[ei] = new_m

        if new_m != old_m:
            if old_m in self.size_to_edges and ei in self.size_to_edges[old_m].pos:
                self.size_to_edges[old_m].remove(ei)
            if new_m not in self.size_to_edges:
                self.size_to_edges[new_m] = IndexSet()
            self.size_to_edges[new_m].add(ei)

            if new_m > self.max_m:
                old_max = self.max_m
                self.max_m = new_m
                self.j_pow = make_pow_table(self.max_m, self.v)

                self._rebuild_split_factor_table(old_max=old_max, new_max=self.max_m)

            self._ensure_type_keys_for_size(new_m)

        if new_m == 2:
            self.j_count[ei] = int(self.state[new_nodes[0]]) + int(self.state[new_nodes[1]])
        elif new_m == 3:
            self.j_count[ei] = (
                int(self.state[new_nodes[0]])
                + int(self.state[new_nodes[1]])
                + int(self.state[new_nodes[2]])
            )
        else:
            self.j_count[ei] = sum(int(self.state[u]) for u in new_nodes)

        self._update_edge_type(ei, old_key)
        self._refresh_total_rates()

    def _rewire_small_edge(self, ei: int, new_nodes: Hyperedge):

        old_nodes = self.hyperedges[ei]
        m = len(new_nodes)
        old_key = (m, int(self.j_count[ei]))

        for u in old_nodes:
            if u not in new_nodes:
                self.node2edges[u].remove(ei)
        for u in new_nodes:
            if u not in old_nodes:
                self.node2edges[u].add(ei)

        self.hyperedges[ei] = new_nodes
        if m == 2:
            self.j_count[ei] = (
                int(self.state[new_nodes[0]]) + int(self.state[new_nodes[1]])
            )
        else:
            self.j_count[ei] = (
                int(self.state[new_nodes[0]])
                + int(self.state[new_nodes[1]])
                + int(self.state[new_nodes[2]])
            )

        self._update_edge_type(ei, old_key)
        self._refresh_total_rates()

    def _sample_edge_node_by_state(self, ei: int, wanted_state: int) -> Optional[int]:

        m = int(self.edge_size[ei])
        j = int(self.j_count[ei])
        count = j if wanted_state == 1 else (m - j)
        if count <= 0:
            return None

        nodes = self.hyperedges[ei]
        if m == 2:
            a, b = nodes
            if count == 1:
                return a if int(self.state[a]) == wanted_state else b
            return nodes[self.rng.randrange(2)]

        if m == 3:
            a, b, c = nodes
            if count == 1:
                if int(self.state[a]) == wanted_state:
                    return a
                return b if int(self.state[b]) == wanted_state else c
            if count == 3:
                return nodes[self.rng.randrange(3)]

            rank = self.rng.randrange(2)
            if int(self.state[a]) != wanted_state:
                return b if rank == 0 else c
            if int(self.state[b]) != wanted_state:
                return a if rank == 0 else c
            return a if rank == 0 else b

        rank = self.rng.randrange(count)
        for u in nodes:
            if int(self.state[u]) == wanted_state:
                if rank == 0:
                    return u
                rank -= 1
        raise RuntimeError(f"Edge {ei} state count is inconsistent with j_count")

    def _S_to_I(self, u: int):
        if self.state[u] != 0:
            return
        self.state[u] = 1
        if u in self.X_nodes.pos:
            self.X_nodes.remove(u)
        self.Y_nodes.add(u)

        for ei in self.node2edges[u].arr:
            old_key = (int(self.edge_size[ei]), int(self.j_count[ei]))
            self.j_count[ei] += 1
            self._update_edge_type(ei, old_key)

        self._refresh_total_rates()
        self.sum_rec = self.mu * len(self.Y_nodes)

    def _I_to_S(self, u: int):
        if self.state[u] != 1:
            return
        self.state[u] = 0
        if u in self.Y_nodes.pos:
            self.Y_nodes.remove(u)
        self.X_nodes.add(u)

        for ei in self.node2edges[u].arr:
            old_key = (int(self.edge_size[ei]), int(self.j_count[ei]))
            self.j_count[ei] -= 1
            self._update_edge_type(ei, old_key)

        self._refresh_total_rates()
        self.sum_rec = self.mu * len(self.Y_nodes)

    @staticmethod
    def _sample_range_excluding(
        rng: random.Random,
        stop: int,
        excluded: Iterable[int],
        k: int,
    ) -> List[int]:

        excluded_sorted = sorted({int(x) for x in excluded if 0 <= int(x) < stop})
        available = stop - len(excluded_sorted)
        if k < 0 or k > available:
            raise ValueError(f"Cannot sample k={k} from {available} available values")
        if k == 0:
            return []

        ranks = rng.sample(range(available), k)
        sampled: List[int] = []
        for rank in ranks:
            value = rank
            for blocked in excluded_sorted:
                if blocked <= value:
                    value += 1
                else:
                    break
            sampled.append(value)
        return sampled

    @staticmethod
    def _sample_small_indices(
        rng: random.Random,
        stop: int,
        k: int,
    ) -> Tuple[int, ...]:

        if k < 0 or k > stop:
            raise ValueError(f"Cannot sample k={k} from range({stop})")
        if k == 0:
            return ()
        if k == 1:
            return (rng.randrange(stop),)
        if k == 2:
            first = rng.randrange(stop)
            second = rng.randrange(stop - 1)
            if second >= first:
                second += 1
            return (first, second)
        return tuple(rng.sample(range(stop), k))

    @staticmethod
    def _rank_excluding(rank: int, excluded_sorted: Tuple[int, ...]) -> int:

        value = rank
        for blocked in excluded_sorted:
            if blocked <= value:
                value += 1
            else:
                break
        return value

    @staticmethod
    def _sorted_small_nodes(nodes: List[int]) -> Tuple[int, ...]:

        n = len(nodes)
        if n == 1:
            return (nodes[0],)
        if n == 2:
            a, b = nodes
            return (a, b) if a <= b else (b, a)
        if n == 3:
            a, b, c = nodes
            if a > b:
                a, b = b, a
            if b > c:
                b, c = c, b
            if a > b:
                a, b = b, a
            return (a, b, c)
        return tuple(sorted(nodes))

    def _sample_new_edge_nodes(self, m: int, anchor: int) -> Tuple[int, ...]:
        if m < 1:
            raise ValueError(f"Hyperedge size must be positive, got m={m}")
        if m > self.N:
            raise ValueError(f"Cannot form a size-{m} hyperedge from N={self.N} nodes")

        anchor = int(anchor)
        if anchor not in self.Y_nodes.pos:
            raise ValueError(f"Anchor node {anchor} must be infected")
        slots = m - 1
        if slots <= 0:
            return (anchor,)

        N_I = len(self.Y_nodes)
        N_S = len(self.X_nodes)
        lam = float(self.lambda_bias)

        denom = lam * N_I + N_S
        if denom > 0.0:
            p_I = (lam * N_I) / float(denom)
            p_I = max(0.0, min(1.0, p_I))
        else:
            p_I = 0.0

        k_I_rest = int(self.np_rng.binomial(slots, p_I))
        k_S_rest = slots - k_I_rest

        if m <= 3:
            chosen_small = [anchor]

            n_I_available = max(0, N_I - 1)
            kI_eff = min(k_I_rest, n_I_available)
            if kI_eff > 0:
                anchor_pos = self.Y_nodes.pos[anchor]
                virtual_positions = self._sample_small_indices(
                    self.rng, n_I_available, kI_eff
                )
                for virtual_pos in virtual_positions:
                    actual_pos = (
                        virtual_pos if virtual_pos < anchor_pos else virtual_pos + 1
                    )
                    chosen_small.append(self.Y_nodes.arr[actual_pos])

            kS_eff = min(k_S_rest, N_S)
            if kS_eff > 0:
                for pos in self._sample_small_indices(self.rng, N_S, kS_eff):
                    chosen_small.append(self.X_nodes.arr[pos])

            if len(chosen_small) < m:
                need = m - len(chosen_small)
                excluded = self._sorted_small_nodes(chosen_small)
                available = self.N - len(excluded)
                for rank in self._sample_small_indices(self.rng, available, need):
                    chosen_small.append(self._rank_excluding(rank, excluded))

            return self._sorted_small_nodes(chosen_small)

        chosen = {anchor}

        n_I_available = max(0, N_I - 1)
        kI_eff = min(k_I_rest, n_I_available)
        if kI_eff > 0:
            anchor_pos = self.Y_nodes.pos[anchor]
            chosen_I_pos = self._sample_range_excluding(
                self.rng, N_I, (anchor_pos,), kI_eff
            )
            chosen.update(self.Y_nodes.arr[pos] for pos in chosen_I_pos)

        kS_eff = min(k_S_rest, N_S)
        if kS_eff > 0:
            chosen.update(self.rng.sample(self.X_nodes.arr, kS_eff))

        if len(chosen) < m:
            need = m - len(chosen)
            chosen.update(
                self._sample_range_excluding(self.rng, self.N, chosen, need)
            )

        return tuple(sorted(chosen))

    def _full_split(self, ei: int) -> bool:
        j = int(self.j_count[ei])
        m = int(self.edge_size[ei])
        if j < 1:
            return False

        anchor = self._sample_edge_node_by_state(ei, 1)
        if anchor is None:
            raise RuntimeError(f"Split edge {ei} has j={j} but no infected anchor")

        new_nodes = self._sample_new_edge_nodes(m, anchor)
        if m in (2, 3):
            self._rewire_small_edge(ei, new_nodes)
        else:
            self._replace_edge_nodes(ei, new_nodes)
        return True

    def run(
        self,
        record_step: int = 100,
        max_steps: Optional[int] = 5000000,
    ) -> SimResult:

        if record_step <= 0:
            raise ValueError("record_step must be a positive integer")

        Yc = len(self.Y_nodes)
        Ys = [Yc]
        last_recorded_step = 0
        steps = 0

        while Yc > 0:
            if (max_steps is not None) and (steps >= max_steps):
                break

            A0 = self.sum_inf + self.sum_split + self.sum_rec
            if A0 <= 0.0:
                break

            steps += 1
            u_rand = self.rng.random()
            p_inf = self.sum_inf / A0
            p_rec = self.sum_rec / A0

            if u_rand < p_inf:
                ei = self._pick_edge_from_type_buckets("infection")
                if ei is not None:
                    u = self._sample_edge_node_by_state(ei, 0)
                    if u is None:
                        raise RuntimeError(
                            f"Infection edge {ei} has positive rate but no susceptible node"
                        )
                    if int(self.state[u]) != 0:
                        raise RuntimeError(
                            f"Infection edge {ei} selected a non-susceptible node {u}"
                        )
                    self._S_to_I(u)

            elif u_rand < p_inf + p_rec:
                if len(self.Y_nodes) > 0:
                    u = self.Y_nodes.sample_uniform(self.rng)
                    if self.state[u] == 1:
                        self._I_to_S(u)

            else:
                ei = self._pick_edge_from_type_buckets("split")
                if ei is not None:
                    self._full_split(ei)

            Yc = len(self.Y_nodes)

            if steps % record_step == 0:
                Ys.append(Yc)
                last_recorded_step = steps

        if last_recorded_step != steps:
            Ys.append(Yc)

        return SimResult(Y_count=Ys)

_GLOBAL_HYPEREDGES = None

def _init_worker(hyperedges_list):
    global _GLOBAL_HYPEREDGES
    _GLOBAL_HYPEREDGES = hyperedges_list

def _normalize_run_config(cfg):
    max_steps_cfg = cfg.get('max_steps', 5000000)
    seed_num_cfg = cfg.get('seed_num', None)
    normalized = {
        'N': int(cfg['N']),
        'beta': float(cfg['beta']),
        'v': float(cfg['v']),
        'mu': float(cfg['mu']),
        'r': float(cfg['r']),
        'h': float(cfg['h']),
        'lam': float(cfg.get('lam', 1.0)),
        'split_rule': str(cfg.get('split_rule', 'split-1')),
        'seed_method': str(cfg['seed_method']),
        'seed_frac': float(cfg.get('seed_frac', 0.01)),
        'seed_num': None if seed_num_cfg is None else int(seed_num_cfg),
        'times': int(cfg['times']),
        'record_step': int(cfg.get('record_step', 100)),
        'last_k_values': int(cfg.get('last_k_values', 100)),
        'max_steps': None if max_steps_cfg is None else int(max_steps_cfg),
        'network_num': int(cfg.get('network_num', 0)),
        'job_id': cfg['job_id'],
    }
    if normalized['N'] <= 0:
        raise ValueError("N must be a positive integer")
    if normalized['record_step'] <= 0:
        raise ValueError("record_step must be a positive integer")
    if normalized['last_k_values'] <= 0:
        raise ValueError("last_k_values must be a positive integer")
    if normalized['times'] <= 0:
        raise ValueError("times must be a positive integer")
    return normalized

def _result_filename(cfg):
    return (
        f"N={cfg['N']}"
        f"_beta={cfg['beta']}"
        f"_v={cfg['v']}"
        f"_mu={cfg['mu']}"
        f"_r={cfg['r']}"
        f"_h={cfg['h']}"
        f"_lam={cfg['lam']}"
        f"_split_rule={cfg['split_rule']}"
        f"_network_num={cfg['network_num']}"
        f"_seed_meth={cfg['seed_method']}"
        f"_seed_frac={cfg['seed_frac']}"
        f"_seed_num={cfg['seed_num']}"
        f"_record_step={cfg['record_step']}"
        f"_last_k={cfg['last_k_values']}"
        f"_max_steps={cfg['max_steps']}"
        f"_times={cfg['times']}.pkl"
    )

def _short_result_filename(cfg):

    original_name = _result_filename(cfg)
    digest = hashlib.blake2b(
        os.fsencode(original_name),
        digest_size=16,
        person=b'sim-filename-v1',
    ).hexdigest()
    return f"sim_{digest}.pkl"

def _is_filename_too_long_error(exc: OSError) -> bool:

    return exc.errno == errno.ENAMETOOLONG or getattr(exc, 'winerror', None) == 206

def _derive_repeat_seeds(master_seed: int, cfg, repeat_index: int) -> Tuple[int, int]:

    identity = (
        cfg['N'], cfg['beta'], cfg['v'], cfg['mu'], cfg['r'], cfg['h'],
        cfg['lam'], cfg['split_rule'], cfg['seed_method'], cfg['seed_frac'],
        cfg['seed_num'], cfg['network_num'], int(repeat_index), int(master_seed),
    )
    digest = hashlib.blake2b(
        repr(identity).encode('utf-8'),
        digest_size=16,
        person=b'sim-repeat-v1',
    ).digest()
    return (
        int.from_bytes(digest[:8], byteorder='little', signed=False),
        int.from_bytes(digest[8:], byteorder='little', signed=False),
    )

def _draw_initial_seeds(cfg, hyperedges_list, init_seed: int):
    rng_local = random.Random(init_seed)
    N = cfg['N']
    seed_method = cfg['seed_method']
    if seed_method == 'random':
        k = max(1, int(round(cfg['seed_frac'] * N)))
        return set(rng_local.sample(range(N), k))
    if seed_method == 'one_edge':
        return set(rng_local.choice(hyperedges_list))
    if seed_method == 'random_node':
        if cfg['seed_num'] is None:
            raise ValueError("seed_num must be provided for 'random_node'")
        return set(rng_local.sample(range(N), int(cfg['seed_num'])))
    raise ValueError(f"Unknown seed_method: {seed_method}")

def _run_single_repeat(cfg, hyperedges_list, init_seed: int, sim_seed: int):
    seeds = _draw_initial_seeds(cfg, hyperedges_list, init_seed)
    params = FullSplitParams(
        beta=cfg['beta'],
        v=cfg['v'],
        mu=cfg['mu'],
        r=cfg['r'],
        h=cfg['h'],
        lam=cfg['lam'],
        split_rule=cfg['split_rule'],
        seed=sim_seed,
    )
    sim = FullSplitHOSIS_Gillespie(cfg['N'], hyperedges_list, params, Y0=seeds)
    result = sim.run(record_step=cfg['record_step'], max_steps=cfg['max_steps'])
    count_dtype = np.min_scalar_type(cfg['N'])
    return np.asarray(result.Y_count, dtype=count_dtype)

def _run_repeat_task(task):
    cfg, repeat_index, init_seed, sim_seed = task
    if _GLOBAL_HYPEREDGES is None:
        raise RuntimeError("Worker hyperedges have not been initialized")
    try:
        counts = _run_single_repeat(
            cfg, _GLOBAL_HYPEREDGES, init_seed=init_seed, sim_seed=sim_seed
        )
    except Exception as exc:
        raise RuntimeError(
            f"Simulation failed for job_id={cfg['job_id']}, repeat={repeat_index}"
        ) from exc
    return cfg['job_id'], repeat_index, counts

def _build_output_result(cfg, count_series_list):
    Y_frac_ts_list = []
    final_Y_frac_list = []
    for counts in count_series_list:
        Y_series = np.asarray(counts, dtype=float) / cfg['N']
        Y_frac_ts_list.append(Y_series)
        tail_len = min(cfg['last_k_values'], Y_series.size)
        tail = Y_series[-tail_len:]
        if tail[-1] == 0.0:
            final_Y_frac_list.append(0.0)
        else:
            final_Y_frac_list.append(float(tail.mean()))

    return {
        'N': cfg['N'],
        'beta': cfg['beta'],
        'v': cfg['v'],
        'mu': cfg['mu'],
        'r': cfg['r'],
        'h': cfg['h'],
        'lam': cfg['lam'],
        'split_rule': cfg['split_rule'],
        'seed_method': cfg['seed_method'],
        'seed_frac': cfg['seed_frac'],
        'seed_num': cfg['seed_num'],
        'times': cfg['times'],
        'record_step': cfg['record_step'],
        'last_k_values': cfg['last_k_values'],
        'max_steps': cfg['max_steps'],
        'network_num': cfg['network_num'],
        'job_id': cfg['job_id'],
        'final_Y_frac_list': final_Y_frac_list,
    }

def _write_parameter_result(output_dir, cfg, count_series_list):
    if len(count_series_list) != cfg['times']:
        raise ValueError(
            f"Expected {cfg['times']} repeats, received {len(count_series_list)}"
        )
    output_res = _build_output_result(cfg, count_series_list)
    os.makedirs(output_dir, exist_ok=True)
    original_name = _result_filename(cfg)
    final_path = os.path.join(output_dir, original_name)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=output_dir, prefix='.sim-', suffix='.tmp', delete=False
        ) as handle:
            temp_path = handle.name
            pickle.dump(output_res, handle)
        try:
            os.replace(temp_path, final_path)
        except OSError as exc:
            if not _is_filename_too_long_error(exc):
                raise
            short_name = _short_result_filename(cfg)
            final_path = os.path.join(output_dir, short_name)
            os.replace(temp_path, final_path)
            print(
                f"Output filename is too long "
                f"({len(os.fsencode(original_name))} bytes); saved as {short_name}"
            )
        temp_path = None
    except BaseException:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return final_path

def run_one(output_dir, cfg):

    if _GLOBAL_HYPEREDGES is None:
        raise RuntimeError("Hyperedges have not been initialized")
    normalized = _normalize_run_config(cfg)
    master_seed = cfg.get('master_seed')
    if master_seed is None:
        master_seed = random.SystemRandom().getrandbits(128)

    count_series_list = []
    for repeat_index in range(normalized['times']):
        init_seed, sim_seed = _derive_repeat_seeds(
            master_seed, normalized, repeat_index
        )
        count_series_list.append(
            _run_single_repeat(
                normalized, _GLOBAL_HYPEREDGES, init_seed=init_seed, sim_seed=sim_seed
            )
        )

    _write_parameter_result(output_dir, normalized, count_series_list)
    return normalized['job_id']

def run_parameter_jobs_parallel(
    output_dir,
    paras,
    hyperedges_list,
    processes=None,
    master_seed=None,
    show_progress=True,
):

    raw_configs = list(paras)
    configs = [_normalize_run_config(cfg) for cfg in raw_configs]
    if not configs:
        return []

    job_ids = [cfg['job_id'] for cfg in configs]
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Each parameter job must have a unique job_id")

    filenames = [_result_filename(cfg) for cfg in configs]
    if len(set(filenames)) != len(filenames):
        raise ValueError("Multiple parameter jobs would write the same output filename")
    short_filenames = [_short_result_filename(cfg) for cfg in configs]
    if len(set(short_filenames)) != len(short_filenames):
        raise ValueError("Multiple parameter jobs would write the same short output filename")

    if master_seed is None:
        configured_seeds = {
            int(cfg['master_seed'])
            for cfg in raw_configs
            if cfg.get('master_seed') is not None
        }
        if len(configured_seeds) > 1:
            raise ValueError(
                "Parameter jobs specify different master_seed values; pass one explicit seed"
            )
        if configured_seeds:
            master_seed = configured_seeds.pop()
        else:
            master_seed = random.SystemRandom().getrandbits(128)
            print(f"master_seed={master_seed}")
    master_seed = int(master_seed)

    tasks = []
    max_times = max(cfg['times'] for cfg in configs)
    for repeat_index in range(max_times):
        for cfg in configs:
            if repeat_index >= cfg['times']:
                continue
            init_seed, sim_seed = _derive_repeat_seeds(
                master_seed, cfg, repeat_index
            )
            tasks.append((cfg, repeat_index, init_seed, sim_seed))

    if processes is None:
        configured = os.environ.get('SIM_PROCESSES')
        processes = int(configured) if configured else min(os.cpu_count() or 1, 4)
    processes = max(1, min(int(processes), len(tasks)))

    pending = {
        cfg['job_id']: {
            'cfg': cfg,
            'counts': [None] * cfg['times'],
            'received': 0,
        }
        for cfg in configs
    }

    os.makedirs(output_dir, exist_ok=True)
    with Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(hyperedges_list,),
    ) as pool:
        results = pool.imap_unordered(_run_repeat_task, tasks, chunksize=1)
        if show_progress:
            results = tqdm(
                results,
                total=len(tasks),
                mininterval=1.0,
                desc='All repeats',
            )

        for job_id, repeat_index, counts in results:
            item = pending.get(job_id)
            if item is None:
                raise RuntimeError(f"Received an unknown or already completed job_id={job_id}")
            if item['counts'][repeat_index] is not None:
                raise RuntimeError(
                    f"Duplicate result for job_id={job_id}, repeat={repeat_index}"
                )
            item['counts'][repeat_index] = counts
            item['received'] += 1

            if item['received'] == item['cfg']['times']:
                if any(series is None for series in item['counts']):
                    raise RuntimeError(f"Missing repeat for job_id={job_id}")
                _write_parameter_result(
                    output_dir, item['cfg'], item['counts']
                )
                del pending[job_id]

    if pending:
        raise RuntimeError(f"Missing completed parameter jobs: {sorted(pending)}")
    return job_ids
