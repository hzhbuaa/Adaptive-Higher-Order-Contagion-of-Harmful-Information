import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from math import comb, exp
from scipy.optimize import root

@dataclass
class Params:
    beta: float
    v: float
    mu: float
    r: float
    h: float
    lam: float
    N: int
    M: int
    k_list: List[int]
    E_m_list: List[float]
    rho0: float

    split_rule: str = "split-1"

    method: str = "hybr"
    constraint_tol: float = 1e-6

class Indexer:

    def __init__(self, k_list: List[int]):
        self.k_list = list(k_list)
        self.base: Dict[int, int] = {}
        pos = 1
        for m in self.k_list:
            self.base[m] = pos
            pos += m
        self.dim = pos

    def slice_free(self, m: int) -> slice:
        b = self.base[m]
        return slice(b, b + m)

    def pack(self, N_I: float, l_free: Dict[Tuple[int, int], float]) -> np.ndarray:
        z = np.zeros(self.dim, dtype=float)
        z[0] = float(N_I)
        for m in self.k_list:
            b = self.base[m]
            for k in range(m):
                z[b + k] = float(l_free.get((m, k), 0.0))
        return z

    def unpack(self, z: np.ndarray) -> Tuple[float, Dict[int, np.ndarray]]:
        N_I = float(z[0])
        l_free_map: Dict[int, np.ndarray] = {}
        for m in self.k_list:
            sl = self.slice_free(m)
            l_free_map[m] = np.array(z[sl], dtype=float)
        return N_I, l_free_map

def build_E_map(k_list: List[int], E_m_list: List[float]) -> Dict[int, float]:
    return {int(m): float(Em) for m, Em in zip(k_list, E_m_list)}

def l_get(m: int, k: int,
          l_free_map: Dict[int, np.ndarray],
          E_map: Dict[int, float]) -> float:

    if k < 0 or k > m:
        return 0.0
    if k < m:
        return float(l_free_map[m][k])
    Em = E_map.get(m, 0.0)
    s = float(l_free_map[m].sum()) if m > 0 else 0.0
    return float(Em - s)

def split_factor(k: int, h: float, split_rule: str) -> float:

    if k <= 0:
        return 0.0

    if split_rule == "split-1":
        return float(h ** (k - 1))

    if split_rule == "split-2":

        x = h * (k - 1)
        if x > 700:
            x = 700
        return float(exp(x))

    raise ValueError(f"Unknown split_rule={split_rule!r}. Use 'split-1' or 'split-2'.")

def residual(z: np.ndarray, p: Params, idx: Indexer, E_map: Dict[int, float]) -> np.ndarray:

    N_I, l_free_map = idx.unpack(z)
    N_I = max(N_I, 0.0)

    N = float(p.N)
    beta, v, mu, r, h, lam = p.beta, p.v, p.mu, p.r, p.h, p.lam
    split_rule = p.split_rule

    N_S = max(N - N_I, 0.0)
    numer = 0.0
    for m in p.k_list:
        for k in range(m + 1):
            lm_k = l_get(m, k, l_free_map, E_map)
            if lm_k > 0.0 and k > 0 and (m - k) > 0:
                numer += (m - k) * (k ** v) * lm_k
    Theta_node = beta * numer / N_S if N_S > 0 else 0.0

    F = np.zeros(idx.dim, dtype=float)

    F[0] = -mu * N_I + N_S * Theta_node

    if N > 0:
        rho = min(max(N_I / N, 0.0), 1.0)
    else:
        rho = 0.0

    denom = lam * rho + (1.0 - rho)
    if denom <= 0.0:
        p_I = 0.0
    else:
        p_I = (lam * rho) / denom
    p_I = min(max(p_I, 0.0), 1.0)

    Q: Dict[Tuple[int, int], float] = {}
    Lambda: Dict[int, float] = {}

    for m in p.k_list:

        Lm = 0.0
        for j in range(1, m + 1):
            lj = l_get(m, j, l_free_map, E_map)
            if lj > 0.0:
                Lm += r * split_factor(j, h, split_rule) * lj
        Lambda[m] = Lm

        Q[(m, 0)] = 0.0
        for k in range(1, m):
            Q[(m, k)] = comb(m - 1, k - 1) * (p_I ** (k - 1)) * ((1.0 - p_I) ** (m - k))

    out = 1
    for m in p.k_list:
        for k in range(m):
            lm_k = l_get(m, k, l_free_map, E_map)

            T_infrec = 0.0

            if k - 1 >= 0:
                lm_prev = l_get(m, k - 1, l_free_map, E_map)
                if lm_prev > 0.0:
                    S_prev = m - (k - 1)
                    if S_prev > 0:
                        rate_prev = beta * ((k - 1) ** v) + Theta_node
                        T_infrec += rate_prev * S_prev * lm_prev

            if k + 1 <= m:
                lm_next = l_get(m, k + 1, l_free_map, E_map)
                if lm_next > 0.0:
                    T_infrec += mu * (k + 1) * lm_next

            if lm_k > 0.0:
                S_here = m - k
                if S_here > 0:
                    rate_here = beta * (k ** v) + Theta_node
                    T_infrec -= rate_here * S_here * lm_k

                if k > 0:
                    T_infrec -= mu * k * lm_k

            T_split_in = Q[(m, k)] * Lambda[m]

            if 1 <= k <= m - 1 and lm_k > 0.0:
                T_split_out = r * split_factor(k, h, split_rule) * lm_k
            else:
                T_split_out = 0.0

            F[out] = T_infrec + (T_split_in - T_split_out)
            out += 1

    return F

def make_initial_guess(p: Params) -> Tuple[np.ndarray, Indexer, Dict[int, float]]:
    idx = Indexer(p.k_list)
    E_map = build_E_map(p.k_list, p.E_m_list)

    N_I0 = float(p.rho0 * p.N)
    z0 = np.zeros(idx.dim, dtype=float)
    z0[0] = N_I0

    pos = 1
    for m, Em in zip(p.k_list, p.E_m_list):
        probs = np.array(
            [comb(m, k) * (p.rho0 ** k) * ((1.0 - p.rho0) ** (m - k))
             for k in range(m + 1)],
            dtype=float
        )
        S = probs.sum()
        if S > 0:
            probs /= S
        l_full = Em * probs
        z0[pos:pos + m] = l_full[:m]
        pos += m

    return z0, idx, E_map

def rebuild_full_lmk(l_free_map: Dict[int, np.ndarray],
                     k_list: List[int],
                     E_map: Dict[int, float]) -> Dict[Tuple[int, int], float]:
    l_full: Dict[Tuple[int, int], float] = {}
    for m in k_list:
        for k in range(m + 1):
            l_full[(m, k)] = l_get(m, k, l_free_map, E_map)
    return l_full

def check_size_constraints(l_full: Dict[Tuple[int, int], float],
                           k_list: List[int],
                           E_map: Dict[int, float],
                           tol: float = 1e-6):
    report: Dict[int, Dict[str, float]] = {}
    for m in k_list:
        total = sum(l_full.get((m, k), 0.0) for k in range(m + 1))
        Em = E_map.get(m, 0.0)
        err = abs(total - Em)
        report[m] = {
            "sum_from_sol": float(total),
            "E_m": float(Em),
            "abs_err": float(err),
            "ok": 1.0 if err <= tol else 0.0
        }
    return report

def check_nonneg_lmk(l_full: Dict[Tuple[int, int], float],
                     k_list: List[int],
                     tol: float = 1e-12):
    report: Dict[Tuple[int, int], Dict[str, float]] = {}
    ok_all = True
    min_val = float("inf")
    for m in k_list:
        for k in range(m + 1):
            v = float(l_full.get((m, k), 0.0))
            ok = 1.0 if v >= -tol else 0.0
            if ok == 0.0:
                ok_all = False
            if v < min_val:
                min_val = v
            report[(m, k)] = {
                "value": v,
                "ok": ok,
                "violation": float(max(0.0, -v))
            }
    return report, ok_all, float(min_val)

def check_NI_consistency_from_lmk(l_full: Dict[Tuple[int, int], float],
                                  beta: float, v: float, mu: float,
                                  N_I_solver: float,
                                  abs_tol: float = 1e-6):

    rhs = 0.0
    for (m, k), val in l_full.items():
        if val > 0.0 and k > 0 and (m - k) > 0:
            rhs += (m - k) * (k ** v) * val
    N_I_hat = (beta / mu) * rhs if mu > 0 else float("inf")
    abs_err = abs(N_I_hat - N_I_solver)
    rel_err = abs_err / max(1.0, abs(N_I_solver))
    ok = 1.0 if abs_err <= abs_tol else 0.0
    return {
        "N_I_from_lmk": float(N_I_hat),
        "abs_err": float(abs_err),
        "rel_err": float(rel_err),
        "ok": float(ok)
    }

def solve_equilibrium(p: Params):
    z0, idx, E_map = make_initial_guess(p)
    fun = lambda z: residual(z, p, idx, E_map)
    sol = root(fun, z0, method=p.method)

    N_I, l_free_map = idx.unpack(sol.x)
    rho_star = N_I / p.N if p.N > 0 else 0.0

    l_full = rebuild_full_lmk(l_free_map, p.k_list, E_map)

    size_report = check_size_constraints(l_full, p.k_list, E_map,
                                         tol=p.constraint_tol)
    size_ok = all(int(info["ok"]) == 1 for info in size_report.values())

    nonneg_report, nonneg_ok, min_lmk_value = check_nonneg_lmk(
        l_full, p.k_list, tol=p.constraint_tol
    )

    NI_consistency = check_NI_consistency_from_lmk(
        l_full, p.beta, p.v, p.mu, N_I, abs_tol=p.constraint_tol
    )

    return {
        "success": bool(sol.success),
        "message": sol.message,
        "method": p.method,
        "split_rule": p.split_rule,
        "rho_star": float(rho_star),
        "N_I_star": float(N_I),
        "l_mk_star": {k: float(v) for k, v in l_full.items()},
        "E_m_input": {int(m): float(Em) for m, Em in zip(p.k_list, p.E_m_list)},
        "size_constraint_report": size_report,
        "size_constraints_ok": bool(size_ok),
        "nonneg_report": nonneg_report,
        "nonneg_ok": bool(nonneg_ok),
        "min_lmk_value": float(min_lmk_value),
        "N_I_consistency": NI_consistency,
        "n_vars": idx.dim,
        "n_func_evals": getattr(sol, "nfev", None),
    }
