from __future__ import annotations

from math import copysign, exp, isfinite, sqrt
from typing import Callable, Literal, Tuple

SplitRule = Literal["split-1", "split-2"]
Bounds = Tuple[float, float]

class ThresholdRootNotFoundError(ValueError):
    pass

def mean_hyperdegree(N: float, E: float) -> float:

    N = _finite("N", N)
    E = _finite("E", E)
    if N <= 0.0:
        raise ValueError("N must be positive.")
    if E <= 0.0:
        raise ValueError("E must be positive.")
    return 3.0 * E / N

def beta_c(
    d: float,
    v: float,
    mu: float,
    r: float,
    h: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
) -> float:

    C2, C1, C0 = _threshold_coefficients(
        d=d, v=v, mu=mu, r=r, h=h, split_rule=split_rule, lam=lam
    )
    discriminant = C1 * C1 + 4.0 * C2 * C0
    if not isfinite(discriminant) or discriminant <= 0.0:
        raise OverflowError("Threshold discriminant is not finite and positive.")

    denominator = sqrt(discriminant) + C1
    if denominator <= 0.0 or not isfinite(denominator):
        raise OverflowError("Unable to evaluate beta_c stably.")
    value = 2.0 * C0 / denominator
    if not isfinite(value) or value < 0.0:
        raise OverflowError("beta_c is not finite and nonnegative.")
    return value

def threshold_polynomial(
    beta: float,
    d: float,
    v: float,
    mu: float,
    r: float,
    h: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
) -> float:

    beta = _finite("beta", beta)
    if beta < 0.0:
        raise ValueError("beta must be nonnegative.")
    C2, C1, C0 = _threshold_coefficients(
        d=d, v=v, mu=mu, r=r, h=h, split_rule=split_rule, lam=lam
    )
    value = C2 * beta * beta + C1 * beta - C0
    if not isfinite(value):
        raise OverflowError("Threshold polynomial is not finite.")
    return value

def r_c_roots(
    beta: float,
    d: float,
    v: float,
    mu: float,
    h: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
    r_bounds: Bounds = (0.0, 100.0),
) -> list[float]:

    beta, d, v, mu, h = _validate_common(beta, d, v, mu, h, lam)
    lower, upper = _validate_bounds("r_bounds", r_bounds)
    eta, xi = _split_factors(h, split_rule)
    c = 2.0**v

    a = eta * xi * (2.0 * d * beta - mu)
    b = (
        4.0 * c * d * xi * beta * beta
        + (2.0 * d * mu * (3.0 * eta + 2.0 * xi) - mu * c * xi) * beta
        - mu * mu * (3.0 * eta + 2.0 * xi)
    )
    c0 = 6.0 * c * d * mu * beta * beta + 12.0 * d * mu * mu * beta - 6.0 * mu**3

    roots = _quadratic_real_roots(a, b, c0)
    return _roots_in_bounds(roots, lower, upper)

def r_c(
    beta: float,
    d: float,
    v: float,
    mu: float,
    h: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
    r_bounds: Bounds = (0.0, 100.0),
    root_index: int = 0,
) -> float:

    roots = r_c_roots(
        beta, d, v, mu, h, split_rule=split_rule, lam=lam, r_bounds=r_bounds
    )
    return _select_root("r", roots, root_index, r_bounds)

def h_c_roots(
    beta: float,
    d: float,
    v: float,
    mu: float,
    r: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
    h_bounds: Bounds = (0.0, 40.0),
    scan_points: int = 4096,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> list[float]:

    beta, d, v, mu, _ = _validate_common(beta, d, v, mu, 0.0, lam)
    r = _finite("r", r)
    if r < 0.0:
        raise ValueError("r must be nonnegative.")
    lower, upper = _validate_bounds("h_bounds", h_bounds)
    if scan_points < 2:
        raise ValueError("scan_points must be at least 2.")
    if rtol <= 0.0 or atol <= 0.0:
        raise ValueError("rtol and atol must be positive.")

    def objective(h: float) -> float:
        return threshold_polynomial(
            beta, d, v, mu, r, h, split_rule=split_rule, lam=lam
        )

    return _scan_sign_change_roots(
        objective, lower, upper, scan_points=scan_points, rtol=rtol, atol=atol
    )

def h_c(
    beta: float,
    d: float,
    v: float,
    mu: float,
    r: float,
    *,
    split_rule: SplitRule = "split-1",
    lam: float | None = None,
    h_bounds: Bounds = (0.0, 40.0),
    root_index: int = 0,
    scan_points: int = 4096,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> float:

    roots = h_c_roots(
        beta,
        d,
        v,
        mu,
        r,
        split_rule=split_rule,
        lam=lam,
        h_bounds=h_bounds,
        scan_points=scan_points,
        rtol=rtol,
        atol=atol,
    )
    return _select_root("h", roots, root_index, h_bounds)

def _threshold_coefficients(
    *,
    d: float,
    v: float,
    mu: float,
    r: float,
    h: float,
    split_rule: SplitRule,
    lam: float | None,
) -> tuple[float, float, float]:
    _, d, v, mu, h = _validate_common(0.0, d, v, mu, h, lam)
    r = _finite("r", r)
    if r < 0.0:
        raise ValueError("r must be nonnegative.")

    eta, xi = _split_factors(h, split_rule)
    c = 2.0**v
    R = r * xi
    U = 2.0 * mu + r * eta
    V = 3.0 * mu + R

    C2 = 2.0 * c * d * (3.0 * mu + 2.0 * R)
    C1 = 2.0 * d * V * U - mu * c * R
    C0 = mu * V * U
    if not all(isfinite(value) for value in (C2, C1, C0)):
        raise OverflowError("Threshold coefficients are not finite.")
    return C2, C1, C0

def _split_factors(h: float, split_rule: SplitRule) -> tuple[float, float]:
    h = _finite("h", h)
    if h < 0.0:
        raise ValueError("h must be nonnegative.")
    if split_rule == "split-1":
        return h, h * h
    if split_rule == "split-2":

        return exp(min(h, 700.0)), exp(min(2.0 * h, 700.0))
    raise ValueError("split_rule must be 'split-1' or 'split-2'.")

def _validate_common(
    beta: float,
    d: float,
    v: float,
    mu: float,
    h: float,
    lam: float | None,
) -> tuple[float, float, float, float, float]:
    beta = _finite("beta", beta)
    d = _finite("d", d)
    v = _finite("v", v)
    mu = _finite("mu", mu)
    h = _finite("h", h)
    if beta < 0.0:
        raise ValueError("beta must be nonnegative.")
    if d <= 0.0:
        raise ValueError("d must be positive.")
    if mu <= 0.0:
        raise ValueError("mu must be positive.")
    if h < 0.0:
        raise ValueError("h must be nonnegative.")
    if lam is not None:
        lam = _finite("lam", lam)
        if lam < 0.0:
            raise ValueError("lam must be nonnegative when supplied.")
    return beta, d, v, mu, h

def _finite(name: str, value: float) -> float:
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value

def _validate_bounds(name: str, bounds: Bounds) -> tuple[float, float]:
    if len(bounds) != 2:
        raise ValueError(f"{name} must contain exactly two values.")
    lower = _finite(f"{name}[0]", bounds[0])
    upper = _finite(f"{name}[1]", bounds[1])
    if lower < 0.0 or upper <= lower:
        raise ValueError(f"{name} must satisfy 0 <= lower < upper.")
    return lower, upper

def _quadratic_real_roots(a: float, b: float, c: float) -> list[float]:

    scale = max(1.0, abs(a), abs(b), abs(c))
    tiny = 1.0e-14 * scale
    if abs(a) <= tiny:
        if abs(b) <= tiny:
            if abs(c) <= tiny:
                raise ThresholdRootNotFoundError(
                    "The inverse equation is identically zero; r is not identifiable."
                )
            return []
        return [-c / b]

    discriminant = b * b - 4.0 * a * c
    if discriminant < -tiny * scale:
        return []
    if discriminant <= tiny * scale:
        return [-b / (2.0 * a)]

    sqrt_discriminant = sqrt(discriminant)
    q = -0.5 * (b + copysign(sqrt_discriminant, b))
    if q == 0.0:
        return [-b / (2.0 * a)]
    return [q / a, c / q]

def _roots_in_bounds(roots: list[float], lower: float, upper: float) -> list[float]:
    tolerance = 1.0e-11 * max(1.0, abs(lower), abs(upper))
    bounded = []
    for root in roots:
        if not isfinite(root) or root < lower - tolerance or root > upper + tolerance:
            continue
        root = min(max(root, lower), upper)
        if not bounded or abs(root - bounded[-1]) > tolerance:
            bounded.append(root)
    return sorted(bounded)

def _scan_sign_change_roots(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    scan_points: int,
    rtol: float,
    atol: float,
) -> list[float]:
    roots: list[float] = []
    x_left = lower
    f_left = function(x_left)
    function_scale = max(1.0, abs(f_left))

    for index in range(1, scan_points + 1):
        x_right = lower + (upper - lower) * index / scan_points
        f_right = function(x_right)
        function_scale = max(function_scale, abs(f_right))
        function_tolerance = 1.0e-11 * function_scale

        if abs(f_left) <= function_tolerance:
            _append_unique(roots, x_left, atol)
        if f_left * f_right < 0.0:
            root = _bisect(function, x_left, x_right, f_left, f_right, rtol, atol)
            _append_unique(roots, root, atol)
        if index == scan_points and abs(f_right) <= function_tolerance:
            _append_unique(roots, x_right, atol)

        x_left, f_left = x_right, f_right

    return roots

def _bisect(
    function: Callable[[float], float],
    left: float,
    right: float,
    f_left: float,
    f_right: float,
    rtol: float,
    atol: float,
) -> float:
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right

    for _ in range(256):
        middle = 0.5 * (left + right)
        f_middle = function(middle)
        if f_middle == 0.0 or (right - left) <= max(atol, rtol * max(1.0, abs(middle))):
            return middle
        if (f_left < 0.0) == (f_middle < 0.0):
            left, f_left = middle, f_middle
        else:
            right, f_right = middle, f_middle
    return 0.5 * (left + right)

def _append_unique(values: list[float], value: float, atol: float) -> None:
    tolerance = max(atol, 1.0e-12 * max(1.0, abs(value)))
    if not values or abs(value - values[-1]) > tolerance:
        values.append(value)

def _select_root(name: str, roots: list[float], root_index: int, bounds: Bounds) -> float:
    if not roots:
        raise ThresholdRootNotFoundError(
            f"No critical {name} was found in {bounds}. "
            f"Use {name}_c_roots or widen the search bounds."
        )
    try:
        return roots[root_index]
    except IndexError as error:
        raise ThresholdRootNotFoundError(
            f"root_index={root_index} is unavailable; roots are {roots}."
        ) from error
