"""Physics engine for the Hubbard model solver.

Supports two modes in a two-step radial box potential:
  V(r)=0 for r<R1, V(r)=mu_offset for R1<=r<R2, V(r)=+infinity for r>=R2.

- atomic: atomic-limit local partition functions (finite U).
- kinetic: non-interacting lattice bands with kinetic dispersion (U forced to 0).
"""

from functools import lru_cache

import numpy as np
from numba import jit
from scipy.optimize import brentq, root

kB = 1.0
DELTA = 1.0  # Band gap — the energy unit. Excited band sits at this energy.
_MIN_T = 1e-4
_WARMED_UP = False


@jit(nopython=True, cache=True)
def _orbital_probs(mu_eff, T, u):
    """Occupation probabilities for a single orbital (numerically stable)."""
    beta = 1.0 / T
    a0 = 0.0
    a1 = beta * mu_eff
    a2 = beta * (2.0 * mu_eff - u)
    a_max = max(a0, max(a1, a2))
    e0 = np.exp(a0 - a_max)
    e1 = np.exp(a1 - a_max)
    e2 = np.exp(a2 - a_max)
    z = e0 + 2.0 * e1 + e2
    return e0 / z, 2.0 * e1 / z, e2 / z


@jit(nopython=True, cache=True)
def _orbital_entropy(p_h, p_s, p_d):
    """Entropy from one orbital given its occupation probabilities."""
    s = 0.0
    if p_h > 1e-15:
        s += p_h * np.log(p_h)
    if p_s > 1e-15:
        s += p_s * np.log(p_s / 2.0)
    if p_d > 1e-15:
        s += p_d * np.log(p_d)
    return -s


@jit(nopython=True, cache=True)
def _integrate_ns_box_atomic(mu, T, r1, r2, mu_offset, u, delta):
    """Closed-form radial integration for atomic-limit local occupations."""
    area_core = np.pi * r1 * r1
    area_ring = np.pi * (r2 * r2 - r1 * r1)

    # Core region: V=0
    p0h_c, p0s_c, p0d_c = _orbital_probs(mu, T, u)
    n_core = p0s_c + 2.0 * p0d_c
    s_core = _orbital_entropy(p0h_c, p0s_c, p0d_c)

    p1h_c, p1s_c, p1d_c = _orbital_probs(mu - delta, T, u)
    n_core += 2.0 * (p1s_c + 2.0 * p1d_c)
    s_core += 2.0 * _orbital_entropy(p1h_c, p1s_c, p1d_c)

    # Ring region: V=mu_offset
    mu_ring = mu - mu_offset
    p0h_r, p0s_r, p0d_r = _orbital_probs(mu_ring, T, u)
    n_ring = p0s_r + 2.0 * p0d_r
    s_ring = _orbital_entropy(p0h_r, p0s_r, p0d_r)

    p1h_r, p1s_r, p1d_r = _orbital_probs(mu_ring - delta, T, u)
    n_ring += 2.0 * (p1s_r + 2.0 * p1d_r)
    s_ring += 2.0 * _orbital_entropy(p1h_r, p1s_r, p1d_r)

    n_tot = area_core * n_core + area_ring * n_ring
    s_tot = area_core * s_core + area_ring * s_ring
    return n_tot, s_tot


def _clip_exponent_arg(x):
    return np.clip(x, -700.0, 700.0)


def _fermi_dirac(x):
    x_clip = _clip_exponent_arg(x)
    return 1.0 / (np.exp(x_clip) + 1.0)


def _binary_entropy_from_f(f):
    f_clip = np.clip(f, 1e-12, 1.0 - 1e-12)
    return -(f_clip * np.log(f_clip) + (1.0 - f_clip) * np.log(1.0 - f_clip))


@lru_cache(maxsize=16)
def _kinetic_dispersion(nk, t_hop):
    """Flattened epsilon_k grid for square-lattice nearest-neighbor hopping."""
    nk_int = int(nk)
    t = float(t_hop)
    k_vals = np.linspace(-np.pi, np.pi, nk_int, endpoint=False)
    kx, ky = np.meshgrid(k_vals, k_vals, indexing="ij")
    eps = -2.0 * t * (np.cos(kx) + np.cos(ky))
    return eps.ravel()


def _kinetic_local_quantities(mu_loc, T, t_hop, nk):
    """Local non-interacting n/s quantities at one chemical potential."""
    eps = _kinetic_dispersion(nk, t_hop)

    # Ground band (g0=2 spin)
    f0 = _fermi_dirac((eps - mu_loc) / T)
    n_spin0 = np.mean(f0)
    n_ground = 2.0 * n_spin0
    s_ground = 2.0 * np.mean(_binary_entropy_from_f(f0))

    # Excited band (g1=4 = 2 orbitals x 2 spin), shifted by DELTA
    f1 = _fermi_dirac((eps + DELTA - mu_loc) / T)
    n_spin1 = np.mean(f1)
    n_excited = 4.0 * n_spin1
    s_excited = 4.0 * np.mean(_binary_entropy_from_f(f1))

    # Effective local hole/singlon/doublon diagnostics for the ground band.
    p_hole = (1.0 - n_spin0) ** 2
    p_singlon = 2.0 * n_spin0 * (1.0 - n_spin0)
    p_doublon = n_spin0 ** 2

    n_site = n_ground + n_excited
    s_site = s_ground + s_excited
    return n_site, s_site, n_ground, n_excited, s_ground, s_excited, p_hole, p_singlon, p_doublon


def _integrate_ns_box_kinetic(mu, T, r1, r2, mu_offset, t_hop, nk):
    """Closed-form radial integration with kinetic lattice occupations."""
    area_core = np.pi * r1 * r1
    area_ring = np.pi * (r2 * r2 - r1 * r1)

    n_core, s_core, *_ = _kinetic_local_quantities(mu, T, t_hop, nk)
    n_ring, s_ring, *_ = _kinetic_local_quantities(mu - mu_offset, T, t_hop, nk)
    return area_core * n_core + area_ring * n_ring, area_core * s_core + area_ring * s_ring


def kinetic_local_quantities(mu_loc, T, t_hop=0.12, nk=220):
    """Return local kinetic-mode observables at a single local chemical potential."""
    t_eff = max(float(T), _MIN_T)
    n_site, s_site, n_ground, n_excited, s_ground, s_excited, p_hole, p_singlon, p_doublon = (
        _kinetic_local_quantities(float(mu_loc), t_eff, float(t_hop), int(nk))
    )
    return {
        "n_site": n_site,
        "s_site": s_site,
        "n_ground": n_ground,
        "n_excited": n_excited,
        "s_ground": s_ground,
        "s_excited": s_excited,
        "p_hole": p_hole,
        "p_singlon": p_singlon,
        "p_doublon": p_doublon,
    }


def kinetic_inner_entropy_curve(T, t_hop=0.12, nk=220, mu_min=None, mu_max=None, n_points=260):
    """Inner-box local entropy density s(mu) curve for kinetic mode."""
    t_eff = max(float(T), _MIN_T)
    t = abs(float(t_hop))
    nk_int = int(nk)

    band_hw = 4.0 * t
    if mu_min is None:
        mu_min = -band_hw - DELTA - 2.0
    if mu_max is None:
        mu_max = band_hw + DELTA + 2.0

    mu_vals = np.linspace(float(mu_min), float(mu_max), int(n_points))
    s_vals = np.empty_like(mu_vals)
    for idx, mu_loc in enumerate(mu_vals):
        _, s_site, *_ = _kinetic_local_quantities(mu_loc, t_eff, t, nk_int)
        s_vals[idx] = s_site
    return mu_vals, s_vals


def _validate_common_inputs(target_n, r1, r2, mu_offset):
    if target_n <= 0.0:
        return "Invalid target N."
    if r1 < 0.0 or r2 <= r1:
        return "Invalid box radii: require 0 <= R1 < R2."
    if mu_offset < 0.0 or mu_offset > 2.0:
        return "Invalid mu offset: require 0 <= mu_offset <= 2."
    return ""


def calculate_ns(mu, T, r1, r2, mu_offset, u, model="atomic", t_hop=0.12, nk=220):
    """Calculate total particle number N and entropy S."""
    t_eff = max(float(T), _MIN_T)
    model_name = str(model).strip().lower()
    if model_name == "atomic":
        return _integrate_ns_box_atomic(mu, t_eff, r1, r2, mu_offset, u, DELTA)
    if model_name == "kinetic":
        return _integrate_ns_box_kinetic(mu, t_eff, r1, r2, mu_offset, float(t_hop), int(nk))
    raise ValueError(f"Unknown model '{model}'.")


def system_equations(
    params, target_n, target_s_per_n, r1, r2, mu_offset, u, model="atomic", t_hop=0.12, nk=220
):
    """Residual equations F(mu, T) = 0 for the root finder."""
    mu, T = params
    T = max(T, _MIN_T)
    n_calc, s_calc = calculate_ns(mu, T, r1, r2, mu_offset, u, model=model, t_hop=t_hop, nk=nk)
    eq1 = (n_calc - target_n) / target_n
    eq2 = (s_calc / n_calc - target_s_per_n) if n_calc > 1e-8 else 0.0
    return [eq1, eq2]


def _mu_for_target_n(
    target_n,
    T,
    r1,
    r2,
    mu_offset,
    u,
    model="atomic",
    t_hop=0.12,
    nk=220,
    mu_center=0.0,
    mu_span=20.0,
):
    """Solve N(mu, T) = target_n for mu using adaptive bracketing."""

    def n_res(mu):
        n, _ = calculate_ns(mu, T, r1, r2, mu_offset, u, model=model, t_hop=t_hop, nk=nk)
        return n - target_n

    model_name = str(model).strip().lower()
    if model_name == "kinetic":
        band_hw = 4.0 * abs(float(t_hop))
        margin = 12.0 + max(2.0, 8.0 * max(T, _MIN_T))
        lo = -band_hw - DELTA - abs(mu_offset) - margin
        hi = DELTA + band_hw + abs(mu_offset) + margin
    else:
        lo = mu_center - mu_span
        hi = mu_center + mu_span

    f_lo = n_res(lo)
    f_hi = n_res(hi)
    for _ in range(8):
        if f_lo * f_hi <= 0.0:
            return brentq(n_res, lo, hi, xtol=1e-3)
        width = hi - lo
        lo -= width
        hi += width
        f_lo = n_res(lo)
        f_hi = n_res(hi)
    raise ValueError("Could not bracket mu for target N.")


def _estimate_low_t_entropy_floor(target_n, r1, r2, mu_offset, u, model="atomic", t_hop=0.12, nk=220):
    """Conservative low-temperature S/N floor estimate for infeasibility checks."""
    floor = np.inf
    for t_probe in (0.02, 0.01):
        try:
            mu_probe = _mu_for_target_n(
                target_n, t_probe, r1, r2, mu_offset, u, model=model, t_hop=t_hop, nk=nk
            )
        except ValueError:
            continue
        n_probe, s_probe = calculate_ns(
            mu_probe, t_probe, r1, r2, mu_offset, u, model=model, t_hop=t_hop, nk=nk
        )
        if n_probe > 1e-8:
            floor = min(floor, s_probe / n_probe)
    return floor


def warmup():
    """Compile hot JIT kernels once to reduce first-interaction latency."""
    global _WARMED_UP
    if _WARMED_UP:
        return
    calculate_ns(0.0, 0.5, 5.0, 15.0, 0.5, 0.0, model="atomic")
    calculate_ns(0.0, 0.5, 5.0, 15.0, 0.5, 0.0, model="kinetic", t_hop=0.12, nk=120)
    _WARMED_UP = True


def solve(
    target_n,
    target_s_per_n,
    r1,
    r2,
    mu_offset,
    u,
    model="atomic",
    t_hop=0.12,
    nk=220,
    init_guess=None,
    return_reason=False,
):
    """Solve for (mu, T) given target N and S/N."""

    def _pack(mu, t, success, reason=""):
        if return_reason:
            return mu, t, success, reason
        return mu, t, success

    err = _validate_common_inputs(target_n, r1, r2, mu_offset)
    if err:
        return _pack(0.0, 0.0, False, err)

    model_name = str(model).strip().lower()
    if model_name not in ("atomic", "kinetic"):
        return _pack(0.0, 0.0, False, f"Unknown model '{model}'.")
    if model_name == "kinetic":
        if t_hop < 0.0:
            return _pack(0.0, 0.0, False, "Invalid t_hop: require t_hop >= 0.")
        if int(nk) < 20:
            return _pack(0.0, 0.0, False, "Invalid nk: require nk >= 20.")

    warning_parts = []
    u_eff = float(u)
    if model_name == "kinetic":
        if abs(u_eff) > 1e-12:
            warning_parts.append("Kinetic mode enforces U=0.")
        u_eff = 0.0

    if target_s_per_n < 1.0:
        s_floor = _estimate_low_t_entropy_floor(
            target_n, r1, r2, mu_offset, u_eff, model=model_name, t_hop=t_hop, nk=int(nk)
        )
        if np.isfinite(s_floor) and target_s_per_n < s_floor - 0.05:
            warning_parts.append(
                f"Requested S/N may be infeasible (estimated floor ~ {s_floor:.3f}). Attempting solve anyway."
            )

    warning_reason = " ".join(warning_parts)

    try:
        cache = {}

        def residuals(params):
            mu = float(params[0])
            T = max(float(params[1]), _MIN_T)
            key = (mu, T)
            if key in cache:
                n_calc, s_calc = cache[key]
            else:
                n_calc, s_calc = calculate_ns(
                    mu,
                    T,
                    r1,
                    r2,
                    mu_offset,
                    u_eff,
                    model=model_name,
                    t_hop=t_hop,
                    nk=int(nk),
                )
                cache[key] = (n_calc, s_calc)
            eq1 = (n_calc - target_n) / target_n
            eq2 = (s_calc / n_calc - target_s_per_n) if n_calc > 1e-8 else 0.0
            return [eq1, eq2]

        if init_guess is not None:
            mu_init, t_init = init_guess
            x0 = [float(mu_init), max(float(t_init), _MIN_T)]
            sol = root(residuals, x0, method="hybr")
            if sol.success and sol.x[1] > _MIN_T:
                return _pack(sol.x[0], sol.x[1], True, warning_reason)

        T_trial = 0.5
        mu0 = _mu_for_target_n(
            target_n,
            T_trial,
            r1,
            r2,
            mu_offset,
            u_eff,
            model=model_name,
            t_hop=t_hop,
            nk=int(nk),
        )
        sol = root(residuals, [mu0, T_trial], method="hybr")
    except ValueError:
        if warning_reason:
            return _pack(
                0.0,
                0.0,
                False,
                f"{warning_reason} Unable to bracket the particle-number equation.",
            )
        return _pack(0.0, 0.0, False, "Unable to bracket the particle-number equation.")

    if sol.success and sol.x[1] > _MIN_T:
        return _pack(sol.x[0], sol.x[1], True, warning_reason)
    if warning_reason:
        return _pack(0.0, 0.0, False, f"{warning_reason} Root solve failed to converge.")
    return _pack(0.0, 0.0, False, "Root solve failed to converge.")


def compute_profiles(
    mu,
    T,
    r1,
    r2,
    mu_offset,
    u,
    model="atomic",
    t_hop=0.12,
    nk=220,
    n_points=300,
):
    """Vectorised radial profiles for plotting."""
    model_name = str(model).strip().lower()
    t_eff = max(float(T), _MIN_T)

    r = np.linspace(0.0, r2, n_points)
    v_trap = np.where(r < r1, 0.0, mu_offset)
    mu_loc = mu - v_trap

    if model_name == "atomic":
        beta = 1.0 / t_eff

        def _band_probs(mu_eff):
            a0 = np.zeros_like(mu_eff)
            a1 = beta * mu_eff
            a2 = beta * (2.0 * mu_eff - u)
            a_max = np.maximum(a0, np.maximum(a1, a2))
            e0 = np.exp(a0 - a_max)
            e1 = np.exp(a1 - a_max)
            e2 = np.exp(a2 - a_max)
            z = e0 + 2.0 * e1 + e2
            return e0 / z, 2.0 * e1 / z, e2 / z

        def _entropy_from_probs(p_h, p_s, p_d):
            s = np.zeros_like(p_h)
            mask_h = p_h > 1e-15
            mask_s = p_s > 1e-15
            mask_d = p_d > 1e-15
            s[mask_h] += p_h[mask_h] * np.log(p_h[mask_h])
            s[mask_s] += p_s[mask_s] * np.log(p_s[mask_s] / 2.0)
            s[mask_d] += p_d[mask_d] * np.log(p_d[mask_d])
            return -s

        p0_h, p0_s, p0_d = _band_probs(mu_loc)
        n_ground = p0_s + 2.0 * p0_d
        s_ground = _entropy_from_probs(p0_h, p0_s, p0_d)

        p1_h, p1_s, p1_d = _band_probs(mu_loc - DELTA)
        n_excited = 2.0 * (p1_s + 2.0 * p1_d)
        s_excited = 2.0 * _entropy_from_probs(p1_h, p1_s, p1_d)

        return {
            "r": r,
            "v_trap": v_trap,
            "n_total": n_ground + n_excited,
            "n_ground": n_ground,
            "n_excited": n_excited,
            "s_ground": s_ground,
            "s_excited": s_excited,
            "s_local": s_ground + s_excited,
            "p_hole": p0_h,
            "p_singlon": p0_s,
            "p_doublon": p0_d,
        }

    if model_name != "kinetic":
        raise ValueError(f"Unknown model '{model}'.")

    u_eff = 0.0
    core_vals = _kinetic_local_quantities(mu, t_eff, t_hop, int(nk))
    ring_vals = _kinetic_local_quantities(mu - mu_offset, t_eff, t_hop, int(nk))
    in_core = r < r1

    n_core, s_core, ng_core, ne_core, sg_core, se_core, ph_core, ps_core, pd_core = core_vals
    n_ring, s_ring, ng_ring, ne_ring, sg_ring, se_ring, ph_ring, ps_ring, pd_ring = ring_vals

    n_total = np.where(in_core, n_core, n_ring)
    n_ground = np.where(in_core, ng_core, ng_ring)
    n_excited = np.where(in_core, ne_core, ne_ring)
    s_ground = np.where(in_core, sg_core, sg_ring)
    s_excited = np.where(in_core, se_core, se_ring)
    s_local = np.where(in_core, s_core, s_ring)
    p_hole = np.where(in_core, ph_core, ph_ring)
    p_singlon = np.where(in_core, ps_core, ps_ring)
    p_doublon = np.where(in_core, pd_core, pd_ring)

    return {
        "r": r,
        "v_trap": v_trap,
        "n_total": n_total,
        "n_ground": n_ground,
        "n_excited": n_excited,
        "s_ground": s_ground,
        "s_excited": s_excited,
        "s_local": s_local,
        "p_hole": p_hole,
        "p_singlon": p_singlon,
        "p_doublon": p_doublon,
        "u_effective": u_eff,
    }
