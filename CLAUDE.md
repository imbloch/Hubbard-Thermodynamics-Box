# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Matplotlib GUI
.venv/bin/python3 Hubbard-Solver.py

# Streamlit web app
.venv/bin/streamlit run app.py
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `numba`, `streamlit`. Installed in `.venv/`.

## What This Project Does

Solves thermodynamic properties of 2D trapped lattice fermions in the Hubbard model using the grand canonical ensemble. Two physics modes are available: **atomic limit** (finite U) and **kinetic U=0** (square-lattice nearest-neighbor hopping). Interactive GUIs (matplotlib or Streamlit) let users adjust physical parameters (particle number N, box radii R1/R2, outer-ring chemical potential offset μ_offset, entropy per particle S/N, interaction strength U, band gap Δ, and hopping t in kinetic mode) and solve for temperature T and chemical potential μ.

## Architecture

- **`physics.py`** — All computation. Shared by both UIs.
  - JIT-compiled atomic kernels (`@jit(nopython=True, cache=True)`): `_orbital_probs`, `_orbital_entropy`, `_integrate_ns_box_atomic`.
  - Kinetic mode computes non-interacting occupations from `epsilon_k = -2t(cos kx + cos ky)` on a cached k-grid.
  - Two-step radial box integration (`R1`, `R2`, `μ_offset`) computes both N and S by core/ring area weighting.
  - `solve()` → (mu, T, success). Initial guess via `brentq` for μ at trial T, then `scipy.optimize.root` (hybr method).
  - `compute_profiles()` → dict of vectorised radial arrays for plotting.

- **`Hubbard-Solver.py`** — Matplotlib GUI. Sliders + "Calculate" button.
- **`app.py`** — Streamlit GUI. Auto-solves on slider change, warmup cached via `@st.cache_resource`.

## Multi-band Physics

- Atomic mode: partition function includes **ground band** (1 s-orbital) and **first excited band** (2 degenerate p-orbitals at energy Δ). In the atomic limit with no inter-orbital interactions, `Z_site = Z_ground × (Z_excited)^2`.
- Kinetic mode: ground band uses spin degeneracy 2, excited band uses degeneracy 4 (2 orbitals × spin), both with nearest-neighbor square-lattice dispersion.

## Key Conventions

- Natural units: kB = 1. Energies and temperatures in units of the tunneling energy.
- Local density approximation: `μ_eff(r) = μ_global − V(r)`, with `V(r)=0` for `r<R1`, `V(r)=μ_offset` for `R1<=r<R2`, and a hard wall at `r=R2`.
- Entropy per orbital uses spin-degeneracy correction for singlons: `p_s · ln(p_s / 2)`.
- Plot shows radial density profile: total density, ground-band singlons/doublons/holes, and excited-band density.
