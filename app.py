import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from physics import (
    compute_profiles,
    kinetic_inner_entropy_curve,
    kinetic_local_quantities,
    solve,
    warmup,
)

# ==========================================
# STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Hubbard Solver", layout="wide")
st.title("Thermodynamics of 2D Lattice Fermions (Atomic + Kinetic)")

@st.cache_resource(show_spinner=False)
def _warmup_once():
    warmup()


@st.cache_data(show_spinner=False)
def _kinetic_entropy_curve_cached(T, t_hop, nk=220, n_points=260):
    return kinetic_inner_entropy_curve(T, t_hop=t_hop, nk=nk, n_points=n_points)


_warmup_once()

if "last_guess" not in st.session_state:
    st.session_state.last_guess = None
if "result" not in st.session_state:
    st.session_state.result = None
if "profile" not in st.session_state:
    st.session_state.profile = None
if "last_params" not in st.session_state:
    st.session_state.last_params = None
if "last_model" not in st.session_state:
    st.session_state.last_model = None

st.sidebar.header("Parameters")
val_model_label = st.sidebar.selectbox(
    "Model",
    ["Kinetic (U=0)", "Atomic limit"],
    index=0,
)
val_model = "kinetic" if "Kinetic" in val_model_label else "atomic"
val_t_hop = st.sidebar.slider(
    "Hopping t (Δ)",
    0.0,
    0.5,
    0.12,
    step=0.01,
    disabled=(val_model != "kinetic"),
)
val_N = st.sidebar.slider("Particle N", 100, 5000, 300, step=10)
val_R1 = st.sidebar.slider("Core Radius R1", 0.0, 40.0, 6.0, step=0.25)
val_R2_min = val_R1 + 0.25
val_R2_default = max(16.0, val_R2_min)
val_R2 = st.sidebar.slider("Outer Radius R2", val_R2_min, 60.0, val_R2_default, step=0.25)
val_mu_offset = st.sidebar.slider(
    "Outer Ring μ Offset (Δ)", 0.0, 2.0, 0.75, step=0.025, format="%.3f"
)
val_S = st.sidebar.slider("Entropy S/N", 0.0, 2.0, 1.5, step=0.1)
val_U_ui = st.sidebar.slider(
    "Interaction U (Δ units)",
    -1.0,
    1.0,
    0.5,
    step=0.05,
    disabled=(val_model == "kinetic"),
)
if val_model == "kinetic":
    st.sidebar.info("Kinetic mode currently enforces U = 0 (finite-U kinetic mode is not implemented yet).")
val_U = 0.0 if val_model == "kinetic" else val_U_ui

params = (
    val_model,
    float(val_t_hop),
    float(val_N),
    float(val_R1),
    float(val_R2),
    float(val_mu_offset),
    float(val_S),
    float(val_U),
)
if st.session_state.last_params != params:
    init_guess = st.session_state.last_guess if st.session_state.last_model == val_model else None
    with st.spinner("Solving..."):
        mu_sol, T_sol, success, reason = solve(
            float(val_N),
            val_S,
            val_R1,
            val_R2,
            val_mu_offset,
            val_U,
            model=val_model,
            t_hop=val_t_hop,
            init_guess=init_guess,
            return_reason=True,
        )
    st.session_state.result = {
        "mu": mu_sol,
        "T": T_sol,
        "success": success,
        "reason": reason,
        "N": val_N,
        "R1": val_R1,
        "R2": val_R2,
        "mu_offset": val_mu_offset,
        "S": val_S,
        "U": val_U,
        "model": val_model,
        "t_hop": val_t_hop,
    }
    if success:
        st.session_state.last_guess = (mu_sol, T_sol)
        st.session_state.profile = compute_profiles(
            mu_sol,
            T_sol,
            val_R1,
            val_R2,
            val_mu_offset,
            val_U,
            model=val_model,
            t_hop=val_t_hop,
        )
    else:
        st.session_state.profile = None
    st.session_state.last_model = val_model
    st.session_state.last_params = params

res = st.session_state.result

if res and res["success"]:
    prof = st.session_state.profile
    if res["model"] == "kinetic":
        st.caption(
            f"Solved for model=kinetic(U=0), t={res['t_hop']:.2f}, N={res['N']}, "
            f"R1={res['R1']:.2f}, R2={res['R2']:.2f}, μoffset={res['mu_offset']:.3f}, S/N={res['S']:.2f}"
        )
    else:
        st.caption(
            f"Solved for model=atomic, N={res['N']}, R1={res['R1']:.2f}, R2={res['R2']:.2f}, "
            f"μoffset={res['mu_offset']:.3f}, S/N={res['S']:.2f}, U={res['U']:.2f}"
        )
    if res["reason"]:
        st.warning(res["reason"])

    # Display results
    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature T", f"{res['T']:.4f} (Δ/kB)")
    col2.metric("Chemical Potential μ", f"{res['mu']:.4f} (Δ)")
    n_check = np.trapezoid(prof['n_total'] * 2.0 * np.pi * prof['r'], prof['r'])
    col3.metric("Calc N", f"{n_check:.1f}  (target {res['N']})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    r = prof['r']
    ax.plot(r, prof['n_total'],   'k-',  lw=2,   label='Total Density')
    ax.plot(r, prof['p_singlon'], 'b--', lw=1.5, label='Singlons')
    ax.plot(r, prof['p_doublon'], 'r-',  lw=2,   label='Doubles (Pairs)')
    ax.plot(r, prof['p_hole'],    'g-.', lw=1.5, label='Holes')
    ax.plot(r, prof['n_excited'], color='orange', ls=':', lw=2, label='Excited Band')
    ax.set_xlabel('Radius $r$ (lattice sites)')
    ax.set_ylabel('Density / Occupation per site')
    ax.set_ylim(-0.05, max(2.05, float(np.max(prof['n_total'])) * 1.1))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right')

    # Inset: two-step trapping potential V(r)
    axins = ax.inset_axes([0.58, 0.50, 0.38, 0.43])
    axins.step(r, prof['v_trap'], where='post', color='tab:blue', lw=1.8)
    axins.axvline(res['R1'], color='gray', ls=':', lw=1.0)
    axins.axvline(res['R2'], color='black', ls='--', lw=1.2)
    y_top = max(2.1, res['mu_offset'] + 0.4)
    axins.text(res['R2'], y_top * 0.96, '∞ wall', ha='right', va='top', fontsize=8)
    axins.set_xlim(0.0, max(1.0, res['R2'] * 1.05))
    axins.set_ylim(-0.05, y_top)
    axins.set_title('Trap V(r)', fontsize=9)
    axins.set_xlabel('r', fontsize=8)
    axins.set_ylabel('V/Δ', fontsize=8)
    axins.tick_params(labelsize=8)

    st.pyplot(fig)
    plt.close(fig)

    # Local entropy profile
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.plot(r, prof['s_local'], 'm-', lw=2, label='Total Local Entropy')
    ax_s.plot(r, prof['s_ground'], 'c--', lw=1.5, label='Ground Band Entropy')
    ax_s.plot(r, prof['s_excited'], color='brown', ls='-.', lw=1.5, label='Excited Band Entropy')
    ax_s.set_xlabel('Radius $r$ (lattice sites)')
    ax_s.set_ylabel('Local Entropy per Site')
    ax_s.set_ylim(bottom=-0.02)
    ax_s.grid(True, linestyle='--', alpha=0.6)
    ax_s.legend(loc='upper right')
    st.pyplot(fig_s)
    plt.close(fig_s)

    if res["model"] == "kinetic":
        mu_curve, s_curve = _kinetic_entropy_curve_cached(float(res["T"]), float(res["t_hop"]))
        inner_local = kinetic_local_quantities(
            float(res["mu"]), float(res["T"]), t_hop=float(res["t_hop"]), nk=220
        )
        mu_sol = float(res["mu"])
        s_inner_at_mu = float(inner_local["s_site"])

        fig_mu, ax_mu = plt.subplots(figsize=(10, 4))
        ax_mu.plot(mu_curve, s_curve, color="navy", lw=2, label=r"$s_{\mathrm{inner}}(\mu)$")
        ax_mu.axvline(mu_sol, color="crimson", ls="--", lw=1.4, label="Solved μ")
        ax_mu.plot([mu_sol], [s_inner_at_mu], "o", color="crimson", ms=6)
        ax_mu.set_xlabel("Chemical potential μ (Δ)")
        ax_mu.set_ylabel("Local entropy in inner box")
        ax_mu.set_title(
            f"Inner-box entropy vs μ (R1={res['R1']:.2f}, R2={res['R2']:.2f}, t={res['t_hop']:.2f})"
        )
        ax_mu.set_xlim(-1.0, 2.0)
        ax_mu.grid(True, linestyle="--", alpha=0.6)
        ax_mu.legend(loc="best")

        # Inset zoom for low-entropy regime.
        axins_mu = ax_mu.inset_axes([0.08, 0.56, 0.37, 0.37])
        axins_mu.plot(mu_curve, s_curve, color="navy", lw=1.6)
        axins_mu.axvline(mu_sol, color="crimson", ls="--", lw=1.1)
        if 0.0 <= s_inner_at_mu <= 0.2:
            axins_mu.plot([mu_sol], [s_inner_at_mu], "o", color="crimson", ms=4.5)
        axins_mu.set_xlim(-1.0, 2.0)
        axins_mu.set_ylim(0.0, 0.2)
        axins_mu.set_title("Inset: s ∈ [0, 0.2]", fontsize=8)
        axins_mu.set_xlabel("μ", fontsize=8)
        axins_mu.set_ylabel("s", fontsize=8)
        axins_mu.tick_params(labelsize=8)

        st.pyplot(fig_mu)
        plt.close(fig_mu)
elif res:
    st.error(res["reason"] or "Solver failed! Try different parameters.")
