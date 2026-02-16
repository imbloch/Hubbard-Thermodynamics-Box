import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider

from physics import compute_profiles, solve, warmup

# ==========================================
# INTERACTIVE PLOT (MATPLOTLIB)
# ==========================================

# Initial Parameters
init_model = "kinetic"
init_t_hop = 0.12
init_N = 300
init_R1 = 6.0
init_R2 = 16.0
init_mu_offset = 0.75
init_S = 1.5
init_U = 0.5
last_guess = [None]
last_params = [None]
last_model = [None]

warmup()

# Setup Figure
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.50, right=0.9, top=0.9)

ax_plot = fig.add_subplot(111)
ax_plot.set_xlabel("Radius $r$ (lattice sites)")
ax_plot.set_ylabel("Density / Occupation per site")
ax_plot.set_title("Thermodynamics of 2D Lattice Fermions (Atomic + Kinetic)")
ax_plot.grid(True, linestyle="--", alpha=0.6)
ax_plot.set_ylim(-0.05, 2.05)

# Initial Plot Lines (Empty)
line_tot, = ax_plot.plot([], [], "k-", lw=2, label="Total Density")
line_sin, = ax_plot.plot([], [], "b--", lw=1.5, label="Singlons")
line_dbl, = ax_plot.plot([], [], "r-", lw=2, label="Doubles (Pairs)")
line_hol, = ax_plot.plot([], [], "g-.", lw=1.5, label="Holes")
line_exc, = ax_plot.plot([], [], color="orange", ls=":", lw=2, label="Excited Band")
ax_plot.legend(loc="upper right")

# Inset: two-step trapping potential
ax_inset = ax_plot.inset_axes([0.58, 0.50, 0.38, 0.43])
line_pot, = ax_inset.plot([], [], color="tab:blue", lw=1.8, drawstyle="steps-post")
line_r1 = ax_inset.axvline(init_R1, color="gray", ls=":", lw=1.0)
line_r2 = ax_inset.axvline(init_R2, color="black", ls="--", lw=1.2)
text_wall = ax_inset.text(init_R2, 1.0, "∞ wall", ha="right", va="top", fontsize=8)
ax_inset.set_title("Trap V(r)", fontsize=9)
ax_inset.set_xlabel("r", fontsize=8)
ax_inset.set_ylabel("V/Δ", fontsize=8)
ax_inset.tick_params(labelsize=8)

# Text Output
text_res = plt.figtext(
    0.1, 0.92, "Adjust controls to solve...", fontsize=12, fontweight="bold", color="blue"
)

# Controls
axcolor = "lightgoldenrodyellow"
ax_model = plt.axes([0.03, 0.05, 0.17, 0.14], facecolor=axcolor)
ax_t = plt.axes([0.25, 0.35, 0.65, 0.03], facecolor=axcolor)
ax_N = plt.axes([0.25, 0.30, 0.65, 0.03], facecolor=axcolor)
ax_R1 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
ax_R2 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_mu_offset = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_S = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_U = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

r_model = RadioButtons(ax_model, ("Kinetic (U=0)", "Atomic limit"), active=0)
ax_model.set_title("Model", fontsize=10)

s_t = Slider(ax_t, "Hopping t (Δ)", 0.0, 0.5, valinit=init_t_hop, valstep=0.01)
s_N = Slider(ax_N, "Particle N", 100, 5000, valinit=init_N, valstep=10)
s_R1 = Slider(ax_R1, "Core Radius R1", 0.0, 40.0, valinit=init_R1, valstep=0.25)
s_R2 = Slider(ax_R2, "Outer Radius R2", 0.25, 60.0, valinit=init_R2, valstep=0.25)
s_mu_offset = Slider(
    ax_mu_offset,
    "Outer Ring μ Offset (Δ)",
    0.0,
    2.0,
    valinit=init_mu_offset,
    valstep=0.025,
)
s_S = Slider(ax_S, "Entropy S/N", 0.0, 2.0, valinit=init_S)
s_U = Slider(ax_U, "Interaction U (Δ)", -1.0, 1.0, valinit=init_U, valstep=0.05)


def _selected_model():
    return "kinetic" if "Kinetic" in r_model.value_selected else "atomic"


def _refresh_control_state():
    kinetic_mode = _selected_model() == "kinetic"
    ax_t.set_facecolor(axcolor if kinetic_mode else "#e4e4e4")
    ax_U.set_facecolor("#e4e4e4" if kinetic_mode else axcolor)


def update(event):
    model = _selected_model()
    val_t_hop = s_t.val
    val_N = s_N.val
    val_R1 = s_R1.val
    val_R2 = s_R2.val
    val_mu_offset = s_mu_offset.val
    val_S = s_S.val
    val_U_raw = s_U.val
    val_U = 0.0 if model == "kinetic" else val_U_raw

    _refresh_control_state()

    params = (
        model,
        val_t_hop if model == "kinetic" else -1.0,
        val_N,
        val_R1,
        val_R2,
        val_mu_offset,
        val_S,
        val_U,
    )
    if last_params[0] == params:
        return
    last_params[0] = params

    text_res.set_text("Solving... Please wait.")
    plt.draw()

    if val_R2 <= val_R1:
        text_res.set_text("Require R2 > R1 for a valid box trap.")
        text_res.set_color("red")
        plt.draw()
        return

    init_guess = last_guess[0] if last_model[0] == model else None
    mu_sol, T_sol, success, reason = solve(
        val_N,
        val_S,
        val_R1,
        val_R2,
        val_mu_offset,
        val_U,
        model=model,
        t_hop=val_t_hop,
        init_guess=init_guess,
        return_reason=True,
    )

    if success:
        last_guess[0] = (mu_sol, T_sol)
        last_model[0] = model
        prof = compute_profiles(
            mu_sol,
            T_sol,
            val_R1,
            val_R2,
            val_mu_offset,
            val_U,
            model=model,
            t_hop=val_t_hop,
        )
        n_check = np.trapezoid(prof["n_total"] * 2.0 * np.pi * prof["r"], prof["r"])

        if model == "kinetic":
            status = (
                "SOLVED:\n"
                f"model = kinetic(U=0), t = {val_t_hop:.2f} Δ\n"
                f"R1 = {val_R1:.2f}, R2 = {val_R2:.2f}, μoffset = {val_mu_offset:.3f} Δ\n"
                f"T = {T_sol:.4f} (Δ/kB)\n"
                f"μ = {mu_sol:.4f} (Δ)\n"
                f"Calc N = {n_check:.1f} (Target {val_N})"
            )
        else:
            status = (
                "SOLVED:\n"
                f"model = atomic, U = {val_U:.2f} Δ\n"
                f"R1 = {val_R1:.2f}, R2 = {val_R2:.2f}, μoffset = {val_mu_offset:.3f} Δ\n"
                f"T = {T_sol:.4f} (Δ/kB)\n"
                f"μ = {mu_sol:.4f} (Δ)\n"
                f"Calc N = {n_check:.1f} (Target {val_N})"
            )
        if reason:
            status += f"\nWarning: {reason}"
        text_res.set_text(status)
        text_res.set_color("darkgreen")

        r = prof["r"]
        line_tot.set_data(r, prof["n_total"])
        line_sin.set_data(r, prof["p_singlon"])
        line_dbl.set_data(r, prof["p_doublon"])
        line_hol.set_data(r, prof["p_hole"])
        line_exc.set_data(r, prof["n_excited"])
        ax_plot.set_xlim(0, r[-1])
        ax_plot.set_ylim(-0.05, max(2.05, float(np.max(prof["n_total"])) * 1.1))

        line_pot.set_data(r, prof["v_trap"])
        line_r1.set_xdata([val_R1, val_R1])
        line_r2.set_xdata([val_R2, val_R2])
        y_top = max(2.1, val_mu_offset + 0.4)
        text_wall.set_position((val_R2, y_top * 0.96))
        ax_inset.set_xlim(0.0, max(1.0, val_R2 * 1.05))
        ax_inset.set_ylim(-0.05, y_top)
    else:
        text_res.set_text(reason or "Solver Failed! Try different parameters.")
        text_res.set_color("red")

    plt.draw()


r_model.on_clicked(update)
s_t.on_changed(update)
s_N.on_changed(update)
s_R1.on_changed(update)
s_R2.on_changed(update)
s_mu_offset.on_changed(update)
s_S.on_changed(update)
s_U.on_changed(update)

_refresh_control_state()
update(None)
plt.show()
