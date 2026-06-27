import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import os
import pandas as pd

colors = {
    "AGRESIVO": "red",
    "INTERMEDIO": "green",
    "SUAVE": "blue",
    "PARTE A": "black"
}

def LQI(Q_a, R, A_a, B_a, E_a, verbose=False):
    K_a, S, poles = ctrl.lqr(A_a, B_a, Q_a, R)

    A_cl = A_a - B_a @ K_a

    sys_cl = ctrl.ss(A_cl, E_a, np.eye(4), np.zeros((4, 1)))

    if verbose: 
        print("K_a =", K_a)
        print("Polos LQI =", poles)

    return {
        "K_a": K_a,
        "S": S,
        "poles": poles,
        "A_cl": A_cl,
        "sys_cl": sys_cl,
        "Q_a": Q_a,
        "R": R
    }

def plot_poles(controllers, save=False, save_dir='plots'):
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(6, 6))

    markers = ["o", "s", "^", "D", "P", "X"]

    for i, (name, controller) in enumerate(controllers.items()):
        poles = controller["poles"]
        plt.scatter(
            poles.real,
            poles.imag,
            label=name,
            s=100,
            alpha=0.65,
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.7,
            color=colors[name]
        )

    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.title("Polos de los controladores LQI")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        fig.savefig(f"{save_dir}/polos.png", dpi=300)

    plt.show()

def get_info(name, controller, T, psi_ref, x0):
    K_a = controller["K_a"]
    sys_cl = controller["sys_cl"]

    T_sim, x_a = ctrl.forced_response(sys_cl, T=T, U=psi_ref, X0=x0)

    v = x_a[0, :]
    r_yaw = x_a[1, :]
    psi = x_a[2, :]
    xi = x_a[3, :]

    tau_r = -(K_a @ x_a).flatten()
    error = psi_ref - psi

    metrics = {
        "name": name,
        "max_abs_tau": np.max(np.abs(tau_r)),
        "rms_tau": np.sqrt(np.mean(tau_r**2)),
        "max_abs_error": np.max(np.abs(error)),
        "iae": np.trapezoid(np.abs(error), T_sim),
        "ise": np.trapezoid(error**2, T_sim),
        "final_error": error[-1],
        "max_abs_v": np.max(np.abs(v)),
        "max_abs_r": np.max(np.abs(r_yaw)),
        "max_abs_psi": np.max(np.abs(psi))
    }

    return {
        "name": name,
        "T": T_sim,
        "psi_ref": psi_ref,
        "x_a": x_a,
        "v": v,
        "r_yaw": r_yaw,
        "psi": psi,
        "xi": xi,
        "tau_r": tau_r,
        "error": error,
        "metrics": metrics,
        "controller": controller
    }

def plot_results(results, show=True, save=False, save_dir='plots'):
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fontsize_labels = 14
    fontsize_title = 14
    fontsize_legend = 10

    # 1) Seguimiento de referencia
    fig1 = plt.figure(figsize=(10, 4))

    first_result = next(iter(results.values()))
    T_ref = first_result["T"]
    psi_ref = first_result["psi_ref"]

    plt.plot(T_ref, psi_ref, "--", label=r"$\psi_{ref}(t)$", linewidth=2, color="orange")

    for name, result in results.items():
        plt.plot(result["T"], result["psi"], label=fr"$\psi(t)$ - {name}", linewidth=2, color=colors[name])

    plt.xlabel("Tiempo [s]", fontsize=fontsize_labels)
    plt.ylabel(r"Rumbo $\psi$ [rad]", fontsize=fontsize_labels)
    plt.title("Seguimiento de referencia", fontsize=fontsize_title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()

    if save:
        fig1.savefig(f"{save_dir}/seguimiento.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig1)

    # 2) Error
    fig2 = plt.figure(figsize=(10, 4))

    for name, result in results.items():
        plt.plot(result["T"], result["error"], label=fr"$e(t)$ - {name}", linewidth=2, color=colors[name])

    plt.xlabel("Tiempo [s]", fontsize=fontsize_labels)
    plt.ylabel("Error [rad]", fontsize=fontsize_labels)
    plt.title("Error de seguimiento", fontsize=fontsize_title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()

    if save:
        fig2.savefig(f"{save_dir}/error.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig2)

    # # 3) Estados
    fig3, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

    ax1, ax2, ax3 = axes
    for name, result in results.items():
        ax1.plot(result["T"], result["v"], label=fr"$v(t)$ - {name}", color=colors[name])
    ax1.set_ylabel(r"$v$ [m/s]", fontsize=fontsize_labels)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=fontsize_legend)

    for name, result in results.items():
        ax2.plot(result["T"], result["r_yaw"], label=fr"$r(t)$ - {name}", color=colors[name])
    ax2.set_ylabel(r"$r$ [rad/s]", fontsize=fontsize_labels)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=fontsize_legend)

    ax3.plot(T_ref, psi_ref, "--", label=r"$\psi_{ref}(t)$", color="orange")
    for name, result in results.items():
        ax3.plot(result["T"], result["psi"], label=fr"$\psi(t)$ - {name}", color=colors[name])
    ax3.set_xlabel("Tiempo [s]", fontsize=fontsize_labels)
    ax3.set_ylabel(r"$\psi$ [rad]", fontsize=fontsize_labels)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=fontsize_legend)

    fig3.suptitle("Estados del sistema", fontsize=fontsize_title)
    fig3.tight_layout()

    if save:
        fig3.savefig(f"{save_dir}/estados.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig3)

    # 4) Integrador
    fig4 = plt.figure(figsize=(10, 4))

    for name, result in results.items():
        plt.plot(result["T"], result["xi"], label=fr"$\xi(t)$ - {name}", linewidth=2, color=colors[name])

    plt.xlabel("Tiempo [s]", fontsize=fontsize_labels)
    plt.ylabel(r"$\xi$", fontsize=fontsize_labels)
    plt.title("Estado del integrador", fontsize=fontsize_title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()

    if save:
        fig4.savefig(f"{save_dir}/integrador.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig4)

    # 5) Control
    fig5 = plt.figure(figsize=(10, 4))

    for name, result in results.items():
        plt.plot(result["T"], result["tau_r"], label=fr"$\tau_r(t)$ - {name}", linewidth=2, color=colors[name])

    plt.xlabel("Tiempo [s]", fontsize=fontsize_labels)
    plt.ylabel(r"$\tau_r$", fontsize=fontsize_labels)
    plt.title("Señal de control", fontsize=fontsize_title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=fontsize_legend)
    plt.tight_layout()

    if save:
        fig5.savefig(f"{save_dir}/control.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig5)

def print_metrics(results):
    rows = []

    for name, result in results.items():
        m = result["metrics"]
        rows.append({
            "Controller": name,
            "Max |tau_r|": m["max_abs_tau"],
            "RMS tau_r": m["rms_tau"],
            "Max |error|": m["max_abs_error"],
            "IAE": m["iae"],
            "ISE": m["ise"],
            "Final error": m["final_error"],
            "Max |v|": m["max_abs_v"],
            "Max |r|": m["max_abs_r"],
            "Max |psi|": m["max_abs_psi"],
        })

    df = pd.DataFrame(rows)
    return df