import numpy as np
import control as ctrl
import matplotlib.pyplot as plt
import os
import pandas as pd


def LQI(Q_a, R, A_a, B_a, E_a, verbose=False):
    K_a, S, poles = ctrl.lqr(A_a, B_a, Q_a, R)

    A_cl = A_a - B_a @ K_a

    sys_cl = ctrl.ss(
        A_cl,
        E_a,
        np.eye(4),
        np.zeros((4, 1))
    )

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

def get_info(name, controller, T, psi_ref, x0):
    K_a = controller["K_a"]
    sys_cl = controller["sys_cl"]

    T_sim, x_a = ctrl.forced_response(
        sys_cl,
        T=T,
        U=psi_ref,
        X0=x0
    )

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

    # 1) Seguimiento de referencia
    fig1 = plt.figure(figsize=(10, 4))

    first_result = next(iter(results.values()))
    T_ref = first_result["T"]
    psi_ref = first_result["psi_ref"]

    plt.plot(T_ref, psi_ref, "--", label=r"$\psi_{ref}(t)$", linewidth=2)

    for name, result in results.items():
        plt.plot(result["T"], result["psi"], label=fr"$\psi(t)$ - {name}", linewidth=2)

    plt.xlabel("Tiempo [s]")
    plt.ylabel(r"Rumbo $\psi$ [rad]")
    plt.title("Seguimiento de referencia")
    plt.grid(True, alpha=0.3)
    plt.legend()
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
        plt.plot(result["T"], result["error"], label=fr"$e(t)$ - {name}", linewidth=2)

    plt.xlabel("Tiempo [s]")
    plt.ylabel("Error [rad]")
    plt.title("Error de seguimiento")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save:
        fig2.savefig(f"{save_dir}/error.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig2)

    # 3) Estados
    fig3 = plt.figure(figsize=(10, 6))

    plt.subplot(3, 1, 1)
    for name, result in results.items():
        plt.plot(result["T"], result["v"], label=fr"$v(t)$ - {name}")
    plt.ylabel(r"$v$ [m/s]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 2)
    for name, result in results.items():
        plt.plot(result["T"], result["r_yaw"], label=fr"$r(t)$ - {name}")
    plt.ylabel(r"$r$ [rad/s]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(T_ref, psi_ref, "--", label=r"$\psi_{ref}(t)$")
    for name, result in results.items():
        plt.plot(result["T"], result["psi"], label=fr"$\psi(t)$ - {name}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel(r"$\psi$ [rad]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.suptitle("Estados del sistema")
    plt.tight_layout()

    if save:
        fig3.savefig(f"{save_dir}/estados.png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig3)

    # 4) Integrador
    fig4 = plt.figure(figsize=(10, 4))

    for name, result in results.items():
        plt.plot(result["T"], result["xi"], label=fr"$\xi(t)$ - {name}", linewidth=2)

    plt.xlabel("Tiempo [s]")
    plt.ylabel(r"$\xi$")
    plt.title("Estado del integrador")
    plt.grid(True, alpha=0.3)
    plt.legend()
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
        plt.plot(result["T"], result["tau_r"], label=fr"$\tau_r(t)$ - {name}", linewidth=2)

    plt.xlabel("Tiempo [s]")
    plt.ylabel(r"$\tau_r$")
    plt.title("Señal de control")
    plt.grid(True, alpha=0.3)
    plt.legend()
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