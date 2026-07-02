import numpy as np

def check_controllability(A, B):
    import control as ctrl

    C = ctrl.ctrb(A, B)
    rank = np.linalg.matrix_rank(C)
    n = A.shape[0]
    print(f"Matriz de controlabilidad:\n{C}")
    print(f"\nRango de C: {rank}")
    print(f"Dimensión del sistema: {n}")
    print(f"\nEl sistema es controlable: {rank == n}")

def reconstruct_trajectory(T, v, psi, u0=2.0, X0=0.0, Y0=0.0):
    dX = u0 * np.cos(psi) - v * np.sin(psi)
    dY = u0 * np.sin(psi) + v * np.cos(psi)
    dt = np.diff(T)

    X = np.concatenate(([X0], X0 + np.cumsum(0.5 * (dX[:-1] + dX[1:]) * dt)))
    Y = np.concatenate(([Y0], Y0 + np.cumsum(0.5 * (dY[:-1] + dY[1:]) * dt)))

    return X, Y

def ideal_trajectory(T, psi_ref, u0=2.0, X0=0.0, Y0=0.0):
    dt = np.diff(T)

    X = np.concatenate(([X0], X0 + np.cumsum(u0 * np.cos(psi_ref[:-1]) * dt)))
    Y = np.concatenate(([Y0], Y0 + np.cumsum(u0 * np.sin(psi_ref[:-1]) * dt)))

    return X, Y
