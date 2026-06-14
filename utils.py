import control as ctrl
import numpy as np

def check_controllability(A, B):
    C = ctrl.ctrb(A, B)
    rank = np.linalg.matrix_rank(C)
    n = A.shape[0]
    print(f"Matriz de controlabilidad:\n{C}")
    print(f"\nRango de C: {rank}")
    print(f"Dimensión del sistema: {n}")
    print(f"\nEl sistema es controlable: {rank == n}")