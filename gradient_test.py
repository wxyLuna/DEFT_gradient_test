import numpy as np
import torch
import time


def func_DX_ICitr(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Inextensibility Constraint Iterative Function

    # Inputs:
    - M_0: [3, 3] mass matrix of vertex i
    - M_1: [3, 3] mass matrix of vertex i+1
    - X_0: [3, 1] position of vertex i
    - X_1: [3, 1] position of vertex i+1
    - X_0_init: [3, 1] undeformed position of vertex i
    - X_1_init: [3, 1] undeformed position of vertex i+1

    # Outputs:
    - DX_0: [3, 1] position change of vertex i
    - DX_1: [3, 1] position change of vertex i+1
    """

    M_param = np.linalg.inv(M_0 + M_1)
    
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init
    Edge_length = np.linalg.norm(Edge)
    Edge_length_init = np.linalg.norm(Edge_init)

    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)

    DX_0 = M_1 @ M_param @ Edge * lambda_param
    DX_1 = -M_0 @ M_param @ Edge * lambda_param

    return DX_0, DX_1



# ---- single-edge inextensibility constraint (torch) ----
def DX_ICitr_torch(M0, M1, X0, X1, X0i, X1i):
    """
    All inputs shape (…,3,3) or (…,3,1) with broadcasting.
    Returns (DX0, DX1) each (…,3,1)
    """
    Mparam = torch.linalg.inv(M0 + M1)
    Edge   = X1 - X0
    Ein    = X1i - X0i
    l2  = (Edge  ** 2).sum(dim=-2, keepdim=True)   # (…,1,1)
    l20 = (Ein   ** 2).sum(dim=-2, keepdim=True)
    lam = (l2 - l20) / (l2 + l20)                  # (…,1,1)
    DX0 =  (M1 @ Mparam @ Edge) * lam
    DX1 = -(M0 @ Mparam @ Edge) * lam
    return DX0, DX1


def grad_DX_X_ICitr(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Gradient of the inextensibility constraint iterative function with respect to the positions X_0 and X_1.

    # Inputs:
    M_0: [3, 3] mass matrix of vertex i
    M_1: [3, 3] mass matrix of vertex i+1
    X_0: [3, 1] position of vertex i
    X_1: [3, 1] position of vertex i+1
    X_0_init: [3, 1] undeformed position of vertex i
    X_1_init: [3, 1] undeformed position of vertex i+1

    # Outputs:
    4 grads: [3, 3] gradient of the inextensibility constraint iterative function output DX_0, DX_1 with respect to X_0, X_1.
    """

    M_param = np.linalg.inv(M_0 + M_1)
    Edge = X_1 - X_0
    Edge = Edge.reshape(3,1)
    Edge_init = X_1_init - X_0_init
    Edge_init = Edge_init.reshape(3,1)
    Edge_length = np.linalg.norm(Edge)
    Edge_length_init = np.linalg.norm(Edge_init)
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)

	# Gradient of DX_0 with respect to X_0
    grad_00 = - M_1 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_00 -= M_1 @ M_param * lambda_param
	# Gradient of DX_0 with respect to X_1
    grad_01 = M_1 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_01 += M_1 @ M_param * lambda_param
	# Gradient of DX_1 with respect to X_0
    grad_10 = M_0 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_10 += M_0 @ M_param * lambda_param
	# Gradient of DX_1 with respect to X_1
    grad_11 = - M_0 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_11 -= M_0 @ M_param * lambda_param

    return grad_00, grad_01, grad_10, grad_11

def grad_DX_Xinit_ICitr(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Gradient of the inextensibility constraint iterative function with respect to the undeformed positions X_0_init and X_1_init.

    # Inputs:
    M_0: [3, 3] mass matrix of vertex i
    M_1: [3, 3] mass matrix of vertex i+1
    X_0: [3, 1] position of vertex i
    X_1: [3, 1] position of vertex i+1
    X_0_init: [3, 1] undeformed position of vertex i
    X_1_init: [3, 1] undeformed position of vertex i+1

    # Outputs:
    4 grads: [3, 3] gradient of the inextensibility constraint iterative function output DX_0, DX_1 with respect to X_0_init, X_1_init.
    """

    M_param = np.linalg.inv(M_0 + M_1)
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init

    Edge_length = np.linalg.norm(Edge)
    Edge_length_init = np.linalg.norm(Edge_init)
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)
    
	# Gradient of DX_0 with respect to X_0_init
    grad_00 = M_1 @ M_param @ np.outer(Edge, Edge_init) * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
	# Gradient of DX_0 with respect to X_1_init
    grad_01 = -M_1 @ M_param @ np.outer(Edge, Edge_init) * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
	# Gradient of DX_1 with respect to X_0_init
    grad_10 = -M_0 @ M_param @ np.outer(Edge, Edge_init)  * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
	# Gradient of DX_1 with respect to X_1_init
    grad_11 = M_0 @ M_param @ np.outer(Edge, Edge_init)  * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)


    return grad_00, grad_01, grad_10, grad_11

def grad_DX_M_ICitr(M_0,M_1,X_0,X_1,X_0_init,X_1_init):
    """
       Gradient of the inextensibility constraint iterative function with respect to the mass matrixes M_0 and M_1.


       # Inputs:
       idx_1: Index of gradient, DX_0 (0) or DX_1 (1)
       idx_2: Index of gradient, M_0 (0) or M_1 (1)
       M_0: [3, 3] mass matrix of vertex i
       M_1: [3, 3] mass matrix of vertex i+1
       X_0: [3, 1] position of vertex i
       X_1: [3, 1] position of vertex i+1
       X_0_init: [3, 1] undeformed position of vertex i
       X_1_init: [3, 1] undeformed position of vertex i+1

       # Outputs:
       4 grads: [3, 1] gradient of the inextensibility constraint iterative function output DX_0, DX_1 with respect to M_0, M_1.
    """
    M_param = np.linalg.inv(M_0+M_1)
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init
    Edge_norm = np.linalg.norm(Edge)
    Edge_init_norm = np.linalg.norm(Edge_init)
    lambda_param = (Edge_norm**2-Edge_init_norm**2)/ (Edge_norm**2+Edge_init_norm**2)
	
    grad_M_00 = - M_1 @ M_param @ M_param @ Edge * lambda_param
    grad_M_01 = (np.eye(3) - M_1 @ M_param) @ M_param @ Edge* lambda_param
    grad_M_10 = -(np.eye(3)- M_0 @ M_param) @ M_param @ Edge*lambda_param
    grad_M_11 = M_0@M_param@M_param@Edge*lambda_param

    return grad_M_00, grad_M_01, grad_M_10, grad_M_11

# Example usage:
import pandas as pd
import numpy as np


def compute_summary(name, diff_x0, diff_x1, diff_x0_init, diff_x1_init, diff_M0, diff_M1, tol=1e-3):
    # Re-evaluate both numerical and analytical differences
    DX_0_m, DX_1_m = func_DX_ICitr(M0-diff_M0, M1-diff_M1, x0-diff_x0, x1-diff_x1, x0_init-diff_x0_init, x1_init-diff_x1_init)
    DX_0_p, DX_1_p = func_DX_ICitr(M0+diff_M0, M1+diff_M1, x0+diff_x0, x1+diff_x1, x0_init+diff_x0_init, x1_init+diff_x1_init)
    num_diff_DX_0 = (DX_0_p - DX_0_m)/2
    num_diff_DX_1 = (DX_1_p - DX_1_m)/2

    grad_diff_DX_0 = J_DX0_X0 @ diff_x0 + J_DX0_X1 @ diff_x1 + J_DX0_X0_init @ diff_x0_init + J_DX0_X1_init @ diff_x1_init + diff_M0 @ J_DX0_M0  + diff_M1@J_DX0_M1
    grad_diff_DX_1 = J_DX1_X0 @ diff_x0 + J_DX1_X1 @ diff_x1 + J_DX1_X0_init @ diff_x0_init + J_DX1_X1_init @ diff_x1_init + diff_M0 @ J_DX1_M0  + diff_M1 @ J_DX1_M1

    # Differences
    err_DX0 = (grad_diff_DX_0-num_diff_DX_0)
    err_DX1 = (grad_diff_DX_1-num_diff_DX_1)
    relative_error = np.vstack([err_DX0/num_diff_DX_0, err_DX1/num_diff_DX_1])
    absolute_error = np.vstack([err_DX0, err_DX1])
    pass_check = np.all(np.abs(relative_error) < tol)
    formatted_relative = np.vectorize(lambda x: f"{x:.3e}")(relative_error)
    formatted_absolute = np.vectorize(lambda x: f"{x:.3e}")(absolute_error)

    return {
        "Perturbation": name,
        "Relative Difference": formatted_relative,
        "Absolute Difference": formatted_absolute,
        "Pass? (tolerance = 1e^{-8})": pass_check
    }

    # return {
    #     "Perturbation": name,
    #     "Numerical Gradient \delta x _{i}":num_diff_DX_0,
    #     "Numerical Gradient \delta x _{i+1}": num_diff_DX_1,
    #     "Analytical Gradient \delta x _{i}": grad_diff_DX_0,
    #     "Analytical Gradient \delta x _{i+1}": grad_diff_DX_1,
    #     "Max \|\delta x _{i}\|difference": np.max(err_DX0),
    #     "Mean \|\delta x _{i}\|difference": np.mean(err_DX0),
    #     "Max \|\delta x _{i+1}\|difference": np.max(err_DX1),
    #     "Mean \|\delta x _{i+1}\|difference": np.mean(err_DX1),
    #     "Pass?(tolerance = 1e^{-8}": np.max(err_DX0) < tol and np.max(err_DX1) < tol
    # }


if __name__ == '__main__':
    # Define sample values:
    # x0 = np.array([0.0020, 0.6336, -0.0126])
    # x1 = np.array([-0.0046, 0.6808, 0.0102])
    #
    # diff_x0 = np.array([0.00001, 0, 0]) # z
    # diff_x1 = np.array([0.0, 0.0, -0.00002])
    #
    # # x0_init = np.array([0.0, 0.0, 0.0])
    # # x1_init = np.array([0.0447, 0.0, 0.0])
    # x0_init = np.array([ -0.0108,0.6790,0.0035])
    # x1_init = np.array([ -0.0104,0.6355,0.0066])
    #
    # diff_x0_init = np.array([0, 0, 0.00001])
    # diff_x1_init = np.array([0,0, -0.00002])
    #
    # # Let e1_bar be a scalar constant.
    # # e1_bar = 0.0447  # for example
    # e1_bar = np.linalg.norm (x1_init-x0_init)# for example
    #
    # # Define scale as a 3x3 matrix, for instance:
    # M0 = np.array([[0.7349, 0.0, 0.0],
    #                [0.0, 0.7349, 0.0],
    #                [0.0, 0.0, 0.7349]])
    #
    # M1 = np.array([[0.2651, 0.0, 0.0],
    #                [0.0, 0.2651, 0.0],
    #                [0.0, 0.0, 0.2651]])
    #
    # diff_M0 = 0*np.eye(3)
    # diff_M1 = np.array([[0.00001, 0.0, 0.00003],
    #                [0.00002, 0, 0.0],
    #                [0.00001, 0.0, 0]])
    #
    # # Compute the function value and Jacobian.
    # DX_0_m, DX_1_m = func_DX_ICitr(M0-diff_M0, M1-diff_M1, x0-diff_x0, x1-diff_x1, x0_init-diff_x0_init, x1_init-diff_x1_init)
    # DX_0_p, DX_1_p = func_DX_ICitr(M0+diff_M0, M1+diff_M1, x0+diff_x0, x1+diff_x1, x0_init+diff_x0_init, x1_init+diff_x1_init)
    # num_diff_DX_0 = (DX_0_p - DX_0_m) / 2
    # num_diff_DX_1 = (DX_1_p - DX_1_m) / 2
    # # t0_analytical = time.perf_counter()
    # J_DX0_X0, J_DX0_X1, J_DX1_X0, J_DX1_X1 = grad_DX_X_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    # J_DX0_X0_init, J_DX0_X1_init, J_DX1_X0_init, J_DX1_X1_init = grad_DX_Xinit_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    # J_DX0_M0, J_DX0_M1, J_DX1_M0, J_DX1_M1 = grad_DX_M_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    # # t_analytical = (time.perf_counter() - t0_analytical)
    # # print('analytical computation time', t_analytical)
    #
    #
    #
    #
    #
    #
    #
    # grad_diff_DX_0 = J_DX0_X0 @ diff_x0 + J_DX0_X1 @ diff_x1 + J_DX0_X0_init @ diff_x0_init + J_DX0_X1_init @ diff_x1_init + diff_M0 @ J_DX0_M0  + diff_M1@J_DX0_M1
    #
    # grad_diff_DX_1 = J_DX1_X0 @ diff_x0 + J_DX1_X1 @ diff_x1 + J_DX1_X0_init @ diff_x0_init + J_DX1_X1_init @ diff_x1_init + diff_M0 @ J_DX1_M0  + diff_M1 @ J_DX1_M1 # for dM/dx, multiply diff_M first to adjust Jacobian characteristic of a 3x3 matrix

    # print("numerical difference")
    # print("DX_0:", num_diff_DX_0)
    # print("DX_1:", num_diff_DX_1)
    #
    # print("gradient difference")
    # print("DX_0:", grad_diff_DX_0)
    # print("DX_1:", grad_diff_DX_1)


    # ------------------------------------------------------------
    # Common CPU data  (NumPy AND torch share same values)
    # ------------------------------------------------------------
    x0_np = np.array([0.0020, 0.6336, -0.0126]).reshape(3, 1)
    x1_np = np.array([-0.0046, 0.6808, 0.0102]).reshape(3, 1)
    x0i_np = np.array([-0.0108, 0.6790, 0.0035]).reshape(3, 1)
    x1i_np = np.array([-0.0104, 0.6355, 0.0066]).reshape(3, 1)
    M0_np = np.diag([0.7349, 0.7349, 0.7349])
    M1_np = np.diag([0.2651, 0.2651, 0.2651])

    # Torch clones (CPU) ------------------------------------------------
    device = torch.device("cpu")
    dtype = torch.float64
    x0 = torch.tensor(x0_np, dtype=dtype, device=device, requires_grad=True)
    x1 = torch.tensor(x1_np, dtype=dtype, device=device, requires_grad=True)
    x0i = torch.tensor(x0i_np, dtype=dtype, device=device, requires_grad=True)
    x1i = torch.tensor(x1i_np, dtype=dtype, device=device, requires_grad=True)
    M0 = torch.tensor(M0_np, dtype=dtype, device=device, requires_grad=True)
    M1 = torch.tensor(M1_np, dtype=dtype, device=device, requires_grad=True)

    # ------------------------------------------------------------
    # 1)  AUTOGRAD  (forward + backward)
    # ------------------------------------------------------------
    t0_auto = time.perf_counter()
    DX0_t, DX1_t = DX_ICitr_torch(M0, M1, x0, x1, x0i, x1i)
    loss = DX0_t.sum() + DX1_t.sum()
    grads_auto = torch.autograd.grad(loss, (x0, x1, x0i, x1i, M0, M1))
    t_autograd = time.perf_counter() - t0_auto

    # ------------------------------------------------------------
    # 2)  ANALYTICAL  (forward + jacobians + collapse to grad vec)
    # ------------------------------------------------------------
    t0_analytical = time.perf_counter()

    # ---------- forward (same math as torch version) -------------
    DX0_np, DX1_np = func_DX_ICitr(M0_np, M1_np, x0_np, x1_np, x0i_np, x1i_np)
    loss_np = DX0_np.sum() + DX1_np.sum()

    # ---------- jacobians ---------------------------------------
    Jx0, Jx1, Jx1_x0, Jx1_x1 = grad_DX_X_ICitr(M0_np, M1_np, x0_np, x1_np, x0i_np, x1i_np)
    Jx0i, Jx1i, Jx1_x0i, Jx1_x1i = grad_DX_Xinit_ICitr(M0_np, M1_np, x0_np, x1_np, x0i_np, x1i_np)
    JM0_0, JM0_1, JM1_0, JM1_1 = grad_DX_M_ICitr(M0_np, M1_np, x0_np, x1_np, x0i_np, x1i_np)

    # collapse to same 6-vector gradient of the scalar loss
    ones = np.ones((3, 1))
    g_x0 = (Jx0 + Jx1_x0).T @ ones
    g_x1 = (Jx1 + Jx1_x1).T @ ones
    g_x0i = (Jx0i + Jx1_x0i).T @ ones
    g_x1i = (Jx1i + Jx1_x1i).T @ ones
    g_M0 = (JM0_0 + JM1_0).flatten()
    g_M1 = (JM0_1 + JM1_1).flatten()
    grads_analytic = (g_x0, g_x1, g_x0i, g_x1i, g_M0, g_M1)

    t_analytic = time.perf_counter() - t0_analytical

    # ------------------------------------------------------------
    print(f"Autograd   : {t_autograd * 1e6:8.2f}  µs")
    print('autograd',grads_auto)
    print(f"Analytical : {t_analytic * 1e6:8.2f}  µs")
    print('analytical grad',grads_analytic)
    print(f'{t_autograd/t_analytic}times faster')

    # results = []
    #
    # results.append(compute_summary(
    #     name="X_i",
    #     diff_x0=np.array([0.00001, 0, 0]),
    #     diff_x1=np.zeros(3),t_analytic
    #     diff_x0_init=np.zeros(3),
    #     diff_x1_init=np.zeros(3),
    #     diff_M0=np.zeros((3, 3)),
    #     diff_M1=np.zeros((3, 3))
    # ))
    # results.append(compute_summary(
    #     name="X_i+1",
    #     diff_x0=np.zeros(3),
    #     diff_x1=np.array([0.00001, -0.000053, 0.000002]),
    #     diff_x0_init=np.zeros(3),
    #     diff_x1_init=np.zeros(3),
    #     diff_M0=np.zeros((3, 3)),
    #     diff_M1=np.zeros((3, 3))
    # ))
    #
    # results.append(compute_summary(
    #     name="Mi",
    #     diff_x0=np.zeros(3),
    #     diff_x1=np.zeros(3),
    #     diff_x0_init=np.zeros(3),
    #     diff_x1_init=np.zeros(3),
    #     diff_M0=np.array([[0.00001, 0.0, 0.00003], [0.00002, 0, 0.0], [0.00001, 0.0, 0]]),
    #     diff_M1=np.zeros(3)
    # ))
    #
    # results.append(compute_summary(
    #     name="M_i+1",
    #     diff_x0=np.zeros(3),
    #     diff_x1=np.zeros(3),
    #     diff_x0_init=np.zeros(3),
    #     diff_x1_init=np.zeros(3),
    #     diff_M0=np.zeros(3),
    #     diff_M1=0.00001*np.eye(3)
    # ))
    #
    # results.append(compute_summary(
    #     name="Xi_init",
    #     diff_x0=np.zeros(3),
    #     diff_x1=np.zeros(3),
    #     diff_x0_init=np.array([0.00004, 0.00002, -0.000009]),
    #     diff_x1_init=np.zeros(3),
    #     diff_M0=np.zeros((3, 3)),
    #     diff_M1=np.zeros((3, 3))
    # ))
    # results.append(compute_summary(
    #     name="Xi+1_init",
    #     diff_x0=np.zeros(3),
    #     diff_x1=np.zeros(3),
    #     diff_x0_init=np.zeros(3),
    #     diff_x1_init=np.array([-0.00002, -0.000015, 0.000003]),
    #     diff_M0=np.zeros((3, 3)),
    #     diff_M1=np.zeros((3, 3))
    # ))
    # df = pd.DataFrame(results)
    # df.to_csv("gradient_check_results_absolute.csv", index=False)
    # print(df.to_latex(index=False, float_format="%.2e"))



