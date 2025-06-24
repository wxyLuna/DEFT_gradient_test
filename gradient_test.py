import numpy as np


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

if __name__ == '__main__':
    # Define sample values:
    x0 = np.array([0.0020, 0.6336, -0.0126])
    x1 = np.array([-0.0046, 0.6808, 0.0102])
    
    diff_x0 = np.array([0.00001, 0, 0]) # z
    diff_x1 = np.array([0.0, 0.0, -0.00002])

    # x0_init = np.array([0.0, 0.0, 0.0])
    # x1_init = np.array([0.0447, 0.0, 0.0])
    x0_init = np.array([ -0.0108,0.6790,0.0035])
    x1_init = np.array([ -0.0104,0.6355,0.0066])
    
    diff_x0_init = np.array([0, 0, 0.00001])
    diff_x1_init = np.array([0,0, -0.00002])

    # Let e1_bar be a scalar constant.
    # e1_bar = 0.0447  # for example
    e1_bar = np.linalg.norm (x1_init-x0_init)# for example

    # Define scale as a 3x3 matrix, for instance:
    M0 = np.array([[0.7349, 0.0, 0.0],
                   [0.0, 0.7349, 0.0],
                   [0.0, 0.0, 0.7349]])
    
    M1 = np.array([[0.2651, 0.0, 0.0],
                   [0.0, 0.2651, 0.0],
                   [0.0, 0.0, 0.2651]])
    
    diff_M0 = 0*np.eye(3)
    diff_M1 = np.array([[0.00001, 0.0, 0.00003],
                   [0.00002, 0, 0.0],
                   [0.00001, 0.0, 0]])

    # Compute the function value and Jacobian.
    DX_0_m, DX_1_m = func_DX_ICitr(M0-diff_M0, M1-diff_M1, x0-diff_x0, x1-diff_x1, x0_init-diff_x0_init, x1_init-diff_x1_init)
    DX_0_p, DX_1_p = func_DX_ICitr(M0+diff_M0, M1+diff_M1, x0+diff_x0, x1+diff_x1, x0_init+diff_x0_init, x1_init+diff_x1_init)
    J_DX0_X0, J_DX0_X1, J_DX1_X0, J_DX1_X1 = grad_DX_X_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    J_DX0_X0_init, J_DX0_X1_init, J_DX1_X0_init, J_DX1_X1_init = grad_DX_Xinit_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    J_DX0_M0, J_DX0_M1, J_DX1_M0, J_DX1_M1 = grad_DX_M_ICitr(M0, M1, x0, x1, x0_init, x1_init)
    
    num_diff_DX_0 = (DX_0_p - DX_0_m)/2
    num_diff_DX_1 = (DX_1_p - DX_1_m)/2


    grad_diff_DX_0 = J_DX0_X0 @ diff_x0 + J_DX0_X1 @ diff_x1 + J_DX0_X0_init @ diff_x0_init + J_DX0_X1_init @ diff_x1_init + diff_M0 @ J_DX0_M0  + diff_M1@J_DX0_M1

    grad_diff_DX_1 = J_DX1_X0 @ diff_x0 + J_DX1_X1 @ diff_x1 + J_DX1_X0_init @ diff_x0_init + J_DX1_X1_init @ diff_x1_init + diff_M0 @ J_DX1_M0  + diff_M1 @ J_DX1_M1 # for dM/dx, multiply diff_M first to adjust Jacobian characteristic of a 3x3 matrix

    print("numerical difference")
    print("DX_0:", num_diff_DX_0)
    print("DX_1:", num_diff_DX_1)

    print("gradient difference")
    print("DX_0:", grad_diff_DX_0)
    print("DX_1:", grad_diff_DX_1)
