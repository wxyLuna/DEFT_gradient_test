import numpy as np

def func_DX_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Batch version of Inextensibility Constraint Iterative Function

    # Inputs:
    - M_0: [batch_size, 3, 3] mass matrix of vertex i
    - M_1: [batch_size, 3, 3] mass matrix of vertex i+1
    - X_0: [batch_size, 3, 1] position of vertex i
    - X_1: [batch_size, 3, 1] position of vertex i+1
    - X_0_init: [batch_size, 3, 1] undeformed position of vertex i
    - X_1_init: [batch_size, 3, 1] undeformed position of vertex i+1

    # Outputs:
    - DX_0: [batch_size, 3, 1] position change of vertex i
    - DX_1: [batch_size, 3, 1] position change of vertex i+1
    """
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_length = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_length_init = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)  # [batch_size, 1, 1]

    # Compute DX_0 and DX_1 for each batch
    DX_0 = np.einsum('bij,bjk,bkl->bil', M_1, M_param, Edge) * lambda_param  # [batch_size, 3, 1]
    DX_1 = -np.einsum('bij,bjk,bkl->bil', M_0, M_param, Edge) * lambda_param  # [batch_size, 3, 1]

    return DX_0, DX_1

def grad_DX_X_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the positions X_0 and X_1.

    # Inputs:
    - M_0: [batch_size, 3, 3] mass matrix of vertex i
    - M_1: [batch_size, 3, 3] mass matrix of vertex i+1
    - X_0: [batch_size, 3, 1] position of vertex i
    - X_1: [batch_size, 3, 1] position of vertex i+1
    - X_0_init: [batch_size, 3, 1] undeformed position of vertex i
    - X_1_init: [batch_size, 3, 1] undeformed position of vertex i+1

    # Outputs:
    - grad_00: [batch_size, 3, 3] gradient of DX_0 with respect to X_0
    - grad_01: [batch_size, 3, 3] gradient of DX_0 with respect to X_1
    - grad_10: [batch_size, 3, 3] gradient of DX_1 with respect to X_0
    - grad_11: [batch_size, 3, 3] gradient of DX_1 with respect to X_1
    """
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_length = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_length_init = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)  # [batch_size, 1, 1]

    # Compute gradients for each batch
    grad_00 = -np.einsum('bij,bjk,bkl->bil', M_1, M_param, np.einsum('bij,bkj->bik', Edge, Edge)) * (4 * Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_00 -= M_1 @ M_param * lambda_param

    grad_01 = np.einsum('bij,bjk,bkl->bil', M_1, M_param, np.einsum('bij,bkj->bik', Edge, Edge)) * (4 * Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_01 += M_1 @ M_param * lambda_param

    grad_10 = np.einsum('bij,bjk,bkl->bil', M_0, M_param, np.einsum('bij,bkj->bik', Edge, Edge)) * (4 * Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_10 += M_0 @ M_param * lambda_param

    grad_11 = -np.einsum('bij,bjk,bkl->bil', M_0, M_param, np.einsum('bij,bkj->bik', Edge, Edge)) * (4 * Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_11 -= M_0 @ M_param * lambda_param

    return grad_00, grad_01, grad_10, grad_11

def grad_DX_Xinit_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the undeformed positions X_0_init and X_1_init.

    # Inputs:
    - M_0: [batch_size, 3, 3] mass matrix of vertex i
    - M_1: [batch_size, 3, 3] mass matrix of vertex i+1
    - X_0: [batch_size, 3, 1] position of vertex i
    - X_1: [batch_size, 3, 1] position of vertex i+1
    - X_0_init: [batch_size, 3, 1] undeformed position of vertex i
    - X_1_init: [batch_size, 3, 1] undeformed position of vertex i+1

    # Outputs:
    - grad_00: [batch_size, 3, 3] gradient of DX_0 with respect to X_0_init
    - grad_01: [batch_size, 3, 3] gradient of DX_0 with respect to X_1_init
    - grad_10: [batch_size, 3, 3] gradient of DX_1 with respect to X_0_init
    - grad_11: [batch_size, 3, 3] gradient of DX_1 with respect to X_1_init
    """
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_length = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_length_init = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)  # [batch_size, 1, 1]

    # Compute gradients for each batch
    grad_00 = np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bij,bkj->bik', Edge, Edge_init)) * (4 * Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_01 = -np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bij,bkj->bik', Edge, Edge_init)) * (4 * Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_10 = -np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bij,bkj->bik', Edge, Edge_init)) * (4 * Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    grad_11 = np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bij,bkj->bik', Edge, Edge_init)) * (4 * Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)

    return grad_00, grad_01, grad_10, grad_11

def grad_DX_M_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the mass matrices M_0 and M_1.

    # Inputs:
    - M_0: [batch_size, 3, 3] mass matrix of vertex i
    - M_1: [batch_size, 3, 3] mass matrix of vertex i+1
    - X_0: [batch_size, 3, 1] position of vertex i
    - X_1: [batch_size, 3, 1] position of vertex i+1
    - X_0_init: [batch_size, 3, 1] undeformed position of vertex i
    - X_1_init: [batch_size, 3, 1] undeformed position of vertex i+1

    # Outputs:
    - grad_M_00: [batch_size, 3, 1] gradient of DX_0 with respect to M_0
    - grad_M_01: [batch_size, 3, 1] gradient of DX_0 with respect to M_1
    - grad_M_10: [batch_size, 3, 1] gradient of DX_1 with respect to M_0
    - grad_M_11: [batch_size, 3, 1] gradient of DX_1 with respect to M_1
    """
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_norm = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_init_norm = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    lambda_param = (Edge_norm**2 - Edge_init_norm**2) / (Edge_norm**2 + Edge_init_norm**2)  # [batch_size, 1, 1]

    # Compute gradients for each batch
    grad_M_00 = -np.einsum('bij,bjk,bkl->bil', M_1, M_param @ M_param, Edge) * lambda_param  # [batch_size, 3, 1]
    grad_M_01 = np.einsum('bij,bjk->bik', (np.eye(3) - M_1 @ M_param), M_param @ Edge) * lambda_param  # [batch_size, 3, 1]
    grad_M_10 = -np.einsum('bij,bjk->bik', (np.eye(3) - M_0 @ M_param), M_param @ Edge) * lambda_param  # [batch_size, 3, 1]
    grad_M_11 = np.einsum('bij,bjk,bkl->bil', M_0, M_param @ M_param, Edge) * lambda_param  # [batch_size, 3, 1]

    return grad_M_00, grad_M_01, grad_M_10, grad_M_11