import numpy as np
import torch

# Gradient Saver
class BackwardGradientIC:
    def __init__(self, batch, num_vertices):
        # the batch here is actually num_batch * num_branch
        self.grad_DX_X = None
        self.grad_DX_Xinit = None
        self.grad_DX_M = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_X = np.zeros((batch, num_vertices*3, num_vertices*3), dtype=np.float32)
        self.grad_DX_Xinit = np.zeros((batch, num_vertices*3, num_vertices*3), dtype=np.float32)
        self.grad_DX_M = np.zeros((batch, num_vertices*3, num_vertices), dtype=np.float32)
        return

class BackwardGradientDamping:
    def __init__(self, batch, brach, num_vertices):
        # the batch here is actually num_batch * num_branch, while the branch is num_branch
        self.grad_DX_damping = None
        self.reset(batch, branch, num_vertices)
        return

    def reset(self, batch, branch, num_vertices):
        self.grad_DX_damping = np.zeros((batch, num_vertices*3, branch), dtype=np.float32)
        return

# Gradient Solver
    # Inextensibility Constraint Enforcement
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

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    X_0, X_1 = X_0.detach().cpu().numpy(), X_1.detach().cpu().numpy()
    X_0_init, X_1_init = np.asarray(X_0_init), np.asarray(X_1_init)
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

    # Edge_outer = np.einsum('bi,bj->bij', Edge, Edge)

    scale = (4 * Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
    scaled_value = scale[:, np.newaxis]

    # Compute gradients for each batch
    grad_00 = -np.einsum('bij,bjk,bkl->bil', M_1, M_param, np.einsum('bi,bj->bij', Edge, Edge)) * scaled_value
    grad_00 -= M_1 @ M_param * lambda_param[:, np.newaxis]

    grad_01 = np.einsum('bij,bjk,bkl->bil', M_1, M_param, np.einsum('bi,bj->bij', Edge, Edge)) * scaled_value
    grad_01 += M_1 @ M_param * lambda_param[:, np.newaxis]

    grad_10 = np.einsum('bij,bjk,bkl->bil', M_0, M_param, np.einsum('bi,bj->bij', Edge, Edge)) * scaled_value
    grad_10 += M_0 @ M_param * lambda_param[:, np.newaxis]

    grad_11 = -np.einsum('bij,bjk,bkl->bil', M_0, M_param,np.einsum('bi,bj->bij', Edge, Edge)) * scaled_value
    grad_11 -= M_0 @ M_param * lambda_param[:, np.newaxis]

    grad_DX_X = np.concatenate(
        (np.concatenate((grad_00, grad_01), axis=2),
         np.concatenate((grad_10, grad_11), axis=2)),
        axis=1
    )


    return grad_DX_X

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

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    batch_size = M_0.shape[0]
    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    X_0, X_1 = X_0.detach().cpu().numpy(), X_1.detach().cpu().numpy()


    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]

    # Compute Edge and Edge_init for each batch
    Edge = X_1 - X_0  # [batch_size, 3, 1]
    Edge_init = X_1_init - X_0_init  # [batch_size, 3, 1]

    # Compute Edge lengths for each batch
    Edge_length = np.linalg.norm(Edge, axis=1, keepdims=True)  # [batch_size, 1, 1]
    Edge_length_init = np.linalg.norm(Edge_init, axis=1, keepdims=True)  # [batch_size, 1, 1]

    # Compute lambda_param for each batch
    # lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)  # [batch_size, 1, 1]
    scale = (4 * Edge_length ** 2 / (Edge_length ** 2 + Edge_length_init ** 2) ** 2)
    scaled_value = scale[:, np.newaxis]

    # Compute gradients for each batch
    grad_00 = np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_01 = -np.einsum('bij,bjk->bik', M_1 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_10 = -np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value
    grad_11 = np.einsum('bij,bjk->bik', M_0 @ M_param, np.einsum('bi,bj->bij', Edge, Edge_init)) * scaled_value

    grad_DX_X_init = np.concatenate(
        (np.concatenate((grad_00, grad_01), axis=2),
         np.concatenate((grad_10, grad_11), axis=2)),
        axis=1
    )

    return grad_DX_X_init

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

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    batch_size = M_0.shape[0]
    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    X_0, X_1 = X_0.detach().cpu().numpy(), X_1.detach().cpu().numpy()
    X_0_init, X_1_init = np.asarray(X_0_init), np.asarray(X_1_init)

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

    Edge = Edge[:,:,np.newaxis]
    # Compute gradients for each batch
    grad_M_00 = -np.einsum('bij,bjk,bkl->bil', M_1, M_param @ M_param, Edge) * lambda_param[:, np.newaxis]  # [batch_size, 3, 1]
    grad_M_01 = np.einsum('bij,bjk->bik', (np.eye(3) - M_1 @ M_param), M_param @ Edge) * lambda_param[:, np.newaxis]  # [batch_size, 3, 1]
    grad_M_10 = -np.einsum('bij,bjk->bik', (np.eye(3) - M_0 @ M_param), M_param @ Edge) * lambda_param[:, np.newaxis]  # [batch_size, 3, 1]
    grad_M_11 = np.einsum('bij,bjk,bkl->bil', M_0, M_param @ M_param, Edge) * lambda_param[:, np.newaxis]  # [batch_size, 3, 1]

    grad_DX_M = np.concatenate(
        (np.concatenate((grad_M_00, grad_M_01), axis=2),
         np.concatenate((grad_M_10, grad_M_11), axis=2)),
        axis=1
    )
    
    return grad_DX_M

    # Damping
def grad_DX_damping_batch(integration_ratio, dt, b_DLOs_velocity, n_branch):
    """
    Batch version of Gradient of the damping constraint iterative function with respect to the damping coefficients.

    # Inputs:
    - integration_ratio: [1] integration ratio for the damping
    - dt: [1] time step size
    - b_DLOs_velocity: [batch_size, n_vert, 3] velocities of the DLOs

    # Outputs:
    - grad_DX_damping: [batch_size, n_vert*3, n_branch] gradient of DX_damping with respect to the damping coefficients

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    batch_size = b_DLOs_velocity.shape[0]
    b_DLOs_velocity_np = b_DLOs_velocity.detach().cpu().numpy()
    integration_ratio_np = integration_ratio.detach().cpu().numpy()
    dt = dt.detach().cpu().numpy()
    
    # separate the velocities for each branch
    b_DLOs_velocity_expand = np.zeros((batch_size, n_vert, 3, n_branch), dtype=np.float32)
    for i in range(batch_size):
        b_DLOs_velocity_expand[i, :, :, i % n_branch] = b_DLOs_velocity_np[i, :, :]
    b_DLOs_velocity_expand = b_DLOs_velocity_expand.reshape(batch_size, n_vert * 3, n_branch)
    grad_DX_damping = - integration_ratio_np * dt**2 * b_DLOs_velocity_expand

    return grad_DX_damping