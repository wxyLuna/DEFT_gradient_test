import numpy as np
import torch

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
    Edge = Edge.numpy().transpose(0, 2, 1)

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

# def Inextensibility_Constraint_Enforcement( batch, current_vertices, nominal_length,
#                                            scale, mass_scale, zero_mask_num):
#     """
#     Enforces inextensibility constraints for a single DLO by adjusting vertex positions
#     so that the edge lengths stay near their nominal values.
#
#     Args:
#         batch (int): Batch size (number of rods or scenes).
#         current_vertices (torch.Tensor): Shape (batch, n_vertices, 3).
#         nominal_length (torch.Tensor): Shape (batch, n_edges). The nominal distances between adjacent vertices.
#         DLO_mass (torch.Tensor): (Not used directly here, but can store mass information)
#         clamped_index (torch.Tensor): Indices in the rods to clamp or fix (not used here).
#         scale (torch.Tensor): Scale factors for each edge, shape (batch, n_edges).
#         mass_scale (torch.Tensor): Another scaling for masses, shape (batch, n_edges).
#         zero_mask_num (torch.Tensor): 0/1 or boolean mask indicating which edges are active.
#
#     Returns:
#         current_vertices (torch.Tensor): Updated vertex positions enforcing length constraints.
#     """
#     # Square of the nominal length for each edge
#
#     nominal_length_square = nominal_length * nominal_length
#     print('used inextensibility enforcement')
#
#     # Loop over each edge
#     for i in range(current_vertices.size()[1] - 1):
#
#
#         # Extract the 'edge' vector, masked by zero_mask_num
#         updated_edges = (current_vertices[:, i + 1] - current_vertices[:, i]) * zero_mask_num[:, i].unsqueeze(-1)
#
#         # denominator = L^2 + updated_edges^2
#         denominator = nominal_length_square[:, i] + (updated_edges * updated_edges).sum(dim=1)
#
#         # l ~ measure of inextensibility mismatch
#         l = torch.zeros_like(nominal_length_square[:, i])
#         mask = zero_mask_num[:, i].bool()
#
#         # l = 1 - 2L^2 / (L^2 + |edge|^2)
#         l[mask] = 1 - 2 * nominal_length_square[mask, i] / denominator[mask]
#
#         # # If all edges are within tolerance, skip
#         # are_all_close_to_zero = torch.all(torch.abs(l) < self.tolerance)
#         # if are_all_close_to_zero:
#         #     continue
#
#         # l_cat used for scaling -> shape (batch,) -> repeated
#         l_cat = (l.unsqueeze(-1).repeat(1, 2).view(-1) / scale[:, i])
#         # l_scale -> (batch,) -> expanded for each dimension
#         l_scale = l_cat.unsqueeze(-1).unsqueeze(-1) * mass_scale[:, i]
#
#         # Update vertices in pair: i, i+1
#         #   new_position = old_position + l_scale * 'edge_vector'
#         #   repeated for each vertex in the pair
#         print('updated_edges', updated_edges[0])
#         current_vertices[:, (i, i + 1)] = current_vertices[:, (i, i + 1)] + (
#                 l_scale @ updated_edges.unsqueeze(dim=1)
#                 .repeat(1, 2, 1)
#                 .view(-1, 3, 1)
#         ).view(-1, 2, 3)
#
#     return current_vertices
#
# if __name__ == '__main__':
#     n_branch =1
#     n_vert = 4
#     n_edge =3
#     batch = 1
#     diff_x0 = np.array([0.00001, 0, 0])
#     diff_x1 = np.array([0.0, 0.0, -0.00002])
#
#     x0_init = np.array([-0.0108, 0.6790, 0.0035])
#     x1_init = np.array([-0.0104, 0.6355, 0.0066])
#
#     M0 = np.array([[0.7349, 0.0, 0.0],
#                    [0.0, 0.7349, 0.0],
#                    [0.0, 0.0, 0.7349]])
#
#     M1 = np.array([[0.2651, 0.0, 0.0],
#                    [0.0, 0.2651, 0.0],
#                    [0.0, 0.0, 0.2651]])
#
#
#     # mass_scale = torch.cat((M0, -M1), dim=1).view(-1, n_edge, 3, 3)
#     mass_matrix = (
#             torch.eye(3)
#             .unsqueeze(dim=0)
#             .unsqueeze(dim=0)
#             .repeat(n_branch, n_vert, 1, 1)
#             * (self.mass_diagonal.unsqueeze(dim=-1).unsqueeze(dim=-1))
#     ).unsqueeze(dim=0).repeat(batch, 1, 1, 1, 1).view(-1, n_vert, 3, 3)
#
#
#     print(mass_scale)
