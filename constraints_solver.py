import time
from itertools import repeat, permutations

import torch
import pytorch3d
import pytorch3d.transforms.rotation_conversions
from click.core import batch
from numpy.core.defchararray import lower

torch.set_default_dtype(torch.float64)
import torch.nn as nn
from iterative_gradients import func_DX_ICitr_batch, func_DX_ICECitr_batch
import gradients
import numpy as np

torch.set_default_dtype(torch.float64)


class constraints_enforcement(nn.Module):
    """
    A class that enforces various geometric constraints (inextensibility, rotation, coupling)
    on discrete linkage objects (DLOs) or 'rods'. Inherits from PyTorch's nn.Module
    to integrate with common PyTorch workflows.

    Args:
        n_branch (int): Number of 'branches' or rods in the system (if relevant).
    """

    def __init__(self, n_branch):
        super().__init__()
        self.tolerance = 5e-3  # Tolerance threshold for checking small angles or lengths
        self.scale = 10.  # A scaling factor used in some constraints

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """
        Computes the rotation matrix that rotates vec1 into vec2 for each batch/branch in the input.

        Args:
            vec1 (torch.Tensor): Tensor of shape (batch, n_branch, 3) - initial vectors.
            vec2 (torch.Tensor): Tensor of shape (batch, n_branch, 3) - target vectors.

        Returns:
            rotation_matrix (torch.Tensor): Shape (batch, n_branch, 3, 3),
                                            the rotation matrices for each pair (vec1, vec2).
        """
        # 1) Normalize vec1 and vec2
        a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)

        # 2) Cross product (axis of rotation) and dot product (cosine of angle)
        v = torch.cross(a, b, dim=-1)
        c = torch.sum(a * b, dim=-1, keepdim=True)
        s = torch.norm(v, dim=-1, keepdim=True)  # Sine of angle is magnitude of cross product

        # 3) Build skew-symmetric cross-product matrix 'kmat' for each element
        kmat = torch.zeros((vec1.shape[0], vec1.shape[1], 3, 3), dtype=torch.float64)
        kmat[:, :, 0, 1] = -v[:, :, 2]
        kmat[:, :, 0, 2] = v[:, :, 1]
        kmat[:, :, 1, 0] = v[:, :, 2]
        kmat[:, :, 1, 2] = -v[:, :, 0]
        kmat[:, :, 2, 0] = -v[:, :, 1]
        kmat[:, :, 2, 1] = v[:, :, 0]

        # 4) Create identity matrix
        eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).unsqueeze(0).repeat(vec1.shape[0], vec1.shape[1], 1, 1)

        # 5) Rodrigues' rotation formula: R = I + [k] + [k]^2 * ((1 - c) / s^2)
        rotation_matrix = eye + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1)

        # 6) Handle near-zero 's' (parallel or anti-parallel vectors)
        s_zero = (s < 1e-30).squeeze(-1)  # bool mask for s ~ 0
        c_positive = (c > 0).squeeze(-1)  # parallel
        c_negative = (c < 0).squeeze(-1)  # anti-parallel

        # Expand for broadcasting
        s_zero_expanded = s_zero.unsqueeze(-1).unsqueeze(-1).expand_as(eye)
        c_positive_expanded = c_positive.unsqueeze(-1).unsqueeze(-1).expand_as(eye)

        # 7) If vectors are parallel (s=0, c>0), use Identity
        rotation_matrix = torch.where(s_zero_expanded & c_positive_expanded, eye, rotation_matrix)

        # 8) Anti-parallel vectors (s=0, c<0): rotate 180 degrees around any perpendicular axis
        for batch in range(vec1.shape[0]):
            for branch in range(vec1.shape[1]):
                if s_zero[batch, branch] and c_negative[batch, branch]:
                    # Choose a fallback axis for cross product
                    axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
                    if torch.allclose(a[batch, branch], axis):
                        axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
                    # Perpendicular axis to 'a'
                    perp_axis = torch.cross(a[batch, branch], axis)
                    perp_axis = perp_axis / torch.norm(perp_axis)

                    # Construct 180-degree rotation matrix around 'perp_axis'
                    kmat_180 = torch.zeros(3, 3, dtype=torch.float64)
                    kmat_180[0, 1] = -perp_axis[2]
                    kmat_180[0, 2] = perp_axis[1]
                    kmat_180[1, 0] = perp_axis[2]
                    kmat_180[1, 2] = -perp_axis[0]
                    kmat_180[2, 0] = -perp_axis[1]
                    kmat_180[2, 1] = perp_axis[0]
                    # R = I + 2 * kmat_180^2
                    rotation_matrix[batch, branch] = eye[batch, branch] + 2 * torch.matmul(kmat_180, kmat_180)

        return rotation_matrix

    def rotation_matrix_from_vectors_lowerdim(self, vec1, vec2):
        """
        Similar to rotation_matrix_from_vectors, but designed for fewer dimensions
        (batch dimension only, no 'branch' dimension).
        Used for a simpler scenario: shape (batch, 3) for vec1/vec2.

        Args:
            vec1 (torch.Tensor): Shape (batch, 3)
            vec2 (torch.Tensor): Shape (batch, 3)

        Returns:
            rotation_matrix (torch.Tensor): Shape (batch, 3, 3)
        """
        # 1) Normalize inputs
        a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)

        # 2) Cross product & dot product
        v = torch.cross(a, b, dim=-1)
        c = torch.sum(a * b, dim=-1, keepdim=True)
        s = torch.norm(v, dim=-1, keepdim=True)

        # 3) Skew-symmetric cross matrix
        kmat = torch.zeros((vec1.shape[0], 3, 3), dtype=torch.float64)
        kmat[:, 0, 1] = -v[:, 2]
        kmat[:, 0, 2] = v[:, 1]
        kmat[:, 1, 0] = v[:, 2]
        kmat[:, 1, 2] = -v[:, 0]
        kmat[:, 2, 0] = -v[:, 1]
        kmat[:, 2, 1] = v[:, 0]

        # 4) Identity matrix
        eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(vec1.shape[0], 1, 1)

        # 5) Rodrigues' formula with safe check for s^2 == 0
        s_squared = s ** 2
        s_squared_safe = s_squared.clone()
        s_squared_safe[s_squared_safe == 0] = 1  # avoid division by zero
        rotation_matrix = eye + kmat + torch.matmul(kmat, kmat) * ((1 - c) / s_squared_safe).unsqueeze(-1)

        # 6) Handle parallel and anti-parallel vectors
        s_zero = (s.squeeze(-1) < 1e-30)
        c_positive = (c.squeeze(-1) > 0)
        c_negative = (c.squeeze(-1) < 0)

        # Vectors are parallel: identity matrix
        rotation_matrix[s_zero & c_positive] = eye[s_zero & c_positive]

        # Vectors are anti-parallel: 180-degree rotation about some perpendicular axis
        for batch in range(vec1.shape[0]):
            if s_zero[batch] and c_negative[batch]:
                # fallback axis
                not_parallel = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
                if torch.allclose(a[batch], not_parallel):
                    not_parallel = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
                perp_axis = torch.cross(a[batch], not_parallel)
                perp_axis = perp_axis / torch.norm(perp_axis)

                kmat_180 = torch.zeros(3, 3, dtype=torch.float64)
                kmat_180[0, 1] = -perp_axis[2]
                kmat_180[0, 2] = perp_axis[1]
                kmat_180[1, 0] = perp_axis[2]
                kmat_180[1, 2] = -perp_axis[0]
                kmat_180[2, 0] = -perp_axis[1]
                kmat_180[2, 1] = perp_axis[0]
                rotation_matrix[batch] = eye[batch] + 2 * torch.matmul(kmat_180, kmat_180)

        return rotation_matrix

    def rotation_matrix_from_vectors_lower(self, vec1, vec2):
        """
        Another variant of rotation_matrix_from_vectors supporting a single batch dimension
        without an extra "branch" dimension. Very similar to rotation_matrix_from_vectors_lowerdim,
        but uses a slightly different code structure.

        Args:
            vec1 (torch.Tensor): Shape (batch, 3)
            vec2 (torch.Tensor): Shape (batch, 3)

        Returns:
            rotation_matrix (torch.Tensor): Shape (batch, 3, 3)
        """
        # Same steps as above: (1) Normalize, (2) Cross/dot, (3) Skew mat, (4) Identity
        a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
        b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
        v = torch.cross(a, b, dim=-1)
        c = torch.sum(a * b, dim=-1, keepdim=True)
        s = torch.norm(v, dim=-1, keepdim=True)

        kmat = torch.zeros((vec1.shape[0], 3, 3), dtype=torch.float64)
        kmat[:, 0, 1] = -v[:, 2]
        kmat[:, 0, 2] = v[:, 1]
        kmat[:, 1, 0] = v[:, 2]
        kmat[:, 1, 2] = -v[:, 0]
        kmat[:, 2, 0] = -v[:, 1]
        kmat[:, 2, 1] = v[:, 0]

        eye = torch.eye(3, dtype=torch.float64).unsqueeze(0).repeat(vec1.shape[0], 1, 1)

        rotation_matrix = eye + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1)

        s_zero = (s < 1e-30).squeeze(-1)
        c_positive = (c > 0).squeeze(-1)
        c_negative = (c < 0).squeeze(-1)

        s_zero_expanded = s_zero.unsqueeze(-1).unsqueeze(-1).expand_as(eye)
        c_positive_expanded = c_positive.unsqueeze(-1).unsqueeze(-1).expand_as(eye)

        # Parallel => identity
        rotation_matrix = torch.where(s_zero_expanded & c_positive_expanded, eye, rotation_matrix)

        # Anti-parallel => rotate 180 degrees around perpendicular
        for batch in range(vec1.shape[0]):
            if s_zero[batch] and c_negative[batch]:
                axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
                if torch.allclose(a[batch], axis):
                    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
                perp_axis = torch.cross(a[batch], axis)
                perp_axis = perp_axis / torch.norm(perp_axis)

                kmat_180 = torch.zeros(3, 3, dtype=torch.float64)
                kmat_180[0, 1] = -perp_axis[2]
                kmat_180[0, 2] = perp_axis[1]
                kmat_180[1, 0] = perp_axis[2]
                kmat_180[1, 2] = -perp_axis[0]
                kmat_180[2, 0] = -perp_axis[1]
                kmat_180[2, 1] = perp_axis[0]

                rotation_matrix[batch] = eye[batch] + 2 * torch.matmul(kmat_180, kmat_180)

        return rotation_matrix

    def Inextensibility_Constraint_Enforcement(self, batch, current_vertices, nominal_length, DLO_mass, clamped_index,
                                               scale, mass_scale, zero_mask_num, undeformed_vertices, bkgrad, n_branch):
        """
        Enforces inextensibility constraints for a single DLO by adjusting vertex positions
        so that the edge lengths stay near their nominal values.

        Args:
            batch (int): Batch size (number of rods or scenes).
            current_vertices (torch.Tensor): Shape (batch, n_vertices, 3).
            nominal_length (torch.Tensor): Shape (batch, n_edges). The nominal distances between adjacent vertices.
            DLO_mass (torch.Tensor): (Not used directly here, but can store mass information)
            clamped_index (torch.Tensor): Indices in the rods to clamp or fix (not used here).
            scale (torch.Tensor): Scale factors for each edge, shape (batch, n_edges).
            mass_scale (torch.Tensor): Another scaling for masses, shape (batch, n_edges).
            zero_mask_num (torch.Tensor): 0/1 or boolean mask indicating which edges are active.
            undeformed_vertices (torch.Tensor): Reference undeformed vertex positions for computing gradients.
            bkgrad (class): A class or function to store gradients from previous iteration.
            n_branch(int): Number of branches.

        Returns:
            current_vertices (torch.Tensor): Updated vertex positions enforcing length constraints.

        """
        # Square of the nominal length for each edge
        nominal_length_square = nominal_length * nominal_length
        # Initialize gradient storage
        grad_per_ICitr = bkgrad

        # Loop over each edge
        for i in range(current_vertices.size()[1] - 1):
            # Extract the 'edge' vector, masked by zero_mask_num

            updated_edges = (current_vertices[:, i + 1] - current_vertices[:, i]) * zero_mask_num[:, i].unsqueeze(-1)

            # denominator = L^2 + updated_edges^2
            denominator = nominal_length_square[:, i] + (updated_edges * updated_edges).sum(dim=1)

            # l ~ measure of inextensibility mismatch
            l = torch.zeros_like(nominal_length_square[:, i])
            mask = zero_mask_num[:, i].bool()

            # l = 1 - 2L^2 / (L^2 + |edge|^2)
            l[mask] = 1 - 2 * nominal_length_square[mask, i] / denominator[mask]

            # print('nominal_length**2',nominal_length_square[:, i])
            # print('edge **2', (updated_edges * updated_edges).sum(dim=1))
            # print('l',l)

            # If all edges are within tolerance, skip
            are_all_close_to_zero = torch.all(torch.abs(l) < self.tolerance)
            if are_all_close_to_zero:

                continue

            # l_cat used for scaling -> shape (batch,) -> repeated
            l_cat = (l.unsqueeze(-1).repeat(1, 2).view(-1) / scale[:, i])
            # l_scale -> (batch,) -> expanded for each dimension
            l_scale = l_cat.unsqueeze(-1).unsqueeze(-1) * mass_scale[:, i]



            #store pre_updated current_vertices for gradient computation
            current_vertices_copy = current_vertices.clone()

            # Update vertices in pair: i, i+1
            #   new_position = old_position + l_scale * 'edge_vector'
            #   repeated for each vertex in the pair
            current_vertices[:, (i, i + 1)] = current_vertices[:, (i, i + 1)] + (
                    l_scale @ updated_edges.unsqueeze(dim=1)
                    .repeat(1, 2, 1)
                    .view(-1, 3, 1)
            ).view(-1, 2, 3)

            #------------------------Compute gradient for backpropagation------------------------
            # Inextensibility scale factors for DX
            DX_0_scale = scale[:, i][0::2]
            DX_1_scale = scale[:, i][1::2]

            # this calculation is only for checking the ICE equation result vs. the ICE function's output
            # turn on the switch if needed, default is off
            calculate_DX = False
            if calculate_DX:
                #ICE function output for DX
                delta_x = (l_scale @ updated_edges.unsqueeze(dim=1)
                           .repeat(1, 2, 1)
                           .view(-1, 3, 1)
                           ).view(-1, 2, 3)
                dx_0 = delta_x[:, 0, :].unsqueeze(-1)
                dx_1 = delta_x[:, 1, :].unsqueeze(-1)
                #ICE equation output for DX


                DX_0, DX_1, active = func_DX_ICitr_batch(
                    DLO_mass[:, i, :, :], DLO_mass[:, i + 1, :, :],
                    current_vertices_copy[:, i, :][:, :, None], current_vertices_copy[:, i + 1, :][:, :, None],
                    undeformed_vertices[:, i, :][:, :, None], undeformed_vertices[:, i + 1, :][:, :, None], mask, i,  n_branch
                )


                dx_0_np = dx_0.detach().numpy()
                dx_1_np = dx_1.detach().numpy()


                DX_0 /= DX_0_scale.view(-1, 1, 1)
                DX_1 /= DX_1_scale.view(-1, 1, 1)
                DX0_ratio = DX_0/dx_0_np
                DX1_ratio = DX_1/dx_1_np
                print(f'DX0 ratio at edge {i}',DX0_ratio)


            # ___Update the gradient for the current vertices___
            # Gradient of the inextensibility constraint w.r.t. the positions of the two vertices

            grad_DX_X_step = gradients.grad_DX_X_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i, :][:, :, None], current_vertices_copy[:, i + 1, :][:, :, None],
                undeformed_vertices[:, i, :][:, :, None], undeformed_vertices[:, i + 1, :][:, :, None], mask
            )
            # print(f'at edge {i}, grad_DX_X',grad_DX_X_step)
            # print('grad_DX_X_step[:, 0:3, :]',grad_DX_X_step[:, 0:3, :])
            # print('grad_DX_X_step[:, 3:6, :]',grad_DX_X_step[:, 3:6, :])

            grad_DX_X_step[:, 0:3, :] /= DX_0_scale.view(-1, 1, 1).repeat(1, 3, 6)
            grad_DX_X_step[:, 3:6, :] /= DX_1_scale.view(-1, 1, 1).repeat(1, 3, 6)
            grad_DX_X_step = grad_DX_X_step.reshape(batch, n_branch, 6, 6)



            for idx_branch in range(n_branch):
                grad_interest_DX_X = grad_per_ICitr.grad_DX_X[:,
                                      idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * i:
                                      idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * (i + 2),
                                      :].copy()

                grad_chain_passed_DX_X = grad_DX_X_step[:,idx_branch,:,:] @ grad_interest_DX_X


                grad_DX_X_step_expanded = np.concatenate((
                    np.zeros((batch, 6, 3 * (idx_branch * grad_per_ICitr.num_vertices + i))),
                    grad_DX_X_step[:,idx_branch,:,:],
                    np.zeros((batch, 6, 3 * (3 * grad_per_ICitr.num_vertices - i - 2 - idx_branch * grad_per_ICitr.num_vertices)))
                ), axis=2)

                grad_per_ICitr.grad_DX_X[:, idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * i:
                                          idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * (i + 2),
                                          :] = grad_DX_X_step_expanded + grad_interest_DX_X + grad_chain_passed_DX_X

            # ___Update the gradient for the mass scale___
            # Gradient of the inextensibility constraint w.r.t. the mass matrices of the two vertices
            grad_DX_M_step = gradients.grad_DX_M_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i, :][:, :, None], current_vertices_copy[:, i + 1, :][:, :, None],
                undeformed_vertices[:, i, :][:, :, None], undeformed_vertices[:, i + 1, :][:, :, None], mask

            )


            grad_DX_M_step[:, 0:3, :] /= DX_0_scale.view(-1, 1, 1).repeat(1, 3, 2)
            grad_DX_M_step[:, 3:6, :] /= DX_1_scale.view(-1, 1, 1).repeat(1, 3, 2)
            grad_DX_M_step = grad_DX_M_step.reshape(batch, n_branch, 6, 2)

            for idx_branch in range(n_branch):
                grad_interest_DX_M = grad_per_ICitr.grad_DX_M[:, 
                                      idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * i:
                                      idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * (i + 2),
                                      :].copy()

                grad_chain_passed_DX_M = grad_DX_X_step[:,idx_branch,:,:] @ grad_interest_DX_M

                grad_M_X_step_expanded = np.concatenate((
                    np.zeros((batch, 6, idx_branch * grad_per_ICitr.num_vertices + i)),
                    grad_DX_M_step[:,idx_branch,:,:],
                    np.zeros((batch, 6, 3 * grad_per_ICitr.num_vertices - i - 2 - idx_branch * grad_per_ICitr.num_vertices))
                ), axis=2)

                grad_per_ICitr.grad_DX_M[:, idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * i:
                                          idx_branch * 3 * grad_per_ICitr.num_vertices + 3 * (i + 2),
                                          :] = grad_M_X_step_expanded + grad_interest_DX_M + grad_chain_passed_DX_M

        return current_vertices, grad_per_ICitr

    def Inextensibility_Constraint_Enforcement_Coupling(self, parent_vertices, child_vertices, coupling_index,
                                                        coupling_mass_scale, selected_parent_index,
                                                        selected_children_index, parent_mass, children_mass, bkgrad):
        """
        Enforces inextensibility or position constraints between a 'parent' rod and a 'child' rod
        at a specific coupling index.

        Args:
            parent_vertices (torch.Tensor): Shape (batch, n_parent_vertices, 3).
            child_vertices (torch.Tensor): Shape (batch, n_child_vertices, 3).
            coupling_index (torch.Tensor): Indices on the parent rod to couple with children.
            coupling_mass_scale (torch.Tensor): Matrix scale for how parent/child share corrections.
            selected_parent_index (list): Which rods in a bigger scene are 'parents'.
            selected_children_index (list): Which rods in the bigger scene are 'children'.
            parent_mass (torch.Tensor): Parent mass matrices, shape (batch,n_parents, n_vertices, 3).
            children_mass (torch.Tensor): Child mass matrices, shape (batch,n_children, n_vertices, 3).

        Returns:
            b_DLOs_vertices (torch.Tensor): Combined or updated vertices for the rods
                                            after enforcing coupling constraints.
        """
        # Initialize store gradient for backpropagation
        grad_per_ICEC = bkgrad
        # Vector from parent to child's first vertex
        updated_edges = child_vertices[:, 0] - parent_vertices[:, coupling_index].view(-1, 3)
        parent_vertices_copy = parent_vertices.clone()
        child_vertices_copy = child_vertices.clone()
        # coupling_mass_scale => (l1, l2)
        l1 = coupling_mass_scale[:, 0]
        l2 = coupling_mass_scale[:, 1]

        # Update parent's coupling_index
        parent_vertices[:, coupling_index] = parent_vertices[:, coupling_index] + (
                l1 @ updated_edges.unsqueeze(dim=-1)
        ).view(-1, len(coupling_index), 3)

        # Update child's first vertex
        child_vertices[:, 0] = child_vertices[:, 0] + (
                l2 @ updated_edges.unsqueeze(dim=-1)
        ).reshape(-1, 3)

        # Combine back into b_DLOs_vertices for a final representation
        b_DLOs_vertices = torch.empty(len(selected_parent_index) + len(selected_children_index),
                                      parent_vertices.size()[1], 3)
        b_DLOs_vertices[selected_parent_index] = parent_vertices
        b_DLOs_vertices[selected_children_index] = child_vertices

        # ------------------------Compute gradient for backpropagation------------------------
        batch = grad_per_ICEC.batch
        for i, child_idx in zip(coupling_index, selected_children_index):
            pm = parent_mass[:, i]  # (batch, 3, 3)
            cm = children_mass[child_idx-1::2][:,0,:,:] # (batch, 3, 3)
            pv = parent_vertices_copy[:, i:i + 1, :].reshape(batch, 3, 1)  # (batch, 3, 1)
            cv = child_vertices_copy[child_idx-1::2][:,0,:].unsqueeze(-1)  # (batch, 3, 1)

            # this calculation is only for checking the ICEC equation result vs. the ICEC function's output
            # turn on the switch if needed, default is off
            calculate_DX = False
            if calculate_DX:
                # ICE function output for DX

                dx_0 = (l1 @ updated_edges.unsqueeze(dim=-1))[child_idx-1::2]
                dx_1 = (l2 @ updated_edges.unsqueeze(dim=-1))[child_idx-1::2]
                # ICE equation output for DX
                dx_0, dx_1 = dx_0.detach().cpu().numpy(),dx_1.detach().cpu().numpy()
                DX_0, DX_1 = func_DX_ICECitr_batch(pm, cm, pv, cv)


                print('DX parent ratio', DX_0 / dx_0)
                print('DX child ratio', DX_1 / dx_1)

            # ___Update the gradients___
            grad_X_pc_pc_step, grad_X_pc_cc_step,grad_X_cc_pc_step,grad_X_cc_cc_step, grad_DX_X_step = gradients.grad_DX_X_ICEC_batch(pm, cm)  # (batch, 6, 6)
            grad_M_pc_pc_step, grad_M_pc_cc_step, grad_M_cc_pc_step, grad_M_cc_cc_step, grad_DX_M_step = gradients.grad_DX_M_ICEC_batch(pm, cm, pv, cv)


            # p_slice = slice(3 * (selected_parent_index * grad_per_ICEC.num_vertices + i),
            #                 3 * (selected_parent_index * grad_per_ICEC.num_vertices + i) + 3)
            # p_mcol = selected_parent_index * grad_per_ICEC.num_vertices + i
            # c_slice = slice(3 * (child_idx * grad_per_ICEC.num_vertices + 0),
            #                 3 * (child_idx * grad_per_ICEC.num_vertices + 0) + 3)
            # c_mcol = child_idx * grad_per_ICEC.num_vertices + 0
            grad_X_pc_interest_list = []
            grad_X_cc_interest_list = []

            for b in range(batch):
                p_index = selected_parent_index[b].item()

                p_start = 3 * (i)
                p_end = p_start + 3
                c_start = 3 * (child_idx * grad_per_ICEC.num_vertices + 0)
                c_end = c_start + 3

                grad_X_pc_interest_list.append(grad_per_ICEC.grad_DX_X[b, p_start:p_end, :].copy())
                grad_X_cc_interest_list.append(grad_per_ICEC.grad_DX_X[b, c_start:c_end, :].copy())

            # Stack into tensor shape (B, 3, N)
            grad_X_pc_interest = np.stack(grad_X_pc_interest_list, axis=0)
            grad_X_cc_interest = np.stack(grad_X_cc_interest_list, axis=0)

            # ___Update the gradient for the current vertices___
            grad_DX_X_interest = np.concatenate((grad_X_pc_interest, grad_X_cc_interest), axis=1)

            grad_chain_passed_DX_X = grad_DX_X_step @ grad_DX_X_interest

            grad_DX_X_step_expanded = np.concatenate((
                np.zeros((batch, 6, 3 * i)),
                grad_DX_X_step[:, :, 0:3],
                np.zeros((batch, 6, 3 * (child_idx * grad_per_ICEC.num_vertices - i - 1))),
                grad_DX_X_step[:, :, 3:6],
                np.zeros((batch, 6, 3 * (grad_per_ICEC.num_branch * grad_per_ICEC.num_vertices - 1 - child_idx * grad_per_ICEC.num_vertices)))
            ), axis=2)

            grad_step_DX_X = grad_DX_X_step_expanded + grad_DX_X_interest + grad_chain_passed_DX_X

            for b in range(batch):
                p_index = selected_parent_index[b].item()

                p_start = 3 * (i)
                p_end = p_start + 3
                c_start = 3 * (child_idx * grad_per_ICEC.num_vertices + 0)
                c_end = c_start + 3
                grad_per_ICEC.grad_DX_X[:, p_start:p_end, :] = grad_step_DX_X[:, :3, :]
                grad_per_ICEC.grad_DX_X[:, c_start:c_end, :] = grad_step_DX_X[:, 3:, :]

            # ___Update the gradient for the mass matrices___
            grad_M_pc_interest_list = []
            grad_M_cc_interest_list = []
            for b in range(batch):
                p_index = selected_parent_index[b].item()

                p_start = 3 * (i)
                p_end = p_start + 3
                c_start = 3 * (child_idx * grad_per_ICEC.num_vertices + 0)
                c_end = c_start + 3

                grad_M_pc_interest_list.append(grad_per_ICEC.grad_DX_M[b, p_start:p_end, :].copy())
                grad_M_cc_interest_list.append(grad_per_ICEC.grad_DX_M[b, c_start:c_end, :].copy())

            # Stack into tensor shape (B, 3, N)
            grad_M_pc_interest = np.stack(grad_M_pc_interest_list, axis=0)
            grad_M_cc_interest = np.stack(grad_M_cc_interest_list, axis=0)

            grad_DX_M_interest = np.concatenate((grad_M_pc_interest, grad_M_cc_interest), axis=1)

            grad_chain_passed_DX_M = grad_DX_X_step @ grad_DX_M_interest

            grad_DX_M_step_expanded = np.concatenate((
                np.zeros((batch, 6, i)),
                grad_DX_M_step[:, :, 0:1],
                np.zeros((batch, 6, child_idx * grad_per_ICEC.num_vertices - i - 1)),
                grad_DX_M_step[:, :, 1:2],
                np.zeros((batch, 6, grad_per_ICEC.num_branch * grad_per_ICEC.num_vertices - 1 - child_idx * grad_per_ICEC.num_vertices))
            ), axis=2)

            grad_step_DX_M = grad_DX_M_step_expanded + grad_DX_M_interest + grad_chain_passed_DX_M
            for b in range(batch):
                p_index = selected_parent_index[b].item()

                p_start = 3 * (i)
                p_end = p_start + 3
                c_start = 3 * (child_idx * grad_per_ICEC.num_vertices + 0)
                c_end = c_start + 3
                grad_per_ICEC.grad_DX_M[:, p_start:p_end, :] = grad_step_DX_M[:, :3, :]
                grad_per_ICEC.grad_DX_M[:, c_start:c_end, :] = grad_step_DX_M[:, 3:, :]

        return b_DLOs_vertices, grad_per_ICEC

    def quaternion_magnitude(self, quaternion):
        """
        Calculate the magnitude (norm) of a quaternion.

        Args:
            quaternion (torch.Tensor): Shape (..., 4), last dim is (w, x, y, z).

        Returns:
            torch.Tensor: Magnitude of the quaternion(s).
        """
        assert quaternion.shape[-1] == 4, "Quaternion should have 4 components (w, x, y, z)"
        magnitude = torch.sqrt(torch.sum(quaternion ** 2, dim=-1))
        return magnitude

    def Rotation_Constraints_Enforcement_Parent_Children(
            self,
            parent_vertices, parent_orientations, previous_parent_vertices,
            children_vertices, children_orientations, previous_children_vertices,
            parent_MOIs, children_MOIs, index_selection, selected_children_index, parent_MOI_index, momentum_scale_previous, n_vert, bkgrad
    ):
        """
        Enforces rotational constraints (continuity) between parent and child rods
        based on how edges have changed from a 'previous' iteration/state to the current one.

        Args:
            parent_vertices (torch.Tensor): Current parent rod vertices.
            parent_orientations (torch.Tensor): Current parent rod orientations (quaternions).
            previous_parent_vertices (torch.Tensor): Previous parent rod vertices.
            children_vertices (torch.Tensor): Current child rod vertices.
            children_orientations (torch.Tensor): Current child rod orientations.
            previous_children_vertices (torch.Tensor): Previous child rod vertices.
            parent_MOIs (torch.Tensor): Parent moments of inertia (not fully used here).
            children_MOIs (torch.Tensor): Child moments of inertia.
            index_selection (torch.Tensor): Indices of the parent rods to apply constraints to.
            selected_children_index (list): Indices of the child rods in the larger scene.
            parent_MOI_index (torch.Tensor): Indices for selecting from parent_MOIs.
            momentum_scale_previous (torch.Tensor): Scale factors for rotational momentum-based correction.

        Returns:
            Tuple of updated parent_vertices, parent_orientations, children_vertices, children_orientations.
        """
        parent_vertices_copy = parent_vertices.clone().detach().numpy()
        children_vertices_copy = children_vertices.clone().detach().numpy()
        parent_orientations_copy = parent_orientations.clone().detach().numpy()
        children_orientations_copy = children_orientations.clone()
        previous_parent_vertices_copy = previous_parent_vertices.clone().detach().numpy()
        previous_children_vertices_copy = previous_children_vertices.clone().detach().numpy()

        n_child_branch = children_vertices.shape[1]
        RCEPC_gradient = gradients.RCEPC_gradient()


        batch = parent_vertices.size()[0]
        n_children = len(index_selection)
        grad_per_RCEPC = bkgrad

        # 1) Collect 'previous' edges and 'current' edges from both parent and children rods
        previous_edges = torch.cat(
            (
                previous_parent_vertices[:, index_selection + 1] - previous_parent_vertices[:, index_selection],
                previous_children_vertices[:, :, 1] - previous_children_vertices[:, :, 0]
            ),
            dim=0
        ).view(-1, 3)

        current_edges = torch.cat(
            (
                parent_vertices[:, index_selection + 1] - parent_vertices[:, index_selection],
                children_vertices[:, :, 1] - children_vertices[:, :, 0]
            ),
            dim=0
        ).view(-1, 3)

        # 2) Collect current orientations, then compute quaternion that rotates 'previous_edges' to 'current_edges'
        orientations = torch.cat((parent_orientations[:, index_selection], children_orientations), dim=0).view(-1, 4)
        quaternion = pytorch3d.transforms.matrix_to_quaternion(
            self.rotation_matrix_from_vectors_lowerdim(previous_edges, current_edges)
        )


        # 3) Combine new rotation quaternion with existing orientation
        quaternion_magnitude = self.quaternion_magnitude(quaternion)
        # (Optional early exit if all are within tolerance, commented out here)
        orientations = pytorch3d.transforms.quaternion_multiply(quaternion, orientations)

        # 4) Split updated orientations back into parent/child
        parent_orientations[:, index_selection] = orientations.view(2 * batch, -1, 4)[:batch]
        children_orientations = orientations.view(2 * batch, -1, 4)[batch:]

        # 5) Re-order parent vertices for rotation application
        parent_desired_order = torch.cat((index_selection.unsqueeze(0), index_selection.unsqueeze(0) + 1),
                                         dim=0).T.flatten()
        parent_rod_vertices = parent_vertices[:, parent_desired_order]
        children_rod_vertices = children_vertices[:, :, 0:2].reshape(-1, children_vertices.size()[1] * 2, 3)

        # 6) Apply further rotation updates based on momentum scale
        parent_rod_vertices, parent_rod_quaternion, children_rod_vertices, children_orientations = self.apply_rotation(
            batch, n_children,
            parent_orientations[:, index_selection],  # sub-set of parent orientations
            children_orientations,
            parent_MOIs[parent_MOI_index], children_MOIs,
            parent_rod_vertices, children_rod_vertices,
            momentum_scale_previous
        )

        # 7) Put updated vertices and orientations back in place
        parent_vertices[:, parent_desired_order] = parent_rod_vertices
        parent_orientations[:, index_selection] = parent_rod_quaternion.view(batch, n_children, 4)
        children_vertices[:, :, 0:2] = children_rod_vertices.reshape(-1, children_vertices.size()[1], 2, 3)

        # -------------------------------gradient implementation-------------------------------


        DR = self.rotation_matrix_from_vectors_lowerdim(previous_edges, current_edges)
        DRpc_indices = []
        DRcc_indices = []

        for b in range(batch):
            base = b * 2 * n_child_branch
            for i in range(n_child_branch):
                DRpc_indices.append(base + i)
                DRcc_indices.append(base + n_child_branch + i)
        DRpc = DR[DRpc_indices].reshape(batch,n_child_branch,3,3)

        DRcc = DR[DRcc_indices].reshape(batch,n_child_branch,3,3)


        pos_map = {int(idx.item()): pos for pos, idx in enumerate(index_selection)}
        children_vertices_copy = children_vertices_copy.reshape(-1, n_vert,3)
        previous_children_vertices_copy = previous_children_vertices_copy.reshape(-1, n_vert,3)
        #initialize the grad_DX_X_step and grad_DX_MOI_step matrix for RCEPC
        grad_DX_X_step = np.zeros((batch, 3 * n_vert * grad_per_RCEPC.num_branch, 3 * n_vert * grad_per_RCEPC.num_branch))
        grad_DX_MOI_step = np.zeros((batch, 3 * n_vert * grad_per_RCEPC.num_branch, 3 * n_vert * grad_per_RCEPC.num_branch))
        for i, child_idx in zip(index_selection, selected_children_index):

            moi_index = pos_map[int(i.item())]
            pmoi = parent_MOIs[parent_MOI_index][moi_index].repeat(batch,1,1)  # (batch, 3, 3)
            cmoi = children_MOIs[moi_index].repeat(batch,1,1)  # (batch, 3, 3)
            pv_0 = parent_vertices_copy[:, i:i + 1, :].reshape(batch, 3)  # (batch, 3)
            pv_1 = parent_vertices_copy[:, i + 1:i + 2, :].reshape(batch, 3)  # (batch, 3)
            cv_0 = children_vertices_copy[child_idx - 1::2][:, 0, :]# (batch, 3)
            cv_1 = children_vertices_copy[child_idx - 1::2][:, 1, :] # (batch, 3)
            pv_init_0 = previous_parent_vertices_copy[:, i:i + 1, :].reshape(batch, 3)  # (batch, 3)
            pv_init_1 = previous_parent_vertices_copy[:, i + 1:i + 2, :].reshape(batch, 3)  # (batch, 3)
            cv_init_0 = previous_children_vertices_copy[child_idx - 1::2][:, 0, :]
            cv_init_1 = previous_children_vertices_copy[child_idx - 1::2][:, 1, :]
            rpc = pv_1 - pv_0
            rcc = cv_1 - cv_0

            Rcc = pytorch3d.transforms.quaternion_to_matrix(children_orientations_copy)[:,moi_index,:,:]

            DRpc_interest = DRpc[:, moi_index, :, :]
            DRcc_interest = DRcc[:, moi_index, :, :]
            Drpc = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(DRpc_interest)
            Drcc = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(DRcc_interest)

            DR_pc_cc = torch.matmul(DRpc_interest, torch.linalg.inv(DRcc_interest))
            Dr = pytorch3d.transforms.rotation_conversions.matrix_to_axis_angle(DR_pc_cc)


            Rcc = Rcc.detach().numpy()
            DRpc_interest = DRpc_interest.detach().numpy()
            DRcc_interest = DRcc_interest.detach().numpy()
            Drpc = Drpc.detach().numpy()
            Drcc = Drcc.detach().numpy()
            Dr = Dr.detach().numpy()
            pmoi = pmoi.detach().numpy()
            cmoi = cmoi.detach().numpy()

            #update DX/MOI, DX/X]

            J_00, J_01, J_02, J_03, J_10, J_11, J_12, J_13, J = RCEPC_gradient.gradient_DX_X(pv_0, pv_1, cv_0, cv_1,
                                                                                             DRpc_interest, DRcc_interest, Drpc, Drcc, Dr,
                                                                                             pmoi, cmoi, Rcc, rpc, rcc,
                                                                                             pv_init_0, pv_init_1,
                                                                                             cv_init_0, cv_init_1)
            grad_DX_X_step_pc = np.concatenate((
                np.zeros((batch, 3, 3 * i)),
                J_00, J_01,
                np.zeros((batch, 3, 3 * (child_idx * n_vert - i - 2))),
                J_02, J_03,
                np.zeros((batch, 3, 3 * (grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert)))
            ), axis=2)
            grad_DX_X_step_cc = np.concatenate((
                np.zeros((batch, 3, 3 * i)),
                J_10, J_11,
                np.zeros((batch, 3, 3 * (child_idx * n_vert - i - 2))),
                J_12, J_13,
                np.zeros((batch, 3, 3 * (grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert)))
            ), axis=2)

            # a very sketchy way to insert the grad step into the full grad_DX_X matrix
            # obtain the intermediate grad_DX_X with each pair of parent-child update

            grad_DX_X_step += np.concatenate((np.zeros((batch,3*(i+1),3*n_vert*grad_per_RCEPC.num_branch)),
                                            grad_DX_X_step_pc,
                                            np.zeros((batch,3*(child_idx*n_vert-i-1),3*n_vert*grad_per_RCEPC.num_branch)),
                                            grad_DX_X_step_cc,
                                            np.zeros((batch,3*(grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert),3*n_vert*grad_per_RCEPC.num_branch))), axis=1)
            # grad_DX_X_step = np.concatenate((grad_DX_X_step_pc,
            #                                np.zeros((batch, 3 * (n_vert - i - child_idx), 3 * n_vert * grad_per_RCEPC.num_branch)),
            #                                grad_DX_X_step_cc), axis=1)


            J_00, J_01, J_10, J_11, J = RCEPC_gradient.gradient_DX_MOI(pv_0, pv_1,cv_0, cv_1,
                                                                       DRpc_interest, DRcc_interest, Drpc, Drcc, Dr,
                                                                       pmoi, cmoi)
            grad_DX_MOI_step_pc = np.concatenate((
                np.zeros((batch, 3, 3 * i)),
                J_00, J_01,
                np.zeros((batch, 3, 3 * (child_idx * n_vert - i - 2))),
                J_02, J_03,
                np.zeros((batch, 3, 3 * (grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert)))
            ), axis=2)
            grad_DX_MOI_step_cc = np.concatenate((
                np.zeros((batch, 3, 3 * i)),
                J_10, J_11,
                np.zeros((batch, 3, 3 * (child_idx * n_vert - i - 2))),
                J_12, J_13,
                np.zeros((batch, 3, 3 * (grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert)))
            ), axis=2)
            grad_DX_MOI_step += np.concatenate((np.zeros((batch, 3 * (i + 1), 3 * n_vert * grad_per_RCEPC.num_branch)),
                                              grad_DX_MOI_step_pc,
                                              np.zeros((batch, 3 * (child_idx * n_vert - i - 1), 3 * n_vert * grad_per_RCEPC.num_branch)),
                                              grad_DX_MOI_step_cc,
                                              np.zeros((batch, 3 * (grad_per_RCEPC.num_branch * n_vert - 2 - child_idx * n_vert),3 * n_vert * grad_per_RCEPC.num_branch))), axis=1)

        return parent_vertices, parent_orientations, children_vertices, children_orientations.view(batch, n_children, 4)

    def apply_rotation(
            self, batch, n_children, edge_q1, edge_q2, rod_MOI1, rod_MOI2,
            rods_vertices1, rods_vertices2, momentum_scale
    ):
        """
        Applies a rotation update to rods based on quaternion differences between
        parent edge orientation (edge_q1) and child edge orientation (edge_q2).

        Args:
            batch (int): Number of samples.
            n_children (int): Number of rods or child edges to process.
            edge_q1 (torch.Tensor): Parent's edge quaternions, shape (batch*n_children, 4).
            edge_q2 (torch.Tensor): Child's edge quaternions, same shape.
            rod_MOI1 (torch.Tensor): Parent's moment of inertia (not fully used here).
            rod_MOI2 (torch.Tensor): Child's moment of inertia.
            rods_vertices1 (torch.Tensor): Parent rod vertex coordinates.
            rods_vertices2 (torch.Tensor): Child rod vertex coordinates.
            momentum_scale (torch.Tensor): A matrix for adjusting how rotation is applied
                                           based on some momentum factor.

        Returns:
            rods_vertices1, rod_orientation1, rods_vertices2, rod_orientation2: Updated rods and their new orientations.
        """
        # 1) Flatten or combine edge quaternions
        edge_q1 = edge_q1.view(-1, 4)
        edge_q2 = edge_q2.view(-1, 4)
        edge_q = torch.cat((edge_q1, edge_q2), 1).view(-1, 4)

        # 2) Compute delta quaternion (difference)
        updated_quaternion = pytorch3d.transforms.quaternion_multiply(
            edge_q1.clone(),
            pytorch3d.transforms.quaternion_invert(edge_q2)
        )
        # Convert delta quaternion to axis-angle, then scale by momentum_scale
        delta_angular = pytorch3d.transforms.rotation_conversions.quaternion_to_axis_angle(updated_quaternion).view(-1,
                                                                                                                    1,
                                                                                                                    3)
        delta_angular = delta_angular.repeat(1, 2, 1).view(-1, 3)
        delta_angular_rod = (momentum_scale @ delta_angular.unsqueeze(dim=-1)).view(-1, 3)

        # 3) Convert that scaled axis-angle back to a quaternion, separate parent & child
        angular_change_quaternion_rod = pytorch3d.transforms.rotation_conversions.axis_angle_to_quaternion(
            delta_angular_rod
        ).view(-1, 2, 4)

        # 4) Multiply new rotation quaternions with existing edge quaternions
        orientation = pytorch3d.transforms.quaternion_multiply(
            angular_change_quaternion_rod.clone().view(-1, 4), edge_q.clone()
        ).view(n_children * batch, 2, 4)
        rod_orientation1, rod_orientation2 = orientation[:, 0], orientation[:, 1]

        # 5) Reshape rods
        angular_change_quaternion_rod = angular_change_quaternion_rod.view(batch, n_children, 2, 4)
        rods_vertices1 = rods_vertices1.view(batch, n_children, 2, 3)
        rods_vertices2 = rods_vertices2.view(batch, n_children, 2, 3)

        # 6) Combine rods for consistent rotation application
        rods_vertices = torch.stack([rods_vertices1, rods_vertices2], dim=2)
        # => shape: [batch, n_children, 2(rods), 2(vertices), 3]

        # 7) Compute each rod's origin so we can rotate around the rod's base
        rod_vertices_origin = rods_vertices[:, :, :, 0:1, :]  # shape: [batch, n_children, 2, 1, 3]
        rod_vertices_originated = rods_vertices - rod_vertices_origin

        angular_change_quaternion_rod_expanded = angular_change_quaternion_rod.unsqueeze(dim=3).expand(-1, -1, -1, 2,
                                                                                                       -1)
        # => shape: [batch, n_children, 2, 2(vertices), 4]

        # 8) Apply rotation to each vertex
        rod_vertices_rotated = pytorch3d.transforms.quaternion_apply(
            angular_change_quaternion_rod_expanded.reshape(-1, 4),
            rod_vertices_originated.reshape(-1, 3)
        ).view(batch, n_children, 2, 2, 3)

        # 9) Add back origin
        rods_vertices_updated = rod_vertices_rotated + rod_vertices_origin

        # 10) Separate the updated rods
        rods_vertices1 = rods_vertices_updated[:, :, 0, :, :].reshape(batch, n_children * 2, 3)
        rods_vertices2 = rods_vertices_updated[:, :, 1, :, :].reshape(batch, n_children * 2, 3)

        return rods_vertices1, rod_orientation1, rods_vertices2, rod_orientation2
