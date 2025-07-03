import time
from itertools import repeat, permutations

import torch
import pytorch3d
import pytorch3d.transforms.rotation_conversions
from click.core import batch
from numpy.core.defchararray import lower
import numpy as np

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import gradient_test
import iterative_gradients

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
        self.undeformed_BDLO = torch.tensor([[[-0.6790, -0.6355, -0.5595, -0.4539, -0.3688, -0.2776, -0.1857,
                                          -0.0991, 0.0102, 0.0808, 0.1357, 0.2081, 0.2404, -0.4279,
                                          -0.4880, -0.5394, -0.5559, 0.0698, 0.0991, 0.1125]],
                                        [[0.0035, -0.0066, -0.0285, -0.0349, -0.0704, -0.0663, -0.0744,
                                          -0.0957, -0.0702, -0.0592, -0.0452, -0.0236, -0.0134, -0.0813,
                                          -0.1233, -0.1875, -0.2178, -0.1044, -0.1858, -0.2165]],
                                        [[0.0108, 0.0104, 0.0083, 0.0104, 0.0083, 0.0145, 0.0133,
                                          0.0198, 0.0155, 0.0231, 0.0199, 0.0154, 0.0169, 0.0160,
                                          0.0153, 0.0090, 0.0121, 0.0205, 0.0155, 0.0148]]]).permute(1, 2, 0)
        n_parent_vertices = 13
        n_children_vertices= (5,4)
        self.parent_vertices_undeform = self.undeformed_BDLO[:, :n_parent_vertices]
        self.child1_vertices_undeform = self.undeformed_BDLO[:, n_parent_vertices: n_parent_vertices + n_children_vertices[0] - 1]
        self.child2_vertices_undeform = self.undeformed_BDLO[:, n_parent_vertices + n_children_vertices[0] - 1:]

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
                                               scale, mass_scale, zero_mask_num ):
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

        Returns:
            current_vertices (torch.Tensor): Updated vertex positions enforcing length constraints.
        """
        # Square of the nominal length for each edge

        nominal_length_square = nominal_length * nominal_length
        print('used inextensibility enforcement')

        # Loop over each edge
        for i in range(current_vertices.size()[1] - 1):


            # current_vertices[:, i + 1] += diff_x1
            # current_vertices[:, i ] += diff_x0


            # Extract the 'edge' vector, masked by zero_mask_num
            updated_edges = (current_vertices[:, i + 1] - current_vertices[:, i]) * zero_mask_num[:, i].unsqueeze(-1)

            # denominator = L^2 + updated_edges^2
            denominator = nominal_length_square[:, i] + (updated_edges * updated_edges).sum(dim=1)

            # l ~ measure of inextensibility mismatch
            l = torch.zeros_like(nominal_length_square[:, i])
            mask = zero_mask_num[:, i].bool()

            # l = 1 - 2L^2 / (L^2 + |edge|^2)
            l[mask] = 1 - 2 * nominal_length_square[mask, i] / denominator[mask]

            # If all edges are within tolerance, skip
            are_all_close_to_zero = torch.all(torch.abs(l) < self.tolerance)
            if are_all_close_to_zero:
                continue

            # l_cat used for scaling -> shape (batch,) -> repeated
            l_cat = (l.unsqueeze(-1).repeat(1, 2).view(-1) / scale[:, i])
            # l_scale -> (batch,) -> expanded for each dimension
            l_scale = l_cat.unsqueeze(-1).unsqueeze(-1) * mass_scale[:, i]

            '''applying gradient check'''


            # M0, M1 = mass_matrix[:, i], mass_matrix[:, i + 1]
            # # Mask where both M0 and M1 are not zero
            # mask = ~((M0 == 0).all(dim=(1, 2)) | (M1 == 0).all(dim=(1, 2)))
            # print('M0',M0)
            if not mask.any():
                continue

            # print(f'{i}th parent vert',self.parent_vertices_undeform[:,i])
            # print(f'{i}th child1 vert',self.child1_vertices_undeform[:,i])
            # print(f'{i}th child1 vert', self.child2_vertices_undeform[:, i])


            # DX_0, DX_1=iterative_gradients.func_DX_ICitr_batch(M0[mask],M1[mask],current_vertices[:, i ].unsqueeze(1) ,current_vertices[:, i +1].unsqueeze(1),x_undeform[:,i],x_undeform[:,i+1])
            # print('DX_0',DX_0.squeeze() )
            # print('DX_1', DX_1.squeeze())
            '''gradient check end'''
            # Update vertices in pair: i, i+1
            #   new_position = old_position + l_scale * 'edge_vector'
            #   repeated for each vertex in the pair


            current_vertices[:, (i, i + 1)] = current_vertices[:, (i, i + 1)] + (
                    l_scale @ updated_edges.unsqueeze(dim=1)
                    .repeat(1, 2, 1)
                    .view(-1, 3, 1)
            ).view(-1, 2, 3)

        return current_vertices

