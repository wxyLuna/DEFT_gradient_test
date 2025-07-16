import time
from itertools import repeat, permutations

import numpy as np
import torch
import pytorch3d
import pytorch3d.transforms.rotation_conversions
from click.core import batch
from numpy.core.defchararray import lower
import numpy as np

torch.set_default_dtype(torch.float64)
import torch.nn as nn

import gradient_IC
import gradient_saver
from iterative_gradients import func_DX_ICitr_batch

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
                                               scale, mass_scale, zero_mask_num, undeformed_vertices, bkgrad, n_branch):
        """
        Enforces inextensibility constraints for a single DLO by adjusting vertex positions
        so that the edge lengths stay near their nominal values.

        Args:
            batch (int): Batch size (number of rods or scenes).
            current_vertices (torch.Tensor): Shape (batch, n_vertices, 3).
            nominal_length (torch.Tensor): Shape (batch, n_edges). The nominal distances between adjacent vertices.
            DLO_mass (torch.Tensor): (Not used directly here, but can store mass information for gradient calculations).
            clamped_index (torch.Tensor): Indices in the rods to clamp or fix (not used here).
            scale (torch.Tensor): Scale factors for each edge, shape (batch, n_edges).
            mass_scale (torch.Tensor): Another scaling for masses, shape (batch, n_edges).
            zero_mask_num (torch.Tensor): 0/1 or boolean mask indicating which edges are active.

        Returns:
            current_vertices (torch.Tensor): Updated vertex positions enforcing length constraints.
        """
        # Square of the nominal length for each edge



        nominal_length_square = nominal_length * nominal_length

        # grad_per_ICitr = gradient_saver.BackwardGradientIC(current_vertices.size()[1])
        grad_per_ICitr = bkgrad
        undeformed_vertices = undeformed_vertices.repeat(batch, 1, 1)

        # Loop over each edge
        for i in range(current_vertices.size()[1] - 1):
        # for i in range(3):


            # Extract the 'edge' vector, masked by zero_mask_num
            updated_edges = (current_vertices[:, i + 1] - current_vertices[:, i]) * zero_mask_num[:, i].unsqueeze(-1)






            # denominator = L^2 + updated_edges^2
            denominator = nominal_length_square[:, i] + (updated_edges * updated_edges).sum(dim=1)#iter_Grad has higher precision


            # l ~ measure of inextensibility mismatch
            l = torch.zeros_like(nominal_length_square[:, i])
            mask = zero_mask_num[:, i].bool()

            # l = 1 - 2L^2 / (L^2 + |edge|^2)
            l[mask] = 1 - 2 * nominal_length_square[mask, i] / denominator[mask]
            # print('l:', l) # in some occasion is off by 1e-4



            # If all edges are within tolerance, skip
            are_all_close_to_zero = torch.all(torch.abs(l) < self.tolerance)
            if are_all_close_to_zero:
                print('all edges are within tolerance, skipping inextensibility constraint enforcement')
                continue

            # l_cat used for scaling -> shape (batch,) -> repeated
            #scale = 1e20*([[1.0, 0.0, 0.0, 1.0]])


            l_cat = (l.unsqueeze(-1).repeat(1, 2).view(-1) / scale[:, i])


            # l_scale -> (batch,) -> expanded for each dimension
            l_scale = l_cat.unsqueeze(-1).unsqueeze(-1) * mass_scale[:, i]

            current_vertices_copy = current_vertices.clone()



            # Update vertices in pair: i, i+1
            #   new_position = old_position + l_scale * 'edge_vector'
            #   repeated for each vertex in the pair
            current_vertices[:, (i, i + 1)] = current_vertices[:, (i, i + 1)] + (
                    l_scale @ updated_edges.unsqueeze(dim=1)
                    .repeat(1, 2, 1)
                    .view(-1, 3, 1)
            ).view(-1, 2, 3)
            delta_x = (l_scale @ updated_edges.unsqueeze(dim=1)
                    .repeat(1, 2, 1)
                    .view(-1, 3, 1)
            ).view(-1, 2, 3)
            # print('DX within inextensibility constraint enforcement:', delta_x)
            # compute func_DX for sanity check
            DX_0, DX_1 = func_DX_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i], current_vertices_copy[:, i + 1],
                undeformed_vertices[:, i], undeformed_vertices[:, i + 1],
            )
            DX_0 /= scale[:, i]
            DX_1  /= scale[:, i]
            # print('DX_0 properly scaled',DX_0)
            # print('DX_1 properly scaled',DX_1)
            # print('multiple within IC', DX_0 / delta_x[:, 0, :].unsqueeze(-1),DX_1 / delta_x[:, 1, :].unsqueeze(-1))





            # Update the gradient for the current vertices


            grad_DX_X_step = gradient_IC.grad_DX_X_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i], current_vertices_copy[:, i + 1],
                undeformed_vertices[:, i], undeformed_vertices[:, i + 1],
            )
            grad_DX_X_step /= np.array(scale[:, i])

            grad_interest_DX_X = grad_per_ICitr.grad_DX_X[:, 3 * i: 3 * (i + 2), :].copy()

            grad_chain_passed_DX_X = grad_DX_X_step @ grad_interest_DX_X
            grad_step_DX_X = np.concatenate((
                np.zeros((n_branch * batch, 6, 3 * i)),
                grad_DX_X_step,
                np.zeros((n_branch * batch, 6, 3 * (current_vertices_copy.size()[1] - i - 2)))
            ), axis=2)

            grad_per_ICitr.grad_DX_X[:, 3 * i: 3 * (i + 2),:] = grad_interest_DX_X + grad_step_DX_X + grad_chain_passed_DX_X

            # Update the gradient for the undeformed vertices
            grad_DX_Xinit_step = gradient_IC.grad_DX_Xinit_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i], current_vertices_copy[:, i + 1],
                undeformed_vertices[:, i], undeformed_vertices[:, i + 1],
            )
            grad_DX_Xinit_step /= np.array(scale[:, i])

            grad_interest_DX_Xinit = grad_per_ICitr.grad_DX_Xinit[:, 3 * i: 3 * (i + 2), :].copy()
            grad_chain_passed_DX_Xinit = grad_DX_X_step @ grad_interest_DX_Xinit
            grad_step_DX_Xinit = np.concatenate((
                np.zeros((n_branch * batch, 6, 3 * i)),
                grad_DX_Xinit_step,
                np.zeros((n_branch * batch, 6, 3 * (current_vertices_copy.size()[1] - i - 2)))
            ), axis=2)

            grad_per_ICitr.grad_DX_Xinit[:, 3 * i: 3 * (i + 2),
            :] = grad_interest_DX_Xinit + grad_step_DX_Xinit + grad_chain_passed_DX_Xinit

            # Update the gradient for the mass scale
            grad_DX_M_step = gradient_IC.grad_DX_M_ICitr_batch(
                DLO_mass[:, i], DLO_mass[:, i + 1],
                current_vertices_copy[:, i], current_vertices_copy[:, i + 1],
                undeformed_vertices[:, i], undeformed_vertices[:, i + 1],
            )

            grad_DX_M_step /= np.array(scale[:, i])

            grad_interest_DX_M = grad_per_ICitr.grad_DX_M[:, 3 * i: 3 * (i + 2), :].copy()
            grad_chain_passed_DX_M = grad_DX_X_step @ grad_interest_DX_M

            grad_step_DX_M = np.concatenate((
                np.zeros((n_branch * batch, 6, i)),
                grad_DX_M_step,
                np.zeros((n_branch * batch, 6, (current_vertices_copy.size()[1] - i - 2)))
            ), axis=2)

            grad_per_ICitr.grad_DX_M[:, 3 * i: 3 * (i + 2),
            :] = grad_interest_DX_M + grad_step_DX_M + grad_chain_passed_DX_M

        return current_vertices, grad_per_ICitr

    def Inextensibility_Constraint_Enforcement_Coupling(self, parent_vertices, child_vertices, coupling_index,
                                                        coupling_mass_scale, selected_parent_index,
                                                        selected_children_index):
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

        Returns:
            b_DLOs_vertices (torch.Tensor): Combined or updated vertices for the rods
                                            after enforcing coupling constraints.
        """
        # Vector from parent to child's first vertex
        updated_edges = child_vertices[:, 0] - parent_vertices[:, coupling_index].view(-1, 3)

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

        return b_DLOs_vertices

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
            parent_MOIs, children_MOIs, index_selection, parent_MOI_index, momentum_scale_previous
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
            parent_MOI_index (torch.Tensor): Indices for selecting from parent_MOIs.
            momentum_scale_previous (torch.Tensor): Scale factors for rotational momentum-based correction.

        Returns:
            Tuple of updated parent_vertices, parent_orientations, children_vertices, children_orientations.
        """
        batch = parent_vertices.size()[0]
        n_children = len(index_selection)

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

