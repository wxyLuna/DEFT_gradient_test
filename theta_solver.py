import torch
import theseus as th
from typing import List, Optional, Tuple
import os
import numpy as np
from matplotlib.colors import to_rgb
from numpy.f2py.crackfortran import verbose

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)


# If needed, uncomment these lines to enforce deterministic behaviors (may affect performance).
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# If needed, uncomment these lines to limit threading (can help ensure reproducibility and control CPU usage).
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# torch.set_num_threads(1)
# torch.set_num_interop_threads(1)


# ----------------------------------------------
#  Custom CostFunction: VectorDifference
# ----------------------------------------------
class VectorDifference(th.CostFunction):
    """
    This class defines a custom cost function for Theseus-based optimization. The cost function
    encapsulates both the bending and twisting energies of a discrete rod-like structure using
    concepts from the Discrete Elastic Rods (DER) formulation.

    Attributes:
        n_branch (int): Number of 'branches' or rod segments treated together in a batch.
        cost_weight (th.CostWeight): Cost weighting used by Theseus in the objective.
        energy_target (th.Variable): (Currently unused) An auxiliary variable intended for
                                     target energy or reference data.
        kb (th.Variable): Discrete curvature binormal (batch_size, n_edge, 3).
        b_u (th.Variable): Bishop frame component 'u' (batch_size, n_edge, 3).
        b_v (th.Variable): Bishop frame component 'v' (batch_size, n_edge, 3).
        m_restW1 (th.Variable): Rest curvature in the material frame component #1 (batch_size, n_edge, 2).
        m_restW2 (th.Variable): Rest curvature in the material frame component #2 (batch_size, n_edge, 2).
        restRegionL (th.Variable): Rest lengths for each segment (batch_size, n_edge).
        inner_theta_variable (th.Vector): The twist angles (theta) to be optimized (batch_size, n_edge).
        end_theta (th.Variable): (Currently unused) Possibly the endpoint twist angle or boundary condition.
        stiff_threshold (th.Variable): Minimum threshold for stiffness values to prevent degeneracies.
        bend_stiffness_last_unsq (th.Variable): Bending stiffness for the "previous" portion of each edge.
        bend_stiffness_next_unsq (th.Variable): Bending stiffness for the "next" portion of each edge.
        bend_stiffness (th.Variable): Bending stiffness (legacy parameter, not directly used if we have last/next).
        twist_stiffness (th.Variable): Twisting stiffness for each edge (batch_size, n_edge).
        optimization_mask (torch.Tensor or tuple): Mask to selectively zero out gradient entries.
        name (Optional[str]): Name to be used for cost function in Theseus.

    Notes:
        - The cost function calculates the sum of bending and twisting energies for
          each segment and accumulates them to get the total rod energy.
        - Bending energy is derived from the difference between the current material
          curvature and the rest curvature.
        - Twisting energy is derived from the difference in angles between consecutive segments.
    """

    def __init__(
            self,
            n_branch,
            cost_weight: th.CostWeight,
            energy_target: th.Variable,
            kb: th.Variable,
            b_u: th.Variable,
            b_v: th.Variable,
            m_restW1: th.Variable,
            m_restW2: th.Variable,
            restRegionL: th.Variable,
            inner_theta_variable: th.Vector,
            end_theta: th.Variable,
            stiff_threshold: th.Variable,
            bend_stiffness_last_unsq: th.Variable,
            bend_stiffness_next_unsq: th.Variable,
            bend_stiffness: th.Variable,
            twist_stiffness: th.Variable,
            optimization_mask,
            name: Optional[str] = None,
    ):
        # Call the parent constructor (CostFunction)
        super().__init__(cost_weight, name=name)

        # A small epsilon value to avoid division by zero.
        self.epsilon = 1e-40

        # Save references to the input variables
        self.cost_weight = cost_weight
        self.energy_target = energy_target
        self.kb = kb
        self.b_u = b_u
        self.b_v = b_v
        self.m_restW1 = m_restW1
        self.m_restW2 = m_restW2
        self.restRegionL = restRegionL
        self.inner_theta_variable = inner_theta_variable
        self.end_theta = end_theta
        self.n_branch = n_branch
        self.stiff_threshold = stiff_threshold
        self.bend_stiffness_last_unsq = bend_stiffness_last_unsq
        self.bend_stiffness_next_unsq = bend_stiffness_next_unsq
        self.bend_stiffness = bend_stiffness # torch.Size([1, 3, 12])
        self.twist_stiffness = twist_stiffness # torch.Size([1, 3, 12]) 
        # print(f"the shape of bend_stiffness_last_unsq: {self.bend_stiffness.tensor.shape}")
        # print(f"the value of bend_stiffness_last_unsq: {self.bend_stiffness_last_unsq.tensor}")
        # print(f"the shape of bend_stiffness_next_unsq: {self.bend_stiffness_next_unsq.tensor.shape}")
        # print(f"the value of bend_stiffness_next_unsq: {self.bend_stiffness_next_unsq.tensor}")
        # print(f"the shape of bend_stiffness: {self.bend_stiffness.tensor.shape}")
        # print(f"the value of bend_stiffness: {self.bend_stiffness.tensor}")
        # print(f"the shape of twist_stiffness: {self.twist_stiffness.tensor.shape} ")

        # Register the optimization variable with Theseus (the variable to be optimized).
        self.register_optim_vars(["inner_theta_variable"])

        # Register auxiliary variables that are not optimized but will be read by the cost function.
        self.register_aux_vars(["energy_target"])
        self.register_aux_vars(["b_u"])
        self.register_aux_vars(["b_v"])
        self.register_aux_vars(["m_restW1"])
        self.register_aux_vars(["m_restW2"])
        self.register_aux_vars(["restRegionL"])
        self.register_aux_vars(["end_theta"])
        self.register_aux_vars(["bend_stiffness"])
        self.register_aux_vars(["twist_stiffness"])

        # Keep track of the mask for partial gradient updates
        self.optimization_mask = optimization_mask,


        # Basic set-up for indexing edges
        n_edge = b_u.tensor.size()[1]
        self.idx = torch.arange(1, n_edge)
        self.idx_prev = torch.arange(0, n_edge - 1)

        # Clamp the bending stiffness to avoid values below the threshold
        bend_stiffness_clamped = torch.clamp(
            self.bend_stiffness.tensor, min=self.stiff_threshold.tensor
        ).view(1, -1, 1, 1)
        self.bend_stiffness_clamped = bend_stiffness_clamped
        # print(f"the shape of bend_stiffness_clamped: {bend_stiffness_clamped.shape}")
        # print(f"the value of bend_stiffness_clamped: {bend_stiffness_clamped}")

        # Build a rotation matrix that rotates by 90 degrees for all edges
        # print("Initializing permutation matrix J")
        J = self.rotation_matrix(torch.pi / 2. * torch.ones(kb.tensor.size()[1]))
        # print(f"the shape of J is {J.shape}")
        self.J = J

        I = self.batched_identity(kb.tensor.size()[1])
        self.I = I
        # print(f"the shape of batched_identity is {self.I.shape}")
        # print(f"the value of batched_identity is {self.I}")

        # Expand rotation matrices J across the batch dimension and edges
        # n_branch: how many rods in parallel we are optimizing
        J_expanded = (
            J.unsqueeze(0)
            .repeat(n_branch, 1, 1, 1)
            .unsqueeze(0)
            .view(1, -1, 2, 2)
        )
        self.J_expanded = J_expanded

        I_expanded = (
            I.unsqueeze(0)
            .repeat(n_branch, 1, 1, 1)
            .unsqueeze(0)
            .view(1, -1, 2, 2)
        )
        self.I_expanded = I_expanded

        # Multiply rotation matrix J by the clamped bending stiffness
        self.JB = J_expanded * bend_stiffness_clamped
        self.JTBJ = I_expanded * bend_stiffness_clamped
        # print(f"the shape of JB is {self.JB.shape}")
        # print(f"the shape of JTBJ is {self.JTBJ.shape}")
        # print(f"the value of JB is {self.JB[0,0,:,:]}")
        # print(f"the value of JTBJ is {self.JTBJ[0,0,:,:]}")


        # Identify where restRegionL is zero; used to zero-out certain computations
        # for edges that effectively do not exist or have zero length.
        restRegionL_unsq = self.restRegionL[:, 1:].unsqueeze(dim=-1)
        self.zero_mask = restRegionL_unsq == 0
        self.zero_mask_twist = self.restRegionL[:, 1:] == 0
        # print(f"the shape of zero_mask_twist: {self.zero_mask_twist[0]}")

        # Batch size is total batch divided by the number of rods (n_branch),
        # e.g., if you have multiple rods each with the same number of edges combined in one batch.
        self.batch = self.kb.tensor.size()[0] // n_branch
        n_edge = self.kb.tensor.size()[1]

        # Repeat the twist stiffness to match the entire batch dimension
        self.twist_stiffness_unsq = self.twist_stiffness.tensor.repeat(self.batch, 1, 1).view(-1, n_edge)

    def computeMaterialFrame(self, m_theta, b_u, b_v):
        """
        Computes the material frame (m1, m2) given the Bishop frame (b_u, b_v) and
        a twist angle m_theta. The material frame is rotated by m_theta about the rod's tangent.

        Parameters:
            m_theta (torch.Tensor): Twist angles (batch_size, n_edge).
            b_u (torch.Tensor): Bishop frame 'u' (batch_size, n_edge, 3).
            b_v (torch.Tensor): Bishop frame 'v' (batch_size, n_edge, 3).

        Returns:
            (m1, m2): The two perpendicular axes of the material frame.
        """
        # Compute cosines and sines of the angles, shape: (batch_size, n_edge, 1)
        cosQ = torch.cos(m_theta.clone()).unsqueeze(dim=2)
        sinQ = torch.sin(m_theta.clone()).unsqueeze(dim=2)

        # m1, m2 are formed by rotating (b_u, b_v) around the rod's tangent
        m1 = cosQ * b_u + sinQ * b_v
        m2 = -sinQ * b_u + cosQ * b_v
        return m1, m2

    def computeW(self, kb, m1, m2):
        """
        Projects the discrete curvature binormal kb onto the local material frame (m1, m2).
        This is used to extract the 2D curvature components in the local cross-section.

        Parameters:
            kb (torch.Tensor): Curvature binormal, shape (batch_size, n_edge, 3).
            m1, m2 (torch.Tensor): The material frame vectors, each (batch_size, n_edge, 3).

        Returns:
            o_wij (torch.Tensor): 2D curvature components in the material frame,
                                  shape (batch_size, n_edge, 2).
        """
        # Dot product of kb with m2 and m1 (with sign changes) to get the local 2D curvature.
        # We create a 2D vector of shape (batch_size, n_edge, 2).
        o_wij = torch.cat((
            (kb * m2).sum(dim=2).unsqueeze(dim=2),
            -(kb * m1).sum(dim=2).unsqueeze(dim=2)
        ), dim=2)
        return o_wij

    def computeMaterialCurvature(self, kb, m1, m2):
        """
        Computes the material curvature at edges 1..n_edge-1 using the derivative
        of twist angles between consecutive edges.

        According to the DER paper, eq. (2), the material curvature is the difference
        between consecutive frames in the bishop or material representation.

        Parameters:
            kb (torch.Tensor): Curvature binormal, shape (batch_size, n_edge, 3).
            m1, m2 (torch.Tensor): Material frames (batch_size, n_edge, 3).

        Returns:
            m_W1, m_W2 (torch.Tensor): The 2D curvature in the material frame
                                       for each edge in [1..n_edge-1].
        """
        # Curvature for the "previous" portion (between indices j-1 and j).
        m_W1 = self.computeW(kb[:, 1:], m1[:, :-1], m2[:, :-1]) # omega_(i+1, i), i start at 0
        # Curvature for the "next" portion (between indices j and j+1).
        m_W2 = self.computeW(kb[:, 1:], m1[:, 1:], m2[:, 1:]) # omega(i, i), i start at 1
        return m_W1, m_W2

    def compute_dE2dTheta2(self, theta_star):
        """
        compute the hessian of the bending energy wrt theta at the theta_star, 
        which is the optimized theta that minimize the potential energy
        """
        batch, n_edge = theta_star.size()
        hessian_theta = torch.zeros(batch, n_edge, n_edge, 
                                    dtype=theta_star.dtype, device=theta_star.device)
        m1, m2 = self.computeMaterialFrame(theta_star, self.b_u.tensor, self.b_v.tensor)
        
        if n_edge > 1:
            # bending enegry part
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)

            temp = (o_W1-self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            JTBJ_j = self.JTBJ.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
            # print(f"the value of JTBJ_j: {JTBJ_j[0,0,:,:]}")
            JTBJ_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]

            term1 = (o_W1 * JTBJ_wij).sum(dim=-1)  # [batch_size, n_edge-1], omega_(i+1, i), i start at 0

            temp = o_W1.unsqueeze(-1)
            B_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            term1 -= (o_W1 * B_wij).sum(dim=-1)

            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JTBJ_j = self.JTBJ.view(self.n_branch, -1, 2, 2)[:, 1:].repeat(self.batch, 1, 1, 1)
            JTBJ_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)
            term2 = (o_W2 * JTBJ_wij).sum(dim=-1)

            temp = o_W2.unsqueeze(-1)
            B_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            term2 -= (o_W2 * B_wij).sum(dim=-1)

            i = torch.arange(n_edge-1, device=hessian_theta.device)
            hessian_theta[:,i,i] += torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )
            hessian_theta[:,i+1,i+1] += torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # twisting energy part
            # diagonal value
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq[:, 1:], min=self.stiff_threshold.tensor
            )
            # print(f"the shape of stiff_threshold: {self.stiff_threshold.tensor.shape}")
            # print(f"the value of stiff_threshold: {self.stiff_threshold.tensor}")
            # print(f"the value of self.twist_stiffness_unsq: {self.twist_stiffness_unsq[0,1:]}")
            # print(f"the shape of twist_stiffness_clamped: {twist_stiffness_clamped.shape}")
            # print(f"the value of twist_stiffness_clamped: {twist_stiffness_clamped[0]}")
            term1 = twist_stiffness_clamped
            term1 = torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )
            # print(f"the shape of restRegionL: {self.restRegionL[:, 1:].shape}")
            # print(f"the value of restRegionL: {self.restRegionL[0,1:]}")
            # print(f"the value of term1: {term1[0]}")
            # print("chris paul 58, kevin durant 1.5")

            hessian_theta[:, i, i] += term1
            hessian_theta[:, i+1, i+1] += term1

            # off-diagonal value
            hessian_theta[:, i+1, i] -= term1
            hessian_theta[:, i, i+1] -= term1
        
        return -hessian_theta


    def compute_dE2dBendingdTheta(self, theta_star):
        """
        compute the hessian of potential energy wrt to bending stiffness and theta at theta_full
        """
        batch, n_edge = theta_star.size()
        hessian_bendingTheta = torch.zeros(batch, n_edge, n_edge, 
                                    dtype=theta_star.dtype, device=theta_star.device)
        m1, m2 = self.computeMaterialFrame(theta_star, self.b_u.tensor, self.b_v.tensor)
        
        if n_edge > 1:
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)

            temp = (o_W1-self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            JTBJ_j = self.J_expanded.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
            # print(f"the value of JTBJ_j: {JTBJ_j[0,0,:,:]}")
            JTBJ_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]

            term1 = (o_W1 * JTBJ_wij).sum(dim=-1)  # [batch_size, n_edge-1], omega_(i+1, i), i start at 0

            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JTBJ_j = self.J_expanded.view(self.n_branch, -1, 2, 2)[:, 1:].repeat(self.batch, 1, 1, 1)
            # print(f"the value of JTBJ_j: {JTBJ_j[0,0,:,:]}")
            JTBJ_wij = torch.matmul(JTBJ_j, temp).squeeze(-1)
            term2 = (o_W2 * JTBJ_wij).sum(dim=-1)


            i = torch.arange(n_edge-1, device=hessian_bendingTheta.device)

            hessian_bendingTheta[:, i+1, i+1] += torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )
            hessian_bendingTheta[:, i, i + 1] += torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )
        # print(hessian_bendingTheta[0])
        return hessian_bendingTheta

    def compute_dE2dTwistdTheta(self, theta_star):
        """
        compute the hessian of potential energy wrt to twisting stiffness and theta at theta_full
        """
        batch, n_edge = theta_star.size()
        hessian_twistTheta = torch.zeros(batch, n_edge, n_edge, 
                                    dtype=theta_star.dtype, device=theta_star.device)
        
        
        if n_edge > 1:
            theta_diff = theta_star[:, 1:] - theta_star[:, :-1]  # shape: (batch, n_edge-1)
            i = torch.arange(n_edge-1, device=hessian_twistTheta.device)
            # diagonal value
            hessian_twistTheta[:, i+1, i+1] += theta_diff/ (self.restRegionL[:, 1:] + self.epsilon)
            # off-diagonal value
            hessian_twistTheta[:, i, i + 1] -= theta_diff / (self.restRegionL[:, 1:] + self.epsilon)

        return hessian_twistTheta

    def compute_dTheta_star_dAlpha(self, theta_star):
        """
        Implicit derivative via dθ*/dα = - H^{-1} ∂^2E/∂θ∂α
        """
        # Each of these returns a (batch, n_edge, n_edge) tensor
        H   = self.compute_dE2dTheta2(   theta_star)[0:1]  # d²E/dθ²
        H_b = self.compute_dE2dBendingdTheta(theta_star)[0:1]  # d²E/dθ d(bend)
        H_t = self.compute_dE2dTwistdTheta(  theta_star)[0:1]  # d²E/dθ d(twist)

        # Invert the Hessian
        invH = torch.inverse(H)   # shape (batch, n_edge, n_edge)

        # Apply the implicit function theorem:
        #   dθ/d(bend)  = – invH @ H_b
        #   dθ/d(twist) = – invH @ H_t
        dTheta_db = -torch.matmul(invH, H_b)
        dTheta_dt = -torch.matmul(invH, H_t)

        return dTheta_db, dTheta_dt

    def error(self) -> torch.Tensor:
        """
        Computes the total energy (bending + twisting) for all edges, which is the
        quantity that gets minimized in the optimization.

        Returns:
            O_E (torch.Tensor): A tensor of shape (batch_size, 1) representing
                                the energy for each batch element.
        """
        # Retrieve the twist angles
        theta_full = self.inner_theta_variable.tensor

        # Compute the material frame from the Bishop frame
        m1, m2 = self.computeMaterialFrame(theta_full, self.b_u.tensor, self.b_v.tensor)

        # For convenience, get the rest lengths for edges 1..n_edge-1 and expand dimension
        restRegionL_unsq = self.restRegionL[:, 1:].unsqueeze(dim=-1)

        # Project discrete curvature binormal onto material frame
        o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)

        # Compute difference between current curvature and rest curvature
        diff_prev = o_W1 - self.m_restW1.tensor[:, 1:]
        diff_next = o_W2 - self.m_restW2.tensor[:, 1:]

        # Bending energy: sum of squared differences scaled by stiffness / restRegion length
        O_E = torch.where(
            self.zero_mask,
            0.,  # Zero out if restRegion is zero
            (
                    0.5 * (
                    torch.clamp(self.bend_stiffness_last_unsq.tensor, self.stiff_threshold.tensor)
                    * diff_prev * diff_prev
                    + torch.clamp(self.bend_stiffness_next_unsq.tensor, self.stiff_threshold.tensor)
                    * diff_next * diff_next
            ) / (restRegionL_unsq + self.epsilon)
            )
        ).sum(dim=(1, 2))

        # Twisting energy: depends on the difference between consecutive theta values
        m = theta_full[:, 1:] - theta_full[:, :-1]

        # Add the twisting part of the energy
        O_E += torch.where(
            self.zero_mask_twist,
            0.,
            0.5 * (
                    torch.clamp(self.twist_stiffness_unsq[:, 1:], self.stiff_threshold.tensor)
                    * m * m
                    / (self.restRegionL[:, 1:] + self.epsilon)
            )
        ).sum(dim=1)

        # Return as (batch_size, 1) for Theseus
        return O_E.view(-1, 1)

    def rotation_matrix(self, theta):
        """
        Constructs a 2D rotation matrix (for each batch element) that rotates by the input angle theta.

        Parameters:
            theta (torch.Tensor): Shape (batch_size,). The angle of rotation for each batch element.

        Returns:
            transform_basis (torch.Tensor): Shape (batch_size, 2, 2). A 2D rotation matrix for each batch.
        """
        batch = theta.size()[0]
        rot_sin = torch.sin(theta)
        rot_cos = torch.cos(theta)

        # Initialize a zero tensor for the rotation matrix
        transform_basis = torch.zeros(batch, 2, 2)

        # Fill in the rotation matrix entries
        transform_basis[:, 0, 0] = rot_cos
        transform_basis[:, 0, 1] = -rot_sin
        transform_basis[:, 1, 0] = rot_sin
        transform_basis[:, 1, 1] = rot_cos
        # print(f'the element at first row and first column of transform_basis: {transform_basis[0, 0, 0]}')
        # print(f'the element at first row and second column of transform_basis: {transform_basis[0, 0, 1]}')
        # print(f'the element at second row and first column of transform_basis: {transform_basis[0, 1, 0]}')
        # print(f'the element at second row and second column of transform_basis: {transform_basis[0, 1, 1]}')
        return transform_basis

    def batched_identity(self, batch_size):
        """
        initialized a batched identity matrix of shape (batch_size, 2, 2)
        """
        batched_I = torch.eye(2).unsqueeze(0).repeat(batch_size, 1, 1)
        return batched_I

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Computes the gradient of the energy function with respect to the twist angles theta.

        Returns:
            A tuple (jacobian_list, error_tensor):
                - jacobian_list: A list of length 1 containing the Jacobian tensor
                                 with shape [batch_size, 1, n_edge].
                - error_tensor: The same as self.error() for direct usage in Theseus.
        """
        # Get the current twist angles
        theta = self.inner_theta_variable.tensor
        # print(theta[0])
        # print(theta[1])
        # print(theta[2])
        # print(f"the shape of theta: {theta.shape}")

        # Compute the material frame
        m1, m2 = self.computeMaterialFrame(theta, self.b_u.tensor, self.b_v.tensor)
        # print(f"the shape of m1: {m1.shape}, m2: {m2.shape}, they represent the material frame")
        


        # Prepare a gradient buffer
        batch_size, n_edge, _ = m1.size()
        dEdtheta = torch.zeros((batch_size, n_edge))  # [batch_size, n_edge]
        # to be more specific, when batch_size=1, n_edges=11

        # Only proceed if n_edge > 1 because we need consecutive edges
        if n_edge > 1:
            # Compute material curvature for consecutive edges
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)
            # print(f"the shape of o_W1: {o_W1.shape}, o_W2: {o_W2.shape}, they represent the material curvature")
            # print(f"the shape of m_restW1: {self.m_restW1.tensor.shape}, m_restW2: {self.m_restW2.tensor.shape}, they represent the material curvature in the undeformed state")

            # 1) Bending energy gradient components
            # Difference from rest
            temp = (o_W1 - self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            # print(f"the shape of temp: {temp.shape} , which is omega_w1 - omega_restW1 (undeformed state)")

            # JB_j for "previous" edges (skip the last one)
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
            # print(f"the shape of JB_j: {JB_j.shape}, which is the bending stiffness dot product with a permuatation matric")
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            # print(f"the shape of JB_wij: {JB_wij.shape}, which is the JB_j dot product with temp")

            # dot product with o_W1
            term1 = (o_W1 * JB_wij).sum(dim=-1)  # [batch_size, n_edge-1]
            # print(f"the shape of term1: {term1.shape}, which is the first term of dP_bending/dtheta")

            # Similarly for "next" edges
            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, 1:].repeat(self.batch, 1, 1, 1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term2 = (o_W2 * JB_wij).sum(dim=-1)
            # print(f"the shape of term2: {term2.shape}, which is the second term of dP_bending/dtheta")


            # 2) Twisting energy gradient components
            # Clamped twist stiffness
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq[:, 1:], min=self.stiff_threshold.tensor
            )
            # print(f"the shape of twist_stiffness_clamped: {twist_stiffness_clamped.shape}")
            # print(f"the shape of (theata[:, 1:] - theta[:, :-1]): {(theta[:, 1:] - theta[:, :-1]).shape}")

            # The difference in theta values
            # gradient wrt theta_j for the difference (theta_j - theta_{j-1})
            term1 -= twist_stiffness_clamped * (theta[:, 1:] - theta[:, :-1])
            term1 = torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # gradient wrt theta_{j-1} for the difference (theta_j - theta_{j-1})
            term2 += twist_stiffness_clamped * (theta[:, 1:] - theta[:, :-1])
            term2 = torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # Accumulate these values appropriately in the gradient
            
            # Adds term1 to the gradient of theta variables at edges [0, 1, 2, ..., n_edge-2]
            # This represents the gradient contribution from the energy term 
            # that depends on the difference (theta[j] - theta[j-1]) 
            # where this theta variable appears as theta[j] (the "current" angle)
            dEdtheta[:, :-1] += term1

            # Adds term2 to the gradient of theta variables at edges [1, 2, 3, ..., n_edge-1]
            # This represents the gradient contribution from the energy term 
            # that depends on the difference (theta[j] - theta[j-1]) 
            # where this theta variable appears as theta[j-1] (the "previous" angle)
            dEdtheta[:, 1:] += term2
        # print("============================================================= jacobians =============================================================")
        # print("input theta is ", theta[0])
        # print("the gradient dEdtheta is ", dEdtheta[0])

        # Expand the gradient to match the shape expected by Theseus: [batch_size, 1, n_edge]
        jacobian_tensor = dEdtheta.unsqueeze(1)
        # print(f"the shape of jacobian_tensor: {jacobian_tensor.shape}, which is the gradient of the energy wrt theta")

        # Apply any given optimization mask (if certain DOFs need to be disabled)
        jacobian_tensor = jacobian_tensor * self.optimization_mask[0]
        # print(f"the shape of optimization_mask: {self.optimization_mask[0].shape}")
        # print(f"the shape of jacobian_tensor after applying mask: {jacobian_tensor.shape}")

        # Return the gradient and the error
        return [jacobian_tensor], self.error()

    def compute_jacobian(self, theta_star):
        # print("theta_star is ", theta_star[0])
        m1, m2 = self.computeMaterialFrame(theta_star, self.b_u.tensor, self.b_v.tensor)
        batch_size, n_edge, _ = m1.size()
        dEdtheta = torch.zeros((batch_size, n_edge)) 
        if n_edge > 1:
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)
            temp = (o_W1 - self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            term1 = (o_W1 * JB_wij).sum(dim=-1)  # [batch_size, n_edge-1]

            # Similarly for "next" edges
            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, 1:].repeat(self.batch, 1, 1, 1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term2 = (o_W2 * JB_wij).sum(dim=-1)


            # 2) Twisting energy gradient components
            # Clamped twist stiffness
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq[:, 1:], min=self.stiff_threshold.tensor
            )
            term1 -= twist_stiffness_clamped * (theta_star[:, 1:] - theta_star[:, :-1])
            term1 = torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # gradient wrt theta_{j-1} for the difference (theta_j - theta_{j-1})
            term2 += twist_stiffness_clamped * (theta_star[:, 1:] - theta_star[:, :-1])
            term2 = torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            dEdtheta[:, :-1] += term1
            dEdtheta[:, 1:] += term2
        return dEdtheta
    
    def compute_jacobian_perturb_bending(self, theta_star, noise_scale):
        # print("theta_star is ", theta_star[0])
        m1, m2 = self.computeMaterialFrame(theta_star, self.b_u.tensor, self.b_v.tensor)
        batch_size, n_edge, _ = m1.size()
        dEdtheta = torch.zeros((batch_size, n_edge)) 
        if n_edge > 1:
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)
            temp = (o_W1 - self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            perturbed_bend_stiffness = self.bend_stiffness_clamped + noise_scale * torch.ones(self.bend_stiffness_clamped.shape, device=self.bend_stiffness_clamped.device)
            perturbed_JB = self.J_expanded * perturbed_bend_stiffness
            JB_j = perturbed_JB.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
           
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            term1 = (o_W1 * JB_wij).sum(dim=-1)  # [batch_size, n_edge-1]

            # Similarly for "next" edges
            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term2 = (o_W2 * JB_wij).sum(dim=-1)


            # 2) Twisting energy gradient components
            # Clamped twist stiffness
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq[:, 1:], min=self.stiff_threshold.tensor
            )
            term1 -= twist_stiffness_clamped * (theta_star[:, 1:] - theta_star[:, :-1])
            term1 = torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # gradient wrt theta_{j-1} for the difference (theta_j - theta_{j-1})
            term2 += twist_stiffness_clamped * (theta_star[:, 1:] - theta_star[:, :-1])
            term2 = torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            dEdtheta[:, :-1] += term1
            dEdtheta[:, 1:] += term2
        return dEdtheta


    def compute_jacobian_perturb_twist(self, theta_star, noise_scale):
        m1, m2 = self.computeMaterialFrame(theta_star, self.b_u.tensor, self.b_v.tensor)
        batch_size, n_edge, _ = m1.size()
        dEdtheta = torch.zeros((batch_size, n_edge)) 
        if n_edge > 1:
            o_W1, o_W2 = self.computeMaterialCurvature(self.kb.tensor, m1, m2)
            temp = (o_W1 - self.m_restW1.tensor[:, 1:]).unsqueeze(-1)  # shape: [batch_size, n_edge-1, 2, 1]
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, :-1].repeat(self.batch, 1, 1, 1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)  # shape: [batch_size, n_edge-1, 2]
            term1 = (o_W1 * JB_wij).sum(dim=-1)  # [batch_size, n_edge-1]

            # Similarly for "next" edges
            temp = (o_W2 - self.m_restW2.tensor[:, 1:]).unsqueeze(-1)
            JB_j = self.JB.view(self.n_branch, -1, 2, 2)[:, 1:].repeat(self.batch, 1, 1, 1)
            JB_wij = torch.matmul(JB_j, temp).squeeze(-1)
            term2 = (o_W2 * JB_wij).sum(dim=-1)


            # 2) Twisting energy gradient components
            # Clamped twist stiffness
            twist_stiffness_clamped = torch.clamp(
                self.twist_stiffness_unsq[:, 1:], min=self.stiff_threshold.tensor
            )
            perturbed_twist_stiffness = twist_stiffness_clamped + noise_scale * torch.ones(twist_stiffness_clamped.shape, device=twist_stiffness_clamped.device)
            term1 -= perturbed_twist_stiffness * (theta_star[:, 1:] - theta_star[:, :-1])
            term1 = torch.where(
                self.zero_mask_twist,
                0.,
                term1 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            # gradient wrt theta_{j-1} for the difference (theta_j - theta_{j-1})
            term2 += perturbed_twist_stiffness * (theta_star[:, 1:] - theta_star[:, :-1])
            term2 = torch.where(
                self.zero_mask_twist,
                0.,
                term2 / (self.restRegionL[:, 1:] + self.epsilon)
            )

            dEdtheta[:, :-1] += term1
            dEdtheta[:, 1:] += term2
        return dEdtheta


    def get_numerial_jacobian_difference_theta(self, theta_star, noise_scale):
        jacobian_perturbed = []
        for i in [noise_scale, -noise_scale]:
            perturbed_theta = theta_star + i * torch.ones(theta_star.shape, device=theta_star.device)
            dEdtheta = self.compute_jacobian(perturbed_theta)
            jacobian_perturbed.append(dEdtheta)
        numerical_jacob_diff = (jacobian_perturbed[0] - jacobian_perturbed[1])/2
        return numerical_jacob_diff
    
    def get_numerical_jacobian_difference_bending(self, theta_star, noise_scale):
        jacobian_perturbed = []
        for ns in [noise_scale, -noise_scale]:
            dEdtheta = self.compute_jacobian_perturb_bending(theta_star, ns)
            jacobian_perturbed.append(dEdtheta)
        numerical_jacob_diff = (jacobian_perturbed[0] - jacobian_perturbed[1])/2
        return numerical_jacob_diff

    def get_numerical_jacobian_difference_twist(self, theta_star, noise_scale):
        jacobian_perturbed = []
        for ns in [noise_scale, -noise_scale]:
            dEdtheta = self.compute_jacobian_perturb_twist(theta_star, ns)
            jacobian_perturbed.append(dEdtheta)
        numerical_jacob_diff = (jacobian_perturbed[0] - jacobian_perturbed[1])/2
        return numerical_jacob_diff
    
    
    def dim(self) -> int:
        """
        This cost function returns a single scalar energy value per batch element,
        so the dimension is 1.
        """
        return 1

    def _copy_impl(self, new_name: Optional[str] = None) -> "VectorDifference":
        """
        Creates a copy of this cost function. Required method for Theseus
        to replicate the cost function with a new name if needed.
        """
        return VectorDifference(  # type: ignore
            self.n_branch,
            self.cost_weight.copy(),
            self.energy_target.copy(),
            self.kb.copy(),
            self.b_u.copy(),
            self.b_v.copy(),
            self.m_restW1.copy(),
            self.m_restW2.copy(),
            self.restRegionL.copy(),
            self.inner_theta_variable.copy(),
            self.end_theta.copy(),
            self.stiff_threshold.copy(),
            self.bend_stiffness_last_unsq.copy(),
            self.bend_stiffness_next_unsq.copy(),
            self.bend_stiffness.copy(),
            self.twist_stiffness.copy(),
            name=new_name,
        )


# ------------------------------------------------------
#  Optimization Routine for theta (the twist angles)
# ------------------------------------------------------
def theta_optimize(
        n_branch,
        cost_weight,
        target_energy,
        kb,
        b_u,
        b_v,
        m_restW1,
        m_restW2,
        restRegionL,
        inner_theta_opt,
        inner_theta_tensor,
        end_theta,
        stiff_threshold,
        bend_stiffness_last_unsq,
        bend_stiffness_next_unsq,
        bend_stiffness,
        twist_stiffness,
        optimization_mask
):
    """
    Sets up a Theseus optimization problem with the VectorDifference cost function.

    Args:
        n_branch (int): Number of parallel rod segments in the batch.
        cost_weight (th.CostWeight): Theseus cost weight parameter.
        target_energy (th.Variable): (Currently unused) Possibly a reference energy target.
        kb (th.Variable): Curvature binormal variable (batch_size, n_edge, 3).
        b_u (th.Variable): Bishop frame u (batch_size, n_edge, 3).
        b_v (th.Variable): Bishop frame v (batch_size, n_edge, 3).
        m_restW1 (th.Variable): Rest curvature (component #1).
        m_restW2 (th.Variable): Rest curvature (component #2).
        restRegionL (th.Variable): Rest lengths for edges.
        inner_theta_opt (th.Vector): The Theseus variable that will be optimized.
        inner_theta_tensor (torch.Tensor): Initial guess for the twist angles.
        end_theta (th.Variable): (Currently unused) Possibly boundary condition for the last edge.
        stiff_threshold (th.Variable): Minimum stiffness threshold.
        bend_stiffness_last_unsq (th.Variable): Bending stiffness for the previous portion of edges.
        bend_stiffness_next_unsq (th.Variable): Bending stiffness for the next portion of edges.
        bend_stiffness (th.Variable): Bending stiffness (legacy).
        twist_stiffness (th.Variable): Twisting stiffness.
        optimization_mask (torch.Tensor): Mask to disable certain optimization DOFs if needed.

    Returns:
        a_i_value (torch.Tensor): The optimized twist angles (batch_size, n_edge).
    """
    # Create a Theseus Objective

    theta_pertubed = []
    # for noise in [1e-8, -1e-8]:
    #     # print(f"perturbation noise: {noise}")
    #     objective = th.Objective()
    #     # build a new perturbation of the bending‐stiffness tensor
    #     bs_tensor_last_unsq = bend_stiffness_last_unsq.tensor  # this is a torch.Tensor
    #     perturbed_bs_last_unsq = bs_tensor_last_unsq + torch.ones(bs_tensor_last_unsq.shape) * noise
    #     bs_tensor_next_unsq = bend_stiffness_next_unsq.tensor  # this is a torch.Tensor
    #     perturbed_bs_next_unsq = bs_tensor_next_unsq + torch.ones(bs_tensor_next_unsq.shape) * noise
    #     bs_tensor = bend_stiffness.tensor              # this is a torch.Tensor
    #     perturbed_bs = bs_tensor + torch.ones(bs_tensor.shape) * noise
    #     # print(perturbed_bs - bs_tensor)
    #     # wrap back into a Theseus Variable (give it a new name so it doesn’t collide)
    #     perturbed_bend_stiffness_last_unsq = th.Variable(
    #         perturbed_bs_last_unsq, name="bend_stiffness_last_unsq_perturbed"
    #     )
    #     perturbed_bend_stiffness_next_unsq = th.Variable(
    #         perturbed_bs_next_unsq, name="bend_stiffness_next_unsq_perturbed"
    #     )
    #     perturbed_bend_stiffness = th.Variable(
    #         perturbed_bs, name="bend_stiffness_perturbed"
    #     )

    #     cost_fn = VectorDifference(
    #         n_branch,
    #         cost_weight,
    #         target_energy,
    #         kb,
    #         b_u,
    #         b_v,
    #         m_restW1,
    #         m_restW2,
    #         restRegionL,
    #         inner_theta_opt,
    #         end_theta,
    #         stiff_threshold,
    #         perturbed_bend_stiffness_last_unsq,
    #         perturbed_bend_stiffness_next_unsq,
    #         perturbed_bend_stiffness,
    #         twist_stiffness,
    #         optimization_mask
    #     )
    #     objective.add(cost_fn)

    #     # Prepare the initial values (dict) for the optimization variables
    #     theseus_inputs = {}
    #     theseus_inputs.update({f"inner_theta_variable": inner_theta_tensor})

    #     # Update the objective with the data
    #     objective.update(theseus_inputs)

    #     # (Optional) We could check initial error:
    #     # error_sq = objective.error_metric()
    #     # print(f"Initial error: {error_sq.item()}")

    #     # Set up the Levenberg-Marquardt optimizer
    #     optimizer = th.LevenbergMarquardt(
    #         objective,
    #         linear_solver_cls=th.LUDenseSolver,  # Use LU solver for dense systems
    #         linearization_cls=th.DenseLinearization,
    #         linear_solver_kwargs={'check_singular': False},
    #         # max_iterations=10,
    #         max_iterations=100, # mannual set a small number for shape checking
    #         step_size=0.5,
    #         abs_err_tolerance=1e-10,
    #         rel_err_tolerance=1e-5,
    #         adaptive_damping=True,
    #     )

    #     # Solve the optimization problem
    #     info = optimizer.optimize(inputs=theseus_inputs)

    #     # (Optional) Check error after optimization:
    #     # error_sq = objective.error_metric()
    #     # print(f"Final error: {error_sq.item()}")

    #     # Retrieve the optimized angles
    #     a_i_value = objective.get_optim_var("inner_theta_variable").tensor
    #     theta_pertubed.append(a_i_value)


    # Instantiate our custom cost function
    objective = th.Objective()
    cost_fn = VectorDifference(
        n_branch,
        cost_weight,
        target_energy,
        kb,
        b_u,
        b_v,
        m_restW1,
        m_restW2,
        restRegionL,
        inner_theta_opt,
        end_theta,
        stiff_threshold,
        bend_stiffness_last_unsq,
        bend_stiffness_next_unsq,
        bend_stiffness,
        twist_stiffness,
        optimization_mask
    )
    # Add the cost function to the objective
    objective.add(cost_fn)

    # Prepare the initial values (dict) for the optimization variables
    theseus_inputs = {}
    theseus_inputs.update({f"inner_theta_variable": inner_theta_tensor})

    # Update the objective with the data
    objective.update(theseus_inputs)

    # (Optional) We could check initial error:
    # error_sq = objective.error_metric()
    # print(f"Initial error: {error_sq.item()}")

    # Set up the Levenberg-Marquardt optimizer
    optimizer = th.LevenbergMarquardt(
        objective,
        linear_solver_cls=th.LUDenseSolver,  # Use LU solver for dense systems
        linearization_cls=th.DenseLinearization,
        linear_solver_kwargs={'check_singular': False},
        # max_iterations=10,
        max_iterations=100, # mannual set a small number for shape checking
        step_size=0.5,
        abs_err_tolerance=1e-10,
        rel_err_tolerance=1e-5,
        adaptive_damping=True,
    )

    # Solve the optimization problem
    info = optimizer.optimize(inputs=theseus_inputs)

    # (Optional) Check error after optimization:
    # error_sq = objective.error_metric()
    # print(f"Final error: {error_sq.item()}")

    # Retrieve the optimized angles
    a_i_value = objective.get_optim_var("inner_theta_variable").tensor




    dTheta_star_dBending, dTheta_star_dTwist = cost_fn.compute_dTheta_star_dAlpha(a_i_value) # two hessians
    print("shape of dTheta_star_dBending:", dTheta_star_dBending.shape)
    # print("dTheta_star_dBending:", dTheta_star_dBending)
    
    # numerial_bending = (theta_pertubed[0] - theta_pertubed[1]) / 2

    # build a noise‐scale vector (length = n_edge)
    noise_scale = 1e-8
    n_edge = dTheta_star_dBending.size(-1)
    noise_vec = noise_scale * torch.ones(n_edge, device=dTheta_star_dBending.device)
    base_dir = f"grad_test_dTheta_star_dAlpha/perturb_scale={noise_scale}"
    os.makedirs(base_dir, exist_ok=True)

    # now write each log under that directory


     # test hessian of energy wrt theta
    print("checking Hessian of energy wrt theta")
    numerical_jacobian_diff = cost_fn.get_numerial_jacobian_difference_theta(a_i_value, noise_scale)
    analytical_hessian = cost_fn.compute_dE2dTheta2(a_i_value)
    analytical_jacobian_diff = torch.matmul(analytical_hessian, noise_vec.unsqueeze(-1)).squeeze(-1)
    log_path = os.path.join(base_dir, "Hessian_theta_theta_numerical_check.txt")
    with open(log_path, "a") as f:
        f.write("=== New run ===\n")
        abs_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0])
        rel_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0]) / torch.abs(numerical_jacobian_diff[0])
        ratio = analytical_jacobian_diff[0] / numerical_jacobian_diff[0]
        f.write(f"absolute_error: {abs_err.tolist()}\n")
        f.write(f"relative_error: {rel_err.tolist()}\n")
        f.write(f"ratio: {ratio.tolist()}\n\n")


    # test hessian of energy wrt partial bending stiffness partial theta
    print("checking Hessian of energy wrt partial bending stiffness partial theta")
    numerical_jacobian_diff = cost_fn.get_numerical_jacobian_difference_bending(a_i_value, noise_scale)
    analytical_hessian = cost_fn.compute_dE2dBendingdTheta(a_i_value)
    analytical_jacobian_diff = torch.matmul(analytical_hessian, noise_vec.unsqueeze(-1)).squeeze(-1)
    log_path = os.path.join(base_dir, "Hessian_bending_theta_numerical_check.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("=== New run ===\n")
        abs_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0])
        rel_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0]) / torch.abs(numerical_jacobian_diff[0])
        ratio = analytical_jacobian_diff[0] / numerical_jacobian_diff[0]
        f.write(f"absolute_error: {abs_err.tolist()}\n")
        f.write(f"relative_error: {rel_err.tolist()}\n")
        f.write(f"ratio: {ratio.tolist()}\n\n")

    # # test hessian of energy wrt partial twisting stiffness partial theta
    print("checking Hessian of energy wrt partial twisting stiffness partial theta")
    numerical_jacobian_diff = cost_fn.get_numerical_jacobian_difference_twist(a_i_value, noise_scale)
    analytical_hessian = cost_fn.compute_dE2dTwistdTheta(a_i_value)
    analytical_jacobian_diff = torch.matmul(analytical_hessian, noise_vec.unsqueeze(-1)).squeeze(-1)
    log_path = os.path.join(base_dir, "Hessian_twist_theta_numerical_check.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("=== New run ===\n")
        abs_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0])
        rel_err = torch.abs(analytical_jacobian_diff[0] - numerical_jacobian_diff[0]) / torch.abs(numerical_jacobian_diff[0])
        ratio = analytical_jacobian_diff[0] / numerical_jacobian_diff[0]
        f.write(f"absolute_error: {abs_err.tolist()}\n")
        f.write(f"relative_error: {rel_err.tolist()}\n")
        f.write(f"ratio: {ratio.tolist()}\n\n")

    # print("analytical_hessian[0]:", analytical_hessian[0])
    # print("noise_vec:", noise_vec)
    # print("numerical_jacobian:", numerical_jacobian_diff[0])
    # print("analytical_jacobian:", analytical_jacobian_diff[0])
    # print("the absolute difference is :", torch.abs(analytical_jacobian_diff[0]  - numerical_jacobian_diff[0]))
    # print("the relative difference is :", torch.abs(analytical_jacobian_diff[0]  - numerical_jacobian_diff[0]) / torch.abs(numerical_jacobian_diff[0]))
    # print("the ratio is:", analytical_jacobian_diff[0] / numerical_jacobian_diff[0])


    # (batch, n_edge, n_edge) @ (n_edge,1) -> (batch, n_edge, 1) -> squeeze -> (batch, n_edge)
    # analytical_bending = torch.matmul(
    #     dTheta_star_dBending,        # [B, n_edge, n_edge]
    #     noise_vec.unsqueeze(-1)      # [n_edge, 1]
    # ).squeeze(-1)               # -> [B, n_edge]
    # print("the shape of analytical_bending:", analytical_bending.shape)
    # print("numerical_bending:", numerial_bending[0])
    # print("analytical_bending:", analytical_bending[0])
    # print("the absolute difference is :", torch.abs(analytical_bending[0] - numerial_bending[0]))
    # print("the relative difference is :", torch.abs(analytical_bending[0] - numerial_bending[0]) / torch.abs(numerial_bending[0]))
    # print("the ratio is:", analytical_bending[0] / numerial_bending[0])

    # Return the final optimized twist angles
    return a_i_value
