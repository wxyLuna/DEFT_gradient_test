import numpy as np
import torch

# Gradient Saver
class BackwardGradientIC:
    def __init__(self, batch, num_branch,num_vertices):
        # the batch here is actually num_batch * num_branch
        self.batch = batch
        self.num_branch = num_branch
        self.num_vertices = num_vertices
        self.grad_DX_X = None
        self.grad_DX_Xinit = None
        self.grad_DX_M = None
        self.reset(batch, num_branch, num_vertices)

        return

    def reset(self, batch,num_branch, num_vertices):
        self.grad_DX_X = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices*3), dtype=np.float32) ## change dimension
        # self.grad_DX_Xinit = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices*3), dtype=np.float32)
        self.grad_DX_M = np.zeros((batch, num_branch*num_vertices*3, num_branch*num_vertices), dtype=np.float32)## change dimension
        return

class BackwardGradientDamping:
    def __init__(self, batch, branch, num_vertices):
        # the batch here is actually num_batch * num_branch, while the branch is num_branch
        self.grad_DX_damping = None
        self.reset(batch, branch, num_vertices)
        return

    def reset(self, batch, branch, num_vertices):
        self.grad_DX_damping = np.zeros((batch, num_vertices*3, branch), dtype=np.float32)
        return

class BackwardGradientIR:
    def __init__(self, batch, num_vertices):
        # the batch here is actually num_batch * num_branch
        self.grad_DX_IR = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_IR = np.zeros((batch, num_vertices*3, 1), dtype=np.float32)
        return

class BackwardGradientCoupling:
    def __init__(self, batch, num_vertices):
        # the batch here is actually num_batch * num_branch
        self.grad_DX_M_Coupling = None
        self.grad_DX_X_Coupling = None
        self.reset(batch, num_vertices)
        return

    def reset(self, batch, num_vertices):
        self.grad_DX_M_Coupling = np.zeros((batch, 2*3, 2), dtype=np.float32)
        self.grad_DX_X_Coupling = np.zeros((batch, 2*3, 6), dtype=np.float32)
        return

# Gradient Solver
#Inextensibility Constraint Enforcement

def grad_DX_X_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init, mask):
    """
    Robust batch gradient for inextensibility wrt positions.
    Inputs:
      M_0, M_1:        (B,3,3)
      X_0, X_1:        (B,3,1)
      X_0_init, X_1_init: (B,3,1)
      mask:            (B,) or (B,1) or (B,1,1); True = active edge
    Output:
      grad_DX_X:       (B,6,6)
    """
    eps = 1e-12

    # ---- to numpy (safe if torch tensors) ----
    def to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    M_0, M_1 = to_np(M_0), to_np(M_1)
    X_0, X_1 = to_np(X_0), to_np(X_1)
    X_0_init, X_1_init = to_np(X_0_init), to_np(X_1_init)
    B = M_0.shape[0]

    # ---- normalize mask to (B,1,1) ----
    mask = to_np(mask).astype(bool)
    if mask.ndim == 1:
        mask = mask[:, None, None]
    elif mask.ndim == 2:
        mask = mask[:, :, None]

    # ---- detect inactive/invalid batches ----
    m0_zero = (np.abs(M_0).sum(axis=(1,2), keepdims=True) == 0)
    m1_zero = (np.abs(M_1).sum(axis=(1,2), keepdims=True) == 0)
    x1_zero = (np.abs(X_1).sum(axis=(1,2), keepdims=True) == 0)
    x1i_zero= (np.abs(X_1_init).sum(axis=(1,2), keepdims=True) == 0)

    bothM_zero = m0_zero & m1_zero                  # cannot invert
    tail_zero  = m1_zero & x1_zero & x1i_zero       # tail vertex absent
    active     = (~bothM_zero) & (~tail_zero) & mask   # (B,1,1)

    # ---- default output zeros ----
    grad_DX_X = np.zeros((B, 6, 6), dtype=np.float64)
    if not np.any(active):
        return grad_DX_X

    idx = np.where(active.reshape(B))[0]
    M0v, M1v = M_0[idx], M_1[idx]
    X0v, X1v = X_0[idx], X_1[idx]
    X0iv, X1iv = X_0_init[idx], X_1_init[idx]
    b = len(idx)

    # ---- edge geometry ----
    Edge      = X1v - X0v           # (b,3,1)
    Edge_init = X1iv - X0iv         # (b,3,1)
    L  = np.linalg.norm(Edge,      axis=1, keepdims=True)   # (b,1,1)
    L0 = np.linalg.norm(Edge_init, axis=1, keepdims=True)   # (b,1,1)

    denom = L**2 + L0**2
    denom = np.where(denom < eps, eps, denom)

    # your scalars
    lam   = (L**2 - L0**2) / denom                 # (b,1,1)
    scale = 4.0 * (L0**2) / (denom**2)             # (b,1,1)

    # ---- safe inverse of (M0+M1) ----
    sumM    = M0v + M1v                            # (b,3,3)
    M_param = np.linalg.inv(sumM)    # (b,3,3) robust
    # outer product Edge*Edge^T: (b,3,3)
    E_outer = np.einsum('bik,bjk->bij', Edge, Edge)

    # Blocks (b,3,3), matching your formulas
    # Note: M_param already acts like your previous M_param;
    # scaled_value == scale; lambda_param == lam.

    term_M1  = M1v @ M_param                       # (b,3,3)
    term_M0  = M0v @ M_param                       # (b,3,3)
    M1_MP_EE = np.einsum('bij,bjk,bkl->bil', M1v, M_param, E_outer)
    M0_MP_EE = np.einsum('bij,bjk,bkl->bil', M0v, M_param, E_outer)

    grad_00 = -M1_MP_EE * scale - term_M1 * lam
    grad_01 =  M1_MP_EE * scale + term_M1 * lam
    grad_10 =  M0_MP_EE * scale + term_M0 * lam
    grad_11 = -M0_MP_EE * scale - term_M0 * lam

    # pack (b,6,6)
    top = np.concatenate([grad_00, grad_01], axis=2)  # (b,3,6)

    bot = np.concatenate([grad_10, grad_11], axis=2)  # (b,3,6)

    packed = np.concatenate([top, bot], axis=1)       # (b,6,6)

    grad_DX_X[idx] = packed
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
    M_param = np.zeros((batch_size, 3, 3))

    for i in range(batch_size):
        # Skip if any row in M_0[i] or M_1[i] is all zeros
        if np.any(np.all(M_0[i] == 0, axis=1)) or np.any(np.all(M_1[i] == 0, axis=1)):
            continue
        sum_M = M_0[i] + M_1[i]
        if np.linalg.det(sum_M) == 0:
            continue
        M_param[i] = np.linalg.inv(sum_M)

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


def grad_DX_M_ICitr_batch(M_0, M_1, X_0, X_1, X_0_init, X_1_init, mask):
    """
    Robust batch gradient for inextensibility wrt mass matrix.
    Inputs:
      M_0, M_1:        (B,3,3)
      X_0, X_1:        (B,3,1)
      X_0_init, X_1_init: (B,3,1)
      mask:            (B,) or (B,1) or (B,1,1); True = active edge
    Output:
      grad_DX_X:       (B,6,6)
    """
    # to numpy (no-ops if already np)
    def to_np(x): return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    M_0, M_1 = to_np(M_0), to_np(M_1)                 # (B,3,3)
    X_0, X_1 = to_np(X_0), to_np(X_1)                 # (B,3,1)
    X_0_init, X_1_init = to_np(X_0_init), to_np(X_1_init)

    mask = to_np(mask).astype(bool).reshape(-1)       # (B,)

    B = M_0.shape[0]
    out = np.zeros((B, 6, 2), dtype=np.float64)       # default zeros

    # ---- build “active” mask per batch entry ----
    m0_zero = (np.abs(M_0).sum(axis=(1,2)) == 0)
    m1_zero = (np.abs(M_1).sum(axis=(1,2)) == 0)
    x1_zero = (np.abs(X_1).sum(axis=(1,2)) == 0)
    x1i_zero= (np.abs(X_1_init).sum(axis=(1,2)) == 0)

    # your skip rules:
    # 1) if both M_0 and M_1 are all-zero -> skip
    bothM_zero = m0_zero & m1_zero
    # 2) if M_1, X_1, X_1_init all zero -> skip
    tail_zero = m1_zero & x1_zero & x1i_zero

    active = (~bothM_zero) & (~tail_zero) & mask      # (B,)

    if not np.any(active):
        return out  # all zeros, nothing to do

    idx = np.where(active)[0]

    # ---- compute only for active rows ----
    M0v, M1v = M_0[idx], M_1[idx]
    X0v, X1v = X_0[idx], X_1[idx]
    X0iv, X1iv = X_0_init[idx], X_1_init[idx]

    # safe edge scalars
    eps = 1e-12
    Edge = X1v - X0v                       # (b,3,1)
    Edge_i = X1iv - X0iv
    L  = np.linalg.norm(Edge,  axis=1, keepdims=True)   # (b,1,1)
    L0 = np.linalg.norm(Edge_i,axis=1, keepdims=True)
    denom = L**2 + L0**2
    denom = np.where(denom < eps, eps, denom)
    lam = (L**2 - L0**2) / denom                           # (b,1,1)

    # safe inverse of (M0+M1)
    sumM = M0v + M1v                                        # (b,3,3)
    Mparam = np.linalg.inv(sumM)              # (b,3,3)
    MM = Mparam @ Mparam

    I = np.broadcast_to(np.eye(3), Mparam.shape)            # (b,3,3)


    # grads (each (b,3,1)) — your earlier formulas
    g00 = -np.einsum('bij,bjk,bkl->bil', M1v, MM, Edge) * lam
    g01 =  np.einsum('bij,bjk->bik', (I - M1v @ Mparam), Mparam @ Edge) * lam
    g10 = -np.einsum('bij,bjk->bik', (I - M0v @ Mparam), Mparam @ Edge) * lam
    g11 =  np.einsum('bij,bjk,bkl->bil', M0v, MM, Edge) * lam

    top = np.concatenate([g00, g01], axis=2)   # (b,3,2)
    bot = np.concatenate([g10, g11], axis=2)   # (b,3,2)
    packed = np.concatenate([top, bot], axis=1)  # (b,6,2)

    out[idx] = packed
    return out


    # Damping
def grad_DX_damping_batch(n_vert, integration_ratio, dt, b_DLOs_velocity, n_branch):
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
    dt_np = dt
    
    # separate the velocities for each branch
    b_DLOs_velocity_expand = np.zeros((batch_size, n_vert, 3, n_branch), dtype=np.float32)
    for i in range(batch_size):
        b_DLOs_velocity_expand[i, :, :, i % n_branch] = b_DLOs_velocity_np[i, :, :]
    b_DLOs_velocity_expand = b_DLOs_velocity_expand.reshape(batch_size, n_vert * 3, n_branch)
    grad_DX_damping = - integration_ratio_np * dt_np**2 * b_DLOs_velocity_expand

    return grad_DX_damping

    # Integration Ratio
def grad_DX_IR_batch(dt, b_DLOs_velocity, mass_matrix, force, damping):
    """
    Batch version of Gradient of the integration ratio constraint iterative function with respect to the integration ratio.

    # Inputs:
    - dt: [1] time step size
    - b_DLOs_velocity: [batch_size, n_vert, 3] velocities of the DLOs
    - mass_matrix: [batch_size, n_vert, 3, 3] mass matrix of the DLOs
    - force: [batch_size, n_vert, 3] forces acting on the DLOs
    - damping: [n_branch] damping forces acting on the DLOs

    # Outputs:
    - grad_DX_IR: [batch_size, n_vert*3, 1] gradient of DX_IR with respect to the integration ratio

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """
    
    batch_size = b_DLOs_velocity.shape[0]
    n_branch = damping.shape

    dt_np = dt
    b_DLOs_velocity_np = b_DLOs_velocity.detach().cpu().numpy()#(1,7,3)
    mass_matrix_np = mass_matrix.detach().cpu().numpy()
    force_np = force.detach().cpu().numpy()#(1,7,3)
    damping_np = damping.detach().cpu().numpy()#(5.0)
    if damping_np.ndim == 0:
        damping_expand = damping_np.reshape(1, 1, 1)*np.ones((batch_size, 1, 1)) #batch_size, 1, 1
    else:
        damping_expand = damping_np.reshape(-1, 1, 1)*np.ones((batch_size, 1, 1))  #batch_size*branch, 1, 1

    M_inv = np.linalg.pinv(mass_matrix_np)  # [batch_size, n_vert, 3, 3]

    acc = np.einsum('ijkl,ijl->ijk', M_inv, force_np) - b_DLOs_velocity_np * damping_expand
    vel = b_DLOs_velocity_np + acc * dt_np
    vel_shrinked = vel.reshape(batch_size, -1, 1)  # [batch_size, n_vert*3, 1]
    grad_DX_IR = vel_shrinked * dt_np

    return grad_DX_IR

def grad_DX_X_ICEC_batch(M_0, M_1):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the positions X_0 and X_1.

    # Inputs:
    - M_0: [batch*n_parent_branch, n_vertices, 3, 3] mass matrix at two coupling index of parent branch
    - M_1: [batch*n_child_branch, 3, 3] mass matrix of the first index of two children branches


    # Outputs:
    - grad_X_pc_pc: [batch, 3, 3] gradient of DX_pc with respect to parent branch mass matrix at two coupling index M_pc
    - grad_X_pc_cc: [batch, 3, 3] gradient of DX_pc with respect to two children branches mass matrix at two coupling index M_cc
    - grad_X_cc_pc: [batch, 3, 3] gradient of two children branch's DX_cc with respect to parent branch mass matrix M_pc
    - grad_X_cc_cc: [batch, 3, 3] gradient of two children branch's DX_cc with respect to their own mass matrix M_cc

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    M_0, M_1 = M_0.detach().cpu().numpy(), M_1.detach().cpu().numpy()
    batch_size = M_0.shape[0]

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_0 + M_1)  # [batch_size, 3, 3]


    grad_00 = -np.einsum('bij,bjk->bik', M_1, M_param)

    grad_01 = np.einsum('bij,bjk->bik', M_1, M_param)

    grad_10 = np.einsum('bij,bjk->bik', M_0, M_param)

    grad_11 = -np.einsum('bij,bjk->bik', M_0, M_param)

    grad_DX_X = np.concatenate(
        (np.concatenate((grad_00, grad_01), axis=2),
         np.concatenate((grad_10, grad_11), axis=2)),
        axis=1
    )

    return grad_00, grad_01, grad_10, grad_11, grad_DX_X

def grad_DX_M_ICEC_batch(M_pc, M_cc, X_pc, X_cc):
    """
    Batch version of Gradient of the inextensibility constraint iterative function with respect to the mass matrices M_0 and M_1.

    # Inputs:
    - M_pc: [batch, n_vertices, 3, 3] mass matrix at two coupling index of parent branch
    - M_cc: [batch, 3, 3] mass matrix of the first index of two children branches
    - X_pc: [batch, 3, 1] position of vertex at two coupling index of parent branch
    - X_cc: [batch, 3, 1] position of vertex of the first index of two children branches

    # Outputs:
    - grad_M_pc_pc: [batch, 3, 1] gradient of DM_pc with respect to parent branch mass matrix at two coupling index M_pc
    - grad_M_pc_cc: [batch, 3, 1] gradient of DM_pc with respect to two children branches mass matrix at two coupling index M_cc
    - grad_M_cc_pc: [batch, 3, 1] gradient of two children branch's DM_cc with respect to parent branch mass matrix M_pc
    - grad_M_cc_cc: [batch, 3, 1] gradient of two children branch's DM_cc with respect to their own mass matrix M_cc

    the batch here is actually num_batch * num_branch, while the branch is num_branch
    """

    M_pc, M_cc = M_pc.detach().cpu().numpy(), M_cc.detach().cpu().numpy()
    X_pc, X_cc = X_pc.detach().cpu().numpy(), X_cc.detach().cpu().numpy()

    # Compute M_param for each batch
    M_param = np.linalg.inv(M_pc + M_cc)  # [2, 3, 3]

    # Compute Edge and Edge_init for each batch ## zero-mask not applied yet
    Edge = X_cc - X_pc  # [2, 3, 1]


    # Compute gradients for each batch
    grad_M_pc_pc = -np.einsum('bij,bjk,bkl->bil', M_cc, M_param @ M_param, Edge)
    grad_M_pc_cc = np.einsum('bij,bjk->bik', (np.eye(3) - M_cc @ M_param), (M_param @ Edge))
    grad_M_cc_pc = -np.einsum('bij,bjk->bik', (np.eye(3) - M_pc @ M_param), M_param @ Edge)  # [batch_size, 3, 1]
    grad_M_cc_cc = np.einsum('bij,bjk,bkl->bil', M_pc, M_param @ M_param, Edge)  # [batch_size, 3, 1]

    grad_DX_M = np.concatenate(
        (np.concatenate((grad_M_pc_pc, grad_M_pc_cc), axis=2),
         np.concatenate((grad_M_cc_pc, grad_M_cc_cc), axis=2)),
        axis=1
    )

    return grad_M_pc_pc, grad_M_pc_cc, grad_M_cc_pc, grad_M_cc_cc, grad_DX_M

class RCEPC_gradient:
    def __init__(self):
        pass
    
    def hat(self, r):
        """
        Compute the hat operator for a 3D vector, converting it to a skew-symmetric matrix.

        Args:
            r: axis_angle, [batch, 3].
        Returns:
            hat_r: skew-symmetric matrix, [batch, 3, 3].
        """
        batch_size = r.shape[0]
        hat_r = np.zeros((batch_size, 3, 3), dtype=np.float64)
        hat_r[:, 0, 1] = -r[:, 2]
        hat_r[:, 0, 2] = r[:, 1]
        hat_r[:, 1, 0] = r[:, 2]
        hat_r[:, 1, 2] = -r[:, 0]
        hat_r[:, 2, 0] = -r[:, 1]
        hat_r[:, 2, 1] = r[:, 0]
        return hat_r
    
    def left_jacobian_so3(self, r):
        """
        Compute the left Jacobian of SO(3) for a batch of rotation vectors.
        Args:
            r: axis_angle, [batch, 3].
        Returns:
            J: left Jacobian, [batch, 3, 3].
        """
        theta = np.linalg.norm(r, axis=-1, keepdims=True)
        theta_safe = np.where(theta == 0, 1, theta)
        hat_r = self.hat(r)  # [batch, 3, 3]

        I = np.broadcast_to(np.eye(3), r.shape[:-1] + (3, 3))

        a = (1 - np.cos(theta)) / (theta_safe ** 2)
        b = (theta - np.sin(theta)) / (theta_safe ** 3)

        # Small-angle fix
        eps = 1e-6
        small = theta < eps
        a = np.where(small, 0.5 - theta**2 / 24 + theta**4 / 720, a)
        b = np.where(small, 1/6 - theta**2 / 120 + theta**4 / 5040, b)

        J = I - a * hat_r + b * (hat_r @ hat_r)
        return J
    
    def ruv_jacobian(self, u, v, eps=1e-8):
        """
        Compute the Jacobian of the rotation vector with respect to the input vectors u and v.

        Args:
            u: first vector, [batch, 3].
            v: second vector, [batch, 3].

        Returns:
            S_u: Jacobian matrix, [batch, 3, 3].
            S_v: Jacobian matrix, [batch, 3, 3].
        """
        batch_size = u.shape[0]

        a = np.cross(u, v)                                # [batch,3]
        c = np.sum(u * v, axis=-1, keepdims=True)         # [batch,1]
        s = np.linalg.norm(a, axis=-1, keepdims=True)     # [batch,1]

        Su = np.zeros((batch_size, 3, 3))
        Sv = np.zeros((batch_size, 3, 3))

        mask = (s[:,0] < eps)    # [batch]

        # small angle approximation
        if np.any(mask):
            Su[mask] = -self.hat(v[mask])   # [m,3,3]
            Sv[mask] =  self.hat(u[mask])

        if np.any(~mask):
            idx = np.where(~mask)[0]
            a_sel, u_sel, v_sel = a[idx], u[idx], v[idx]
            c_sel, s_sel = c[idx], s[idx]

            theta = np.arctan2(s_sel, c_sel)
            n = a_sel / s_sel

            I = np.eye(3)[None, :, :]
            A = (theta/s_sel)[:,None,None] * I + (c_sel - theta/s_sel)[:, :, None] * np.einsum('bi,bj->bij', n, n)

            Su[idx] = A @ (-self.hat(v_sel)) - s_sel[:, :, None] * np.einsum('bi,bj->bij', n, v_sel)
            Sv[idx] = A @ self.hat(u_sel) - s_sel[:, :, None] * np.einsum('bi,bj->bij', n, u_sel)

        return Su, Sv
    
    def gradient_DX_MOI(self, Xpc0, Xpc1, Xcc0, Xcc1, DRpc, DRcc, Drpc, Drcc, Dr, MOIpc, MOIcc):
        """
        Compute the gradient wrt moments of inertia
        Args:
            Xpc0: parent rod vertices at pc, [batch, 3].
            Xpc1: parent rod vertices at pc+1, [batch, 3].
            Xcc0: child rod vertices at cc, [batch, 3].
            Xcc1: child rod vertices at cc+1, [batch, 3].
            DRpc: rotational offset of parent rod, in the form of SO(3) rotation, [batch, 3, 3].
            DRcc: rotational offset of child rod, in the form of SO(3) rotation, [batch, 3, 3].
            Drpc: axis-angle representation of parent rod rotational offset, [batch, 3].
            Drcc: axis-angle representation of child rod rotational offset, [batch, 3].
            Dr: axis-angle representation of the difference between parent and child rods, [batch, 3].
            MOIpc: moments of inertia for parent rods, [batch, 3, 3].
            MOIcc: moments of inertia for child rods, [batch, 3, 3].
        """

        Epc = Xpc1 - Xpc0  # parent edge vector, [batch, 3]
        Ecc = Xcc1 - Xcc0  # child edge vector, [batch, 3]
        
        J_00 = - DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ np.diag(Dr) @ MOIcc @ np.linalg.inv(MOIpc + MOIcc) @ np.linalg.inv(MOIpc + MOIcc) # [batch, 3, 3], DX_{pc+1} / MOI_{pc}
        J_01 = DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ np.diag(Dr) @ MOIpc @ np.linalg.inv(MOIpc + MOIcc) @ np.linalg.inv(MOIpc + MOIcc) # [batch, 3, 3], DX_{pc+1} / MOI_{cc}
        J_10 = - DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ np.diag(Dr) @ MOIcc @ np.linalg.inv(MOIpc + MOIcc) @ np.linalg.inv(MOIpc + MOIcc) # [batch, 3, 3], DX_{cc+1} / MOI_{pc}
        J_11 = DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ np.diag(Dr) @ MOIpc @ np.linalg.inv(MOIpc + MOIcc) @ np.linalg.inv(MOIpc + MOIcc) # [batch, 3, 3], DX_{cc+1} / MOI_{cc}

        J = np.zeros((Xpc0.shape[0], 6, 6), dtype=np.float64)
        J[:, :3, :3] = J_00
        J[:, :3, 3:] = J_01
        J[:, 3:, :3] = J_10
        J[:, 3:, 3:] = J_11

        return J_00, J_01, J_10, J_11, J

    def gradient_DX_X(self, Xpc0, Xpc1, Xcc0, Xcc1, DRpc, DRcc, Drpc, Drcc, Dr, MOIpc, MOIcc, Rcc, rpc, rcc, Xpc0_init, Xpc1_init, Xcc0_init, Xcc1_init):
        """
        Compute the gradient wrt rod vertices
        Args:
            Xpc0: parent rod vertices at pc, [batch, 3].
            Xpc1: parent rod vertices at pc+1, [batch, 3].
            Xcc0: child rod vertices at cc, [batch, 3].
            Xcc1: child rod vertices at cc+1, [batch, 3].
            DRpc: rotational offset of parent rod, in the form of SO(3) rotation, [batch, 3, 3].
            DRcc: rotational offset of child rod, in the form of SO(3) rotation, [batch, 3, 3].
            Drpc: axis-angle representation of parent rod rotational offset, [batch, 3].
            Drcc: axis-angle representation of child rod rotational offset, [batch, 3].
            Dr: axis-angle representation of the difference between parent and child rods, [batch, 3].
            MOIpc: moments of inertia for parent rods, [batch, 3, 3].
            MOIcc: moments of inertia for child rods, [batch, 3, 3].
            Rcc: rotation matrix for child rods, [batch, 3, 3].
            rpc: parent rod edge vector, [batch, 3].
            rcc: child rod edge vector, [batch, 3].
            Xpc0_init: initial parent rod vertices at pc, [batch, 3].
            Xpc1_init: initial parent rod vertices at pc+1, [batch, 3].
            Xcc0_init: initial child rod vertices at cc, [batch, 3].
            Xcc1_init: initial child rod vertices at cc+1, [batch, 3].
        """
        batch_size = Xpc0.shape[0]

        Epc = Xpc1 - Xpc0  # parent edge vector, [batch, 3]
        Ecc = Xcc1 - Xcc0  # child edge vector, [batch, 3]
        Epc_init = Xpc1_init - Xpc0_init  # initial parent edge vector, [batch, 3]
        Ecc_init = Xcc1_init - Xcc0_init  # initial child edge vector, [batch, 3]

        epc = Epc / np.linalg.norm(Epc, axis=-1, keepdims=True)  # normalized parent edge vector, [batch, 3]
        ecc = Ecc / np.linalg.norm(Ecc, axis=-1, keepdims=True)  # normalized child edge vector, [batch, 3]
        epc_init = Epc_init / np.linalg.norm(Epc_init, axis=-1, keepdims=True) # normalized initial parent edge vector, [batch, 3]
        ecc_init = Ecc_init / np.linalg.norm(Ecc_init, axis=-1, keepdims=True) # normalized initial child edge vector, [batch, 3]

        _, Sv_pc = self.ruv_jacobian(epc_init, epc)  # Jacobian for parent edge
        _, Sv_cc = self.ruv_jacobian(ecc_init, ecc)  # Jacobian for child edge

        j_norm_pc = ((I - np.einsum('bi,bj->bij', epc, epc)) / np.linalg.norm(Epc, axis=-1, keepdims=True))
        j_norm_cc = ((I - np.einsum('bi,bj->bij', ecc, ecc)) / np.linalg.norm(Ecc, axis=-1, keepdims=True))

        MOIs0 = (-MOIcc) @ np.linalg.inv(MOIpc + MOIcc)
        MOIs1 = MOIpc @ np.linalg.inv(MOIpc + MOIcc)

        I = np.broadcast_to(np.eye(3), (batch_size, 3, 3))
        J_00 = (DRpc - I) - DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ MOIs0 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rpc) @ Sv_pc @ j_norm_pc # [batch, 3, 3], DX_{pc+1} / X_{pc+1}
        J_01 = -(DRpc - I) + DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ MOIs0 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rpc) @ Sv_pc @ j_norm_pc # [batch, 3, 3], DX_{pc+1} / X_{pc}
        J_02 = DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ MOIs0 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rcc) @ Sv_cc @ j_norm_cc # [batch, 3, 3], DX_{pc+1} / X_{cc+1}
        J_03 = -DRpc @ self.hat(Epc) @ self.left_jacobian_so3(Drpc) @ MOIs0 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rcc) @ Sv_cc @ j_norm_cc # [batch, 3, 3], DX_{pc+1} / X_{cc}
        J_10 = -DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ MOIs1 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rpc) @ Sv_pc @ j_norm_pc # [batch, 3, 3], DX_{cc+1} / X_{pc+1}
        J_11 = DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ MOIs1 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rpc) @ Sv_pc @ j_norm_pc # [batch, 3, 3], DX_{cc+1} / X_{pc}
        J_12 = (DRcc - I) + DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ MOIs1 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rcc) @ Sv_cc @ j_norm_cc # [batch, 3, 3], DX_{cc+1} / X_{cc+1}
        J_13 = -(DRcc - I) - DRcc @ self.hat(Ecc) @ self.left_jacobian_so3(Drcc) @ MOIs1 @ np.linalg.inv(self.left_jacobian_so3(Dr)) @ Rcc @ self.left_jacobian_so3(rcc) @ Sv_cc @ j_norm_cc # [batch, 3, 3], DX_{cc+1} / X_{cc+1}

        J = np.zeros((Xpc0.shape[0], 6, 12), dtype=np.float64)
        J[:, :3, 0:3] = J_01
        J[:, :3, 3:6] = J_00
        J[:, :3, 6:9] = J_03
        J[:, :3, 9:12] = J_02
        J[:, 3:, 0:3] = J_11
        J[:, 3:, 3:6] = J_10
        J[:, 3:, 6:9] = J_13
        J[:, 3:, 9:12] = J_12

        return J_00, J_01, J_02, J_03, J_10, J_11, J_12, J_13, J