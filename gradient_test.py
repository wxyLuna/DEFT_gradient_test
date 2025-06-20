import numpy as np


def f_x1(x1, x2, e1_bar, scale):
    """
    Compute the function:
      f(x1) = scale @ ( -2 * e1_bar^2 / (e1_bar^2 + ||x2 - x1||^2) ) * (x2 - x1)

    Parameters:
      x1      : numpy array of shape (3,), decision variable.
      x2      : numpy array of shape (3,), constant vector.
      e1_bar  : scalar, whose square is used.
      scale   : numpy array of shape (3,3), a constant matrix.

    Returns:
      f_val   : numpy array of shape (3,), the computed function value.
    """
    # Compute A = e1_bar^2 (a constant)
    A = e1_bar ** 2

    # Compute e1 = x2 - x1 and its squared norm B
    e1 = x2 - x1
    B = np.dot(e1, e1)

    # Compute the scalar multiplier Q = -2*A/(A + B)
    Q = -2 * A / (A + B)

    # Compute f(x1) = scale @ (Q * e1)
    f_val = scale @ (Q * e1)
    return f_val


def jacobian_f_x1(x1, x2, e1_bar, scale):
    """
    Compute the Jacobian (derivative with respect to x1) of the function:
      f(x1) = scale @ ( -2 * e1_bar^2 / (e1_bar^2 + ||x2-x1||^2) ) * (x2 - x1)

    The Jacobian is given by:
      J = scale @ [ (2*A/(A+B))*I - (4*A/(A+B)^2)*(x2-x1)(x2-x1)^T ]

    where:
      A = e1_bar^2 and B = ||x2-x1||^2.

    Parameters:
      x1      : numpy array of shape (3,), decision variable.
      x2      : numpy array of shape (3,), constant vector.
      e1_bar  : scalar, whose square is used.
      scale   : numpy array of shape (3,3), a constant matrix.

    Returns:
      J       : numpy array of shape (3,3), the Jacobian matrix of f with respect to x1.
    """
    # Compute A = e1_bar^2 and e1 = x2 - x1
    A = e1_bar ** 2
    e1 = x2 - x1
    # Compute B = ||x2 - x1||^2
    B = np.dot(e1, e1)

    # Create the 3x3 identity matrix
    I = np.eye(3)

    # Compute the two terms for the inner Jacobian:
    term1 = (2 * A / (A + B)) * I
    term2 = (4 * A / (A + B) ** 2) * np.outer(e1, e1)

    # The Jacobian of the inner function Q*e1 with respect to x1:
    J_inner = term1 - term2

    # Full Jacobian is left-multiplied by the scale matrix.
    J = scale @ J_inner
    return J


def func_DX_ICitr(M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Inextensibility Constraint Function

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

def grad_DX_X(idx_1, idx_2, M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Gradient of the inextensibility constraint function with respect to the positions X_0 and X_1.

    # Inputs:
    idx_1: Index of gradient, DX_0 (0) or DX_1 (1)
    idx_2: Index of gradient, X_0 (0) or X_1 (1)
    M_0: [3, 3] mass matrix of vertex i
    M_1: [3, 3] mass matrix of vertex i+1
    X_0: [3, 1] position of vertex i
    X_1: [3, 1] position of vertex i+1
    X_0_init: [3, 1] undeformed position of vertex i
    X_1_init: [3, 1] undeformed position of vertex i+1

    # Outputs:
    grad: [3, 3] gradient of the inextensibility constraint function output DX_0 or DX_1 with respect to X_0 or X_1.
    """

    M_param = np.linalg.inv(M_0 + M_1)
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init
    Edge_length = np.linalg.norm(Edge)
    Edge_length_init = np.linalg.norm(Edge_init)
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)

    if idx_1 == 0 and idx_2 == 0:
        # Gradient of DX_0 with respect to X_0
        grad = - M_1 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
        grad -= M_1 @ M_param * lambda_param
    elif idx_1 == 0 and idx_2 == 1:
        # Gradient of DX_0 with respect to X_1
        grad = M_1 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
        grad += M_1 @ M_param * lambda_param
    elif idx_1 == 1 and idx_2 == 0:
        # Gradient of DX_1 with respect to X_0
        grad = M_0 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
        grad += M_0 @ M_param * lambda_param
    elif idx_1 == 1 and idx_2 == 1:
        # Gradient of DX_1 with respect to X_1
        grad = - M_0 @ M_param @ Edge @ Edge.T * (4*Edge_length_init**2 / (Edge_length**2 + Edge_length_init**2)**2)
        grad -= M_0 @ M_param * lambda_param
    else:
        raise ValueError("Invalid indices for gradient computation. Use (0,0), (0,1), (1,0), or (1,1).")

    return grad

def grad_DX_Xinit(idx_1, idx_2, M_0, M_1, X_0, X_1, X_0_init, X_1_init):
    """
    Gradient of the inextensibility constraint function with respect to the undeformed positions X_0_init and X_1_init.

    # Inputs:
    idx_1: Index of gradient, DX_0 (0) or DX_1 (1)
    idx_2: Index of gradient, X_0 (0) or X_1 (1)
    M_0: [3, 3] mass matrix of vertex i
    M_1: [3, 3] mass matrix of vertex i+1
    X_0: [3, 1] position of vertex i
    X_1: [3, 1] position of vertex i+1
    X_0_init: [3, 1] undeformed position of vertex i
    X_1_init: [3, 1] undeformed position of vertex i+1

    # Outputs:
    grad: [3, 3] gradient of the inextensibility constraint function output DX_0 or DX_1 with respect to X_0_init or X_1_init.
    """

    M_param = np.linalg.inv(M_0 + M_1)
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init
    Edge_length = np.linalg.norm(Edge)
    Edge_length_init = np.linalg.norm(Edge_init)
    lambda_param = (Edge_length**2 - Edge_length_init**2) / (Edge_length**2 + Edge_length_init**2)
    if idx_1 == 0 and idx_2 == 0:
        # Gradient of DX_0 with respect to X_0_init
        grad = M_1 @ M_param @ Edge @ Edge_init.T * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    elif idx_1 == 0 and idx_2 == 1:
        # Gradient of DX_0 with respect to X_1_init
        grad = -M_1 @ M_param @ Edge @ Edge_init.T * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    elif idx_1 == 1 and idx_2 == 0:
        # Gradient of DX_1 with respect to X_0_init
        grad = -M_0 @ M_param @ Edge @ Edge_init.T * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    elif idx_1 == 1 and idx_2 == 1:
        # Gradient of DX_1 with respect to X_1_init
        grad = M_0 @ M_param @ Edge @ Edge_init.T * (4*Edge_length**2 / (Edge_length**2 + Edge_length_init**2)**2)
    else:
        raise ValueError("Invalid indices for gradient computation. Use (0,0), (0,1), (1,0), or (1,1).")

    return grad

def grad_DX_M(idx_1,idx_2,M_0,M_1,X_0,X_1,X_0_init,X_1_init):
    """
       Gradient of the inextensibility constraint function with respect to the mass matrixes M_0 and M_1.


       # Inputs:
       idx_1: Index of gradient, DX_0 (0) or DX_1 (1)
       idx_2: Index of gradient, X_0 (0) or X_1 (1)
       M_0: [3, 3] mass matrix of vertex i
       M_1: [3, 3] mass matrix of vertex i+1
       X_0: [3, 1] position of vertex i
       X_1: [3, 1] position of vertex i+1
       X_0_init: [3, 1] undeformed position of vertex i
       X_1_init: [3, 1] undeformed position of vertex i+1

       # Outputs:
       grad: [3, 3] gradient of the inextensibility constraint function output DX_0 or DX_1 with respect to X_0 or X_1.
    """
    M_param = np.linalg.inv(M_0+M_1)
    Edge = X_1 - X_0
    Edge_init = X_1_init - X_0_init
    Edge_norm = np.linalg.norm(Edge)
    Edge_init_norm = np.linalg.norm(Edge_init)
    lambda_param = (Edge_norm**2-Edge_init_norm**2)/ (Edge_norm**2+Edge_init_norm**2)
    if idx_1 == 0 and idx_2 == 0:
        grad_M = -M_1*M_param**2*lambda_param*Edge
    if idx_1 ==0 and idx_2 ==1:
        grad_M = (np.eye(3)-M_1*M_param)*M_param*lambda_param*Edge
    if idx_1 == 1 and idx_2 ==0:
        grad_M = (np.eye(3)-M_0*M_param)*M_param*lambda_param*Edge
    if idx_1 == 1 and idx_2 == 1:
        grad_M = -M_0*M_param**2*lambda_param*Edge
    else:
        raise ValueError("Invalid indices for gradient computation. Use (0,0), (0,1), (1,0), or (1,1).")
    return grad_M
# Example usage:
if __name__ == '__main__':
    # Define sample values:
    x1 = np.array([0.0020,  0.6336, -0.0126])
    x2 = np.array([-0.0046,  0.6808,  0.0102])

    # Let e1_bar be a scalar constant.
    e1_bar = 0.0447  # for example

    # Define scale as a 3x3 matrix, for instance:
    scale = np.array([[0.7349, 0., 0.],
                      [0.0, 0.7349, 0.],
                      [0.0, 0.0, 0.7349]])

    # Compute the function value and Jacobian.
    f_val = f_x1(x1, x2, e1_bar, scale)
    J = np.eye(3) - jacobian_f_x1(x1, x2, e1_bar, scale)

    print("Function f(x1):")
    print(f_val)
    print("\nJacobian df/dx1:")
    print(J)
