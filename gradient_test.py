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
