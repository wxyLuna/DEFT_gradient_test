import numpy as np

# ------------------------------
# Problem Setup and Parameters
# ------------------------------

# Inner (lower-level) problem parameters:
alpha = 0.1  # inner learning rate for gradient descent in y
inner_steps = 50  # number of inner loop iterations (unrolling steps)
c = np.array([1.0, -0.5, 2.0])  # constant vector in R^3 shifting the inner optimum

# Outer (upper-level) problem parameters:
lambda_out = 1.0  # weight for the outer part that depends on y
d = np.array([0.5, 0.5, 0.5])  # target vector in R^3 for the outer objective

# Outer optimization hyperparameters:
outer_lr = 0.1  # learning rate for updating x
num_outer_iterations = 20  # total number of outer iterations


# ------------------------------
# Helper Functions
# ------------------------------

def inner_update(x, y, alpha, c):
    """
    Perform one gradient descent update on the inner problem:
      y_new = y - alpha * (y - (x+c)).
    """
    grad_y = y - (x + c)  # Analytical gradient: ∇_y g(x,y) = y - (x+c)
    return y - alpha * grad_y


def unroll_inner(x, y0, inner_steps, alpha, c):
    """
    Unroll the inner optimization steps.

    Returns:
      - y: the result after unrolling inner_steps starting from y0,
      - M: the accumulated sensitivity matrix (dy/dx) computed iteratively.

    The update for y:
        y_new = (1 - alpha)* y + alpha*(x+c)
    and for the sensitivity M (a 3x3 matrix):
        M_new = (1 - alpha)*M + alpha*I,
    where I is the 3x3 identity matrix.
    """
    y = y0.copy()
    M = np.zeros((3, 3))  # derivative of y with respect to x (initially zero since y0 is independent of x)
    I = np.eye(3)

    for _ in range(inner_steps):
        # Update y with gradient descent on the inner objective
        y = inner_update(x, y, alpha, c)
        # Update sensitivity (analytical derivative w.r.t. x)
        M = (1 - alpha) * M + alpha * I
    return y, M


def outer_objective(x, y, lambda_out, d):
    """
    Compute the outer objective:
      f(x,y) = 1/2 * ||x||^2 + (lambda_out/2) * ||y-d||^2
    """
    return 0.5 * np.dot(x, x) + 0.5 * lambda_out * np.dot((y - d), (y - d))


def grad_outer(x, y, M, lambda_out, d):
    """
    Compute the analytical gradient of the outer objective with respect to x.

    Using the chain rule:
       ∇_x f = ∇_x (1/2 ||x||^2) + lambda_out * (dy/dx)^T (y-d)
             = x + lambda_out * M^T (y-d)
    """
    return x + lambda_out * (M.T @ (y - d))


# ------------------------------
# Bilevel Optimization Procedure
# ------------------------------

# Initialize upper-level variable x in R^3 randomly and set y0 for the inner loop.
np.random.seed(0)
x = np.random.randn(3)  # initial x ∈ R^3
y0 = np.zeros(3)  # initial guess for y (independent of x)

print("Initial x:", x)

# Outer loop optimization: update x based on the outer gradient (accounting for the inner unrolling).
for outer_iter in range(num_outer_iterations):
    # Solve (approximately) the inner problem by unrolling inner_steps iterations.
    y, M = unroll_inner(x, y0, inner_steps, alpha, c)

    # Evaluate the outer objective using current x and the obtained y.
    f_val = outer_objective(x, y, lambda_out, d)

    # Compute the gradient of the outer objective with respect to x (analytically unrolled).
    grad_x = grad_outer(x, y, M, lambda_out, d)

    # Perform a gradient descent update on x.
    x = x - outer_lr * grad_x

    # Print intermediate results.
    print(f"Iteration {outer_iter + 1}: x = {x}, outer f = {f_val}")

print("\nFinal upper-level variable x:", x)
print("Final inner solution y (approximate minimizer):", y)
