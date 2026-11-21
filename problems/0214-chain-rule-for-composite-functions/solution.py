import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """
    Compute derivative of composite functions using chain rule.
    
    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative
    
    Returns:
        Derivative value at x
    
    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    mapping_function = {
        "square": lambda v: v ** 2,
        "sin": np.sin,
        "exp": np.exp,
        "log": np.log,
    }
    mapping_derivative = {
        "square": lambda v: 2 * v,
        "sin": np.cos,
        "exp": np.exp,
        "log": lambda v: 1 / v,
    }
    val = x
    grad = 1.0
    for f in reversed(functions):
        grad *= mapping_derivative[f](val)
        val = mapping_function[f](val)
    return grad