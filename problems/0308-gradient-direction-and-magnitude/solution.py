import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    """
    Calculate the magnitude and direction of a gradient vector.
    
    Args:
        gradient: A list representing the gradient vector
    
    Returns:
        Dictionary containing:
        - magnitude: The L2 norm of the gradient
        - direction: Unit vector in direction of steepest ascent
        - descent_direction: Unit vector in direction of steepest descent
    """
    magnitude = np.sqrt(sum(gi ** 2 for gi in gradient))
    unit_vector = np.zeros(len(gradient))
    if magnitude != 0:
        unit_vector = np.asarray(gradient, dtype=float) / magnitude 
    return {
        'magnitude' : magnitude,
        'direction' : unit_vector,
        'descent_direction' : - unit_vector
    }