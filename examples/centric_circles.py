import math
import numpy as np


def get_centric_circles(
    n_samples: int = 500,
    noise: float = 0.0) -> np.ndarray:
    """
    Generate concentric circles with different radii and point densities.
    
    Creates three concentric circles:
    - Inner circle (R=0.7): 20% of points
    - Middle circle (R=2.0): 30% of points  
    - Outer circle (R=3.0): 50% of points
    
    Parameters
    ----------
    n_samples : int
        Total number of points to generate across all circles.
        
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2) containing (x, y) coordinates.
    """
    # Distribute points across three circles with different densities
    n_1 = int(n_samples * 0.2)  # Inner circle: 20% of points
    n_2 = int(n_samples * 0.3)  # Middle circle: 30% of points
    n_3 = int(n_samples * 0.5)  # Outer circle: 50% of points
    
    pts = []
    # Generate points for each circle with specified radius and count
    for R, n in [(0.7, n_1), (2.0, n_2), (3.0, n_3)]:
        # Random angles uniformly distributed around the circle
        ang = np.random.random(n) * 2 * math.pi
        R_ = R  # Fixed radius for perfect circles
        # Convert polar to Cartesian coordinates
        xs = R_ * np.cos(ang)
        ys = R_ * np.sin(ang)
        pts.extend(list(zip(xs, ys)))
    
    # Convert to numpy array with consistent dtype
    pts = np.asarray(pts, dtype=np.float64)
    
    return pts