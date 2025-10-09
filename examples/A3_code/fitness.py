import numpy as np
from typing import List, Tuple

# Define a global target position (can be changed if needed)
TARGET_POSITION: Tuple[float, float, float] = (5.0, 0.0, 0.5)

def fitness_function(history: List[List[float]]) -> float:
    """
    Compute fitness based on final distance to the target.

    Parameters
    ----------
    history : List[List[float]]
        A list of positions over time.
        Each position is [x, y, z].

    Returns
    -------
    float
        Fitness score (negative distance to target).
    """
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]  # final position of the robot

    # Minimize distance --> maximize negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2
    )
    return -cartesian_distance
