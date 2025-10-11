import numpy as np
from typing import List, Tuple
from examples.A3_code.evolve.config import TARGET_POS

# Define a global target position (can be changed if needed)

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
    xt, yt, zt = TARGET_POS
    xc, yc, zc = history[-1]  # final position of the robot

    # Minimize distance --> maximize negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2
    )
    return -cartesian_distance

# From Assignment 2

def distance_to_target(pos_history) -> float:
    """
    Dense reward: reward the agent for moving closer to the 3D target at each step.

    Args:
        pos_history (list of [x, y, z]): trajectory of robot positions.
        target_pos (tuple): 3D target position.

    Returns:
        float: accumulated fitness over the trajectory.
    """
    fitness = 0.0
    for i in range(1, len(pos_history)):
        prev = np.array(pos_history[i-1])
        curr = np.array(pos_history[i])
        prev_dist = np.linalg.norm(prev - np.array(TARGET_POS))
        curr_dist = np.linalg.norm(curr - np.array(TARGET_POS))
        fitness += (prev_dist - curr_dist)  # positive if moved closer

    final_dist = np.linalg.norm(np.array(pos_history[-1]) - np.array(TARGET_POS))
    fitness += 1.0 / (1.0 + final_dist)  # closer → larger bonus

    return fitness

