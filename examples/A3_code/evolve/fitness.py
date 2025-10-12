import numpy as np
from typing import List, Tuple
from examples.A3_code.evolve.config import TARGET_POS


# Olympic Arena terrain boundaries
FLAT_END = -0.5
RUGGED_START = -0.5
RUGGED_END = 2.5
INCLINE_START = 2.5
INCLINE_END = 4.5
FINISH_LINE = 5.43


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

def olympic_arena_fitness_simple(pos_history: List[np.ndarray]) -> float:
    """
    Simplified fitness function focused purely on forward progress.
    Use this if the complex version causes issues.
    
    Args:
        pos_history: List of [x, y, z] positions over time
    
    Returns:
        Fitness score (higher is better)
    """
    if len(pos_history) < 2:
        return -999.0
    
    start_x = pos_history[0][0]
    end_x = pos_history[-1][0]
    
    # Just reward forward distance
    forward_distance = end_x - start_x
    
    # Small bonus for reaching further
    if forward_distance > 10:
        forward_distance += 5  # Bonus for significant progress
    if forward_distance > 20:
        forward_distance += 10  # Larger bonus for reaching far
    
    return forward_distance



def olympic_arena_fitness_progressive(pos_history) -> float:
    """Progressive fitness for Olympic Arena - use this first!"""
    if len(pos_history) < 2:
        return -999.0
    
    start_x = pos_history[0][0]
    max_x = max(pos[0] for pos in pos_history)
    forward_distance = max_x - start_x
    
    if forward_distance < 0:
        return -100.0 + forward_distance
    
    fitness = (forward_distance ** 1.15) * 10.0
    
    # Milestones based on actual arena
    if forward_distance > 0.5: fitness += 5
    if forward_distance > 1.0: fitness += 10
    if forward_distance > 2.0: fitness += 20
    if forward_distance > 3.0: fitness += 35
    if forward_distance > 3.5: fitness += 50
    if forward_distance > 4.0: fitness += 75
    if forward_distance > 5.0: fitness += 120
    if forward_distance > 5.43: fitness += 250  # Finish line!
    
    return fitness


def olympic_arena_fitness_enhanced(pos_history, qvel_history=None) -> float:
    """Enhanced fitness with stability and efficiency metrics"""
    if len(pos_history) < 2:
        return -999.0
    
    start_x = pos_history[0][0]
    max_x = max(pos[0] for pos in pos_history)
    forward_distance = max_x - start_x
    
    if forward_distance < 0:
        return -100.0 + forward_distance
    
    # Base distance reward
    fitness = (forward_distance ** 1.15) * 10.0
    
    # Milestone bonuses
    milestones = [(0.5, 5), (1.0, 10), (2.0, 20), (3.0, 35), 
                  (3.5, 50), (4.0, 75), (5.0, 120), (5.43, 250)]
    for threshold, bonus in milestones:
        if forward_distance > threshold:
            fitness += bonus
    
    # --- NEW: Path efficiency bonus ---
    total_path_length = sum(
        np.linalg.norm(pos_history[i] - pos_history[i-1])
        for i in range(1, len(pos_history))
    )
    if total_path_length > 0:
        efficiency = forward_distance / total_path_length
        fitness += efficiency * 20  # Reward straight paths
    
    # --- NEW: Penalize sideways drift ---
    lateral_drift = abs(pos_history[-1][1] - pos_history[0][1])
    fitness -= lateral_drift * 5
    
    # --- NEW: Height stability bonus ---
    heights = [pos[2] for pos in pos_history]
    height_variance = np.var(heights)
    fitness -= height_variance * 10  # Penalize bouncing
    
    return fitness




def olympic_arena_fitness_dense(pos_history: List[np.ndarray]) -> float:
    """
    Dense reward shaping for Olympic Arena navigation.
    Provides learning signal even for early random robots.
    """
    if len(pos_history) < 2:
        return -999.0
    
    fitness = 0.0
    
    # === 1. Stability bonus (didn't fall or explode) ===
    heights = [pos[2] for pos in pos_history]
    if min(heights) > -0.5 and max(heights) < 5.0:
        fitness += 20.0  # Big bonus for staying alive
    else:
        return -500.0  # Severe penalty for instability
    
    # === 2. Forward progress (main objective) ===
    start_x = pos_history[0][0]
    max_x = max(pos[0] for pos in pos_history)
    forward_distance = max_x - start_x
    
    if forward_distance < 0:
        return -100.0  # Moved backwards
    
    # Exponential reward for distance
    fitness += (forward_distance ** 1.2) * 15.0
    
    # === 3. Path efficiency bonus ===
    total_path_length = sum(
        np.linalg.norm(pos_history[i] - pos_history[i-1])
        for i in range(1, len(pos_history))
    )
    if total_path_length > 0 and forward_distance > 0:
        efficiency = forward_distance / total_path_length
        fitness += efficiency * 25.0  # Reward straight paths
    
    # === 4. Lateral stability (stay on course) ===
    lateral_positions = [pos[1] for pos in pos_history]
    lateral_drift = max(lateral_positions) - min(lateral_positions)
    fitness -= lateral_drift * 3.0  # Penalize side-to-side wobbling
    
    # === 5. Height consistency (reduce bouncing) ===
    height_variance = np.var(heights)
    fitness -= height_variance * 8.0
    
    # === 6. Progressive milestone bonuses ===
    # These bootstrap early learning
    if forward_distance > 0.1:  fitness += 10   # Any movement
    if forward_distance > 0.3:  fitness += 15   # Significant movement
    if forward_distance > 0.5:  fitness += 25   # Past flat section
    if forward_distance > 1.0:  fitness += 40   # Into rugged terrain
    if forward_distance > 2.0:  fitness += 60   # Through rugged
    if forward_distance > 3.0:  fitness += 100  # Reached incline
    if forward_distance > 4.0:  fitness += 150  # Up the incline
    if forward_distance > 5.0:  fitness += 250  # Near finish
    if forward_distance > 5.43: fitness += 500  # CROSSED FINISH LINE!
    
    # === 7. Final position bonus ===
    final_x = pos_history[-1][0]
    if final_x > FINISH_LINE:
        fitness += 1000.0  # Massive bonus for completion
    
    return fitness