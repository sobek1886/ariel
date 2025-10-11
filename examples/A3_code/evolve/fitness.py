import numpy as np
from typing import List, Tuple
from examples.A3_code.evolve.config import TARGET_POS, SPAWN_POS

# Define a global target position (can be changed if needed)

def dist_to_target(history: List[List[float]]) -> float:
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

def forward_progress_fitness(pos_history: List[List[float]]) -> float:
    """
    Fitness based on forward progress with speed bonus.
    
    Best for Olympic Arena because:
    - Rewards moving toward the goal (not just distance)
    - Penalizes time taken (faster = better)
    - Handles the directional nature of the track
    
    Args:
        pos_history: List of [x, y, z] positions over time
        
    Returns:
        float: Fitness score (higher is better)
    """
    if len(pos_history) < 2:
        return -999.0
    
    start_pos = np.array(SPAWN_POS)
    target_pos = np.array(TARGET_POS)
    final_pos = np.array(pos_history[-1])
    
    # 1. Forward progress (main component)
    # How far did we move toward the target?
    initial_dist = np.linalg.norm(target_pos - start_pos)
    final_dist = np.linalg.norm(target_pos - final_pos)
    progress = initial_dist - final_dist  # positive = moved closer
    
    # 2. Speed bonus: reward reaching target faster
    # If we're close to target, give bonus inversely proportional to time
    if final_dist < 0.5:  # within 0.5m of target
        speed_bonus = 10.0 / len(pos_history)  # faster = higher bonus
    else:
        speed_bonus = 0.0
    
    # 3. Penalty for deviation from straight path
    # Encourage efficient movement, not zigzagging
    path_length = 0.0
    for i in range(1, len(pos_history)):
        path_length += np.linalg.norm(
            np.array(pos_history[i]) - np.array(pos_history[i-1])
        )
    
    straight_line_dist = np.linalg.norm(final_pos - start_pos)
    efficiency = straight_line_dist / (path_length + 1e-6)  # closer to 1 = more efficient
    efficiency_bonus = efficiency * 0.5
    
    # Total fitness
    fitness = progress + speed_bonus + efficiency_bonus
    
    return fitness


def distance_to_target_improved(pos_history: List[List[float]]) -> float:
    """
    Improved dense reward with better balance and anti-jittering.
    
    Args:
        pos_history: List of [x, y, z] positions over time
        
    Returns:
        float: Fitness score (higher is better)
    """
    if len(pos_history) < 2:
        return -999.0
    
    target_pos = np.array(TARGET_POS)
    fitness = 0.0
    
    # Track minimum distance achieved (anti-jittering)
    min_dist_achieved = float('inf')
    
    for i in range(1, len(pos_history)):
        prev = np.array(pos_history[i-1])
        curr = np.array(pos_history[i])
        
        prev_dist = np.linalg.norm(prev - target_pos)
        curr_dist = np.linalg.norm(curr - target_pos)
        
        # Update minimum distance
        min_dist_achieved = min(min_dist_achieved, curr_dist)
        
        # Reward movement toward target
        step_progress = prev_dist - curr_dist
        if step_progress > 0:  # Only reward actual progress
            fitness += step_progress
    
    # Strong bonus for getting close
    final_dist = np.linalg.norm(np.array(pos_history[-1]) - target_pos)
    
    # Exponential bonus as you get closer
    if final_dist < 1.0:
        fitness += 5.0 * (1.0 - final_dist)  # 0-5 bonus for last meter
    
    # Huge bonus for reaching target
    if final_dist < 0.3:  # "finished" the race
        fitness += 20.0
    
    # Bonus for best distance achieved (rewards exploration)
    fitness += 2.0 / (1.0 + min_dist_achieved)
    
    return fitness


def olympic_arena_fitness(pos_history: List[List[float]]) -> float:
    """
    Specialized fitness for Olympic Arena terrain (flat → rough → uphill).
    
    Gives extra rewards for passing terrain checkpoints.
    
    Args:
        pos_history: List of [x, y, z] positions over time
        
    Returns:
        float: Fitness score (higher is better)
    """
    if len(pos_history) < 2:
        return -999.0
    
    start_pos = np.array(SPAWN_POS)
    target_pos = np.array(TARGET_POS)
    final_pos = np.array(pos_history[-1])
    
    # Approximate terrain boundaries (adjust based on actual arena)
    # Assuming x-axis progression: flat (x < 1), rough (1 < x < 3), uphill (x > 3)
    FLAT_END = 1.0
    ROUGH_END = 3.0
    
    fitness = 0.0
    
    # 1. Base progress reward
    initial_dist = np.linalg.norm(target_pos[:2] - start_pos[:2])  # 2D distance
    final_dist = np.linalg.norm(target_pos[:2] - final_pos[:2])
    progress = initial_dist - final_dist
    fitness += progress * 2.0  # doubled weight for progress
    
    # 2. Milestone bonuses for terrain sections
    x_pos = final_pos[0]
    
    if x_pos >= start_pos[0]:  # Moving forward
        if x_pos >= FLAT_END:
            fitness += 3.0  # Crossed flat section
        if x_pos >= ROUGH_END:
            fitness += 5.0  # Crossed rough section
        if x_pos >= target_pos[0] - 0.5:
            fitness += 10.0  # Almost at finish
    
    # 3. Height bonus (reward for climbing uphill section)
    height_gain = max(0, final_pos[2] - start_pos[2])
    fitness += height_gain * 2.0
    
    # 4. Final distance bonus
    if final_dist < 0.5:
        fitness += 15.0 * (0.5 - final_dist)  # Big reward for getting very close
    
    # 5. Completion bonus
    if final_dist < 0.3:
        fitness += 30.0  # Massive bonus for reaching target
    
    return fitness


def hybrid_fitness(pos_history: List[List[float]]) -> float:
    """
    Combination of multiple fitness components for robust evolution.
    
    Recommended for final submission.
    """
    # Get individual fitness scores
    progress_score = forward_progress_fitness(pos_history)
    dense_score = distance_to_target_improved(pos_history)
    terrain_score = olympic_arena_fitness(pos_history)
    
    # Weighted combination
    fitness = (
        0.3 * progress_score +
        0.3 * dense_score +
        0.4 * terrain_score
    )
    
    return fitness


# 1. Start with: distance_to_target_improved (good baseline)
# 2. Experiment with: forward_progress_fitness (if robots get stuck)
# 3. Fine-tune with: olympic_arena_fitness (terrain-aware)
# 4. Final submission: hybrid_fitness (most robust)