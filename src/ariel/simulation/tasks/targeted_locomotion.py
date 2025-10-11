"""Targeted locomotion."""


def distance_to_target(
    initial_position: tuple[float, float],
    target_position: tuple[float, float],
) -> float:
    """
    Euclidean distance between the current position and the target position.

     Args:
         initial_position (tuple): The current position as (x, y).
         target_position (tuple): The target position as (x, y).

     Returns:
         float: The distance between the two positions.
     """
    return (
         (initial_position[0] - target_position[0]) ** 2
         + (initial_position[1] - target_position[1]) ** 2
    ) ** 0.5

def distance_to_target_ff(xy_history, target_xy=(2.0, 0.0)) -> float:
    """
    Dense reward: reward the agent for moving closer to the target at each step.
    
    Args:
        xy_history (list of [x, y]): trajectory of robot positions.
        target_xy (tuple): target position.

    Returns:
        float: accumulated fitness over the trajectory.
    """
    fitness = 0.0
    for i in range(1, len(xy_history)):
        prev = np.array(xy_history[i-1])
        curr = np.array(xy_history[i])
        prev_dist = np.linalg.norm(prev - np.array(target_xy))
        curr_dist = np.linalg.norm(curr - np.array(target_xy))
        fitness += (prev_dist - curr_dist)  # positive if moved closer

    final_dist = np.linalg.norm(np.array(xy_history[-1]) - np.array(target_xy))
    fitness += 1.0 / (1.0 + final_dist)  # closer → larger bonus

    return fitness



