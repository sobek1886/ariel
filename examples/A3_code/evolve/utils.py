from pathlib import Path
import csv, json
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
import networkx as nx

from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from examples.A3_code.evolve.nn import make_controller_from_genome, decode_genome, build_controller, infer_input_size
from examples.A3_code.evolve.config import DURATION, SPAWN_POS, TARGET_POS, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, STATE_FEATURES
from examples.A3_code.evolve.fitness import olympic_arena_fitness

def count_num_joints(robot_graph):
    """Calculates number of joints directly from a NetworkX DiGraph."""
    if not hasattr(robot_graph, "nodes"):
        raise TypeError(f"Expected NetworkX graph, got {type(robot_graph)}")

    # Now count nodes of type "HINGE"
    hinge_count = sum(1 for _, attrs in robot_graph.nodes(data=True)
                      if attrs.get("type") == "HINGE")

    return hinge_count

def make_world():
    mj.set_mjcb_control(None)
    world = OlympicArena()
    return world

# === Helper: run_simulation (reuses common setup) ===
def run_simulation(genome, robot_graph):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(SPAWN_POS))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    controller = make_controller_from_genome(genome, num_joints, tracker=tracker)
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
    simple_runner(model, data, duration=DURATION)

    traj = np.array(tracker.history["xpos"][0])
    return traj, model, data, tracker, controller


# === Log saving ===
def save_log_csv(log, csv_path: Path):
    """Save log with flexible column handling."""
    if not log:
        return
    
    # Get all keys from first record
    headers = list(log[0].keys())
    
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for rec in log:
            w.writerow([rec[h] for h in headers])
    print(f"Log saved to {csv_path}")


# === Plotting ===
def plot_fitness(log, dest_dir: Path, pop: int, task: str, out_name: str = "plot_fitness.png"):
    
    gens = [rec["gen"] for rec in log]
    avg  = np.array([rec["avg"] for rec in log])
    median = np.array([rec.get("median", rec["avg"]) for rec in log])  # fallback to avg if no median
    q25  = np.array([rec.get("q25", rec["avg"] - rec.get("std", 0)) for rec in log])
    q75  = np.array([rec.get("q75", rec["avg"] + rec.get("std", 0)) for rec in log])
    maxv = np.array([rec["max"] for rec in log])
    minv = np.array([rec.get("min", rec["avg"] - 2*rec.get("std", 0)) for rec in log])
    
    plt.figure(figsize=(10, 6))
    
    # Plot median (more robust than mean for skewed data)
    plt.plot(gens, median, label="Median Fitness", color='blue', linewidth=2)
    
    # Fill between 25th and 75th percentiles (interquartile range)
    plt.fill_between(gens, q25, q75, alpha=0.3, color='blue', label="IQR (25th-75th percentile)")
    
    # Plot best fitness
    plt.plot(gens, maxv, label="Best Fitness", color='green', linewidth=2, linestyle='--')
    
    # Optionally plot average
    plt.plot(gens, avg, label="Average Fitness", color='orange', linewidth=1, linestyle=':')
    
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title(f"{task.upper()} Evolution\n(pop={pop}, duration={DURATION}s", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(dest_dir / out_name, dpi=150)
    plt.close()
    
    print(f"[SAVED] Fitness plot: {dest_dir / out_name}")


def plot_best_trajectory(
    genome, robot_graph, plots_dir: Path,
    out_name: str = "trajectory.png"
):
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph)
    start, end = traj[0], traj[-1]
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    dist_start = np.linalg.norm(start - np.array(TARGET_POS))
    dist_end = np.linalg.norm(end - np.array(TARGET_POS))
    print(f"[DEBUG] trajectory: Path={path_len:.3f}, StartDist={dist_start:.3f}, EndDist={dist_end:.3f}")
    print(f"Start: {start}, end: {end}")

    plt.figure(figsize=(8, 5))
    plt.plot(traj[:,0], traj[:,1], "b-", label="Trajectory (XZ)")
    plt.scatter(traj[0,0], traj[0,1], c="g", marker="o", label="Start")
    plt.scatter(traj[-1,0], traj[-1,1], c="r", marker="x", label="End")
    plt.scatter(TARGET_POS[0], TARGET_POS[1], c="k", marker="*", label="Target (XZ)")
    plt.xlabel("X"); plt.ylabel("Z")
    fitness = olympic_arena_fitness(traj)
    plt.title(f"Best Controller Trajectory (XZ projection) fit={fitness}")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()

# --- Helper: generic plotting function ---
def _plot_curve(x, y, label, color, xlabel, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, color=color, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] {save_path}")

# --- Helper: compute ALL fitness curves ---
def compute_all_fitness_curves(traj, step: int = 100):
    """Compute all fitness function values along the trajectory."""
    from examples.A3_code.evolve.fitness import (
        dist_to_target,
        forward_progress_fitness,
        distance_to_target_improved,
        olympic_arena_fitness,
        hybrid_fitness
    )
    
    # Store all fitness values
    all_fitness = {
        'dist_to_target': [],
        'forward_progress_fitness': [],
        'distance_to_target_improved': [],
        'olympic_arena_fitness': [],
        'hybrid_fitness': [],
    }
    
    timesteps = []
    
    for i in range(step, len(traj), step):
        partial_traj = traj[:i+1]
        timesteps.append(i)
        
        # Compute each fitness function
        all_fitness['dist_to_target'].append(
            dist_to_target(partial_traj)
        )
        all_fitness['forward_progress_fitness'].append(
            forward_progress_fitness(partial_traj)
        )
        all_fitness['distance_to_target_improved'].append(
            distance_to_target_improved(partial_traj)
        )
        all_fitness['olympic_arena_fitness'].append(
            olympic_arena_fitness(partial_traj)
        )
        all_fitness['hybrid_fitness'].append(
            hybrid_fitness(partial_traj)
        )
    
    # Add final point
    if len(traj) - 1 not in timesteps:
        timesteps.append(len(traj) - 1)
        all_fitness['dist_to_target'].append(dist_to_target(traj))
        all_fitness['forward_progress_fitness'].append(forward_progress_fitness(traj))
        all_fitness['distance_to_target_improved'].append(distance_to_target_improved(traj))
        all_fitness['olympic_arena_fitness'].append(olympic_arena_fitness(traj))
        all_fitness['hybrid_fitness'].append(hybrid_fitness(traj))
    
    return timesteps, all_fitness


# --- Main function ---
def plot_best_fitness_over_time(
    genome, robot_graph, plots_dir: Path,
    out_name_combined: str = "fitness_over_time_combined.png",
    step: int = 100
):
    """
    Plot fitness progression over time for ALL fitness functions.
    Creates individual plots for each function and one combined plot.
    """
    # Run simulation
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph)

    # Compute all fitness curves
    timesteps, all_fitness = compute_all_fitness_curves(traj, step)

    # Define colors and line styles for each fitness function
    fitness_styles = {
        'dist_to_target': {'color': 'red', 'linestyle': '-', 'label': 'Simple Distance (Original)'},
        'forward_progress_fitness': {'color': 'orange', 'linestyle': '--', 'label': 'Forward Progress'},
        'distance_to_target_improved': {'color': 'green', 'linestyle': '-', 'label': 'Dense Improved'},
        'olympic_arena_fitness': {'color': 'purple', 'linestyle': '--', 'label': 'Olympic Arena'},
        'hybrid_fitness': {'color': 'black', 'linestyle': '-', 'label': 'Hybrid (Combined)'}
    }

    # 1️⃣ Create individual plots for each fitness function
    # for func_name, style in fitness_styles.items():
    #     _plot_curve(
    #         timesteps, all_fitness[func_name],
    #         label=style['label'],
    #         color=style['color'],
    #         xlabel="Timestep", 
    #         ylabel="Fitness",
    #         title=f"{style['label']} Progression (step={step})",
    #         save_path=plots_dir / f"fitness_{func_name}.png"
    #     )

    # 2️⃣ Create combined plot with all fitness functions
    plt.figure(figsize=(12, 7))
    
    for func_name, style in fitness_styles.items():
        plt.plot(
            timesteps, 
            all_fitness[func_name], 
            label=style['label'],
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2,
            alpha=0.8
        )
    
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title(f"All Fitness Functions Comparison (step={step})", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(plots_dir / out_name_combined, dpi=150)
    plt.close()

    print(f"[SAVED] Combined fitness plot: {plots_dir / out_name_combined}")

    # 3️⃣ Create a subplot grid for better comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (func_name, style) in enumerate(fitness_styles.items()):
        axes[idx].plot(
            timesteps, 
            all_fitness[func_name],
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=2
        )
        axes[idx].set_title(style['label'], fontsize=11)
        axes[idx].set_xlabel("Timestep", fontsize=10)
        axes[idx].set_ylabel("Fitness", fontsize=10)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "fitness_comparison_grid.png", dpi=150)
    plt.close()
    
    print(f"[SAVED] Fitness comparison grid: {plots_dir / 'fitness_comparison_grid.png'}")

    # 4️⃣ Print final fitness values for comparison
    print("\n" + "="*60)
    print("FINAL FITNESS VALUES COMPARISON:")
    print("="*60)
    for func_name, style in fitness_styles.items():
        final_fitness = all_fitness[func_name][-1]
        print(f"{style['label']:30s}: {final_fitness:10.3f}")
    print("="*60 + "\n")

# === Save/Load Robot ===
def save_robot(dest_dir: Path, robot_graph, ctrl_genes, input_size, num_joints):
    robot_path = dest_dir / "robot.json"

    graph_data = nx.node_link_data(robot_graph, edges="links")

    if NN_DEPTH == 1:
        w1, b1, w2, b2 = decode_genome(ctrl_genes, input_size, HIDDEN_SIZE, num_joints, depth=1)
        controller_data = {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2.tolist()}
    else:
        w1, b1, w2, b2, w3, b3 = decode_genome(
            ctrl_genes, input_size, HIDDEN_SIZE, num_joints, depth=2, hidden_size2=HIDDEN_SIZE2
        )
        controller_data = {
            "w1": w1.tolist(), "b1": b1.tolist(),
            "w2": w2.tolist(), "b2": b2.tolist(),
            "w3": w3.tolist(), "b3": b3.tolist(),
        }

    data = {"graph": graph_data, "controller": controller_data}
    with open(robot_path, "w") as f:
        json.dump(data, f)
    print(f"Saved robot to {robot_path}")


def load_robot(dir_path: Path):
    robot_path = dir_path / "robot.json"
    with open(robot_path, "r") as f:
        data = json.load(f)
    robot_graph = nx.node_link_graph(data["graph"], edges="links")
    controller_weights = data["controller"]

    # num_joints = count_num_joints(robot_graph)
    # input_size = infer_input_size(num_joints, STATE_FEATURES)

    # controller = build_controller(controller_data, input_size, HIDDEN_SIZE, num_joints,
    #                               depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2)
    return robot_graph, controller_weights
