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
from examples.A3_code.evolve.fitness import distance_to_target
from examples.A3_code.evolve.nn import make_controller_from_genome, decode_genome, build_controller, infer_input_size
from examples.A3_code.evolve.config import DURATION, SPAWN_POS, TARGET_POS, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, STATE_FEATURES

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

# # === Helper: run_simulation (reuses common setup) ===
# def run_simulation(genome, robot_graph):
#     world = make_world()
#     core = construct_mjspec_from_graph(robot_graph)
#     world.spawn(core.spec, list(SPAWN_POS))
#     model = world.spec.compile()
#     data = mj.MjData(model)
#     mj.mj_resetData(model, data)

#     num_joints = model.nu
#     tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

#     controller = make_controller_from_genome(genome, num_joints, tracker=tracker)
#     if controller.tracker is not None:
#         controller.tracker.setup(world.spec, data)

#     mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
#     simple_runner(model, data, duration=DURATION)

#     traj = np.array(tracker.history["xpos"][0])
#     return traj, model, data, tracker, controller
def run_simulation(genome, robot_graph):
    """Build world, run simulation, return trajectory and model data."""
    import random
    from examples.A3_code.evolve.config import (
        USE_CPG, CPG_HYBRID, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, STATE_FEATURES
    )
    from examples.A3_code.evolve.nn import (
        make_controller_from_genome, 
        make_cpg_controller_from_genome,
        infer_input_size,
        genome_length,
        cpg_genome_length,
        hybrid_cpg_genome_length
    )
    
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec)  # FIXED - no position parameter
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    input_size = infer_input_size(num_joints, STATE_FEATURES)
    
    # Calculate required controller length based on type
    if USE_CPG:
        if CPG_HYBRID:
            required_len = hybrid_cpg_genome_length(num_joints, input_size)
        else:
            required_len = cpg_genome_length(num_joints)
    else:
        required_len = genome_length(
            input_size, HIDDEN_SIZE, num_joints,
            depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
        )
    
    # Resize genome if needed (same logic as evaluate_robot)
    ctrl_genes = list(genome)
    if len(ctrl_genes) != required_len:
        print(f"⚠️ Resizing controller for plotting: {len(ctrl_genes)} → {required_len}")
        if len(ctrl_genes) < required_len:
            padding = [random.uniform(-0.1, 0.1) for _ in range(required_len - len(ctrl_genes))]
            ctrl_genes = ctrl_genes + padding
        else:
            ctrl_genes = ctrl_genes[:required_len]
    
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    
    # Choose controller type based on config
    if USE_CPG:
        controller = make_cpg_controller_from_genome(
            ctrl_genes, num_joints, tracker=tracker, use_hybrid=CPG_HYBRID
        )
    else:
        controller = make_controller_from_genome(
            ctrl_genes, num_joints, tracker=tracker
        )
    
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)
    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
    simple_runner(model, data, duration=DURATION)
    traj = np.array(tracker.history["xpos"][0])
    
    return traj, model, data, tracker, controller

# === Log saving ===
def save_log_csv(log, csv_path: Path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "avg", "std", "max"])
        for rec in log:
            w.writerow([rec["gen"], rec["avg"], rec["std"], rec["max"]])
    print(f"Log saved to {csv_path}")


# === Plotting ===
def plot_fitness(log, dest_dir: Path, pop: int, task: str, out_name: str = "plot_fitness.png"):
    gens = [rec["gen"] for rec in log]
    avg  = np.array([rec["avg"] for rec in log])
    std  = np.array([rec["std"] for rec in log])
    maxv = np.array([rec["max"] for rec in log])
    plt.figure(figsize=(10, 6))
    plt.plot(gens, avg, label="Average Fitness")
    plt.fill_between(gens, avg - std, avg + std, alpha=0.3, label="Std Dev")
    plt.plot(gens, maxv, label="Best Fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness")
    plt.title(f"{task.upper()} evolution (pop={pop}, duration={DURATION})")
    plt.legend(); plt.grid(True)
    plt.savefig(dest_dir / out_name); plt.close()


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
    plt.title("Best Controller Trajectory (XZ projection)")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from examples.A3_code.evolve.config import TARGET_POS

# --- Helper: compute fitness curves ---
def compute_fitness_curves(traj, step: int = 100):
    """Compute dense and final-distance fitness values along the trajectory."""
    fitness_dense, fitness_finaldist, total_dense = [], [], 0.0

    for i in range(1, len(traj)):
        prev, curr = traj[i-1], traj[i]
        # use x,z plane if your order is [x,z,y]
        prev_dist = np.linalg.norm(prev[[0, 1]] - np.array(TARGET_POS)[[0, 1]])
        curr_dist = np.linalg.norm(curr[[0, 1]] - np.array(TARGET_POS)[[0, 1]])

        total_dense += (prev_dist - curr_dist)
        bonus = 1.0 / (1.0 + curr_dist)
        dense_val = total_dense + bonus
        final_val = -curr_dist

        if i % step == 0 or i == len(traj) - 1:
            fitness_dense.append(dense_val)
            fitness_finaldist.append(final_val)

    timesteps = [i for i in range(step, len(fitness_dense) * step + 1, step)]
    if len(timesteps) > len(fitness_dense):
        timesteps = timesteps[:len(fitness_dense)]

    return timesteps, fitness_dense, fitness_finaldist


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


# --- Main function ---
def plot_best_fitness_over_time(
    genome, robot_graph, plots_dir: Path,
    out_name_combined: str = "fitness_over_time_combined.png",
    step: int = 100
):
    # Run simulation
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph)

    # Compute curves
    timesteps, fitness_dense, fitness_finaldist = compute_fitness_curves(traj, step)

    # 1️⃣ Plot dense fitness alone
    _plot_curve(
        timesteps, fitness_dense,
        label="Dense fitness (distance_to_target)",
        color="blue",
        xlabel="Timestep", ylabel="Fitness",
        title=f"Dense Fitness Progression (step={step})",
        save_path=plots_dir / "fitness_dense.png"
    )

    # 2️⃣ Plot final distance fitness alone
    _plot_curve(
        timesteps, fitness_finaldist,
        label="Final distance fitness",
        color="red",
        xlabel="Timestep", ylabel="Fitness",
        title=f"Final Distance Fitness Progression (step={step})",
        save_path=plots_dir / "fitness_finaldist.png"
    )

    # 3️⃣ Combined plot
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, fitness_dense, label="Dense fitness (distance_to_target)", color="blue")
    plt.plot(timesteps, fitness_finaldist, label="Final distance fitness", color="red", linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Fitness")
    plt.title(f"Combined Fitness Progression (step={step})")
    plt.legend()
    plt.grid(True)
    plt.savefig(plots_dir / out_name_combined)
    plt.close()

    print(f"[SAVED] Combined fitness plot: {plots_dir / out_name_combined}")


def plot_best_fitness_over_time_OG(
    genome, robot_graph, plots_dir: Path,
    out_name: str = "fitness_over_time.png"
):
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph)
    fitness_values = [distance_to_target(traj[:i+1]) for i in range(len(traj))]
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_values, label="Fitness over time")
    plt.xlabel("Timestep"); plt.ylabel("Fitness")
    plt.title("Best Controller Fitness Progression")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()

# === Save/Load Robot ===
# def save_robot(dest_dir: Path, robot_graph, ctrl_genes, input_size, num_joints):
#     robot_path = dest_dir / "robot.json"

#     graph_data = nx.node_link_data(robot_graph, edges="links")

#     if NN_DEPTH == 1:
#         w1, b1, w2, b2 = decode_genome(ctrl_genes, input_size, HIDDEN_SIZE, num_joints, depth=1)
#         controller_data = {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2.tolist()}
#     else:
#         w1, b1, w2, b2, w3, b3 = decode_genome(
#             ctrl_genes, input_size, HIDDEN_SIZE, num_joints, depth=2, hidden_size2=HIDDEN_SIZE2
#         )
#         controller_data = {
#             "w1": w1.tolist(), "b1": b1.tolist(),
#             "w2": w2.tolist(), "b2": b2.tolist(),
#             "w3": w3.tolist(), "b3": b3.tolist(),
#         }

#     data = {"graph": graph_data, "controller": controller_data}
#     with open(robot_path, "w") as f:
#         json.dump(data, f)
#     print(f"Saved robot to {robot_path}")
def save_robot(dest_dir: Path, robot_graph, ctrl_genes, input_size, num_joints):
    """Save robot body and controller to a single robot.json file."""
    from examples.A3_code.evolve.config import USE_CPG, CPG_HYBRID, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    robot_path = dest_dir / "robot.json"
    
    # Save body graph
    graph_data = nx.node_link_data(robot_graph, edges="links")
    
    # Save controller based on type
    if USE_CPG:
        # For CPG, save the raw genome
        controller_data = {
            "type": "CPG_HYBRID" if CPG_HYBRID else "CPG",
            "genome": ctrl_genes,
            "num_joints": num_joints,
            "input_size": input_size
        }
        print(f"Saved CPG controller with {len(ctrl_genes)} parameters")
    else:
        # For MLP, decode and save weights
        if NN_DEPTH == 1:
            w1, b1, w2, b2 = decode_genome(
                ctrl_genes, input_size, HIDDEN_SIZE, num_joints, depth=1
            )
            controller_data = {
                "type": "MLP_1layer",
                "w1": w1.tolist(), 
                "b1": b1.tolist(), 
                "w2": w2.tolist(), 
                "b2": b2.tolist(),
                "num_joints": num_joints,
                "input_size": input_size
            }
        else:
            w1, b1, w2, b2, w3, b3 = decode_genome(
                ctrl_genes, input_size, HIDDEN_SIZE, num_joints, 
                depth=2, hidden_size2=HIDDEN_SIZE2
            )
            controller_data = {
                "type": "MLP_2layer",
                "w1": w1.tolist(),
                "b1": b1.tolist(),
                "w2": w2.tolist(),
                "b2": b2.tolist(),
                "w3": w3.tolist(),
                "b3": b3.tolist(),
                "num_joints": num_joints,
                "input_size": input_size
            }
        print(f"Saved MLP controller weights")
    
    # Combine everything into one file
    data = {
        "graph": graph_data, 
        "controller": controller_data
    }
    
    with open(robot_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved robot to {robot_path}")




def load_robot(dir_path: Path):
    
    """Load robot from robot.json file - works with both MLP and CPG controllers."""
    robot_path = dir_path / "robot.json"

    with open(robot_path, "r") as f:
        data = json.load(f)
    
    robot_graph = nx.node_link_graph(data["graph"], edges="links")
    controller_data = data["controller"]
    
    # Check controller type
    controller_type = controller_data.get("type", "MLP")
    
    print(f"Loaded robot with {controller_type} controller")
    
    return robot_graph, controller_data