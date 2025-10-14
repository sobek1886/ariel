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
from examples.A3_code.evolve.config import SPAWN_POS, TARGET_POS, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, STATE_FEATURES
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
def run_simulation(ctrl_genes, robot_graph, duration=None):
    # Initialize variables for cleanup
    world = None
    core = None
    model = None
    data = None
    tracker = None
    controller = None
    
    try:
        world = make_world()
        core = construct_mjspec_from_graph(robot_graph)
        world.spawn(core.spec, list(SPAWN_POS))
        model = world.spec.compile()
        data = mj.MjData(model)
        mj.mj_resetData(model, data)
        
        num_joints = model.nu
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        controller = make_controller_from_genome(ctrl_genes, num_joints, tracker=tracker)
        
        if controller.tracker is not None:
            controller.tracker.setup(world.spec, data)
        
        mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
        
        # ✅ Check for instability before running
        mj.mj_forward(model, data)
        if not np.all(np.isfinite(data.qacc)):
            raise RuntimeError("Simulation unstable before starting")
        
        simple_runner(model, data, duration=duration)
        
        # Convert to array
        traj = np.array(tracker.history["xpos"][0], dtype=np.float32)
        
        # ✅ Validate trajectory
        if not np.all(np.isfinite(traj)):
            raise RuntimeError("NaN/Inf in trajectory")
            
    except Exception as e:
        print(f"[!] Simulation failed: {e}")
        # Return minimal valid trajectory (single point at spawn)
        traj = np.array([list(SPAWN_POS)], dtype=np.float32)
    
    finally:
        # CRITICAL: Aggressive cleanup
        mj.set_mjcb_control(None)
        
        if tracker is not None:
            tracker.history.clear()
        
        # Delete objects in reverse order of creation
        del controller
        del tracker
        del data
        del model
        del core
        del world
    
    return traj


# === Log saving ===
def save_log_csv(log, csv_path: Path):
    """Save or append to fitness log with automatic header handling."""
    if not log:
        return

    headers = list(log[0].keys())

    # If the file exists, append rows without rewriting header
    if csv_path.exists():
        with open(csv_path, "a", newline="") as f:
            for rec in log:
                f.write(",".join(str(rec[h]) for h in headers) + "\n")
        print(f"Appended {len(log)} entries to {csv_path}")
    else:
        # Otherwise, create new file with headers
        with open(csv_path, "w", newline="") as f:
            f.write(",".join(headers) + "\n")
            for rec in log:
                f.write(",".join(str(rec[h]) for h in headers) + "\n")
        print(f"Created new log file at {csv_path}")


# === Plotting ===
def plot_fitness(log, dest_dir: Path, pop: int, out_name: str = "plot_fitness.png", duration=None):
    
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
    # plt.plot(gens, avg, label="Average Fitness", color='orange', linewidth=1, linestyle=':')
    
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title(f"Fitness Evolution\n(pop={pop}, duration={duration}s)", fontsize=12)
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
    out_name: str = "trajectory.png",
    duration=None
):
    traj = run_simulation(genome, robot_graph, duration)
    print(f"    Len of trajectory {len(traj)}")
    print(f"    before last pos: {traj[-1]}")
    
    # SPEED UP: Subsample trajectory (plot every 10th point)
    traj_subsampled = traj[::10]  # Every 10th point

    start, end = traj[0], traj[-1]
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    dist_start = np.linalg.norm(start - np.array(TARGET_POS))
    dist_end = np.linalg.norm(end - np.array(TARGET_POS))
    print(f"[DEBUG] trajectory: Path={path_len:.3f}, StartDist={dist_start:.3f}, EndDist={dist_end:.3f}")
    print(f"Plot_best_trajectory Start: {start}, end: {end}")

    plt.figure(figsize=(8, 5))
    plt.plot(traj_subsampled[:,0], traj_subsampled[:,1], "b-", label="Trajectory (XY)")
    plt.scatter(traj[0,0], traj[0,1], c="g", marker="o", label="Start")
    plt.scatter(traj[-1,0], traj[-1,1], c="r", marker="x", label="End")
    plt.scatter(TARGET_POS[0], TARGET_POS[1], c="k", marker="*", label="Target (XY)")
    plt.xlabel("X"); plt.ylabel("Z")
    fitness = olympic_arena_fitness(traj)
    plt.title(f"Best Controller Trajectory (XY projection) fit={fitness}, dur={duration}s")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()

    # Clean up
    del traj
    del traj_subsampled

# --- Main function ---
def plot_best_fitness_over_time(
    genome, robot_graph, plots_dir: Path,
    out_name: str = "fitness_over_time.png",
    step: int = 250, duration=None
):
    """
    Plot Olympic Arena fitness progression over time.
    """
    # Run simulation
    traj = run_simulation(genome, robot_graph, duration=duration)
    
    # Compute Olympic Arena fitness at intervals
    timesteps = []
    fitness_values = []
    
    for i in range(step, len(traj), step):
        partial_traj = traj[:i+1]
        timesteps.append(i)
        fitness_values.append(olympic_arena_fitness(partial_traj))
    
    # Add final point
    if len(traj) - 1 not in timesteps:
        timesteps.append(len(traj) - 1)
        fitness_values.append(olympic_arena_fitness(traj))
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, fitness_values, color='purple', linewidth=2, label='Olympic Arena Fitness')
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Fitness", fontsize=12)
    plt.title(f"Olympic Arena Fitness Progression (step={step}, dur={duration}s)", fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(plots_dir / out_name, dpi=150)
    plt.close()
    
    print(f"[SAVED] Olympic Arena fitness plot: {plots_dir / out_name}")
    print(f"Final Olympic Arena Fitness: {fitness_values[-1]:.3f}")
    
    # Clean up
    del traj


# === Save/Load Robot ===
def save_robot(dest_dir: Path, robot_graph, ctrl_genes, input_size, num_joints, out="robot.json"):
    robot_path = dest_dir / out

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
