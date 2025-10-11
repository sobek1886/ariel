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
from examples.A3_code.evolve.nn import make_controller_from_genome, decode_genome, build_controller
from examples.A3_code.evolve.config import DURATION, TARGET_POS, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH


def make_world():
    mj.set_mjcb_control(None)
    world = OlympicArena()
    return world

# === Helper: run_simulation (reuses common setup) ===
def run_simulation(genome, robot_graph, spawn_pos):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
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
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gen", "avg", "std", "max"])
        for rec in log:
            w.writerow([rec["gen"], rec["avg"], rec["std"], rec["max"]])
    print(f"Log saved to {csv_path}")


# === Plotting ===
def plot_fitness(log, dest_dir: Path, pop: int, task: str, out_name: str = "plot_fitness.png"):
    gens = log.select("gen")
    avg = np.array(log.select("avg"))
    std = np.array(log.select("std"))
    maxv = np.array(log.select("max"))
    plt.figure(figsize=(10, 6))
    plt.plot(gens, avg, label="Average Fitness")
    plt.fill_between(gens, avg - std, avg + std, alpha=0.3, label="Std Dev")
    plt.plot(gens, maxv, label="Best Fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness")
    plt.title(f"{task.upper()} evolution (pop={pop}, duration={DURATION})")
    plt.legend(); plt.grid(True)
    plt.savefig(dest_dir / out_name); plt.close()


def plot_best_trajectory(
    genome, robot_graph, spawn_pos, plots_dir: Path,
    target_pos: tuple[float,float,float] = TARGET_POS,
    out_name: str = "trajectory.png"
):
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph, spawn_pos)
    start, end = traj[0], traj[-1]
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    dist_start = np.linalg.norm(start - np.array(target_pos))
    dist_end = np.linalg.norm(end - np.array(target_pos))
    print(f"[DEBUG] trajectory: Path={path_len:.3f}, StartDist={dist_start:.3f}, EndDist={dist_end:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(traj[:,0], traj[:,1], "b-", label="Trajectory")
    plt.scatter(traj[0,0], traj[0,1], c="g", marker="o", label="Start")
    plt.scatter(traj[-1,0], traj[-1,1], c="r", marker="x", label="End")
    plt.scatter(target_pos[0], target_pos[1], c="k", marker="*", label="Target (XY)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Best Controller Trajectory (XY projection)")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()


def plot_best_fitness_over_time(
    genome, robot_graph, spawn_pos, plots_dir: Path,
    target_pos: tuple[float, float, float] = TARGET_POS,
    out_name: str = "fitness_over_time.png",
    step: int = 100
):
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph, spawn_pos)
    fitness_dense, fitness_finaldist, total_dense = [], [], 0.0

    for i in range(1, len(traj)):
        prev, curr = traj[i-1], traj[i]
        prev_dist = np.linalg.norm(prev - np.array(target_pos))
        curr_dist = np.linalg.norm(curr - np.array(target_pos))
        total_dense += (prev_dist - curr_dist)
        bonus = 1.0 / (1.0 + curr_dist)
        dense_val = total_dense + bonus
        final_val = -curr_dist
        if i % step == 0 or i == len(traj) - 1:
            fitness_dense.append(dense_val)
            fitness_finaldist.append(final_val)

    timesteps = [i for i in range(step, len(fitness_dense)*step+1, step)]
    if len(timesteps) > len(fitness_dense):
        timesteps = timesteps[:len(fitness_dense)]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, fitness_dense, label="Dense fitness (distance_to_target)", color="blue")
    plt.plot(timesteps, fitness_finaldist, label="Final distance fitness", color="red", linestyle="--")
    plt.xlabel("Timestep"); plt.ylabel("Fitness")
    plt.title(f"Best Controller Fitness Progression (update={step})")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()
    print(f"Fitness-over-time plot saved to {plots_dir / out_name}")


def plot_best_fitness_over_time_OG(
    genome, robot_graph, spawn_pos, plots_dir: Path,
    target_pos: tuple[float,float,float] = TARGET_POS,
    out_name: str = "fitness_over_time.png"
):
    traj, model, data, tracker, controller = run_simulation(genome, robot_graph, spawn_pos)
    fitness_values = [distance_to_target(traj[:i+1]) for i in range(len(traj))]
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_values, label="Fitness over time")
    plt.xlabel("Timestep"); plt.ylabel("Fitness")
    plt.title("Best Controller Fitness Progression")
    plt.legend(); plt.grid(True)
    plt.savefig(plots_dir / out_name); plt.close()

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


def load_robot(path: Path, input_size, num_joints):
    with open(path, "r") as f:
        data = json.load(f)
    robot_graph = nx.node_link_graph(data["graph"], edges="links")
    controller_data = data["controller"]
    controller = build_controller(controller_data, input_size, HIDDEN_SIZE, num_joints,
                                  depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2)
    return robot_graph, controller
