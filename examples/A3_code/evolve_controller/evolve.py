from __future__ import annotations
from pathlib import Path
import random, csv, string, json
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from deap import base, creator, tools, algorithms

from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from fitness import distance_to_target   # 3D fitness

from .config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH,
    NUM_EVAL_STEPS, NUM_POP, NUM_GENS,
    TARGET_POS, SPAWN_POS, infer_input_size, get_state_vector,
    SAVE_MODELS, SAVE_PLOTS, SAVE_LOGS
)
from .nn import build_controller, genome_length, decode_genome

# === Global Debug Flag ===
DEBUG_PROGRESS = True   # set to False for quiet runs

# === Helpers ===
def make_world():
    return SimpleFlatWorld()

def evaluate_genome(
    genome,
    *,
    task: str,
    sim_steps: int,
    spawn_pos: tuple[float,float,float],
    robot_graph,
    target_pos: tuple[float,float,float] = TARGET_POS,
):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    num_joints = model.nu
    input_size = infer_input_size(num_joints, STATE_FEATURES)

    controller = build_controller(
        genome, input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    pos_history: list[np.ndarray] = []
    for _ in range(sim_steps):
        state = get_state_vector(data, num_joints, STATE_FEATURES, target_pos=target_pos)
        action = controller(state)
        data.ctrl[:] = np.clip(action, -np.pi/2, np.pi/2)
        mujoco.mj_step(model, data)
        pos_history.append(data.qpos[:3].copy())   # track (x, y, z)

    if task == "nav":
        fitness = distance_to_target(pos_history, target_pos=target_pos)
    else:
        raise ValueError(f"Unknown task {task!r}")
    return (fitness,)  # DEAP expects tuple

# === Debug print helper ===
def debug_best_controller(genome, steps, robot_graph, spawn_pos, target_pos, label=""):
    """Run best genome and print distances."""
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    num_joints = model.nu
    input_size = infer_input_size(num_joints, STATE_FEATURES)
    controller = build_controller(
        genome, input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    traj = []
    for _ in range(steps):
        state = get_state_vector(data, num_joints, STATE_FEATURES, target_pos=target_pos)
        action = controller(state)
        data.ctrl[:] = np.clip(action, -np.pi/2, np.pi/2)
        mujoco.mj_step(model, data)
        traj.append(data.qpos[:3].copy())
    traj = np.array(traj)

    start, end = traj[0], traj[-1]
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    dist_start = np.linalg.norm(start - np.array(target_pos))
    dist_end = np.linalg.norm(end - np.array(target_pos))

    print(f"[DEBUG] {label} Path={path_len:.3f}, Start→Target={dist_start:.3f}, End→Target={dist_end:.3f}")

# === Plotting ===
def plot_fitness(log, save_path: Path, pop: int, steps: int, task: str):
    gens = log.select("gen")
    avg = np.array(log.select("avg"))
    std = np.array(log.select("std"))
    maxv = np.array(log.select("max"))
    plt.figure(figsize=(10, 6))
    plt.plot(gens, avg, label="Average Fitness")
    plt.fill_between(gens, avg - std, avg + std, alpha=0.3, label="Std Dev")
    plt.plot(gens, maxv, label="Best Fitness")
    plt.xlabel("Generation"); plt.ylabel("Fitness")
    plt.title(f"{task.upper()} evolution (pop={pop}, steps={steps})")
    plt.legend(); plt.grid(True)
    plt.savefig(save_path); plt.close()
    if DEBUG_PROGRESS:
        print(f"Fitness plot saved to {save_path}")

def plot_best_trajectory(
    best_controller, robot_graph, steps, spawn_pos, plots_dir: Path,
    target_pos: tuple[float,float,float] = (5.0,0.0,0.5),
    out_name: str = "trajectory.png"
):
    """Run the best controller and plot its 2D XY trajectory."""
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    num_joints = model.nu
    _ = infer_input_size(num_joints, STATE_FEATURES)

    traj = []
    for _ in range(steps):
        state = get_state_vector(
            data, num_joints, STATE_FEATURES, target_pos=target_pos
        ).astype(np.float32)
        action = best_controller(state)
        data.ctrl[:] = np.clip(action, -np.pi/2, np.pi/2)
        mujoco.mj_step(model, data)
        traj.append(data.qpos[:3].copy())

    traj = np.array(traj)

    plt.figure(figsize=(8, 5))
    plt.plot(traj[:,0], traj[:,1], "b-", label="Trajectory")
    plt.scatter(traj[0,0], traj[0,1], c="g", marker="o", label="Start")
    plt.scatter(traj[-1,0], traj[-1,1], c="r", marker="x", label="End")
    plt.scatter(target_pos[0], target_pos[1], c="k", marker="*", label="Target (XY)")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Best Controller Trajectory (XY projection)")
    plt.legend(); plt.grid(True)
    out = plots_dir / out_name
    plt.savefig(out); plt.close()
    if DEBUG_PROGRESS:
        print(f"Trajectory plot saved to {out}")

# === EA driver ===
def run_controller_evolution(
    task: str,
    robot_graph,
    steps: int = NUM_EVAL_STEPS,
    pop_size: int = NUM_POP,
    gens: int = NUM_GENS,
    seed: int | None = None,
    spawn_pos: tuple[float, float, float] = SPAWN_POS,
    target_pos: tuple[float, float, float] = TARGET_POS,
    dest_dir: Path | None = None,
):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)


    dest_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = dest_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # sizing
    tmp_world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    tmp_world.spawn(core.spec, spawn_position=list(spawn_pos))
    tmp_model = tmp_world.spec.compile()
    num_joints = tmp_model.nu
    input_size = infer_input_size(num_joints, STATE_FEATURES)
    g_len = genome_length(
        input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    # === DEBUG PRINTS ===
    print("---- Dimension Check ----")
    print(f"Number of joints (actuators/outputs): {num_joints}")
    print(f"Chosen state features: {STATE_FEATURES}")
    print(f"Input size to NN (state vector length): {input_size}")
    if NN_DEPTH == 1:
        print(f"NN architecture: Input ({input_size}) -> {HIDDEN_SIZE} -> {num_joints}")
    elif NN_DEPTH == 2:
        print(f"NN architecture: Input ({input_size}) -> {HIDDEN_SIZE} -> {HIDDEN_SIZE2} -> {num_joints}")
    print(f"Total genome length (flat vector size): {g_len}")
    print("-------------------------")

    mut_sigma, mut_indpb = 0.2, 0.10

    # DEAP setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=g_len)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register(
        "evaluate", evaluate_genome, task=task, sim_steps=steps,
        spawn_pos=spawn_pos, robot_graph=robot_graph, target_pos=target_pos
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("std", np.std); stats.register("max", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=gens,
        stats=stats, halloffame=hof, verbose=DEBUG_PROGRESS
    )

    if DEBUG_PROGRESS:
        best_genome = np.array(hof[0])
        debug_best_controller(best_genome, steps, robot_graph, spawn_pos, target_pos, label="Best Overall")

    model_path = None

    if SAVE_LOGS:
        csv_path = plots_dir / f"log.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["gen","avg","std","max"])
            for rec in log: w.writerow([rec["gen"], rec["avg"], rec["std"], rec["max"]])
        if DEBUG_PROGRESS:
            print(f"Log saved to {csv_path}")

    if SAVE_PLOTS:
        plot_fitness(log, plots_dir / f"plot_fitness.png", pop_size, steps, task)

    if SAVE_MODELS:
        # Decode for saving in a simple JSON structure
        if NN_DEPTH == 1:
            w1, b1, w2, b2 = decode_genome(np.array(hof[0]), input_size, HIDDEN_SIZE, num_joints, depth=1)
            weights = {"w1": w1.tolist(), "b1": b1.tolist(), "w2": w2.tolist(), "b2": b2.tolist()}
        else:
            w1, b1, w2, b2, w3, b3 = decode_genome(
                np.array(hof[0]), input_size, HIDDEN_SIZE, num_joints, depth=2, hidden_size2=HIDDEN_SIZE2
            )
            weights = {
                "w1": w1.tolist(), "b1": b1.tolist(),
                "w2": w2.tolist(), "b2": b2.tolist(),
                "w3": w3.tolist(), "b3": b3.tolist(),
            }

        controller_path = dest_dir / f"controller.json"
        with open(controller_path, "w") as f:
            json.dump(weights, f)
        if DEBUG_PROGRESS:
            print(f"Controller weights saved to {controller_path}")

    best_genome = np.array(hof[0])
    best_controller = build_controller(
        best_genome, input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    if SAVE_PLOTS:
        plot_best_trajectory(
            best_controller, robot_graph, steps, spawn_pos, plots_dir,
            target_pos=target_pos,
            out_name=f"trajectory.png"
        )

    return best_controller
