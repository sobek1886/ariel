from __future__ import annotations
from pathlib import Path
import random, csv, string, json
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
import copy
import multiprocessing, os, time
from deap import base, creator, tools, algorithms

from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from fitness import distance_to_target, fitness_function   # 3D fitness

from .config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH,
    NUM_POP, NUM_GENS,
    TARGET_POS, SPAWN_POS, infer_input_size,
    SAVE_MODELS, SAVE_PLOTS, SAVE_LOGS, DURATION
)
from .nn import build_controller, genome_length, decode_genome, make_controller_from_genome

DEBUG_PROGRESS = True   # set to False for quiet runs
is_timer = False
start_time = 0.0

def make_world():
    mj.set_mjcb_control(None)
    return OlympicArena()

def evaluate_genome(
    genome,
    *,
    task: str,
    spawn_pos: tuple[float,float,float],
    robot_graph,
    target_pos: tuple[float,float,float] = TARGET_POS,
):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    controller = make_controller_from_genome(
        genome, num_joints,
        tracker=tracker,
    )

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    simple_runner(model, data, duration=DURATION)

    # ✅ use tracker history instead of manual traj
    traj = np.array(tracker.history["xpos"][0])

    if task == "nav":
        fitness = distance_to_target(traj, target_pos=target_pos)
    else:
        raise ValueError(f"Unknown task {task!r}")
    return (fitness,)


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
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    controller = make_controller_from_genome(
        genome, num_joints,
        tracker=tracker,
    )

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    simple_runner(model, data, duration=DURATION)

    # ✅ use tracker history
    traj = np.array(tracker.history["xpos"][0])

    start, end = traj[0], traj[-1]
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    dist_start = np.linalg.norm(start - np.array(target_pos))
    dist_end = np.linalg.norm(end - np.array(target_pos))

    print(
        f"[DEBUG] trajectory: "
        f"Path length traveled = {path_len:.3f}, "
        f"Distance from start to target = {dist_start:.3f}, "
        f"Final distance to target = {dist_end:.3f}"
    )

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

def plot_best_fitness_over_time(
    genome, robot_graph, spawn_pos, plots_dir: Path,
    target_pos: tuple[float, float, float] = TARGET_POS,
    out_name: str = "fitness_over_time.png",
    step: int = 100
):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    controller = make_controller_from_genome(
        genome, num_joints,
        tracker=tracker,
    )

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
    simple_runner(model, data, duration=DURATION)

    traj = np.array(tracker.history["xpos"][0])

    fitness_dense = []
    fitness_finaldist = []
    total_dense = 0.0

    for i in range(1, len(traj)):
        prev, curr = traj[i-1], traj[i]
        prev_dist = np.linalg.norm(prev - np.array(target_pos))
        curr_dist = np.linalg.norm(curr - np.array(target_pos))

        total_dense += (prev_dist - curr_dist)
        # ✅ recompute bonus from *current last point* just like original
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
    plt.xlabel("Timestep")
    plt.ylabel("Fitness")
    plt.title(f"Best Controller Fitness Progression (step={step})")
    plt.legend(); plt.grid(True)

    out = plots_dir / out_name
    plt.savefig(out); plt.close()
    if DEBUG_PROGRESS:
        print(f"Fitness-over-time plot saved to {out}")

def plot_best_fitness_over_time_OG(
    genome, robot_graph, spawn_pos, plots_dir: Path,
    target_pos: tuple[float,float,float] = TARGET_POS,
    out_name: str = "fitness_over_time.png"
):
    world = make_world()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, spawn_position=list(spawn_pos))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    controller = make_controller_from_genome(
        genome, num_joints,
        tracker=tracker,
    )

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    # Run sim
    simple_runner(model, data, duration=DURATION)

    # Use tracker history (positions over time)
    traj = np.array(tracker.history["xpos"][0])

    # Compute fitness at each timestep
    fitness_values = [distance_to_target(traj[:i+1], target_pos=target_pos) for i in range(len(traj))]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_values, label="Fitness over time")
    plt.xlabel("Timestep")
    plt.ylabel("Fitness")
    plt.title("Best Controller Fitness Progression")
    plt.legend(); plt.grid(True)
    out = plots_dir / out_name
    plt.savefig(out); plt.close()

    if DEBUG_PROGRESS:
        print(f"Fitness over time plot saved to {out}")



def tap_timer(timer_name: str = "Timer"):
    global is_timer, start_time
    if not is_timer:
        start_time = time.time()
        is_timer = True
    else:
        total_time = time.time() - start_time
        print(f"--- {timer_name} timer took: {total_time:.2f} sec ---")
        start_time = 0.0
        is_timer = False

# === EA driver ===
def run_controller_evolution(
    task: str,
    robot_graph,
    pop_size: int = NUM_POP,
    gens: int = NUM_GENS,
    seed: int | None = None,
    spawn_pos: tuple[float, float, float] = SPAWN_POS,
    target_pos: tuple[float, float, float] = TARGET_POS,
    dest_dir: Path | None = None,
):
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    print(f"------ Evolution parameters ------")
    print(f"Population size: {pop_size}")
    print(f"Number of generations: {gens}")
    print(f"Duration per evaluation: {DURATION} sec")
    print(f"Seed: {seed}")
    print(f"Spawn position: {spawn_pos}")
    print(f"Target position: {target_pos}")
    print(f"Destination directory: {dest_dir}")
    print(f"----------------------------------")

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
        "evaluate", evaluate_genome, task=task,
        spawn_pos=spawn_pos, robot_graph=robot_graph, target_pos=target_pos
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=mut_sigma, indpb=mut_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("std", np.std); stats.register("max", np.max)

    # === Parallelism setup ===
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(n_cpus)
    toolbox.register("map", pool.map)

    # === Evolution loop with timing ===
    tap_timer("Evolution")
    pop, log = algorithms.eaSimple(
        pop, toolbox, cxpb=0.5, mutpb=0.3, ngen=gens,
        stats=stats, halloffame=hof, verbose=DEBUG_PROGRESS
    )
    print(f"-------- End of Evolution --------")
    print()
    tap_timer("Evolution")
    print()
    print(f"-------- Best controller fitness: {hof[0].fitness.values[0]:.3f} --------")

    # Clean up pool
    pool.close()
    pool.join()

    # === Save results ===

    tap_timer("Save logs")
    if SAVE_LOGS:
        csv_path = plots_dir / f"log.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["gen","avg","std","max"])
            for rec in log: w.writerow([rec["gen"], rec["avg"], rec["std"], rec["max"]])
            
        print(f"    [1/5] Log saved to {csv_path}")
    tap_timer("Save logs")

    tap_timer("Save controller")
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
            
        print(f"    [2/5] Controller weights saved to {controller_path}")
    tap_timer("Save controller")

    best_genome = np.array(hof[0])

    if SAVE_PLOTS:
        tap_timer("Save fitness plot")
        plot_fitness(log, plots_dir, pop_size, task, out_name="plot_fitness_generations.png")
        print(f"    [3/5] Fitness plot saved to {plots_dir / f'plot_fitness_generations.png'}")
        tap_timer("Save fitness plot")

        tap_timer("Save trajectory plot")
        plot_best_trajectory(
            best_genome, robot_graph, spawn_pos, plots_dir,
            target_pos=target_pos,
            out_name=f"trajectory.png"
        )
        print(f"    [4/5] Trajectory plot saved to {plots_dir / f'trajectory.png'}")
        tap_timer("Save trajectory plot")

        tap_timer("Save fitness over time plot")
        plot_best_fitness_over_time(
            best_genome, robot_graph, spawn_pos, plots_dir,
            target_pos=target_pos,
            out_name="fitness_over_time.png"
        )
        print(f"    [5/5] Fitness over time plot saved to {plots_dir / f'fitness_over_time.png'}")
        tap_timer("Save fitness over time plot")

    mj.set_mjcb_control(None)

    best_controller = build_controller(
        best_genome, input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    return best_controller
