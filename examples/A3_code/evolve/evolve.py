from __future__ import annotations
from pathlib import Path
import random, json, os, time, multiprocessing, string
import numpy as np
import mujoco as mj
from deap import base, creator, tools
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from examples.A3_code.evolve.fitness import distance_to_target, fitness_function, olympic_arena_fitness_dense

from examples.A3_code.evolve.config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH,
    NUM_POP, NUM_GENS, TARGET_POS, SPAWN_POS,
    SAVE_PLOTS, SAVE_LOGS, DURATION,
    DEBUG_PROGRESS, GENOTYPE_SIZE, NUM_OF_MODULES,
    BODY_GENE_LENGTH, TASK, DATA_PATH, IMMOBILE_THRESH, MUT_INDPB, MUT_SIGMA, CX_PROB, MUT_PROB, TOURNAMENT_SIZE, USE_CPG, CPG_HYBRID
)
from examples.A3_code.evolve.nn import(
    genome_length, make_controller_from_genome, infer_input_size,
    cpg_genome_length, hybrid_cpg_genome_length, make_cpg_controller_from_genome  # ADD THESE
)
from examples.A3_code.evolve.utils import (
    make_world, plot_fitness,
    plot_best_trajectory, plot_best_fitness_over_time,
    save_log_csv, save_robot, count_num_joints
)

timers = {}

# ===============================================================
# HELPERS
# ===============================================================
def split_body_genes(flat):
    type_vec = flat[0:GENOTYPE_SIZE]
    conn_vec = flat[GENOTYPE_SIZE:2*GENOTYPE_SIZE]
    rot_vec  = flat[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE]
    return [type_vec, conn_vec, rot_vec]

def decode_body(body_genes):
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    genotype_vectors = split_body_genes(body_genes)
    p_matrices = nde.forward([np.array(v, dtype=np.float32) for v in genotype_vectors])
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    return hpd.probability_matrices_to_graph(*p_matrices)



# ===============================================================
# DATA CLASSES
# ===============================================================
# class Robot:
#     """Represents an evolutionary individual with body and controller genomes."""
#     def __init__(self, body_genes: list[float], ctrl_genes: list[float]):
#         self.body = body_genes
#         self.ctrl = ctrl_genes
#         self.fitness = (-999.0,)

#     def clone(self):
#         c = Robot(self.body.copy(), self.ctrl.copy())
#         c.fitness = self.fitness
#         return c

# ===============================================================
# DATA CLASSES
# ===============================================================
class Robot:
    """Represents an evolutionary individual with body and controller genomes."""
    _next_id = 0  # static counter for unique IDs

    def __init__(self, body_genes: list[float], ctrl_genes: list[float]):
        self.id = Robot._next_id
        Robot._next_id += 1

        self.body = body_genes
        self.body_graph = decode_body(body_genes)  # decode once on creation
        self.ctrl = ctrl_genes
        self.num_joints = count_num_joints(self.body_graph)
        self.input_size = infer_input_size(self.num_joints, STATE_FEATURES)
        self.fitness = (-999.0,)
        self.prev_num_joints = None
        self.limb_history = []  # record num_joints across evaluations
        self.mutation_count = 0
        self.gen_created = 0  # for tracking when it appeared
        self.disp = 0.0

    def update_body(self):
        self.body_graph = decode_body(self.body)
        self.num_joints = count_num_joints(self.body_graph)
        self.input_size = infer_input_size(self.num_joints, STATE_FEATURES)

    def clone(self):
        c = Robot(self.body.copy(), self.ctrl.copy())

        # Restore original ID
        c.id = self.id
        
        # Roll back the class-wide counter increment
        Robot._next_id -= 1 

        # Lineage tracking
        c.clone_num = getattr(self, "clone_num", 0) + 1

        c.fitness = self.fitness
        c.prev_num_joints = self.prev_num_joints
        c.limb_history = self.limb_history.copy()
        c.mutation_count = self.mutation_count
        c.gen_created = self.gen_created
        c.disp = self.disp
        c.body_graph = self.body_graph  # keep cached body
        c.num_joints = self.num_joints
        c.input_size = self.input_size
        return c

    def record_limb_count(self, num_joints: int):
        """Log limb count and detect structural changes."""
        if self.prev_num_joints is None:
            self.prev_num_joints = num_joints
        elif self.prev_num_joints != num_joints:
            # print(f"[LIMB CHANGE] Robot {self.id} changed limbs: {self.prev_num_joints} → {num_joints}")
            self.prev_num_joints = num_joints
        self.limb_history.append(num_joints)

    def stats(self):
        return (f"Robot {self.id}.{getattr(self, 'clone_num', 0)} | "
                f"fitness={self.fitness[0]:.1f} | "
                f"mutations={self.mutation_count} | "
                f"limbs={self.num_joints} | "
                f"input_size={self.input_size} | "
                f"evaluations={len(self.limb_history)} | "
                f"disp={self.disp:.1f}")

def adaptive_mutate_robot(bot: Robot, gen: int, max_gens: int):
    """Decrease mutation as evolution progresses"""
    progress = gen / max_gens
    # Start high, decay to 20% of initial
    current_sigma = MUT_SIGMA * (1.0 - 0.8 * progress)
    current_indpb = MUT_INDPB * (1.0 - 0.7 * progress)
    
    is_mutate = False
    for i in range(len(bot.body)):
        if random.random() < current_indpb:
            bot.body[i] += np.random.normal(0, current_sigma)
            is_mutate = True
    
    for i in range(len(bot.ctrl)):
        if random.random() < current_indpb:
            bot.ctrl[i] += np.random.normal(0, current_sigma)
    
    if is_mutate:
        bot.mutation_count += 1
        bot.update_body()
    
    return bot

# ===============================================================
# SIMULATION
# ===============================================================
def run_simulation(ctrl_genes, robot_graph):
    """Build world, run simulation, return trajectory and model data."""
    mj.set_mjcb_control(None)
    world = OlympicArena()
    core = construct_mjspec_from_graph(robot_graph)
    world.spawn(core.spec, list(SPAWN_POS))
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)


    num_joints = model.nu
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
    return traj, model, data, tracker

# ===============================================================
# GENETIC OPERATORS
# ===============================================================
def crossover_robots(bot1: Robot, bot2: Robot):
    """Crossover both body and controller with awareness of different sizes."""
    # --- Body crossover ---
    for start in [0, GENOTYPE_SIZE, 2 * GENOTYPE_SIZE]:
        end = start + GENOTYPE_SIZE
        if random.random() < 0.5:
            cxp = random.randint(1, GENOTYPE_SIZE - 1)
            bot1.body[start:end], bot2.body[start:end] = (
                bot1.body[start:cxp] + bot2.body[cxp:end],
                bot2.body[start:cxp] + bot1.body[cxp:end],
            )

    # --- Controller crossover ---
    L1, L2 = len(bot1.ctrl), len(bot2.ctrl)
    if L1 > 1 and L2 > 1:
        L = min(L1, L2)
        cxp = random.randint(1, L - 1)
        new1 = bot1.ctrl[:cxp] + bot2.ctrl[cxp:L] + (bot1.ctrl[L:] if L1 > L else [])
        new2 = bot2.ctrl[:cxp] + bot1.ctrl[cxp:L] + (bot2.ctrl[L:] if L2 > L else [])
        bot1.ctrl, bot2.ctrl = new1, new2
    return bot1, bot2

def mutate_robot(bot: Robot):
    """Gaussian mutation for both body and controller."""
    is_mutate = False

    for i in range(len(bot.body)):
        if random.random() < MUT_INDPB:
            bot.body[i] += np.random.normal(0, MUT_SIGMA)
            is_mutate = True

    for i in range(len(bot.ctrl)):
        if random.random() < MUT_INDPB:
            bot.ctrl[i] += np.random.normal(0, MUT_SIGMA)

    if is_mutate:
        # After mutating the body, rebuild the morphology
        # print(f"[MUTATION] Robot {bot.id}.{getattr(bot, 'clone_num', 0)} body mutated, rebuilding morphology.")
        bot.mutation_count += 1
        # print(f"    HAD limbs: {bot.num_joints} and input_size: {bot.input_size}")
        bot.update_body()
        # print(f"        NOW limbs: {bot.num_joints} and input_size: {bot.input_size}")

    return bot

# ===============================================================
# EVALUATION
# ===============================================================
def evaluate_robot(bot: Robot):
    """Evaluate fitness with dynamic controller adjustment."""
    try:
        robot_graph = bot.body_graph  # use cached graph

        # Build temporary model to check joint count
        tmp_world = make_world()
        core = construct_mjspec_from_graph(robot_graph)
        tmp_world.spawn(core.spec, list(SPAWN_POS))
        tmp_model = tmp_world.spec.compile()
        num_joints = tmp_model.nu

        input_size = infer_input_size(num_joints, STATE_FEATURES)

        # Resize controller if body changed
        required_len = genome_length(input_size, HIDDEN_SIZE, num_joints,
                                    depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2)

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
        
        ctrl_genes = bot.ctrl
        # Resize controller if needed
        if len(bot.ctrl) != required_len:
            if len(bot.ctrl) < required_len:
                # Body grew - pad with small random values
                padding = [random.uniform(-0.1, 0.1) for _ in range(required_len - len(bot.ctrl))]
                ctrl_genes = bot.ctrl + padding
            else:
                # Body shrunk - truncate
                ctrl_genes = bot.ctrl[:required_len]
            
            bot.ctrl = ctrl_genes
        # Simulate
        traj, model, data, tracker = run_simulation(ctrl_genes, robot_graph)
        disp = np.linalg.norm(traj[-1, [0, 1]] - traj[0, [0, 1]])

        # print(f"  displacement = {disp:.3f} m")
        # print("-----------------------------")
        
        # Compute fitness
        if disp < IMMOBILE_THRESH:
            fitness = -100.0 + (disp * 10.0)
        else:
            # Use forward distance for Olympic Arena race
            fitness = olympic_arena_fitness_dense(traj)
        # Return all data needed for updating bot in main process
        return {
            'fitness': fitness,
            'disp': disp,
            'num_joints': num_joints,
            'input_size': input_size,
            'ctrl_genes': ctrl_genes
        }

    except Exception as e:
        print(f"[!] Robot evaluation failed: {e}")
        return {
            'fitness': -200.0,
            'disp': 0.0,
            'num_joints': None,
            'input_size': None,
            'ctrl_genes': bot.ctrl
        }

def _eval_robot_helper(bot):
    """Helper wrapper for multiprocessing pool.map."""
    return evaluate_robot(bot)

def tap_timer(timer_name: str = "Timer"):
    global timers
    if timer_name not in timers:
        # Start the timer
        timers[timer_name] = time.time()
    else:
        # Stop the timer and print elapsed time
        total_time = time.time() - timers[timer_name]
        print(f"[TIMER] {timer_name} timer took: {total_time:.2f} sec")
        del timers[timer_name]  # Reset the timer

def smart_resize_controller(old_ctrl: list, required_len: int, 
                           old_input_size: int, new_input_size: int,
                           old_num_joints: int, new_num_joints: int) -> list:
    """Intelligently resize controller preserving learned patterns"""
    
    if len(old_ctrl) == required_len:
        return old_ctrl
    
    # Calculate layer sizes
    old_w1_size = old_input_size * HIDDEN_SIZE
    new_w1_size = new_input_size * HIDDEN_SIZE
    
    if len(old_ctrl) < required_len:
        # Body grew - expand strategically
        new_ctrl = old_ctrl.copy()
        
        # Add small noise for new input->hidden weights
        padding_size = required_len - len(old_ctrl)
        new_ctrl.extend([random.uniform(-0.05, 0.05) for _ in range(padding_size)])
        
    else:
        # Body shrunk - preserve most important weights
        new_ctrl = []
        
        # Try to preserve center of weight matrices
        old_w1 = np.array(old_ctrl[:old_w1_size]).reshape(old_input_size, HIDDEN_SIZE)
        
        # Keep central features if possible
        if new_input_size < old_input_size:
            # Trim from edges
            trim = (old_input_size - new_input_size) // 2
            new_w1 = old_w1[trim:trim+new_input_size, :]
        else:
            new_w1 = old_w1[:new_input_size, :]
        
        new_ctrl.extend(new_w1.flatten().tolist())
        
        # Keep rest of network
        remaining = old_ctrl[old_w1_size:]
        needed = required_len - len(new_ctrl)
        new_ctrl.extend(remaining[:needed])
        
        # Pad if still short
        if len(new_ctrl) < required_len:
            new_ctrl.extend([random.uniform(-0.05, 0.05) 
                           for _ in range(required_len - len(new_ctrl))])
    
    return new_ctrl[:required_len]


# ===============================================================
# MAIN EVOLUTION LOOP
# ===============================================================
def run_evolve_robot(
    seed: int | None = None,
):
    log = []
    if seed is not None:
        np.random.seed(seed); random.seed(seed)

    DATA_PATH.mkdir(parents=True, exist_ok=True)
    plots_dir = DATA_PATH / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
        # ✓ VERIFY THE FIX WORKS
        

    # --- Initialize population ---
    pop: list[Robot] = []
    for i in range(NUM_POP):
        body = [random.uniform(-1, 1) for _ in range(BODY_GENE_LENGTH)]
        ctrl = [random.uniform(-1, 1) for _ in range(100)]  # temp ctrl, resized in eval
        bot = Robot(body, ctrl)
        bot.gen_created = 0  # generation 0
        pop.append(bot)
    # --- Initialize population ---
    # pop: list[Robot] = []
    # for i in range(NUM_POP):
    #     body = [random.uniform(-1, 1) for _ in range(BODY_GENE_LENGTH)]
        
    #     # Create a temporary robot to determine required controller size
    #     temp_bot = Robot(body, [])  # Empty controller initially
    #     required_ctrl_len = genome_length(
    #         temp_bot.input_size, 
    #         HIDDEN_SIZE, 
    #         temp_bot.num_joints,
    #         depth=NN_DEPTH, 
    #         hidden_size2=HIDDEN_SIZE2
    #     )
        
    #     # Now create controller with correct size
    #     ctrl = [random.uniform(-0.5, 0.5) for _ in range(required_ctrl_len)]
        
    #     # Create the actual robot
    #     bot = Robot(body, ctrl)
    #     bot.gen_created = 0
    #     pop.append(bot)

    # --- Parallel evaluation setup ---
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    pool = multiprocessing.Pool(n_cpus)

    def eval_map(f, items): return pool.map(f, items)

    tap_timer("evolution")
    # --- Evolution ---
    for gen in range(NUM_GENS):
        tap_timer("one generation")
        print(f"\n===== Generation {gen+1}/{NUM_GENS} =====")

        # Parallel evaluate
        results = eval_map(_eval_robot_helper, [b for b in pop])

        # Single core evaluate
        # results = [evaluate_robot(bot, SPAWN_POS, TARGET_POS) for bot in pop]

        # Update robots with results from workers
        for bot, result in zip(pop, results):
            bot.fitness = (result['fitness'],)
            bot.disp = result['disp']
            
            # Update bot state from worker results
            if result['num_joints'] is not None:
                bot.num_joints = result['num_joints']
                bot.input_size = result['input_size']
                bot.ctrl = result['ctrl_genes']  # Update with potentially resized controller
                bot.record_limb_count(result['num_joints'])

        # Replace failed robots
        num_replaced = 0
        for i, bot in enumerate(pop):
            if bot.fitness[0] <= -150.0:  # Crashed or extremely immobile
                body = [random.uniform(-1, 1) for _ in range(BODY_GENE_LENGTH)]
                ctrl = [random.uniform(-1, 1) for _ in range(100)]
                pop[i] = Robot(body, ctrl)
                pop[i].gen_created = gen
                num_replaced += 1

        if num_replaced > 0:
            print(f"[!] Replaced {num_replaced}/{NUM_POP} failed robots")

        if DEBUG_PROGRESS:
            print("\n--- Robot Stats Summary start generation ---")
            for bot in pop:
                print(bot.stats())
            print("---------------------------")

        # Selection, crossover, mutation
        fits = [bot.fitness[0] for bot in pop]
        best_idx = np.argmax(fits)
        best = pop[best_idx].clone()
        selected = tools.selTournament(pop, len(pop), tournsize=TOURNAMENT_SIZE)

        next_pop = []
        for i in range(0, len(selected), 2):
            b1, b2 = selected[i].clone(), selected[min(i + 1, len(selected) - 1)].clone()
            if random.random() < CX_PROB:
                b1, b2 = crossover_robots(b1, b2)
            adaptive_mutate_robot(b1, gen, NUM_GENS)
            adaptive_mutate_robot(b2, gen, NUM_GENS)
            next_pop += [b1, b2]

        pop = next_pop[:NUM_POP]
        pop[0] = best  # elitism

        if DEBUG_PROGRESS:
            print("\n--- Robot Stats Summary end generation ---")
            for bot in pop:
                print(bot.stats())
            print("---------------------------")

        if SAVE_PLOTS:
            gen_dir = plots_dir / f"generations"
            gen_dir.mkdir(parents=True, exist_ok=True)

            plot_best_trajectory(best.ctrl, best.body_graph, gen_dir, out_name=f"trajectory_{gen}.png")

        avg_fit = np.mean([b.fitness[0] for b in pop])
        best_fit = np.max([b.fitness[0] for b in pop])
        std_fit = np.std([b.fitness[0] for b in pop])
        log.append({
            "gen": gen + 1,
            "avg": float(avg_fit),
            "std": float(std_fit),
            "max": float(best_fit)
        })
        print(f"Gen {gen + 1}: avg={float(avg_fit)}, std: {float(std_fit)}, best={float(best_fit)}")

        tap_timer("one generation")

    # --- Finalize ---
    pool.close()
    pool.join()
    tap_timer("evolution")

    best_robot = max(pop, key=lambda b: b.fitness[0])
    print(f"\n\n---------------------------")
    print(f"🏆 Best fitness: {best_robot.fitness[0]:.3f}")
    print(f"Best robot: {best_robot.stats()}")
    print(f"\n\n---------------------------")

    # Use cached body_graph for saving
    traj, model, data, tracker = run_simulation(best_robot.ctrl, best_robot.body_graph)
    save_robot(DATA_PATH, best_robot.body_graph, best_robot.ctrl, input_size=best_robot.input_size, num_joints=best_robot.num_joints)

    if SAVE_LOGS:
        # === Log saving ===
        csv_path = DATA_PATH / "fitness_log.csv"
        save_log_csv(log, csv_path)

        # === Plotting ===
        plot_fitness(
            log,
            dest_dir=DATA_PATH,
            pop=NUM_POP,
            task=TASK,
            out_name="fitness_plot.png"
        )

    if SAVE_PLOTS:
        tap_timer("best_trajectory")
        plot_best_trajectory(best_robot.ctrl, best_robot.body_graph, plots_dir)
        tap_timer("best_trajectory")

        tap_timer("best_fitness")
        plot_best_fitness_over_time(best_robot.ctrl, best_robot.body_graph, plots_dir)
        tap_timer("best_fitness")

if __name__ == "__main__":
    tap_timer("Total")
    run_evolve_robot()
    tap_timer("Total")