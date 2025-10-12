from __future__ import annotations
from pathlib import Path
import random, json, os, time, multiprocessing, string
import numpy as np
import mujoco as mj
import gc
import psutil
from deap import base, creator, tools
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

from examples.A3_code.evolve.config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH,
    NUM_POP, NUM_GENS, TARGET_POS, SPAWN_POS,
    SAVE_PLOTS, SAVE_LOGS, SAVE_CHECKPOINTS,
    DEBUG_PROGRESS, GENOTYPE_SIZE, NUM_OF_MODULES,
    BODY_GENE_LENGTH, TASK, DATA_PATH, IMMOBILE_THRESH, MUT_INDPB, MUT_SIGMA, CX_PROB, MUT_PROB, TOURNAMENT_SIZE,
    DURATION_RAMP_GENS, MAX_DURATION, BASE_DURATION
)
import examples.A3_code.evolve.config as cfg
from examples.A3_code.evolve.nn import genome_length, make_controller_from_genome, infer_input_size
from examples.A3_code.evolve.fitness import olympic_arena_fitness
from examples.A3_code.evolve.utils import (
    plot_fitness, count_num_joints, run_simulation,
    plot_best_trajectory, plot_best_fitness_over_time,
    save_log_csv, save_robot
)

timers = {}
last_gen_time = 0
last_gen_dur = 0

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
        self.clone_num = 0

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
        c.clone_num = self.clone_num + 1

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
        self.prev_num_joints = self.num_joints

        if self.prev_num_joints != num_joints:
            # CRITICAL: Limit history size to prevent memory growth
            self.limb_history.append(num_joints)
            if len(self.limb_history) > 50:  # Keep only last 100 evaluations
                print(f"[INFO] Truncating limb history for bot: {self.id}.{self.clone_num}")
                self.limb_history = self.limb_history[-50:]  # Trim to 50

    def stats(self):
        return (f"Robot {self.id}.{self.clone_num} | "
                f"fitness={self.fitness[0]:.1f} | "
                f"mutations={self.mutation_count} | "
                f"limbs={self.num_joints} | "
                f"input_size={self.input_size} | "
                f"evaluations={len(self.limb_history)} | "
                f"disp={self.disp:.1f}")

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
        # print(f"[MUTATION] Robot {self.id}.{self.clone_num} body mutated, rebuilding morphology.")
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

        num_joints = count_num_joints(robot_graph)
        input_size = infer_input_size(num_joints, STATE_FEATURES)

        # Resize controller if body changed
        required_len = genome_length(input_size, HIDDEN_SIZE, num_joints,
                                    depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2)

        # --- DEBUG PRINTS ---
        # print(f"\n[DEBUG] Robot evaluation info robot: {bot.id}")
        # print(f"  num_joints = {num_joints}")
        # print(f"  input_size = {input_size}")
        # print(f"  required_len = {required_len}")
        # print(f"  current_ctrl_len = {len(bot.ctrl)}")
        # print(f"  BODY_GENE_LENGTH = {BODY_GENE_LENGTH}")
        # print(f"  GENOTYPE_SIZE = {GENOTYPE_SIZE}")
        # print(f"  NUM_OF_MODULES = {NUM_OF_MODULES}")
        
        ctrl_genes = bot.ctrl.copy()
        curr_len = len(ctrl_genes)

        if curr_len < required_len:
            # Add new random genes to fill missing values
            ctrl_genes += [random.uniform(-1.0, 1.0) for _ in range(required_len - curr_len)]
        elif curr_len > required_len:
            # Truncate excess genes
            ctrl_genes = ctrl_genes[:required_len]

        # Simulate
        traj = run_simulation(ctrl_genes, robot_graph)

        # --- EARLY MOBILITY CHECK ---
        # sample 25%, 50%, 75% of the trajectory
        samples = [len(traj)//4, len(traj)//2, 3*len(traj)//4]
        early_disps = [
            np.linalg.norm(traj[s, [0, 1]] - traj[0, [0, 1]])
            for s in samples
        ]

        # If all displacements are below threshold -> immobile
        if all(d < IMMOBILE_THRESH for d in early_disps):
            # Skip rest of computation — robot clearly not moving
            del traj
            return {
                'fitness': -999.0,
                'disp': 0.0,
                'num_joints': num_joints,
                'input_size': input_size,
                'ctrl_genes': ctrl_genes
            }

        # --- Otherwise compute full displacement and fitness ---
        disp = np.linalg.norm(traj[-1, [0, 1]] - traj[0, [0, 1]])

        # print(f"  displacement = {disp:.3f} m")
        # print("-----------------------------")
        
        # Compute fitness
        if disp < IMMOBILE_THRESH:
            fitness = -999.0  # immobile
        else:
            fitness = olympic_arena_fitness(traj)

        del traj

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
            'fitness': -999.0,
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
        return 0
    else:
        # Stop the timer and print elapsed time
        total_time = time.time() - timers[timer_name]
        print(f"[TIMER] {timer_name} timer took: {total_time:.2f} sec")
        del timers[timer_name]  # Reset the timer
        return total_time

def print_memory_usage(label=""):
    """Print current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    rss_gb = mem_info.rss / (1024 ** 3)
    vms_gb = mem_info.vms / (1024 ** 3)
    print(f"[MEMORY {label}] RSS: {rss_gb:.2f} GB, VMS: {vms_gb:.2f} GB")

def print_summary_pop(pop, gen):
    print(f"\n[SUMMARY] Gen {gen+1}: "
      f"Mobile: {sum(1 for b in pop if b.disp >= IMMOBILE_THRESH)}/{len(pop)}, "
      f"Avg limbs: {np.mean([b.num_joints for b in pop]):.1f}")
    
    # Only show top 3 and bottom 
    sorted_pop = sorted(pop, key=lambda b: b.fitness[0], reverse=True)

    print("Top 3:")
    for bot in sorted_pop[:3]:
        print(f"  {bot.stats()}")

    print("Bottom 3:")
    for bot in sorted_pop[-3:]:
        print(f"  {bot.stats()}")

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

    # --- Initialize population ---
    pop: list[Robot] = []
    for i in range(NUM_POP):
        body = [random.uniform(-1, 1) for _ in range(BODY_GENE_LENGTH)]
        ctrl = [random.uniform(-1, 1) for _ in range(100)]  # temp ctrl, resized in eval
        bot = Robot(body, ctrl)
        pop.append(bot)

    # --- Parallel evaluation setup ---
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    # CRITICAL: Pool refresh settings
    POOL_REFRESH_INTERVAL = 10  # Recreate pool every 10 generations
    pool = None

    def eval_map(f, items):
        return pool.map(f, items)

    tap_timer("evolution")
    # --- Evolution ---
    for gen in range(NUM_GENS):
        tap_timer("one generation")

        current_duration = min(
            BASE_DURATION + (MAX_DURATION - BASE_DURATION) * (gen / DURATION_RAMP_GENS),
            MAX_DURATION
        )
        current_duration = int(current_duration)
        cfg.DURATION = current_duration

        print(f"\n===== Start of Generation {gen+1}/{NUM_GENS} (duration={current_duration}s)=====")

        # CRITICAL: Recreate pool periodically to prevent memory buildup
        if gen % POOL_REFRESH_INTERVAL == 0:
            if pool is not None:
                pool.close()
                pool.join()
                del pool
                gc.collect()  # Force garbage collection
            pool = multiprocessing.Pool(n_cpus)
            print(f"[MEMORY] Refreshed multiprocessing pool at generation {gen+1}")

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
                bot.record_limb_count(result['num_joints'])
                bot.num_joints = result['num_joints']
                bot.input_size = result['input_size']
                bot.ctrl = result['ctrl_genes']  # Update with potentially resized controller

        # Replace immobile robots
        for i, bot in enumerate(pop):
            if bot.disp < IMMOBILE_THRESH:
                old_id = bot.id
                body = [random.uniform(-1, 1) for _ in range(BODY_GENE_LENGTH)]
                ctrl = [random.uniform(-1, 1) for _ in range(100)]
                pop[i] = Robot(body, ctrl)
                pop[i].gen_created = gen
                # if DEBUG_PROGRESS:
                    # print(f"[!] Replacing immobile robot {old_id} with robot {pop[i].id} (disp={bot.disp:.3f})")

        if DEBUG_PROGRESS:
            print("\n--- Robot Stats Summary start generation ---")
            print_summary_pop(pop, gen)
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
            mutate_robot(b1)
            mutate_robot(b2)
            next_pop += [b1, b2]

        pop = next_pop[:NUM_POP]
        pop[0] = best  # elitism

        if DEBUG_PROGRESS:
            print("\n--- Robot Stats Summary end generation ---")
            print_summary_pop(pop, gen)
            print("---------------------------")


        gc.collect()
        print_memory_usage(f"Gen {gen+1}")
        
        # Save checkpoint every 10 generations
        if SAVE_CHECKPOINTS:
            chk_pnt_path = DATA_PATH / f"checkpoints"
            chk_pnt_path.mkdir(parents=True, exist_ok=True)

            save_robot(chk_pnt_path, best.body_graph, best.ctrl,
                        input_size=best.input_size, num_joints=best.num_joints, out=f"robot_gen_{gen+1}.json")
            print(f"[CHECKPOINT] Saved at generation {gen+1}")

        if SAVE_PLOTS:
            gen_dir = plots_dir / f"generations"
            gen_dir.mkdir(parents=True, exist_ok=True)

            plot_best_trajectory(best.ctrl, best.body_graph, gen_dir, out_name=f"trajectory_{gen}.png")

        # Compute statistics (use quartiles for skewed distributions)
        fits_array = np.array([b.fitness[0] for b in pop])
        avg_fit = np.mean(fits_array)
        best_fit = np.max(fits_array)
        worst_fit = np.min(fits_array)
        median_fit = np.median(fits_array)
        q25 = np.percentile(fits_array, 25)
        q75 = np.percentile(fits_array, 75)
        
        log.append({
            "gen": gen + 1,
            "avg": float(avg_fit),
            "median": float(median_fit),
            "q25": float(q25),
            "q75": float(q75),
            "min": float(worst_fit),
            "max": float(best_fit)
        })
        print(f"Gen {gen + 1}: avg={float(avg_fit):.3f}, median={float(median_fit):.3f}, best={float(best_fit):.3f}")

        if SAVE_PLOTS:
            if (gen + 1) % 10 == 0 or gen == NUM_GENS - 1:
                csv_path = DATA_PATH / "fitness_log.csv"

                save_log_csv(log, csv_path)

                print(f"[LOG] Appended up to generation {gen+1}")

                # Optionally replot
                gen_dir = plots_dir / f"generations"
                gen_dir.mkdir(parents=True, exist_ok=True)
                plot_fitness(
                    log,
                    dest_dir=gen_dir,
                    pop=NUM_POP,
                    task=TASK,
                    out_name=f"fitness_plot_gen_{gen+1}.png"
                )

            gc.collect()

        print("\n")
        new_gen_time = tap_timer("one generation")
        print(f"    Gen duration increased by {current_duration - last_gen_dur}, time took to run gen increased by: {new_gen_time - last_gen_time}")
        last_gen_time = new_gen_time
        last_gen_dur = current_duration
        print("\n")


    # --- Finalize ---
    pool.close()
    pool.join()
    tap_timer("evolution")

    best_robot = max(pop, key=lambda b: b.fitness[0])
    print(f"\n\n---------------------------")
    print(f"🏆 Best fitness: {best_robot.fitness[0]:.3f}")
    print(f"Best robot: {best_robot.stats()}")
    print(f"Best robot limb history: {best_robot.limb_history}")
    print(f"\n\n---------------------------")

    # Use cached body_graph for saving
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

        tap_timer("best_fitness_over_time")
        plot_best_fitness_over_time(best_robot.ctrl, best_robot.body_graph, plots_dir)
        tap_timer("best_fitness_over_time")

if __name__ == "__main__":
    tap_timer("Total")
    run_evolve_robot()
    tap_timer("Total")
    print(f"Finished evolve, pop: {NUM_POP}, gens: {NUM_GENS}, max duration: {cfg.DURATION}")