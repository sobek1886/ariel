import json
import numpy as np
from pathlib import Path
import mujoco as mj
import networkx as nx

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena

from evolve_controller.evolve import run_controller_evolution
import string, random

# Where to save
DATA = Path.cwd() / "__data__" / "saved_robots"
DATA.mkdir(parents=True, exist_ok=True)

NUM_OF_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]

def generate_random_controller(input_size=None, output_size=None):
    # --- 4. Generate dummy weights with correct shapes ---
    hidden_size = 8
    rng = np.random.default_rng(42)
    w1 = rng.normal(0, 0.5, size=(input_size, hidden_size))
    w2 = rng.normal(0, 0.5, size=(hidden_size, hidden_size))
    w3 = rng.normal(0, 0.5, size=(hidden_size, output_size))

    weights = {
        "w1": w1.tolist(),
        "w2": w2.tolist(),
        "w3": w3.tolist(),
    }

    with open(DATA / "controller_weights.json", "w") as f:
        json.dump(weights, f)

    print(f"✅ Saved robot genotype + graph + controller weights to {DATA}")

def main():
    # Pick a 9-letter random tag for model
    tag = ''.join(random.choice(string.ascii_lowercase) for _ in range(9))
    dest_dir = DATA / tag
    dest_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Fixed Genotype (robot body DNA) ---
    genotype_size = 64
    type_p_genes = [1] * genotype_size
    conn_p_genes = [1] * genotype_size
    rot_p_genes  = [1] * genotype_size
    genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    # with open(dest_dir / "morphology.json", "w") as f:
    #     json.dump({"genotype": genotype}, f)

    # --- 2. Build body once (inside OlympicArena) to get sizes ---
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward([np.array(v, dtype=np.float32) for v in genotype])

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    # save the graph so run_view can reload exactly this body
    with open(dest_dir / "robot_graph.json", "w") as f:
        json.dump(nx.node_link_data(robot_graph, edges="links"), f)

    print(f"✅ Saved robot genotype + graph to {dest_dir}")

    # ✅ Build world + spawn robot (same as run_view)
    core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)   # now matches run_view
    output_size = model.nu

    print(f"Robot built with {input_size} inputs and {output_size} outputs")


    # --- Generate random controller with correct sizes ---
    # generate_random_controller(input_size=input_size, output_size=output_size)
    # exit(0)  # TEMPORARY EXIT TO SKIP EVOLUTION FOR NOW

    # --- 3. Evolve controller ---

    best_controller = run_controller_evolution(
        task="nav",
        robot_graph=robot_graph,
        gens=2,      # smaller for quick test, use NUM_GENS for real
        pop_size=5,
        dest_dir=dest_dir,
    )

    print(f"✅ Saved evolved controller weights to {dest_dir}")

if __name__ == "__main__":
    main()
