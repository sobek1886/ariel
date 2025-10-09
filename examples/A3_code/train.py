import json
import numpy as np
from pathlib import Path
import mujoco as mj
import networkx as nx

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena

# Where to save
DATA = Path.cwd() / "__data__" / "saved_robot"
DATA.mkdir(parents=True, exist_ok=True)

NUM_OF_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]

def main():
    # --- 1. Fixed Genotype (robot body DNA) ---
    genotype_size = 64
    type_p_genes = [1] * genotype_size
    conn_p_genes = [1] * genotype_size
    rot_p_genes  = [1] * genotype_size
    genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    with open(DATA / "robot_genotype.json", "w") as f:
        json.dump({"genotype": genotype}, f)

    # --- 2. Build body once (inside OlympicArena) to get sizes ---
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward([np.array(v, dtype=np.float32) for v in genotype])

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    # save the graph so run_view can reload exactly this body
    with open(DATA / "robot_graph.json", "w") as f:
        json.dump(nx.node_link_data(robot_graph, edges="links"), f, indent=2)

    # ✅ Build world + spawn robot (same as run_view)
    core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, spawn_position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(data.qpos)   # now matches run_view
    output_size = model.nu
    hidden_size = 8

    print(f"Robot built with {input_size} inputs and {output_size} outputs")

    # --- 3. Save controller weights ---
    weights = {
        "w1": np.full((input_size, hidden_size), 0.1).tolist(),
        "w2": np.full((hidden_size, hidden_size), 0.2).tolist(),
        "w3": np.full((hidden_size, output_size), 0.3).tolist(),
    }

    with open(DATA / "controller_weights.json", "w") as f:
        json.dump(weights, f)

    print(f"Saved fixed robot + brain to {DATA}")

if __name__ == "__main__":
    main()
