# evo_turning_in_place.py

# Third-party libraries
import numpy as np
from cma import CMAEvolutionStrategy
import mujoco
from mujoco import viewer

# Local libraries
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.tasks.turning_in_place import turning_in_place
from ariel.utils.renderers import video_renderer

# Import Hopf CPG
from ariel.simulation.controllers.hopfs_cpg import HopfCPG

# Keep track of positions
HISTORY = []


def evaluate_cpg(params, steps=500):
    global HISTORY
    HISTORY = []

    # World + robot
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

    # Important: compile the world spec, not the robot spec
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Genome unpacking
    omega, A, h = params

    num_joints = model.nu
    adjacency_list = {i: [(i + 1) % num_joints] for i in range(num_joints)}
    cpg = HopfCPG(num_neurons=num_joints,
                  adjacency_list=adjacency_list,
                  dt=model.opt.timestep,
                  h=h)
    cpg.omega[:] = omega
    cpg.A[:] = A

    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)
        HISTORY.append(to_track[0].xpos.copy())

    mujoco.set_mjcb_control(controller)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    return turning_in_place([pos[:2] for pos in HISTORY])



def run_evolution(generations=10, popsize=8):
    """Run CMA-ES optimization for turning-in-place."""
    # Initial guess [omega, A, h]
    x0 = [2 * np.pi, 1.0, 0.1]
    sigma0 = 0.5  # initial search spread

    es = CMAEvolutionStrategy(x0, sigma0, {'popsize': popsize})

    for gen in range(generations):
        solutions = es.ask()
        fitnesses = [-evaluate_cpg(s) for s in solutions]  # CMA-ES minimizes
        es.tell(solutions, fitnesses)
        es.disp()

    best_params = es.result.xbest
    best_fitness = -es.result.fbest
    print("\n=== Evolution Complete ===")
    print("Best params:", best_params)
    print("Best fitness:", best_fitness)

    return best_params


def replay(params, steps=1000):
    """Replay the best controller in MuJoCo viewer."""
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Unpack genome
    omega, A, h = params

    # Build Hopf CPG
    num_joints = model.nu
    adjacency_list = {i: [(i + 1) % num_joints] for i in range(num_joints)}
    cpg = HopfCPG(num_neurons=num_joints,
                  adjacency_list=adjacency_list,
                  dt=model.opt.timestep,
                  h=h)
    cpg.omega[:] = omega
    cpg.A[:] = A

    # Define controller
    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)

    mujoco.set_mjcb_control(controller)

    # Launch viewer
    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    best_params = run_evolution(generations=10, popsize=8)
    replay(best_params)
