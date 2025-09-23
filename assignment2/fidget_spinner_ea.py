# fidget_spinner_ea.py

# Third-party libraries
import argparse
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from cma import CMAEvolutionStrategy

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers.hopfs_cpg import HopfCPG
from ariel.simulation.controllers.cpg_with_sensory_feedback import CPGSensoryFeedback
from ariel.simulation.tasks.turning_in_place import turning_in_place

# Global history
HISTORY = []


# -------------------------
# Controller definitions
# -------------------------

def random_controller_builder(model, to_track, params=None):
    def controller(model, data):
        num_joints = model.nu
        rand_moves = np.random.uniform(-np.pi/2, np.pi/2, num_joints)
        delta = 0.05
        data.ctrl += rand_moves * delta
        data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
        HISTORY.append(to_track[0].xpos.copy())
    return controller


def hopf_cpg_controller_builder(model, to_track, params=None):
    num_joints = model.nu
    adjacency_list = {i: [(i + 1) % num_joints] for i in range(num_joints)}

    cpg = HopfCPG(num_neurons=num_joints,
                  adjacency_list=adjacency_list,
                  dt=model.opt.timestep)

    if params is not None:
        omega, A, h = params
        cpg.omega[:] = omega
        cpg.A[:] = A
        cpg.h = h
    else:
        cpg.A[:] = 5.0

    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)
        HISTORY.append(to_track[0].xpos.copy())
    return controller


def feedback_cpg_controller_builder(model, to_track, params=None):
    num_joints = model.nu
    coupling = np.zeros((num_joints, num_joints))
    for i in range(num_joints):
        coupling[i, (i+1) % num_joints] = 0.1

    cpg = CPGSensoryFeedback(num_neurons=num_joints,
                             dt=model.opt.timestep,
                             coupling_weights=coupling)

    if params is not None:
        omega, A, sensory = params
        cpg.omega[:] = omega
        cpg.amplitude[:] = A
        cpg.sensory_term = sensory
    else:
        cpg.amplitude[:] = 5.0

    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)
        HISTORY.append(to_track[0].xpos.copy())
    return controller


# Map argument to builder
CONTROLLERS = {
    "random": random_controller_builder,
    "hopf": hopf_cpg_controller_builder,
    "feedback": feedback_cpg_controller_builder,
}


# -------------------------
# Utility
# -------------------------

def show_qpos_history(history:list):
    pos_data = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.axis('equal')
    plt.show()


# -------------------------
# EA evaluation
# -------------------------

def evaluate(params, controller_name, steps=1000):
    global HISTORY
    HISTORY = []

    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    controller_builder = CONTROLLERS[controller_name]
    mujoco.set_mjcb_control(controller_builder(model, to_track, params))

    for _ in range(steps):
        mujoco.mj_step(model, data)

    return turning_in_place([pos[:2] for pos in HISTORY])


def evolve(controller_name, generations=100, popsize=20):
    if controller_name == "hopf":
        x0, sigma0 = [2*np.pi, 5.0, 0.1], 0.5
    elif controller_name == "feedback":
        x0, sigma0 = [2*np.pi, 5.0, 0.1], 0.5
    else:  # random has no parameters
        print("Random controller cannot be evolved.")
        return None

    es = CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize})

    for gen in range(generations):
        solutions = es.ask()
        fitnesses = [-evaluate(s, controller_name) for s in solutions]
        es.tell(solutions, fitnesses)
        print(f"Gen {gen}: best fitness = {-min(fitnesses):.4f}")

    best_params = es.result.xbest
    print("\n=== Evolution Complete ===")
    print("Best params:", best_params)
    print("Best fitness:", -es.result.fbest)

    return best_params


# -------------------------
# Main entry
# -------------------------

def main(controller_name, use_ea=False):
    if use_ea and controller_name != "random":
        best_params = evolve(controller_name, generations=100, popsize=20)
        print("Replay with best parameters...")
    else:
        best_params = None

    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    controller_builder = CONTROLLERS[controller_name]
    mujoco.set_mjcb_control(controller_builder(model, to_track, best_params))

    viewer.launch(model=model, data=data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=CONTROLLERS.keys(), default="random")
    parser.add_argument("--evolve", action="store_true", help="Run CMA-ES before replay")
    args = parser.parse_args()

    main(args.controller, use_ea=args.evolve)
