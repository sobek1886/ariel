# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from cma import CMAEvolutionStrategy

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.amphitheatre_heightmap import AmphitheatreTerrainWorld
from ariel.simulation.environments.djoser_pyramid import PyramidWorld

# Import gate learning task functions
from ariel.simulation.tasks.gate_learning import xy_displacement, x_speed, y_speed
from ariel.simulation.tasks.turning_in_place import turning_in_place

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# import prebuilt controllers
from ariel.simulation.controllers.hopfs_cpg import HopfCPG
from ariel.simulation.controllers.cpg_with_sensory_feedback import CPGSensoryFeedback


# Keep track of data / history
HISTORY = []


def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints."""
    num_joints = model.nu
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low=-hinge_range,
                                   high=hinge_range,
                                   size=num_joints)
    delta = 0.05
    data.ctrl += rand_moves * delta
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    HISTORY.append(to_track[0].xpos.copy())


def hopf_cpg_controller_builder(model, to_track, params=None):
    """
    Build a Hopf CPG controller callback for MuJoCo.
    Genome format: [omega, amplitude, coupling]
    """
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
        cpg.A[:] = 10  # default amplitude

    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)
        HISTORY.append(to_track[0].xpos.copy())

    return controller


def show_qpos_history(history: list):
    pos_data = np.array(history)
    plt.figure(figsize=(10, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.show()


def evaluate(params, steps=500):
    """Run headless simulation with given params and return spinning fitness."""
    global HISTORY
    HISTORY = []

    mujoco.set_mjcb_control(None)
    world = AmphitheatreTerrainWorld() #SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    mujoco.set_mjcb_control(hopf_cpg_controller_builder(model, to_track, params))

    for _ in range(steps):
        mujoco.mj_step(model, data)

    return turning_in_place([pos[:2] for pos in HISTORY])


def evolve(generations=5, popsize=8):
    """Run CMA-ES evolution on the Hopf CPG parameters."""
    x0 = [2 * np.pi, 1.0, 0.1]  # omega, amplitude, coupling
    sigma0 = 0.5
    es = CMAEvolutionStrategy(x0, sigma0, {"popsize": popsize})

    for gen in range(generations):
        solutions = es.ask()
        fitnesses = [-evaluate(s) for s in solutions]  # CMA-ES minimizes
        es.tell(solutions, fitnesses)
        print(f"Gen {gen}: best fitness = {-min(fitnesses):.4f}")

    best_params = es.result.xbest
    print("\n=== Evolution Complete ===")
    print("Best params:", best_params)
    print("Best fitness:", -es.result.fbest)
    return best_params


def main(params=None):
    """Run the simulation with viewer, using either default or evolved params."""
    mujoco.set_mjcb_control(None)
    world = AmphitheatreTerrainWorld() #SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    mujoco.set_mjcb_control(hopf_cpg_controller_builder(model, to_track, params))

    viewer.launch(model=model, data=data)
    show_qpos_history(HISTORY)


if __name__ == "__main__":
    # Run evolution first
    best_params = evolve(generations=100, popsize=20)

    # Replay best solution in the viewer
    main(best_params)
