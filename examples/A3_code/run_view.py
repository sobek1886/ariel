import json
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
import networkx as nx
import time

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from examples.A3_code.evolve.nn import make_controller_from_weights
from examples.A3_code.evolve.fitness import olympic_arena_fitness
from examples.A3_code.evolve.config import DATA_PATH, SPAWN_POS, TARGET_POS, TASK
from examples.A3_code.evolve.utils import load_robot, plot_best_trajectory, plot_best_fitness_over_time

if TYPE_CHECKING:
    from networkx import DiGraph

type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# SPAWN_POS = [-0.8, 0, 0.1]
# TARGET_POSITION = [5, 0, 0.5]

def load_controller_weights(path: Path) -> dict[str, np.ndarray]:
    with open(path, "r") as f:
        data = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in data.items()}

def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    print("DEBUG: pixel_to_dist =", pixel_to_dist)
    print("DEBUG: Start world =", pos_data[0])
    print("DEBUG: End world   =", pos_data[-1])
    print("DEBUG: Start pixel =", pos_data_pixel[0])
    print("DEBUG: End pixel   =", pos_data_pixel[-1])

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    save_path = str(DATA / "trajectory.png")
    plt.savefig(save_path)
    print(f"Trajectory plot saved to {save_path}")

    # plt.show()

def experiment(robot: Any, weights=None, tracker=None, duration: int = 15, mode: ViewerTypes = "viewer") -> None:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, SPAWN_POS)

    #place ball
    # Add target marker
    world.spec.worldbody.add_site(
        name="target_site",
        pos=[TARGET_POS[0], TARGET_POS[1], TARGET_POS[2]],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    
    world.spec.worldbody.add_site(
        name="flat_start",
        pos=[-1.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    
    world.spec.worldbody.add_site(
        name="flat_finish",
        pos=[0.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    world.spec.worldbody.add_site(
        name="rug_start",
        pos=[1.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    world.spec.worldbody.add_site(
        name="rug_finish",
        pos=[2.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    world.spec.worldbody.add_site(
        name="inc_start",
        pos=[3.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    world.spec.worldbody.add_site(
        name="inc_finish",
        pos=[4.5, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    world.spec.worldbody.add_site(
        name="finish_start",
        pos=[4.7, 0, 0],
        size=[0.1, 0.1, 0.1],
        rgba=[1, 0, 0, 1],
        type=mj.mjtGeom.mjGEOM_SPHERE,
        )
    

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    num_joints = model.nu
    controller = make_controller_from_weights(weights, num_joints, tracker=tracker)

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "frame":
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            video_recorder = VideoRecorder(output_folder=str(DATA / "videos"))
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        case "launcher":
            viewer.launch(model=model, data=data)

def main() -> None:
    start_time = time.time()
    # --- Load saved robot graph ---
    robot_graph, controller_weights = load_robot(DATA_PATH)
    core = construct_mjspec_from_graph(robot_graph)

    # --- Tracker ---
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    # --- DEBUG controller weights ---
    print("DEBUG: Loaded graph has", len(robot_graph.nodes()), "nodes and", len(robot_graph.edges()), "edges")
    print("DEBUG: Loaded controller weight keys:", controller_weights.keys())

    # --- Run experiment ---
    experiment(robot=core, weights=controller_weights, tracker=tracker, mode="launcher")

    history = tracker.history["xpos"][0]
    show_xpos_history(history)

    fitness = olympic_arena_fitness(history)
    console.log(f"Fitness of generated robot: {fitness}")

    # Debug fitness with plots
    debug_dir = DATA_PATH / f"debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    genome = []
    genome.extend(np.ravel(controller_weights["w1"]))
    genome.extend(controller_weights["b1"])
    genome.extend(np.ravel(controller_weights["w2"]))
    genome.extend(controller_weights["b2"])
    genome.extend(np.ravel(controller_weights["w3"]))
    genome.extend(controller_weights["b3"])

    total_time_run = int(time.time() - start_time)
    plot_best_trajectory(genome, robot_graph, debug_dir, duration=total_time_run)
    plot_best_fitness_over_time(genome, robot_graph, debug_dir, duration=total_time_run)

    print(f"Last pos: {history[-1]}")

if __name__ == "__main__":
    main()
