# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.amphitheatre_heightmap import AmphitheatreTerrainWorld
from ariel.simulation.environments.djoser_pyramid import PyramidWorld


# Import gate learning task functions
from ariel.simulation.tasks.gate_learning import xy_displacement, x_speed, y_speed

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# import prebuilt controller
from ariel.simulation.controllers.hopfs_cpg import HopfCPG

# Keep track of data / history
HISTORY = []

def random_move(model, data, to_track) -> None:
    """Generate random movements for the robot's joints.
    
    The mujoco.set_mjcb_control() function will always give 
    model and data as inputs to the function. Even if you don't use them,
    you need to have them as inputs.

    Parameters
    ----------

    model : mujoco.MjModel
        The MuJoCo model of the robot.
    data : mujoco.MjData
        The MuJoCo data of the robot.

    Returns
    -------
    None
        This function modifies the data.ctrl in place.
    """

    # Get the number of joints
    num_joints = model.nu 
    
    # Hinges take values between -pi/2 and pi/2
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range, # -pi/2
                                   high=hinge_range, # pi/2
                                   size=num_joints) 

    # There are 2 ways to make movements:
    # 1. Set the control values directly (this might result in junky physics)
    # data.ctrl = rand_moves

    # 2. Add to the control values with a delta (this results in smoother physics)
    delta = 0.05
    data.ctrl += rand_moves * delta 

    # Bound the control values to be within the hinge limits.
    # If a value goes outside the bounds it might result in jittery movement.
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)

    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())

    ##############################################
    #
    # Take all the above into consideration when creating your controller
    # The input size, output size, output range
    # Your network might return ranges [-1,1], so you will need to scale it
    # to the expected [-pi/2, pi/2] range.
    # 
    # Or you might not need a delta and use the direct controller outputs
    #
    ##############################################

def hopf_cpg_controller_builder(model, to_track):
    """
    Build a Hopf CPG controller callback for MuJoCo.

    Parameters
    ----------
    model : mujoco.MjModel
        The MuJoCo model of the robot.
    to_track : list
        List of geoms to track (not used here, but kept for API consistency).

    Returns
    -------
    callable
        A function (model, data) -> None that updates data.ctrl.
    """
    num_joints = model.nu

    # Example: ring topology where each joint connects to the next
    adjacency_list = {i: [(i + 1) % num_joints] for i in range(num_joints)}

    # Instantiate the Hopf CPG
    cpg = HopfCPG(num_neurons=num_joints,
                  adjacency_list=adjacency_list,
                  dt=model.opt.timestep)
    
    # Change parameters manually
    #cpg.omega[:] = 2 * np.pi * 1   # slower oscillation (0.5 Hz instead of 1 Hz)
    cpg.A[:] = 10                   # bigger amplitude
    #cpg.h = 0.3                      # stronger coupling

    def controller(model, data):
        x, y = cpg.step()
        data.ctrl[:] = np.tanh(x) * (np.pi / 2)

        # Save trajectory like in random_move
        HISTORY.append(to_track[0].xpos.copy())

    return controller


def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def main():
    """Main function to run the simulation with random movements."""
    # Initialise controller to controller to None, always in the beginning.
    mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()
    #world = RuggedTerrainWorld()
    #world = PyramidWorld()
    #world = AmphitheatreTerrainWorld()

    # Initialise robot body
    # YOU MUST USE THE GECKO BODY
    gecko_core = gecko()     # DO NOT CHANGE

    # Spawn robot in the world
    # Check docstring for spawn conditions
    #world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    world.spawn(gecko_core.spec)

    
    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model) # type: ignore

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # Record initial state
    start_xy = tuple(to_track[0].xpos[:2])

    # Set the control callback function
    # This is called every time step to get the next action. 

    # Random controller
    #mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))
    # Hopf CPG controller
    mujoco.set_mjcb_control(hopf_cpg_controller_builder(model, to_track))

    '''# Run the simulation for fixed steps
    steps = 10000
    for _ in range(steps):
        mujoco.mj_step(model, data)

    # Show the XY path
    show_qpos_history(HISTORY)

    # Record final state and evaluate fitness
    end_xy = tuple(to_track[0].xpos[:2])
    dt = model.opt.timestep * steps

    displacement = xy_displacement(start_xy, end_xy)
    speed_x = x_speed(start_xy, end_xy, dt)
    speed_y = y_speed(start_xy, end_xy, dt)

    print(f"Displacement: {displacement:.4f}")
    print(f"X speed:      {speed_x:.4f}")
    print(f"Y speed:      {speed_y:.4f}")'''

    # This opens a viewer window and runs the simulation with the controller you defined
    # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    viewer.launch(
        model=model,  # type: ignore
        data=data,
    )

    show_qpos_history(HISTORY)
    # If you want to record a video of your simulation, you can use the video renderer.

    # # Non-default VideoRecorder options
    # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # Render with video recorder
    # video_renderer(
    #     model,
    #     data,
    #     duration=30,
    #     video_recorder=video_recorder,
    # )

if __name__ == "__main__":
    main()


