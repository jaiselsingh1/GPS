import time
import pathlib
import numpy as np
from mppi_gps.envs.xarm7_env import Xarm7
# from mppi_gps.envs.robosuite_xarm7 import make_pick_place_model

cur_path = pathlib.Path(__file__)
mppi_gps_dir = cur_path.parent.parent

# point to the actual models/assets location
xml_rel_path = mppi_gps_dir / "src/mppi_gps/models/assets/plate_dishwasher_task.xml"
xml_rel_path.parent.mkdir(parents=True, exist_ok=True)

# make_pick_place_model(save_xml_path=xml_rel_path)

# point Xarm7 at that XML - use the path relative to src/mppi_gps
env = Xarm7(model_path=xml_rel_path, render_mode="human")

steps = 1000
obs, _ = env.reset()
action = np.zeros_like(env.action_space.sample())

for i in range(steps):
    if i % 10 == 0 and action.shape[0] > 7:
        action[7] = 255.0
    env.step(action)
    time.sleep(0.002)
    env.render()