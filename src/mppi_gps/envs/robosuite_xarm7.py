from robosuite.models import MujocoWorldBase 
from robosuite.models.robots.manipulators import XArm7
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject
import pathlib
from jaxtyping import Float, Array
import numpy as np
import mujoco
from robosuite.models import MujocoWorldBase 
# from robosuite.models.robots.manipulators import XArm7
from mppi_gps.envs.xarm7_env import Xarm7
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CylinderObject, BoxObject
import pathlib
from jaxtyping import Float, Array
import numpy as np
import mujoco

cur_path = pathlib.Path(__file__)
mppi_gps_dir = cur_path.parent.parent

def make_pick_place_model(
    save_xml_path: str | None = None,  # Changed default to None
    robot_base: Float[Array, "3"] = np.array([0.0, 0.0, 0.0]),
    table_translation: Float[Array, "3"] = np.array([0.8, 0.0, 0.0]),
) -> mujoco.MjModel:
    world = MujocoWorldBase()
    
    # Add robot
    robot = XArm7()
    gripper = gripper_factory("XArm7Gripper")
    robot.add_gripper(gripper)
    robot.set_base_xpos(robot_base)
    world.merge(robot)
    
    # Add table
    table_origin = robot_base + table_translation
    arena = TableArena()
    arena.set_origin(table_origin)
    world.merge(arena)
    
    # Object parameters
    radius = 0.04
    table_height = 0.8
    z_obj = table_height + radius
    
    # Create and merge can (cylinder)
    can = CylinderObject(
        name="can",
        size=[radius, radius * 1.2],  # [radius, half-height]
        rgba=[0.9, 0.2, 0.2, 1.0],
    )
    can_body = can.get_obj()
    can_body.set("pos", f"{table_origin[0]} {table_origin[1]} {z_obj}")
    world.merge_assets(can)
    world.worldbody.append(can_body)
    
    # Create and merge box
    box = BoxObject(
        name="box",
        size=[radius, radius, radius * 0.5],  # [half-x, half-y, half-z]
        rgba=[0.2, 0.6, 0.9, 1.0],
    )
    box_body = box.get_obj()
    box_body.set("pos", f"{table_origin[0] - 0.1} {table_origin[1] + 0.1} {z_obj}")
    world.merge_assets(box)
    world.worldbody.append(box_body)
    
    # Get MuJoCo model
    model = world.get_model(mode="mujoco")
    
    # Save if path provided (must be a .xml file path, not directory)
    if save_xml_path is not None:
        world.save_model(save_xml_path)
    
    return model
cur_path = pathlib.Path(__file__)
mppi_gps_dir = cur_path.parent.parent

def make_pick_place_model(
    save_xml_path: str | None = None,  # Changed default to None
    robot_base: Float[Array, "3"] = np.array([0.0, 0.0, 0.0]),
    table_translation: Float[Array, "3"] = np.array([0.8, 0.0, 0.0]),
) -> mujoco.MjModel:
    world = MujocoWorldBase()
    
    # Add robot
    robot = Xarm7()
    gripper = gripper_factory("XArm7Gripper")
    robot.add_gripper(gripper)
    robot.set_base_xpos(robot_base)
    world.merge(robot)
    
    # Add table
    table_origin = robot_base + table_translation
    arena = TableArena()
    arena.set_origin(table_origin)
    world.merge(arena)
    
    # Object parameters
    radius = 0.04
    table_height = 0.8
    z_obj = table_height + radius
    
    # Create and merge can (cylinder)
    can = CylinderObject(
        name="can",
        size=[radius, radius * 1.2],  # [radius, half-height]
        rgba=[0.9, 0.2, 0.2, 1.0],
    )
    can_body = can.get_obj()
    can_body.set("pos", f"{table_origin[0]} {table_origin[1]} {z_obj}")
    world.merge_assets(can)
    world.worldbody.append(can_body)
    
    # Create and merge box
    box = BoxObject(
        name="box",
        size=[radius, radius, radius * 0.5],  # [half-x, half-y, half-z]
        rgba=[0.2, 0.6, 0.9, 1.0],
    )
    box_body = box.get_obj()
    box_body.set("pos", f"{table_origin[0] - 0.1} {table_origin[1] + 0.1} {z_obj}")
    world.merge_assets(box)
    world.worldbody.append(box_body)
    
    # Get MuJoCo model
    model = world.get_model(mode="mujoco")
    
    # Save if path provided (must be a .xml file path, not directory)
    if save_xml_path is not None:
        world.save_model(save_xml_path)
    
    return model