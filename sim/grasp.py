"""Headless MuJoCo grasp simulator.

Scene: floating parallel-gripper, single box on flat floor.
The gripper drives to (obj_x, obj_y), closes along world-x, lifts.
Whether the grasp succeeds depends on friction, mass, CoM offset, and
the object's yaw (which controls how fingers contact the box faces).
"""
from __future__ import annotations

import numpy as np
import mujoco

# Object geometry (fixed across all scenes; only physics + pose vary)
OBJ_HALF = (0.015, 0.015, 0.025)  # 3cm × 3cm × 5cm box
OBJ_REST_Z = OBJ_HALF[2]            # sit on floor

# Gripper kinematics. Wrist body is positioned at world origin and its
# slide joints carry the entire pose, so position-actuator targets are
# world-frame coordinates. We initialize wrist_z qpos to APPROACH_Z_HIGH.
APPROACH_Z_HIGH = 0.20
# fingers hang 4cm below wrist, half-height 2.5cm → finger center = wrist_z-0.04
# For object center at z=0.025 we want finger center ≈ 0.025 → wrist z ≈ 0.065.
APPROACH_Z_LOW = 0.065
FINGER_OPEN_Q = 0.000   # joint q for fully open (4cm from center)
FINGER_CLOSED_Q = 0.025 # commands fingers ~1.5cm from center (=> squeeze on 1.5cm half-box)

# Time
TIMESTEP = 0.002        # 500 Hz
T_PHASE_DESCEND = 0.6
T_PHASE_CLOSE = 0.4
T_PHASE_LIFT = 1.0
T_PHASE_HOLD = 0.4
T_TOTAL = T_PHASE_DESCEND + T_PHASE_CLOSE + T_PHASE_LIFT + T_PHASE_HOLD  # 2.4s
N_STEPS = int(T_TOTAL / TIMESTEP)

# Trajectory recording
N_RECORD = 120  # ~50Hz
RECORD_INTERVAL = max(1, N_STEPS // N_RECORD)

# Success threshold (height of object above floor after lift)
SUCCESS_Z = 0.10

# Param ranges (used by experiments). Symmetric/sensible physical ranges.
PARAM_BOUNDS = {
    # Group A — geometric (near-known)
    "obj_x":   (-0.02, 0.02),   # 2cm jitter
    "obj_y":   (-0.02, 0.02),
    "obj_yaw": (-0.6, 0.6),     # ±34 deg
    # Group B — physical (hidden)
    "friction": (0.10, 1.20),
    "mass":     (0.05, 0.45),   # 50g–450g
    "com_x":    (-0.010, 0.010) # ±1cm CoM offset along object x
}
GROUP_A = ["obj_x", "obj_y", "obj_yaw"]
GROUP_B = ["friction", "mass", "com_x"]
ALL_PARAMS = GROUP_A + GROUP_B


SCENE_XML_TMPL = """
<mujoco model="grasp">
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 -9.81">
    <flag warmstart="enable"/>
  </option>
  <compiler angle="radian"/>

  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>

  <worldbody>
    <light pos="0 0 1" dir="0 0 -1"/>

    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.8 0.8 0.8 1"
          friction="0.05 0.005 0.0001"/>

    <body name="object" pos="{obj_x} {obj_y} {obj_z}" euler="0 0 {obj_yaw}">
      <freejoint name="obj_free"/>
      <geom name="obj_main" type="box" size="{sx} {sy} {sz}"
            mass="{main_mass}"
            friction="{obj_friction} 0.02 0.001"
            rgba="0.2 0.4 0.8 1"/>
      <geom name="obj_com_blob" type="box" pos="{com_geom_x} 0 0"
            size="0.0015 0.0015 0.0015"
            mass="{offset_mass}"
            contype="0" conaffinity="0"
            rgba="1 0 0 0.8"/>
    </body>

    <body name="wrist" pos="0 0 0">
      <joint name="wrist_x" type="slide" axis="1 0 0" damping="80" limited="false"/>
      <joint name="wrist_y" type="slide" axis="0 1 0" damping="80" limited="false"/>
      <joint name="wrist_z" type="slide" axis="0 0 1" damping="80" limited="false"/>
      <geom name="wrist_body" type="box" size="0.025 0.025 0.015" mass="0.6"
            rgba="0.5 0.5 0.5 1" contype="0" conaffinity="0"/>

      <body name="finger_left" pos="0.04 0 -0.04">
        <joint name="finger_l" type="slide" axis="-1 0 0" damping="5" range="0 0.04"/>
        <geom name="finger_l_pad" type="box" size="0.004 0.018 0.025" mass="0.05"
              friction="0.05 0.005 0.0001" rgba="0.2 0.2 0.2 1"/>
      </body>
      <body name="finger_right" pos="-0.04 0 -0.04">
        <joint name="finger_r" type="slide" axis="1 0 0" damping="5" range="0 0.04"/>
        <geom name="finger_r_pad" type="box" size="0.004 0.018 0.025" mass="0.05"
              friction="0.05 0.005 0.0001" rgba="0.2 0.2 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <contact>
    <pair geom1="obj_main" geom2="finger_l_pad" friction="{obj_friction} {obj_friction} 0.02 0.001 0.001"/>
    <pair geom1="obj_main" geom2="finger_r_pad" friction="{obj_friction} {obj_friction} 0.02 0.001 0.001"/>
    <pair geom1="obj_main" geom2="floor"        friction="{obj_friction} {obj_friction} 0.02 0.001 0.001"/>
  </contact>

  <actuator>
    <position name="wrist_x_act" joint="wrist_x" kp="3000" forcerange="-300 300"/>
    <position name="wrist_y_act" joint="wrist_y" kp="3000" forcerange="-300 300"/>
    <position name="wrist_z_act" joint="wrist_z" kp="3000" forcerange="-300 300"/>
    <position name="finger_l_act" joint="finger_l" kp="800" forcerange="-30 30"/>
    <position name="finger_r_act" joint="finger_r" kp="800" forcerange="-30 30"/>
  </actuator>
</mujoco>
"""


def _split_mass_for_com(total_mass: float, com_x: float, sx: float):
    """Realize CoM offset by splitting the object's mass between a centered
    main geom and a tiny blob geom offset along +x or -x.

    Returns (main_mass, offset_mass, com_geom_x).
    """
    eps = 1e-6
    if abs(com_x) < eps:
        return total_mass, 1e-6, 0.0
    # Place the blob inside the box at ±80% of half-extent
    com_geom_x = float(np.sign(com_x) * sx * 0.8)
    # CoM = (m_blob * com_geom_x) / (m_main + m_blob) = com_x
    # => m_blob = total_mass * com_x / com_geom_x  (works whichever sign)
    blob = total_mass * com_x / com_geom_x
    blob = float(np.clip(blob, 1e-6, total_mass * 0.95))
    main = float(total_mass - blob)
    return main, blob, com_geom_x


def build_xml(params: dict) -> str:
    sx, sy, sz = OBJ_HALF
    main_mass, blob_mass, com_geom_x = _split_mass_for_com(
        float(params["mass"]), float(params["com_x"]), sx
    )
    return SCENE_XML_TMPL.format(
        timestep=TIMESTEP,
        obj_x=float(params["obj_x"]),
        obj_y=float(params["obj_y"]),
        obj_yaw=float(params["obj_yaw"]),
        obj_z=OBJ_REST_Z,
        sx=sx, sy=sy, sz=sz,
        obj_friction=float(params["friction"]),
        main_mass=main_mass,
        offset_mass=blob_mass,
        com_geom_x=com_geom_x,
    )


def _ctrl_for_step(step: int, params: dict) -> np.ndarray:
    """Return desired actuator targets [wx, wy, wz, fl, fr] for a step."""
    n1 = int(T_PHASE_DESCEND / TIMESTEP)
    n2 = n1 + int(T_PHASE_CLOSE / TIMESTEP)
    n3 = n2 + int(T_PHASE_LIFT / TIMESTEP)

    obj_x = float(params["obj_x"])
    obj_y = float(params["obj_y"])

    wx, wy = obj_x, obj_y

    if step < n1:
        alpha = step / max(1, n1)
        wz = APPROACH_Z_HIGH + alpha * (APPROACH_Z_LOW - APPROACH_Z_HIGH)
        fl = fr = FINGER_OPEN_Q
    elif step < n2:
        alpha = (step - n1) / max(1, (n2 - n1))
        wz = APPROACH_Z_LOW
        fl = fr = FINGER_OPEN_Q + alpha * (FINGER_CLOSED_Q - FINGER_OPEN_Q)
    elif step < n3:
        alpha = (step - n2) / max(1, (n3 - n2))
        wz = APPROACH_Z_LOW + alpha * (APPROACH_Z_HIGH - APPROACH_Z_LOW)
        fl = fr = FINGER_CLOSED_Q
    else:
        wz = APPROACH_Z_HIGH
        fl = fr = FINGER_CLOSED_Q

    return np.array([wx, wy, wz, fl, fr], dtype=np.float64)


def run_grasp(params: dict, record: bool = True, seed: int | None = None) -> dict:
    """Run one grasp episode. Returns success flag and (optionally) trajectory.

    params: dict with keys obj_x, obj_y, obj_yaw, friction, mass, com_x.
    """
    xml = build_xml(params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    if seed is not None:
        np.random.seed(seed)

    obj_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    # qpos layout: obj freejoint [0:7], then wrist_x [7], wrist_y [8],
    # wrist_z [9], finger_l [10], finger_r [11]. Lift the wrist to its
    # starting height so it doesn't sweep up through the table first.
    data.qpos[9] = APPROACH_Z_HIGH
    mujoco.mj_forward(model, data)

    traj_pos = []
    traj_quat = []
    traj_t = []

    for step in range(N_STEPS):
        data.ctrl[:] = _ctrl_for_step(step, params)
        mujoco.mj_step(model, data)
        if record and (step % RECORD_INTERVAL == 0):
            traj_pos.append(data.xpos[obj_body_id].copy())
            traj_quat.append(data.xquat[obj_body_id].copy())
            traj_t.append(step * TIMESTEP)

    final_obj_z = float(data.xpos[obj_body_id][2])
    success = bool(final_obj_z > SUCCESS_Z)

    return {
        "success": success,
        "final_obj_z": final_obj_z,
        "traj_pos": np.asarray(traj_pos) if record else None,
        "traj_quat": np.asarray(traj_quat) if record else None,
        "traj_t": np.asarray(traj_t) if record else None,
    }


def sample_uniform_params(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sample N param vectors uniformly within PARAM_BOUNDS. Shape (n, 6)."""
    out = np.zeros((n, len(ALL_PARAMS)))
    for i, k in enumerate(ALL_PARAMS):
        lo, hi = PARAM_BOUNDS[k]
        out[:, i] = rng.uniform(lo, hi, size=n)
    return out


def params_to_dict(vec: np.ndarray) -> dict:
    return {k: float(vec[i]) for i, k in enumerate(ALL_PARAMS)}


def dict_to_vec(d: dict) -> np.ndarray:
    return np.array([d[k] for k in ALL_PARAMS], dtype=np.float64)
