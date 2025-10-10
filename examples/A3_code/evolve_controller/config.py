import numpy as np
from typing import Iterable, Sequence, Tuple, Optional

# ===== Experiment Parameters =====
SPAWN_POS: Tuple[float, float, float] = (-0.8, 0.0, 0.1)
TARGET_POS: Tuple[float, float, float] = (5.0, 0.0, 0.5)   # full 3D

DURATION = 60
NUM_POP = 100
NUM_GENS = 50

# ===== Feature set toggle =====
# Can be "minimal" or "rich"
STATE_SET = "rich"   # change to "rich" to include extra dynamics features

# Default feature sets
STATE_FEATURES_MINIMAL: Sequence[str] = (
    "joint_pos", "joint_vel", "torso_quat", "torso_vel", "target_dir"
)
STATE_FEATURES_RICH: Sequence[str] = (
    "joint_pos", "joint_vel", "torso_quat", "torso_vel", "target_dir",
    "root_pos", "root_quat", "root_ext_forces", "subtree_com",
    "qfrc_bias", "qfrc_actuator"
)

def active_features() -> Sequence[str]:
    """Return feature set depending on STATE_SET."""
    if STATE_SET == "minimal":
        return STATE_FEATURES_MINIMAL
    elif STATE_SET == "rich":
        return STATE_FEATURES_RICH
    else:
        raise ValueError(f"Unknown STATE_SET {STATE_SET}")

# Preserve older code paths that import STATE_FEATURES directly.
STATE_FEATURES: Sequence[str] = active_features()

# ===== Network depth/width =====
# Depth:
#   1 -> input -> HIDDEN_SIZE -> output
#   2 -> input -> HIDDEN_SIZE -> HIDDEN_SIZE2 -> output
NN_DEPTH: int = 2        # you asked for deeper by default
HIDDEN_SIZE: int = 64
HIDDEN_SIZE2: Optional[int] = 32 if NN_DEPTH == 2 else None

# ===== Saving / Plotting Options =====
SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_LOGS = True


def infer_input_size(num_joints: int, features: Iterable[str]) -> int:
    feats = set(features)
    size = 0
    if "joint_pos"        in feats: size += num_joints
    if "joint_vel"        in feats: size += num_joints
    if "torso_quat"       in feats: size += 4
    if "torso_vel"        in feats: size += 6
    if "target_dir"       in feats: size += 3   # full 3D direction
    if "root_pos"         in feats: size += 3
    if "root_quat"        in feats: size += 4
    if "root_ext_forces"  in feats: size += 6
    if "subtree_com"      in feats: size += 3   # only the global COM (body 0)
    if "qfrc_bias"        in feats: size += num_joints     # trimmed to num_joints for stability
    if "qfrc_actuator"    in feats: size += num_joints
    return size


def get_state_vector(
    data,
    num_joints: int,
    features: Sequence[str],
    target_pos: Tuple[float, float, float] = TARGET_POS
) -> np.ndarray:
    """
    Build the state vector.
    - "minimal": proprioception + torso state + target dir.
    - "rich": adds contact forces, subtree COM, bias/actuator forces.
    """

    parts = []

    # --- Minimal state ---
    parts.append(data.qpos[7:7+num_joints])        # joint positions
    parts.append(data.qvel[6:6+num_joints])        # joint velocities
    parts.append(data.qpos[3:7])                   # torso orientation (quat)
    parts.append(data.qvel[:6])                    # torso linear+angular vel
    parts.append(np.asarray(target_pos) - data.qpos[:3])  # direction to target

    if features == STATE_FEATURES_RICH:
        # --- Rich extras ---
        parts.append(data.xpos[0])           # root pos
        parts.append(data.xquat[0])          # root orientation
        parts.append(data.cfrc_ext[0])       # external forces on root
        parts.append(data.subtree_com[0])    # global COM
        parts.append(data.qfrc_bias[:num_joints])     # bias forces per joint
        parts.append(data.qfrc_actuator[:num_joints]) # actuator forces per joint

    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
