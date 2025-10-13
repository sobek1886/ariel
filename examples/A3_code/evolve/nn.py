import numpy as np
from typing import Optional, Sequence, Iterable
from ariel.simulation.controllers.controller import Controller
from examples.A3_code.evolve.config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, TARGET_POS, STATE_FEATURES_RICH, STATE_FEATURES_MINIMAL
)

def _decode_1layer(genome: np.ndarray, input_size: int, hidden_size: int, output_size: int):
    """input -> hidden -> output"""
    idx = 0
    w1 = np.array(genome[idx: idx + input_size * hidden_size]).reshape(input_size, hidden_size)
    idx += input_size * hidden_size
    b1 = np.array(genome[idx: idx + hidden_size])
    idx += hidden_size
    w2 = np.array(genome[idx: idx + hidden_size * output_size]).reshape(hidden_size, output_size)
    idx += hidden_size * output_size
    b2 = np.array(genome[idx: idx + output_size])
    idx += output_size
    return (w1, b1, w2, b2)


def _decode_2layer(genome: np.ndarray, input_size: int, hidden_size: int, hidden_size2: int, output_size: int):
    """input -> hidden(=hidden_size) -> hidden2(=hidden_size2) -> output"""
    idx = 0
    w1 = np.array(genome[idx: idx + input_size * hidden_size]).reshape(input_size, hidden_size)
    idx += input_size * hidden_size
    b1 = np.array(genome[idx: idx + hidden_size])
    idx += hidden_size

    w2 = np.array(genome[idx: idx + hidden_size * hidden_size2]).reshape(hidden_size, hidden_size2)
    idx += hidden_size * hidden_size2
    b2 = np.array(genome[idx: idx + hidden_size2])
    idx += hidden_size2

    w3 = np.array(genome[idx: idx + hidden_size2 * output_size]).reshape(hidden_size2, output_size)
    idx += hidden_size2 * output_size
    b3 = np.array(genome[idx: idx + output_size])
    idx += output_size

    return (w1, b1, w2, b2, w3, b3)


def decode_genome(
    genome,
    input_size: int,
    hidden_size: int,
    output_size: int,
    *,
    depth: int = 1,
    hidden_size2: Optional[int] = None,
):
    """
    Turn a flat genome vector into NN weights and biases.

    depth=1  -> returns (w1, b1, w2, b2)
    depth=2  -> returns (w1, b1, w2, b2, w3, b3)  (requires hidden_size2)
    """
    g = np.asarray(genome, dtype=np.float32)
    if depth == 1:
        return _decode_1layer(g, input_size, hidden_size, output_size)
    elif depth == 2:
        if hidden_size2 is None:
            raise ValueError("decode_genome(depth=2) requires hidden_size2.")
        return _decode_2layer(g, input_size, hidden_size, hidden_size2, output_size)
    else:
        raise ValueError(f"Unsupported depth={depth}. Use 1 or 2.")


def build_controller(
    genome,
    input_size: int,
    hidden_size: int,
    output_size: int,
    *,
    depth: int = 1,
    hidden_size2: Optional[int] = None,
):
    """
    Returns a controller function(state) -> action using decoded weights.

    depth=1: input -> hidden -> output
    depth=2: input -> hidden -> hidden2 -> output
    """
    if depth == 1:
        w1, b1, w2, b2 = decode_genome(genome, input_size, hidden_size, output_size, depth=1)
        def controller(state: np.ndarray) -> np.ndarray:
            h = np.tanh(np.dot(state, w1) + b1)
            out = np.tanh(np.dot(h, w2) + b2)
            return out * (np.pi / 2)
        return controller

    elif depth == 2:
        if hidden_size2 is None:
            raise ValueError("build_controller(depth=2) requires hidden_size2.")
        w1, b1, w2, b2, w3, b3 = decode_genome(
            genome, input_size, hidden_size, output_size, depth=2, hidden_size2=hidden_size2
        )
        def controller(state: np.ndarray) -> np.ndarray:
            h1 = np.tanh(np.dot(state, w1) + b1)
            h2 = np.tanh(np.dot(h1, w2) + b2)
            out = np.tanh(np.dot(h2, w3) + b3)
            return out * (np.pi / 2)
        return controller

    else:
        raise ValueError(f"Unsupported depth={depth}. Use 1 or 2.")


def genome_length(
    input_size: int,
    hidden_size: int,
    output_size: int,
    *,
    depth: int = 1,
    hidden_size2: Optional[int] = None,
) -> int:
    """
    Compute required length of a genome vector for the chosen NN architecture.
    """
    if depth == 1:
        # (in*hid) + hid + (hid*out) + out
        return (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
    elif depth == 2:
        if hidden_size2 is None:
            raise ValueError("genome_length(depth=2) requires hidden_size2.")
        # (in*h1) + h1 + (h1*h2) + h2 + (h2*out) + out
        return (
            (input_size * hidden_size) + hidden_size +
            (hidden_size * hidden_size2) + hidden_size2 +
            (hidden_size2 * output_size) + output_size
        )
    else:
        raise ValueError(f"Unsupported depth={depth}. Use 1 or 2.")

def make_controller_from_genome(
    genome,
    num_joints: int,
    *,
    record_pos: list[np.ndarray] | None = None,
    ctrl_every: int = 1,
    save_every: int = 1,
    alpha: float = 1.0,
    tracker=None,
) -> Controller:
    """
    Wrap build_controller(genome, ...) into an Ariel Controller.
    - record_pos: optional list to collect data.qpos[:3] each control step.
    - ctrl_every/save_every=1 keeps identical behavior to old loop.
    """
    input_size = infer_input_size(num_joints, STATE_FEATURES)
    action_fn = build_controller(
        genome, input_size, HIDDEN_SIZE, num_joints,
        depth=NN_DEPTH, hidden_size2=HIDDEN_SIZE2
    )

    def _callback(model, data):
        state = get_state_vector(
            data, num_joints, STATE_FEATURES
        ).astype(np.float32)
        if record_pos is not None:
            record_pos.append(data.qpos[:3].copy())
        return action_fn(state)

    return Controller(
        controller_callback_function=_callback,
        time_steps_per_ctrl_step=ctrl_every,
        time_steps_per_save=save_every,
        alpha=alpha,
        tracker=tracker,
    )


def make_controller_from_weights(
    weights: dict[str, np.ndarray],
    num_joints: int,
    *,
    record_pos: list[np.ndarray] | None = None,
    ctrl_every: int = 1,
    save_every: int = 1,
    alpha: float = 1.0,
    tracker=None,
) -> Controller:
    """
    Make a Controller from saved weight matrices.
    Supports depth=1 (w1,b1,w2,b2) and depth=2 (w1,b1,w2,b2,w3,b3).
    """
    # CRITICAL: Convert lists to numpy arrays with correct dtype
    w1 = np.array(weights["w1"], dtype=np.float32)
    b1 = np.array(weights["b1"], dtype=np.float32)
    w2 = np.array(weights["w2"], dtype=np.float32)
    b2 = np.array(weights["b2"], dtype=np.float32)
    w3 = np.array(weights["w3"], dtype=np.float32) if "w3" in weights else None
    b3 = np.array(weights["b3"], dtype=np.float32) if "b3" in weights else None

    def _forward(state: np.ndarray) -> np.ndarray:
        # Match the exact logic from build_controller
        h1 = np.tanh(np.dot(state, w1) + b1)  # Remove None check
        if w3 is None:
            out = np.tanh(np.dot(h1, w2) + b2)
        else:
            h2 = np.tanh(np.dot(h1, w2) + b2)
            out = np.tanh(np.dot(h2, w3) + b3)
        return out * (np.pi / 2)

    def _callback(model, data):
        # Remove .copy() to match exactly
        state = get_state_vector(
            data, num_joints, STATE_FEATURES
        ).astype(np.float32)
        if record_pos is not None:
            record_pos.append(data.qpos[:3].copy())
        return _forward(state)

    return Controller(
        controller_callback_function=_callback,
        time_steps_per_ctrl_step=ctrl_every,
        time_steps_per_save=save_every,
        alpha=alpha,
        tracker=tracker,
    )
    
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
    parts.append(np.asarray(TARGET_POS) - data.qpos[:3])  # direction to target

    if features == STATE_FEATURES_RICH:
        # --- Rich extras ---
        parts.append(data.xpos[0])           # root pos
        parts.append(data.xquat[0])          # root orientation
        parts.append(data.cfrc_ext[0])       # external forces on root
        parts.append(data.subtree_com[0])    # global COM
        parts.append(data.qfrc_bias[:num_joints])     # bias forces per joint
        parts.append(data.qfrc_actuator[:num_joints]) # actuator forces per joint

    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
