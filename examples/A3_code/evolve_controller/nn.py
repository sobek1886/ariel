import numpy as np
from typing import Optional
from ariel.simulation.controllers.controller import Controller
from .config import (
    STATE_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE2, NN_DEPTH, TARGET_POS,
    get_state_vector, infer_input_size
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
    target_pos=TARGET_POS,
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
            data, num_joints, STATE_FEATURES, target_pos=target_pos
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
    target_pos=TARGET_POS,
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
    w1 = weights["w1"]; b1 = weights.get("b1")
    w2 = weights["w2"]; b2 = weights.get("b2")
    w3 = weights.get("w3"); b3 = weights.get("b3")

    def _forward(state: np.ndarray) -> np.ndarray:
        # print(f"state shape: {state.shape}, w1 shape: {w1.shape}")
        h1 = np.tanh(state @ w1 + (b1 if b1 is not None else 0))
        if w3 is None:
            out = np.tanh(h1 @ w2 + (b2 if b2 is not None else 0))
        else:
            h2 = np.tanh(h1 @ w2 + (b2 if b2 is not None else 0))
            out = np.tanh(h2 @ w3 + (b3 if b3 is not None else 0))
        return out * (np.pi / 2)

    def _callback(model, data):
        state = get_state_vector(
            data, num_joints, STATE_FEATURES, target_pos=target_pos
        ).astype(np.float32).copy()
        # print("State vector:", state)
        # print("infer_input_size:", infer_input_size(num_joints, STATE_FEATURES))
        # print("actual state length:", len(get_state_vector(data, num_joints, STATE_FEATURES)))
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