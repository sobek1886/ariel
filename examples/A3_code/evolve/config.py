import numpy as np
from typing import Sequence, Tuple, Optional
from pathlib import Path

# ===== Experiment Parameters =====
SPAWN_POS: Tuple[float, float, float] = (-0.8, 0.0, 0.1)
TARGET_POS: Tuple[float, float, float] = (5.0, 0.0, 0.5)

IMMOBILE_THRESH = 0.1  # meters — below this, the robot is considered immobile
MUT_SIGMA, MUT_INDPB = 0.05, 0.1
CX_PROB, MUT_PROB = 0.5, 0.3
TOURNAMENT_SIZE = 5

DURATION = 400
NUM_POP = 200
NUM_GENS = 100

STATE_SET = "rich"
TASK = "nav"

# ===== Controller Type =====
USE_CPG = True  # Set to True to use CPG instead of MLP
CPG_HYBRID = True  # If USE_CPG=True, use hybrid CPG+MLP (recommended)
USE_LAYER_NORM = False  # Layer normalization for MLP

DATA_PATH = Path.cwd() / "__robot_data__" / "saved_robots" / "co_evo_run"

STATE_FEATURES_MINIMAL: Sequence[str] = (
    "joint_pos", "joint_vel", "torso_quat", "torso_vel", "target_dir"
)
STATE_FEATURES_RICH: Sequence[str] = (
    "joint_pos", "joint_vel", "torso_quat", "torso_vel", "target_dir",
    "root_pos", "root_quat", "root_ext_forces", "subtree_com",
    "qfrc_bias", "qfrc_actuator"
)
STATE_FEATURES: Sequence[str] = (
    STATE_FEATURES_RICH if STATE_SET == "rich" else STATE_FEATURES_MINIMAL
)

NN_DEPTH: int = 2
HIDDEN_SIZE: int = 64
HIDDEN_SIZE2: Optional[int] = 32 if NN_DEPTH == 2 else None


SAVE_PLOTS = True
SAVE_LOGS = True

# ===== Body genome sizing =====
DEBUG_PROGRESS = True
GENOTYPE_SIZE = 64
NUM_OF_MODULES = 30
BODY_GENE_LENGTH = GENOTYPE_SIZE * 3