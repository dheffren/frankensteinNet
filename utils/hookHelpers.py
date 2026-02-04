from enum import Enum, auto
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional
class Trigger(Enum):
    TRAIN_BEGIN = auto()
    EPOCH_BEGIN = auto()
    BEFORE_STEP = auto()
    AFTER_FORWARD = auto()
    AFTER_BACKWARD = auto()
    BEFORE_OPT_STEP = auto()
    AFTER_OPT_STEP = auto()
    LR_STEP = auto()
    EPOCH_END = auto()
    EVAL_BEGIN = auto()
    EVAL_STEP = auto()
    EVAL_END = auto()
    TRAIN_END = auto()
    EXCEPTION = auto()

@dataclass(frozen=True)
class StepCtx:
    phase: str         # "train" | "val"
    epoch: int
    step: int          # global train step (train-only)
    batch_idx: int     # -1 at non-step triggers
    lr: Any            # float or dict for param groups
    loss: float | None
    metrics: dict      # shallow scalars only\
    trigger: str
    hook: Optional[str]
@dataclass # TODO: What is this for? 
class Services:
    model: Any                 # torch.nn.Module (read/write allowed)
    train_loader: Any | None
    val_loader: Any | None
    logger: Any                # must expose .log(step, dict) and .artifact(path,name?)
    cfg: dict
    meta: dict
    run_dir: str
    # Optional helpers the trainer provides to hooks:
    run_eval: Callable[[str | None], dict]    # e.g., evaluate current model on named split
    checkpoint: Callable[[str], str]          # save a tagged checkpoint
    device: str                               # "cuda" or "cpu"
@dataclass(frozen = True)
class ACtx:  # minimal ArtifactContext shape
    run_id: str; epoch: int; step: int; trigger: str; hook: str; split: str|None=None