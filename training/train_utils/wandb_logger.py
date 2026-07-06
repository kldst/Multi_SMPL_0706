import atexit
import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from .distributed import get_machine_local_and_dist_rank
from .general import safe_makedirs


class WandbLogger:
    """A minimal logger wrapper exposing the same interface as TensorBoardLogger."""

    def __init__(
        self,
        path: str,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        mode: str = "online",
        resume: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._run = None
        self._path = path
        _, self._rank = get_machine_local_and_dist_rank()

        if self._rank != 0:
            logging.debug("Not logging to Weights & Biases because rank %s != 0", self._rank)
            return

        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "wandb is not installed. Install it with `pip install wandb` to use logging.use_wandb=True."
            ) from exc

        safe_makedirs(path)
        logging.info("Weights & Biases run directory: %s", path)
        self._run = wandb.init(
            dir=path,
            project=project,
            entity=entity,
            name=name,
            mode=mode,
            resume=resume,
            config=config,
            **kwargs,
        )

        # Define separate step metrics for train and val.
        wandb.define_metric("train/step", hidden=True, summary="none")
        wandb.define_metric("val/step", hidden=True, summary="none")

        wandb.define_metric("Values/train/*", step_metric="train/step")
        wandb.define_metric("Values/val/*", step_metric="val/step")
        wandb.define_metric("Visuals/train*", step_metric="train/step")
        wandb.define_metric("Visuals/val*", step_metric="val/step")
        wandb.define_metric("Optim/*", step_metric="train/step")

        atexit.register(self.close)

    @property
    def path(self) -> str:
        return self._path

    def flush(self) -> None:
        if self._run is not None:
            self._run.log({}, commit=False)

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None

    def _build_payload(self, name: str, data: Any, step: int) -> Dict[str, Any]:
        payload: Dict[str, Any] = {name: data}

        if name.startswith("Values/train/") or name.startswith("Optim/") or name.startswith("Visuals/train"):
            payload["train/step"] = step
        elif name.startswith("Values/val/") or name.startswith("Visuals/val"):
            payload["val/step"] = step
        else:
            # Fallback: still log without explicit custom step axis.
            # Avoid passing wandb global step here.
            payload["train/step"] = step

        return payload

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if self._run is None:
            return

        merged_payload: Dict[str, Any] = {}
        for name, value in payload.items():
            item_payload = self._build_payload(name, value, step)
            merged_payload.update(item_payload)

        self._run.log(merged_payload)

    def log(self, name: str, data: Any, step: int) -> None:
        if self._run is None:
            return
        payload = self._build_payload(name, data, step)
        self._run.log(payload)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, np.ndarray, Any],
        step: int,
        fps: int = 4,
    ) -> None:
        if self._run is None:
            return

        import wandb

        if torch.is_tensor(data):
            data = data.detach().cpu().numpy()

        if data.ndim == 3:
            payload = self._build_payload(name, wandb.Image(data), step)
            self._run.log(payload)
        elif data.ndim == 5:
            video = data[0] if data.shape[0] > 1 else data.squeeze(0)
            payload = self._build_payload(name, wandb.Video(video, fps=fps), step)
            self._run.log(payload)
        else:
            raise ValueError(
                f"Unsupported data dimensions: {data.ndim}. Expected 3D for images or 5D for videos."
            )