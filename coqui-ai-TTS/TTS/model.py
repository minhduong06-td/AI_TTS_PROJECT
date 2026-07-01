import os
from abc import abstractmethod
from typing import Any
import torch
from coqpit import Coqpit
from trainer import TrainerModel
from trainer.io import load_fsspec

from TTS.config.shared_configs import BaseTrainingConfig


class BaseTrainerModel(TrainerModel):
    config: BaseTrainingConfig

    @staticmethod
    @abstractmethod
    def init_from_config(config: Coqpit) -> "BaseTrainerModel":
        ...

    @abstractmethod
    def inference(self, input: torch.Tensor, aux_input: dict[str, Any] = {}) -> dict[str, Any]:
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str | os.PathLike[Any],
        *,
        eval: bool = False,
        strict: bool = True,
        cache: bool = False,
        **kwargs: Any,
    ) -> None:
        state = load_fsspec(checkpoint_path, map_location="cpu", cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
