"""W&B integrations."""

from typing import Any, Dict, Mapping, Tuple, Union

import stable_baselines3.common.logger as sb_logger
import wandb


class WandbOutputFormat(sb_logger.KVWriter):
    """A stable-baseline logger that writes to wandb."""

    def __init__(
        self,
        wb_options: Mapping[str, Any],
        config: Mapping[str, Any],
    ):
        """Builds WandbOutputFormat.

        Args:
            wb_options: A dictionary of options to pass to wandb.init.
            config: A dictionary of config values to log to wandb.

        """
        wandb.init(config=config, **wb_options)

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()),
            sorted(key_excluded.items()),
        ):
            if excluded is not None and "wandb" in excluded:
                continue
            wandb.log({key: value}, step=step)

    def close(self) -> None:
        wandb.finish()
