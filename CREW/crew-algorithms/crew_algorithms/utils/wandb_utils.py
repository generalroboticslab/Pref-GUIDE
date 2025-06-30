from attrs import define
from omegaconf import MISSING


@define(auto_attribs=True)
class WandbConfig:
    entity: str = "grl-crew"
    """The entity to set in WandB."""
    project: str = MISSING
    """The project to set in WandB."""
