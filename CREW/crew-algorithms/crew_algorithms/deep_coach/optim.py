import torch
from attrs import define
from tensordict import TensorDictBase
from torch.optim import Optimizer


@define(auto_attribs=True)
class DeepCoachConfig:
    _target_: str = "crew_algorithms.deep_coach.optim.DeepCoach"
    _partial_: bool = True
    decay_rate: float = 0.35
    """Exponential decay of the rate at which to decay past
    policy gradients. Must be between (0, 1)."""
    entropy_coefficient: float = 1.5
    """How much importance should be given to the entropy
    regularization."""
    learning_rate: float = 2.5e-2
    """Rate at which to learn."""


class DeepCoach(Optimizer):
    def __init__(
        self,
        params,
        decay_rate: float,
        entropy_coefficient: float,
        learning_rate: float,
    ) -> None:
        """Creates an optimizer as specified by the DeepCoach paper.

        This optimizer is novel in that it works directly with human feedback by
        treating it as a critique of the current policy rather than just a reward
        signal. Please see the paper for more details:
        https://arxiv.org/pdf/1902.04257.pdf.

        Args:
            params: Parameters to be optimized.
            decay_rate: Exponential decay of the rate at which to decay past
                policy gradients. Must be between (0, 1).
            entropy_coefficient: How much importance should be given to the
                entropy regularization.
            learning_rate: Rate at which to learn.
        """
        defaults = dict(
            decay_rate=decay_rate,
            entropy_weight=entropy_coefficient,
            learning_rate=learning_rate,
        )
        super().__init__(params, defaults)
        for idx, group in enumerate(self.param_groups):
            group["name"] = f"GROUP-{idx}"
        self._eligibility_trace = self._create_eligibility_trace()
        self.batch_len = 0

    @torch.no_grad()
    def _create_eligibility_trace(self):
        eligibility_trace = {
            group["name"]: [torch.zeros_like(p) for p in group["params"]]
            for group in self.param_groups
        }
        return eligibility_trace

    def update_eligibility_trace(self, window: list[TensorDictBase]):
        """Updates the eligibility trace based on a window of experiences.

        Args:
            window: The window of experiences to update the eligibility
                trace from.
        """
        self.batch_len += 1
        local_eligibility_trace = self._create_eligibility_trace()
        final_feedback = window[-1]["feedback"]
        for td in window:
            log_prob_now = td["sample_log_prob_now"]
            log_prob_now.backward()
            prob_now = torch.exp(log_prob_now)
            prob_pre = torch.exp(td["sample_log_prob"])
            for group in self.param_groups:
                for i, param in enumerate(group["params"]):
                    local_eligibility_trace[group["name"]][i] = (
                        group["decay_rate"] * local_eligibility_trace[group["name"]][i]
                        + prob_now / prob_pre * param.grad
                    )
        for group in self.param_groups:
            for i, _ in enumerate(group["params"]):
                self._eligibility_trace[group["name"]][i] += (
                    final_feedback * local_eligibility_trace[group["name"]][i]
                )

    def update_entropy_regularization(
        self, distribution: torch.distributions.Distribution
    ):
        """Performs entropy regularization based on the distribution provided.

        Args:
            The distribution of actions at the current timestep.
        """
        entropy = distribution.entropy()
        entropy.backward()
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                self._eligibility_trace[group["name"]][i] = (
                    self._eligibility_trace[group["name"]][i] / self.batch_len
                    + group["entropy_weight"] * param.grad
                )

    def step(self) -> None:
        """Updates the model parameters based on the results
        of the eligibility traces."""
        for group in self.param_groups:
            for i, param in enumerate(group["params"]):
                with torch.no_grad():
                    param.copy_(
                        param
                        + group["learning_rate"]
                        * self._eligibility_trace[group["name"]][i]
                    )

        self._eligibility_trace = self._create_eligibility_trace()
        self.batch_len = 0
