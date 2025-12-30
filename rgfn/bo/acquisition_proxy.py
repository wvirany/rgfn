from typing import List

import gin
import torch

from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.bo.gp_surrogate import GPSurrogate
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)


@gin.configurable()
class AcquisitionProxy(ProxyBase[ReactionState]):
    """
    Acquisition function that acts as proxy for SCENT training with BO.

    Currently only supports UCB acquisition function.

    Args:
        gp_surrogate: GP surrogate model
        beta_1: Reward-temperature parameter for SCENT
        beta_2: Exploration-exploitation trade-off parameter for UCB
    """

    def __init__(self, gp_surrogate: GPSurrogate, beta_1: float = 8.0, beta_2: float = 1.0):
        super().__init__()
        self.gp_surrogate = gp_surrogate
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def compute_proxy_output(self, states: List[ReactionState]) -> ProxyOutput:
        """
        Compute the acquisition function output for a batch of states.

        Note: We handle early terminal states by returning 0.0 for the acquisition value.

        Args:
            states: A list of reaction states.

        Returns:
            A proxy output object.
        """

        # Proess valid states
        valid_mask = [not isinstance(s, ReactionStateEarlyTerminal) for s in states]
        valid_states = [s for s, valid in zip(states, valid_mask) if valid]
        smiles_list = [state.molecule.smiles for state in valid_states]

        # Get GP predictions on valid states
        mean, std = self.gp_surrogate.predict(smiles_list)

        # Compute UCB acquisition function, SCENT expects float32
        beta = self.beta_2 / self.beta_1 # Decouple SCENT reward temperature from UCB beta
        ucb = (mean + beta * std).float()

        acq_values = torch.zeros(len(states), dtype=torch.float32, device=ucb.device)

        # Fill in UCB values for valid states
        acq_values[valid_mask] = ucb

        return ProxyOutput(value=acq_values, components=None)

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_beta(self, beta: float):
        self.beta_2 = beta
