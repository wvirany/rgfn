from typing import List

import gin
import gpytorch
import numpy as np
import torch
from gauche.kernels.fingerprint_kernels.minmax_kernel import MinMaxKernel
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# Compute ECFP fingerprints:
def smiles_to_morgan_fp(smiles: str, radius: int = 2, fp_size: int = 4096, as_numpy: bool = True):
    mol = Chem.MolFromSmiles(smiles)

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Return fingerprint as numpy array
    if as_numpy:
        return fp_gen.GetCountFingerprintAsNumPy(mol)

    return fp_gen.GetCountFingerprint(mol)


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model with MinMax kernel for molecular fingerprints"""

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(MinMaxKernel())

        if train_x.shape[1] != 2048 and train_x.shape[1] != 4096:
            raise ValueError(
                "Currently only supports molecular fingerprints w/ dimension 2048 or 4096"
            )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@gin.configurable()
class GPSurrogate:
    """GP Surrogate model wrapper."""

    def __init__(
        self,
        fp_radius: int = 2,
        fp_size: int = 4096,
        training_iter: int = 100,
        learning_rate: float = 0.1,
        device: str = "cpu",
    ):
        """
        Args:
            fp_radius: Radius of the fingerprint to use.
            fp_size: Dimension of the fingerprint to use.
            training_iter: Number of training iterations.
            learning_rate: Learning rate.
            device: Device to use.
        """
        self.fp_radius = fp_radius
        self.fp_size = fp_size
        self.training_iter = training_iter
        self.learning_rate = learning_rate
        self.device = device

        self.gp_model = None
        self.likelihood = None
        self.train_x = None
        self.train_y = None

    def fit(self, smiles_list: List[str], train_y: torch.Tensor):
        """
        Fit GP model to (SMILES, score) pairs.

        Args:
            smiles_list: List of SMILES strings.
            train_y: Tensor of scores.
        """

        assert len(smiles_list) == len(train_y), "Length mismatch"

        # Convert SMILES to fingerprints tensor
        fps = np.array(
            [
                smiles_to_morgan_fp(s, radius=self.fp_radius, fp_size=self.fp_size)
                for s in smiles_list
            ]
        )
        self.train_x = torch.tensor(fps, dtype=torch.float64).to(self.device)
        self.train_y = train_y.clone().detach().to(self.device, dtype=torch.float64)

        # Initialize GP model and likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(
            self.device, dtype=torch.float64
        )
        self.gp_model = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(
            self.device, dtype=torch.float64
        )

        # Initialize optimizer and MLL (we use this as the loss function)
        self.optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=self.learning_rate)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        self.gp_model.train()
        self.likelihood.train()

        # Fit the GP model to training data
        for i in range(self.training_iter):
            self.optimizer.zero_grad()
            output = self.gp_model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                print(
                    f"Iteration {i+1}/{self.training_iter} - Loss: {loss.item():.4f} - Outputscale: {self.gp_model.covar_module.outputscale.item():.4f} - Mean: {self.gp_model.mean_module.constant.item():.4f} - Noise: {self.likelihood.noise.item():.4f}"
                )
        print(
            f"Final - Loss: {loss.item():.4f} - Outputscale: {self.gp_model.covar_module.outputscale.item():.4f} - Mean: {self.gp_model.mean_module.constant.item():.4f} - Noise: {self.likelihood.noise.item():.4f}"
        )

        self.gp_model.eval()
        self.likelihood.eval()

    def predict(self, smiles_list: list[str]):
        """
        Predict scores for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings.
        """
        # Convert SMILES to fingerprints tensor
        fps = np.array(
            [
                smiles_to_morgan_fp(s, radius=self.fp_radius, fp_size=self.fp_size)
                for s in smiles_list
            ]
        )
        test_x = torch.tensor(fps, dtype=torch.float64).to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.gp_model(test_x))
            mean = pred.mean
            std = pred.stddev

        return mean, std
