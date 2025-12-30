import gin
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import DataStructs

from rgfn.api.proxy_base import ProxyBase
from rgfn.api.trajectories import TrajectoriesContainer
from rgfn.bo.bo_loop import BOLoop
from rgfn.bo.gp_surrogate import GPSurrogate
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateEarlyTerminal
from rgfn.trainer.metrics.metric_base import MetricsBase


@gin.configurable()
class AcquisitionOracleComparison(MetricsBase):
    """
    Compare the acquisition function values vs oracle values for sampled molecules.
    """

    def __init__(self, gp_surrogate: GPSurrogate, oracle_proxy: ProxyBase):
        super().__init__()
        self.gp_surrogate = gp_surrogate
        self.oracle_proxy = oracle_proxy

    def compute_metrics(self, trajectories_container: TrajectoriesContainer) -> Dict[str, float]:
        trajectories = trajectories_container.forward_trajectories
        terminal_states = trajectories.get_last_states_flat()

        # Filter valid states
        valid_states = [state for state in terminal_states if not isinstance(state, ReactionStateEarlyTerminal)]
        smiles_list = [state.molecule.smiles for state in valid_states]

        # Get mean values from acquisition proxy
        surrogate_mean, _ = self.gp_surrogate.predict(smiles_list)

        # Get oracle values
        oracle_output = self.oracle_proxy.compute_proxy_output(valid_states)

        # Compute Spearman correlation
        correlation, _ = spearmanr(surrogate_mean.cpu().numpy(), oracle_output.value.cpu().numpy())

        return {
            "surrogate mean": surrogate_mean.mean().item(),
            "oracle mean": oracle_output.value.mean().item(),
            "surrogate max": surrogate_mean.max().item(),
            "oracle max": oracle_output.value.max().item(),
            "surrogate_oracle_corr": correlation,
        }


# ============================================================================
# Helper Functions
# ============================================================================


# Compute ECFP fingerprints:
def smiles_to_morgan_fp(smiles: str, radius: int = 2, fp_size: int = 4096, as_numpy: bool = True):
    mol = Chem.MolFromSmiles(smiles)
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)

    # Return fingerprint as numpy array
    if as_numpy:
        return fp_gen.GetCountFingerprintAsNumPy(mol)
    return fp_gen.GetCountFingerprint(mol)


def get_centroid_indices(
    smiles_list: List[str], scores_list: torch.Tensor, tanimoto_threshold: float = 0.5
):
    fps = np.array([smiles_to_morgan_fp(smi, as_numpy=False) for smi in smiles_list])

    sorted_indices = torch.argsort(scores_list, descending=True)
    centroids = []

    for fp, score, idx in zip(fps[sorted_indices], scores_list[sorted_indices], sorted_indices):
        if len(centroids) == 0:
            centroids.append((fp, score, idx))
        else:
            is_close = False
            for centroid in centroids:
                if DataStructs.TanimotoSimilarity(fp, centroid[0]) > 0.5:
                    is_close = True
            if not is_close:
                centroids.append((fp, score, idx))

    centroid_indices = [centroid[2].item() for centroid in centroids]
    return centroid_indices


# ============================================================================
# Metric Functions
# ============================================================================


def best_score(scores_list: torch.Tensor):
    """Compute best (maximum) score."""
    return scores_list.max().item()


def compute_topk_mean(scores_list: torch.Tensor, k: int = 10):
    """Compute mean of top-k scores."""
    if len(scores_list) < k:
        return scores_list.mean().item()
    topk_scores = scores_list.topk(k).values
    return topk_scores.mean().item()


def compute_avg_tanimoto_similarity(fps: List[np.ndarray]):
    """Compute average pairwise Tanimoto similarity."""
    total_similarity = 0
    n = len(fps)
    for fp in fps:
        for other_fp in fps:
            sim = DataStructs.TanimotoSimilarity(fp, other_fp)
            total_similarity += sim
    return total_similarity / (n**2)


def compute_diversity(smiles_list: np.ndarray):
    """Compute diversity as 1 - average Tanimoto similarity."""
    fps = [smiles_to_morgan_fp(smiles, as_numpy=False) for smiles in smiles_list]
    return 1 - compute_avg_tanimoto_similarity(fps)


def compute_topk_diversity(smiles_list: np.ndarray, scores_list: torch.Tensor, k: int = 10):
    """Compute diversity of top-k molecules by score."""
    if len(smiles_list) < k:
        return compute_diversity(smiles_list)

    topk_indices = torch.argsort(scores_list, descending=True)[:k]
    topk_smiles = smiles_list[topk_indices]
    return compute_diversity(topk_smiles)


def compute_topk_modes(smiles_list: np.ndarray, scores_list: torch.Tensor, k: int = 10):
    """Compute mean score of top-k mode representatives."""
    centroid_indices = get_centroid_indices(list(smiles_list), scores_list)
    topk_indices = centroid_indices[: min(k, len(centroid_indices))]
    topk_mode_scores = scores_list[topk_indices]
    return topk_mode_scores.mean().item()


def compute_num_scaffolds(smiles_list: np.ndarray, scores_list: torch.Tensor, thresholds: List[float] = [7, 8]):
    scaffold_counts = {}

    for threshold in thresholds:
        scaffolds = set()
        for smiles, score in zip(smiles_list, scores_list):
            if score > threshold:
                try:
                    scaffold = MurckoScaffoldSmiles(smiles)
                    scaffolds.add(scaffold)
                except:
                    continue
        scaffold_counts[threshold] = len(scaffolds)

    return scaffold_counts


def compute_and_log_bo_metrics(bo_loop: BOLoop):

    batch_size = bo_loop.top_k

    smiles_list = bo_loop.dataset["smiles"]
    scores_list = bo_loop.dataset["scores"]

    for i in range(batch_size, len(scores_list) + 1, batch_size):
        smiles_batch = smiles_list[:i]
        scores_batch = scores_list[:i]

        num_scaffolds = compute_num_scaffolds(smiles_batch, scores_batch)

        metrics = {
            'n_oracle_calls': i,
            'best_score': best_score(scores_batch),
            'top10_mean': compute_topk_mean(scores_batch, k=10),
            'top10_diversity': compute_topk_diversity(smiles_batch, scores_batch, k=10),
            'top10_modes': compute_topk_modes(smiles_batch, scores_batch, k=10),
            'num_scaffolds_7': num_scaffolds[7],
            'num_scaffolds_8': num_scaffolds[8],
        }

        bo_loop.logger.log_metrics(metrics, prefix="results")