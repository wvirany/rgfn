from pathlib import Path
from typing import List, Dict

import gin
import numpy as np
import pandas as pd
import torch

from rgfn.api.proxy_base import ProxyBase
from rgfn.bo.acquisition_proxy import AcquisitionProxy
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.shared.policies.uniform_policy import UniformPolicy
from rgfn.shared.samplers.random_sampler import RandomSampler
from rgfn.trainer.trainer import Trainer


@gin.configurable()
class BOLoop:
    """
    BO outer loop for SCENT training.

    Procedure:
        1. Initialize dataset with random samples.
        2. For each BO iteration:
            - Train GP surrogate on dataset
            - Update acquisition proxy
            - Train SCENT policy for T iterations w/ acquisition proxy as reward
            - Sample M candidates from trained SCENT policy
            - Select top-K by acquisition value
            - Evaluate top-K with true oracle
            - Add to dataset and repeat

    The inner SCENT training loop is handled by the existing Trainer class.

    Right now, we want to evaluate ~10k molecules on oracle, similar to PMO setup. So, we perform 20 BO iterations,
    each with top-K = 500 candidates acquired at each iteration. Note: we currently sample M = 1000 candidates from SCENT
    at each iteration, which is different than the number of candidates we actually evaluate on oracle (top-K of these
    are chosen).
    """

    def __init__(
        self,
        trainer: Trainer,
        oracle_proxy: ProxyBase[ReactionState],
        acquisition_proxy: AcquisitionProxy,
        # BO parameters
        n_bo_iterations: int = 20,
        initial_samples: int = 50,
        initial_hits_filepath: str = None,
        candidates_per_iteration: int = 1000,
        top_k: int = 500,
        # Beta schedule for UCB (currently only using constant beta)
        beta: float = 1.0,
        beta_schedule: str = "constant",
        beta_threshold: int = 10000,
        # Batch size for forward sampling from SCENT
        sample_batch_size: int = 32,
        results_dir: str = "results",
        verbose: bool = True,
    ):
        """
        Initialize BO trainer.

        Args:
            trainer: SCENT trainer (configured with T inner-loop training iterations)
            oracle_proxy: True oracle for evaluation
            acquisition_proxy: Acquisition function proxy (will be updated each BO iteration)

            n_bo_iterations: Number of BO outer loop iterations
            initial_samples: Number of samples to initialize the dataset
            candidates_per_iteration: Number of candidates to sample from SCENT during each BO iteration
            top_k: Number of candidates to select from acquisition proxy during each BO iteration

            beta: Exploration-exploitation trade-off parameter for UCB
            beta_schedule: Schedule for beta (constant, log_uniform)
            beta_threshold: Threshold for switching to constant beta
            sample_batch_size: Batch size for sampling candidates from trained SCENT policy

            results_dir: Directory for saving csv output

            verbose: Whether to print verbose output
        """
        self.trainer = trainer
        self.oracle_proxy = oracle_proxy
        self.acquisition_proxy = acquisition_proxy
        self.gp_surrogate = acquisition_proxy.gp_surrogate

        # BO parameters
        self.n_bo_iterations = n_bo_iterations
        self.initial_samples = initial_samples
        self.initial_hits_filepath = initial_hits_filepath
        self.candidates_per_iteration = candidates_per_iteration
        self.top_k = top_k
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.beta_threshold = beta_threshold
        self.sample_batch_size = sample_batch_size

        # Dataset
        self.dataset = {
            "smiles": np.array([]),
            "scores": torch.tensor([], dtype=torch.float64),
        }
        self.dataset_smiles_set = set() # For fast lookup of duplicate smiles
        self.num_oracle_evaluations = 0

        # Get references to trainer components
        self.env = trainer.train_forward_sampler.env
        self.forward_policy = trainer.train_forward_sampler.policy
        self.reward = trainer.train_forward_sampler.reward
        self.inference_sampler = RandomSampler(
            policy=self.forward_policy.reaction_forward_policy,
            env=self.env,
            reward=None,
        )

        # Use WandB logger from Trainer
        self.logger = trainer.logger

        self.verbose = verbose

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _print(self, msg: str):
        """Print if verbose."""
        if self.verbose:
            print(msg)

    def get_beta(self, iteration: int) -> float:
        """Get beta value based on schedule"""
        if self.beta_schedule == "constant":
            return self.beta
        elif self.beta_schedule == "log_uniform":
            if len(self.dataset["smiles"]) < self.beta_threshold:
                # Sample beta from (-2, -1, 0) and convert to log scale: (0.01, 0.1, 1.0)
                log_beta = np.random.randint(0, 3)
                beta = 10.0 ** (-log_beta)
                return beta
            else:
                return 0.0

    def filter_invalid_and_duplicate_states(
        self, states: List[ReactionState]
    ) -> List[ReactionState]:
        """Filter invalid and duplicate states"""
        valid_states = [
            state for state in states if not isinstance(state, ReactionStateEarlyTerminal)
        ]
        non_duplicate_states = [
            state for state in valid_states if state.molecule.smiles not in self.dataset_smiles_set
        ]
        return non_duplicate_states

    def initialize_dataset(self):
        """Initialize dataset with initial hits or random samples"""
        if self.initial_hits_filepath is not None:
            df = pd.read_csv(self.initial_hits_filepath)
            df_initial = df.head(self.initial_samples)

            initial_smiles = df_initial['molecule'].values
            initial_scores = torch.tensor(df_initial['proxy'].values, dtype=torch.float64)

            self.dataset["smiles"] = np.concatenate([self.dataset["smiles"], initial_smiles])
            self.dataset["scores"] = torch.cat([self.dataset["scores"], initial_scores])
            self.dataset_smiles_set.update(set(initial_smiles))
        else:
            self._random_initialization()
        
    def _random_initialization(self):
        # Create uniform policy sampler
        uniform_policy = UniformPolicy()
        uniform_sampler = RandomSampler(policy=uniform_policy, env=self.env, reward=None)

        sampled_states = []
        sampled_smiles = set()
        while len(sampled_states) < self.initial_samples:
            # Sample trajectories until we have M non-duplicate samples
            trajectories = uniform_sampler.sample_trajectories_batch(
                n_total_trajectories=self.initial_samples - len(sampled_states),
                batch_size=self.sample_batch_size,
            )
            terminal_states = trajectories.get_last_states_flat()

            # Filter out invalid and duplicate states
            non_duplicate_states = self.filter_invalid_and_duplicate_states(terminal_states)

            for state in non_duplicate_states:
                if state.molecule.smiles not in sampled_smiles:
                    sampled_states.append(state)
                    sampled_smiles.add(state.molecule.smiles)

        oracle_output = self.oracle_proxy.compute_proxy_output(sampled_states)
        oracle_scores = oracle_output.value

        self.dataset["smiles"] = np.concatenate(
            [self.dataset["smiles"], [state.molecule.smiles for state in sampled_states]]
        )
        self.dataset["scores"] = torch.cat([self.dataset["scores"], oracle_scores])
        self.dataset_smiles_set.update(set([state.molecule.smiles for state in sampled_states]))
    

    def fit_gp(self):
        """Fit GP surrogate on current dataset"""
        n_samples = len(self.dataset["smiles"])
        if n_samples > 5000:
            # Take top 2500 samples, and 2500 random samples from non-top samples
            top_score_indices = torch.argsort(self.dataset["scores"])[-2500:]
            all_indices = torch.arange(n_samples)
            non_top_indices = np.setdiff1d(all_indices, top_score_indices)
            random_indices = torch.from_numpy(
                non_top_indices[torch.randperm(len(non_top_indices))[:2500]]
            )
            indices = torch.cat([top_score_indices, random_indices])
            smiles = self.dataset["smiles"][indices]
            scores = self.dataset["scores"][indices]
        else:
            # Train on all samples
            smiles = self.dataset["smiles"]
            scores = self.dataset["scores"]
        self.gp_surrogate.fit(smiles, scores)

    def train_scent(self):
        """Train SCENT policy for T iterations w/ acquisition proxy as reward (instantiated in gin config)"""
        self.trainer.train()

    def sample_candidates(self):
        """
        Sample M candidates from trained SCENT policy. Removes any samples that have already been evaluated on oracle.

        Returns:
            List of terminal states from the sampled trajectories
        """
        sampled_states = []
        sampled_smiles = set()

        total_sampled = 0
        total_duplicates = 0

        while len(sampled_states) < self.candidates_per_iteration:
            trajectories = self.inference_sampler.sample_trajectories_batch(
                n_total_trajectories=self.candidates_per_iteration - len(sampled_states),
                batch_size=self.sample_batch_size,
            )
            terminal_states = trajectories.get_last_states_flat()

            total_sampled += len(terminal_states)

            # Filter out invalid and duplicate states
            non_duplicate_states = self.filter_invalid_and_duplicate_states(terminal_states)

            batch_duplicates = len(terminal_states) - len(non_duplicate_states)
            total_duplicates += batch_duplicates

            # Only add states that haven't been sampled yet
            for state in non_duplicate_states:
                if state.molecule.smiles not in sampled_smiles:
                    sampled_states.append(state)
                    sampled_smiles.add(state.molecule.smiles)

        sampling_metrics = {
            'total_sampled': total_sampled,
            'total_duplicates': total_duplicates,
            'total_unique_samples': len(sampled_states),
        }

        return sampled_states, sampling_metrics

    def select_top_k(self, states: List[ReactionState]) -> List[int]:
        """Select top-K by acquisition value"""
        acquisition_values = self.acquisition_proxy.compute_proxy_output(states).value
        top_k_indices = torch.argsort(acquisition_values)[-self.top_k :]
        return top_k_indices

    def bo_iteration(self, iteration: int):
        """Perform one BO iteration"""
        self._print(f"\n{'='*60}")
        self._print(f"BO Iteration {iteration}/{self.n_bo_iterations}")
        self._print(f"{'='*60}")

        # 1. Fit GP on current dataset
        self.fit_gp()

        # 2. Update acquisition proxy
        beta = self.get_beta(iteration)
        self.acquisition_proxy.set_beta(beta)

        # 3. Train SCENT policy for T iterations w/ acquisition proxy as reward
        self.train_scent()

        # 4. Sample M candidates from trained SCENT policy
        candidate_states, sampling_metrics = self.sample_candidates()

        # 5. Select top-K by acquisition value
        top_k_indices = self.select_top_k(candidate_states)
        top_k_states = [candidate_states[i] for i in top_k_indices]
        top_k_smiles = np.array([state.molecule.smiles for state in top_k_states])

        # 6. Evaluate top-K with oracle
        oracle_output = self.oracle_proxy.compute_proxy_output(top_k_states)
        top_k_scores = oracle_output.value

        # 7. Add to dataset
        self.dataset["smiles"] = np.concatenate([self.dataset["smiles"], top_k_smiles])
        self.dataset["scores"] = torch.cat([self.dataset["scores"], top_k_scores])
        self.dataset_smiles_set.update(set(top_k_smiles))

        # 8. Compute and log metrics
        bo_metrics = self.compute_bo_metrics(iteration)

        mean_acquired_scores = top_k_scores.mean().item()
        best_acquired_score = top_k_scores.max().item()

        bo_metrics['mean_acquired_scores'] = mean_acquired_scores
        bo_metrics['best_acquired_score'] = best_acquired_score
        bo_metrics['beta'] = beta
        bo_metrics.update(sampling_metrics)

        self.logger.log_metrics(bo_metrics, prefix="bo")

        return bo_metrics
    
    def compute_bo_metrics(self, iteration: int) -> Dict[str, float]:
        """Compute BO metrics"""
        n_oracle_calls = len(self.dataset["smiles"])
        best_score = self.dataset["scores"].max().item()
        mean_top_10_scores = (
            self.dataset["scores"].topk(min(10, len(self.dataset["scores"]))).values.mean().item()
        )

        return {
            "bo_iteration": iteration,
            "n_oracle_calls": n_oracle_calls,
            "best_score": best_score,
            "mean_top_10_scores": mean_top_10_scores,
        }

    def run(self):
        """
        Run full BO loop.
        """
        self._print("\n" + "=" * 60)
        self._print("Starting BO Experiment")
        self._print("=" * 60)

        # Initialize dataset with random samples
        self._print(f"\nInitializing dataset with {self.initial_samples} random samples...")
        self.initialize_dataset()
        self._print(f"Initialization complete. Dataset size: {len(self.dataset['smiles'])}")

        bo_metrics = self.compute_bo_metrics(iteration=0)
        self.logger.log_metrics(bo_metrics, prefix="bo")

        top_k = self.top_k
        for iteration in range(1, self.n_bo_iterations + 1):
            if iteration == 1 and self.initial_hits_filepath is not None:
                self.top_k = top_k - self.initial_samples
                if self.top_k <= 0:
                    continue
            else:
                self.top_k = top_k

            bo_metrics = self.bo_iteration(iteration)

            self._print(f"\nIteration {iteration}/{self.n_bo_iterations}")
            self._print(f"  Total oracle calls: {len(self.dataset['smiles'])}")
            self._print(f"  Best score: {bo_metrics['best_score']:.4f}")
            self._print(f"  Mean top-10: {bo_metrics['mean_top_10_scores']:.4f}")

        self._print("\n" + "=" * 60)
        self._print("BO Experiment Complete")
        self._print("=" * 60)
        self._print(f"Final dataset size: {len(self.dataset['smiles'])}")
        self._print(f"Final best score: {self.dataset['scores'].max().item():.4f}")

    def save_results(self, save_path: str = None):
        """Save results to pickle file"""
        df = pd.DataFrame(
            {"smiles": self.dataset["smiles"], "oracle_score": self.dataset["scores"].cpu().numpy()}
        )
        df.to_csv(save_path, index=False)
        self._print(f"Results saved to {save_path}")
