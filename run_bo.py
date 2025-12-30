import argparse
from pathlib import Path

import gin

from gin_config import get_time_stamp
from rgfn.bo.bo_loop import BOLoop
from rgfn.bo.bo_metrics import AcquisitionOracleComparison, compute_and_log_bo_metrics
from rgfn.utils.helpers import seed_everything

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    seed = args.seed
    config = args.cfg

    seed_everything(seed)

    config_path = Path(config)
    relative_path = config_path.relative_to("configs").with_suffix("")
    timestamp = get_time_stamp()

    exp_dir = Path("experiments") / relative_path / timestamp
    exp_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"{relative_path / timestamp}"

    gin.clear_config()
    gin.parse_config_files_and_bindings([config], bindings=[f'run_name="{run_name}"'])

    bo_loop = BOLoop()
    bo_loop.run()

    # Compute and log post-hoc BO metrics
    compute_and_log_bo_metrics(bo_loop)

    # Save latest results
    results_path = Path("experiments") / f"{relative_path}" / "results.csv"
    bo_loop.save_results(results_path)
    bo_loop.logger.log_files([results_path])

    bo_loop.logger.log_to_file(gin.operative_config_str(), "operative_config")
    bo_loop.logger.log_to_file(gin.config_str(), "config")

    bo_loop.logger.close()
