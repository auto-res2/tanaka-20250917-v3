import argparse
import yaml
import os
import time
import torch
import json

from .preprocess import load_data
from .train import get_model, run_training
from .evaluate import evaluate_experiment, plot_variance_curves

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="HyperVAD Experiment Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full experiment.')
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        print("--- Running Smoke Test ---")
    else:
        config_path = 'config/full_experiment.yaml'
        print("--- Running Full Experiment ---")

    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not torch.cuda.is_available():
        print("Warning: CUDA not available. Running on CPU will be very slow.")

    all_results = {}
    # Loop through each experiment defined in the config
    for experiment_type, experiment_config in config['experiments'].items():
        print(f"\n{'='*20} Starting Experiment: {experiment_type.upper()} {'='*20}")
        
        # Loop through each run within an experiment
        for run_config in experiment_config['runs']:
            run_name = run_config['name']
            print(f"\n>>> Running: {run_name}")

            # 1. Load Data
            data_loaders = load_data(
                data_config=run_config['data'],
                batch_size=experiment_config['training']['batch_size'],
                num_workers=config.get('num_workers', 4)
            )

            # 2. Initialize Model
            model = get_model(run_config['model'])

            # 3. Run Training
            start_time = time.time()
            trained_model = run_training(
                model=model,
                data_loader=data_loaders['train'],
                train_config=experiment_config['training'],
                device=device,
                experiment_type=experiment_type
            )
            end_time = time.time()
            duration_seconds = end_time - start_time

            # 4. Evaluate
            run_results = evaluate_experiment(
                model=trained_model,
                loaders=data_loaders,
                eval_config=experiment_config.get('evaluation', {}),
                duration_s=duration_seconds,
                experiment_type=experiment_type,
                run_name=run_name
            )
            all_results[run_name] = run_results

    # Post-experiment analysis and plotting
    if 'variance_benchmark' in config['experiments']:
        plot_variance_curves(all_results, '.research/iteration1/images')

    print("\n--- All experiments complete ---")
    print("Summary of all runs:")
    print(json.dumps(all_results, indent=4))


if __name__ == '__main__':
    main()
