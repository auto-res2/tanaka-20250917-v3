import argparse
import yaml
import torch
import numpy as np
import random
import os
import logging

from .preprocess import get_dataloaders
from .train import train_radiance, adapt_zeroth_order, get_models
from .evaluate import evaluate_model, plot_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path):
    """Loads a YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise

def run_experiment(config, smoke_test):
    """Main function to run a single experimental configuration."""
    if not torch.cuda.is_available():
        logging.warning("CUDA not available. Running on CPU. This will be very slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        logging.info(f"Using device: {torch.cuda.get_device_name(0)}")

    # Set seed for the current run
    set_seed(config['seeds'][0])

    # --- 1. Data Loading ---
    # Dataloaders are created but not used in this simplified training loop
    # to keep the example self-contained and runnable without the large dataset.
    # In a real run, they would be passed to the training function.
    try:
        train_loader, val_loader = get_dataloaders(config, smoke_test)
    except Exception as e:
        logging.error("Could not prepare dataloaders. Aborting experiment.")
        return False, None

    # --- 2. Training Phase ---
    # Experiment 1 & 2: Train RL controller and power surrogate
    controller, power_surrogate = train_radiance(config, device)

    # --- 3. Adaptation Phase ---
    # Experiment 3: Zeroth-Order Adaptation
    adapt_zeroth_order(config, device)

    # --- 4. Evaluation Phase ---
    all_results = []
    for model_name in config['models']:
        base_model = get_models(model_name, device)
        for device_type, budgets in config['budgets'].items():
            for budget in budgets:
                for seed in config['seeds']:
                    set_seed(seed)
                    logging.info(f"Running eval for model:{model_name}, device:{device_type}, budget:{budget}, seed:{seed}")
                    results = evaluate_model(controller, power_surrogate, base_model, device, budget, config)
                    all_results.append(results)
    
    # --- 5. Plotting --- 
    plot_results(all_results)
    
    logging.info("Experiment finished successfully.")
    return True, all_results

def main():
    parser = argparse.ArgumentParser(description="RADIANCE: Risk-bounded Adaptive Diffusion Transformer Acceleration Engine")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke-test', action='store_true', help='Run a small-scale smoke test.')
    group.add_argument('--full-experiment', action='store_true', help='Run the full-scale experiment.')
    
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        is_smoke_test = True
        logging.info("--- Starting Smoke Test ---")
    else: # args.full_experiment
        config_path = 'config/full_experiment.yaml'
        is_smoke_test = False
        logging.info("--- Starting Full Experiment ---")

    config = load_config(config_path)

    # First, run a smoke test if in full experiment mode to verify the setup
    if args.full_experiment:
        logging.info("Running a pre-flight smoke test before the full experiment...")
        smoke_config = load_config('config/smoke_test.yaml')
        success, _ = run_experiment(smoke_config, smoke_test=True)
        if not success:
            logging.error("Smoke test failed. Aborting full experiment.")
            return
        logging.info("Smoke test passed. Proceeding with the full experiment.")

    run_experiment(config, smoke_test=is_smoke_test)

if __name__ == '__main__':
    main()
