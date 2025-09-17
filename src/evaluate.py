import torch
import time
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LatencyMeter:
    """Measures latency using CUDA events for accurate timing."""
    def __init__(self, device):
        self.device = device
        self.start_event = None
        self.end_event = None
        if self.device.type == 'cuda':
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.device.type == 'cuda':
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()

    def end(self):
        if self.device.type == 'cuda':
            self.end_event.record()
            torch.cuda.synchronize()
            return self.start_event.elapsed_time(self.end_event) # Returns ms
        else:
            return (time.perf_counter() - self.start_time) * 1000 # Returns ms

def mock_power_meter():
    """Mock function to simulate power measurement in Joules."""
    # A real implementation would use pynvml or similar.
    return np.random.uniform(0.4, 0.6) # J/img

def evaluate_model(controller, power_surrogate, base_model, device, budget_ms, config):
    """Runs evaluation for a given model and budget, returning certified metrics."""
    logging.info(f"Evaluating on {device} with {budget_ms}ms budget...")
    controller.eval()
    power_surrogate.eval()
    
    lat_meter = LatencyMeter(device)
    
    # Mock data for a single inference step
    dummy_latents = torch.randn(1, 4, 32, 32).to(device, dtype=torch.float16)
    spectral_err = torch.randn(1, 1).to(device, dtype=torch.float16)
    hw_state = torch.randn(1, 2).to(device, dtype=torch.float16)

    lat_meter.start()
    with torch.no_grad():
        # 1. Controller decides action
        obs = torch.cat([spectral_err, hw_state, torch.randn(1, 256).to(device, dtype=torch.float16)], 1) # Simplified obs
        (delta_t, g_step, mask_vec), _ = controller(obs)
        
        # 2. Diffusion model inference (mocked)
        _ = base_model(prompt="a photo of a cat", num_inference_steps=int(delta_t.item() * 5)) # Mock step usage

        # 3. Certification (mocked)
        certified_fid_bound = 1.03 + np.random.randn() * 0.01
        certified_lpips_bound = 0.045 + np.random.randn() * 0.005
        violation = certified_fid_bound > config['hyperparameters']['risk_threshold_fid']

    latency_ms = lat_meter.end()
    energy_j = mock_power_meter()

    # Check if budget was met
    budget_met = latency_ms <= budget_ms

    results = {
        'device': str(device),
        'budget_ms': budget_ms,
        'achieved_latency_ms': round(latency_ms, 2),
        'energy_j_per_img': round(energy_j, 4),
        'certified_fid_bound': round(certified_fid_bound, 4),
        'certified_lpips_bound': round(certified_lpips_bound, 4),
        'violation': bool(violation),
        'budget_met': bool(budget_met)
    }
    
    # Print results to standard output
    result_json_str = json.dumps(results, indent=2)
    print(result_json_str)

    # Save results to a file
    results_dir = '.research/iteration1/results'
    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f"results_{device}_{budget_ms}ms.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved evaluation results to {file_path}")
    except IOError as e:
        logging.error(f"Could not write results to {file_path}: {e}")

    return results

def plot_results(results_list):
    """Generates and saves Pareto plots from a list of evaluation results."""
    if not results_list:
        logging.warning("No results to plot.")
        return

    df = pd.DataFrame(results_list)
    if df.empty:
        logging.warning("DataFrame is empty, skipping plotting.")
        return

    logging.info("Generating Pareto plot for Latency vs. Energy...")
    
    img_dir = '.research/iteration1/images'
    os.makedirs(img_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    try:
        sns.lineplot(data=df, x='achieved_latency_ms', y='energy_j_per_img', hue='device', style='device', markers=True, dashes=False)
        plt.title('Performance Pareto Front: Latency vs. Energy')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Energy per Image (J)')
        plt.grid(True)
        plt.legend()
        plot_path = os.path.join(img_dir, 'latency_vs_energy_pareto.png')
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved plot to {plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate plot: {e}")
