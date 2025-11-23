"""
Export trained SNN weights for hardware implementation.

This script loads a trained model and exports all parameters needed
for discrete component hardware deployment.
"""

import torch
import numpy as np
import json
from pathlib import Path

from snn_layers import SpikingNetwork


def export_for_hardware(checkpoint_path, output_dir='./hardware_export'):
    """
    Export trained model for hardware implementation.

    Args:
        checkpoint_path: Path to saved model checkpoint
        output_dir: Directory to save exported parameters

    Exports:
        - Weight matrices (numpy arrays)
        - Bias vectors
        - LIF neuron parameters
        - Network architecture
        - Input/output specifications
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get hardware parameters
    hardware_params = checkpoint['hardware_params']
    config = checkpoint['config']
    accuracy = checkpoint['accuracy']

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("EXPORTING FOR HARDWARE IMPLEMENTATION")
    print("="*60)
    print(f"Model accuracy: {accuracy:.2f}%")
    print(f"Export directory: {output_path.absolute()}")

    # Export Layer 1 weights
    layer1_weight = hardware_params['layer1']['weight_matrix']
    layer1_bias = hardware_params['layer1']['bias']
    layer1_lif = hardware_params['layer1']['lif_params']

    print("\n--- Layer 1: Input → Hidden ---")
    print(f"Weight matrix shape: {layer1_weight.shape}")
    print(f"Bias vector shape: {layer1_bias.shape}")
    print(f"LIF parameters: {layer1_lif}")

    np.save(output_path / 'layer1_weights.npy', layer1_weight)
    np.save(output_path / 'layer1_bias.npy', layer1_bias)

    # Export Layer 2 weights
    layer2_weight = hardware_params['layer2']['weight_matrix']
    layer2_bias = hardware_params['layer2']['bias']
    layer2_lif = hardware_params['layer2']['lif_params']

    print("\n--- Layer 2: Hidden → Output ---")
    print(f"Weight matrix shape: {layer2_weight.shape}")
    print(f"Bias vector shape: {layer2_bias.shape}")
    print(f"LIF parameters: {layer2_lif}")

    np.save(output_path / 'layer2_weights.npy', layer2_weight)
    np.save(output_path / 'layer2_bias.npy', layer2_bias)

    # Export architecture and parameters
    architecture = hardware_params['architecture']
    print("\n--- Network Architecture ---")
    for key, value in architecture.items():
        print(f"{key}: {value}")

    # Create comprehensive hardware specification
    hardware_spec = {
        'model_info': {
            'accuracy': float(accuracy),
            'training_epochs': checkpoint['epoch'],
            'framework': 'PyTorch SNN with Surrogate Gradients'
        },
        'architecture': architecture,
        'layer1': {
            'type': 'Fully Connected + LIF',
            'input_size': layer1_weight.shape[1],
            'output_size': layer1_weight.shape[0],
            'weight_file': 'layer1_weights.npy',
            'bias_file': 'layer1_bias.npy',
            'lif_params': layer1_lif,
            'weight_range': [float(layer1_weight.min()), float(layer1_weight.max())],
            'weight_mean': float(layer1_weight.mean()),
            'weight_std': float(layer1_weight.std())
        },
        'layer2': {
            'type': 'Fully Connected + LIF',
            'input_size': layer2_weight.shape[1],
            'output_size': layer2_weight.shape[0],
            'weight_file': 'layer2_weights.npy',
            'bias_file': 'layer2_bias.npy',
            'lif_params': layer2_lif,
            'weight_range': [float(layer2_weight.min()), float(layer2_weight.max())],
            'weight_mean': float(layer2_weight.mean()),
            'weight_std': float(layer2_weight.std())
        },
        'hardware_requirements': {
            'neurons': {
                'total': architecture['hidden_size'] + architecture['output_size'],
                'layer1': architecture['hidden_size'],
                'layer2': architecture['output_size']
            },
            'synapses': {
                'total': (architecture['input_size'] * architecture['hidden_size'] +
                         architecture['hidden_size'] * architecture['output_size']),
                'layer1': architecture['input_size'] * architecture['hidden_size'],
                'layer2': architecture['hidden_size'] * architecture['output_size']
            },
            'timesteps_per_inference': architecture['num_timesteps']
        },
        'inference_spec': {
            'input_encoding': 'Poisson spike encoding',
            'spike_probability': 'pixel_value / num_timesteps',
            'output_decoding': 'Sum spike counts over time, argmax for classification',
            'expected_accuracy': f"{accuracy:.2f}%"
        }
    }

    # Save hardware specification
    spec_path = output_path / 'hardware_specification.json'
    with open(spec_path, 'w') as f:
        json.dump(hardware_spec, f, indent=2)

    print("\n" + "="*60)
    print("EXPORT COMPLETE")
    print("="*60)
    print(f"\nFiles saved to {output_path.absolute()}:")
    print("  - layer1_weights.npy")
    print("  - layer1_bias.npy")
    print("  - layer2_weights.npy")
    print("  - layer2_bias.npy")
    print("  - hardware_specification.json")

    print("\n--- Hardware Requirements ---")
    print(f"Total neurons: {hardware_spec['hardware_requirements']['neurons']['total']}")
    print(f"Total synapses: {hardware_spec['hardware_requirements']['synapses']['total']:,}")
    print(f"Expected accuracy: {accuracy:.2f}%")

    return hardware_spec


if __name__ == "__main__":
    checkpoint_path = './checkpoints/best_model.pt'

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
    else:
        export_for_hardware(checkpoint_path)
