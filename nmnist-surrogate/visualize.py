"""
Visualize SNN training results and spike patterns.
"""

import torch
import numpy as np

# Set matplotlib to use non-interactive backend (for headless environments)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import json
from pathlib import Path

from snn_layers import SpikingNetwork
from data_loader import load_mnist


def plot_training_history(history_path='./checkpoints/training_history.json'):
    """Plot training loss and accuracy curves."""
    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history['train_loss'], linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot accuracy
    axes[1].plot(history['train_acc'], linewidth=2, label='Training Accuracy', marker='o')
    axes[1].plot(history['test_acc'], linewidth=2, label='Test Accuracy', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim([80, 100])

    plt.tight_layout()
    plt.savefig('./checkpoints/training_curves.png', dpi=150, bbox_inches='tight')
    print("Saved: ./checkpoints/training_curves.png")
    plt.close()


def plot_spike_rasters(model, test_loader, device, num_samples=5):
    """Plot spike raster plots for sample images."""
    model.eval()

    # Get a batch of test data
    data, labels = next(iter(test_loader))
    data, labels = data[:num_samples].to(device), labels[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3*num_samples))

    with torch.no_grad():
        for i in range(num_samples):
            sample = data[i:i+1]
            label = labels[i].item()

            # Forward pass to get spike trains
            spike_counts, output_spike_trains = model(sample)
            prediction = spike_counts.argmax(1).item()

            # Get spike trains: (timesteps, 1, num_outputs)
            spikes = output_spike_trains.squeeze(1).cpu().numpy()  # (timesteps, 10)

            # Plot input image
            img = sample.cpu().numpy().reshape(28, 28)
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Input (Label: {label})', fontweight='bold')
            axes[i, 0].axis('off')

            # Plot spike raster
            for neuron_idx in range(10):
                spike_times = np.where(spikes[:, neuron_idx] > 0)[0]
                axes[i, 1].scatter(spike_times, [neuron_idx]*len(spike_times),
                                 marker='|', s=100, c='black', alpha=0.7)

            axes[i, 1].set_xlabel('Time Step', fontsize=10)
            axes[i, 1].set_ylabel('Output Neuron', fontsize=10)
            axes[i, 1].set_title(f'Spike Raster (Pred: {prediction})', fontweight='bold')
            axes[i, 1].set_yticks(range(10))
            axes[i, 1].set_ylim([-0.5, 9.5])
            axes[i, 1].grid(True, alpha=0.3, axis='x')

            # Plot spike counts
            counts = spike_counts.cpu().numpy()[0]
            colors = ['red' if j == label else 'steelblue' for j in range(10)]
            axes[i, 2].bar(range(10), counts, color=colors, alpha=0.7)
            axes[i, 2].set_xlabel('Output Neuron', fontsize=10)
            axes[i, 2].set_ylabel('Total Spike Count', fontsize=10)
            axes[i, 2].set_title('Output Spike Counts', fontweight='bold')
            axes[i, 2].set_xticks(range(10))
            axes[i, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('./checkpoints/spike_rasters.png', dpi=150, bbox_inches='tight')
    print("Saved: ./checkpoints/spike_rasters.png")
    plt.close()


def plot_confusion_matrix(model, test_loader, device):
    """Plot confusion matrix."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            spike_counts, _ = model(data)
            preds = spike_counts.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(all_labels, all_preds)

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./checkpoints/confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Saved: ./checkpoints/confusion_matrix.png")
    plt.close()

    # Print classification report
    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))


def plot_weight_visualization(model):
    """Visualize learned weights from input layer."""
    # Get weights from first layer
    weights = model.layer1.fc.weight.detach().cpu().numpy()  # (128, 784)

    # Plot first 25 neurons' receptive fields
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for i in range(25):
        ax = axes[i // 5, i % 5]
        w = weights[i].reshape(28, 28)

        # Normalize for visualization
        vmax = np.abs(w).max()
        im = ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Neuron {i}', fontsize=10)
        ax.axis('off')

    plt.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    plt.suptitle('Learned Receptive Fields (First 25 Hidden Neurons)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('./checkpoints/weight_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: ./checkpoints/weight_visualization.png")
    plt.close()


def main():
    """Generate all visualizations."""
    print("\n" + "="*60)
    print("VISUALIZING SNN RESULTS")
    print("="*60 + "\n")

    # Check if checkpoint exists
    checkpoint_path = './checkpoints/best_model.pt'
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return

    # Setup device (use CPU for visualization - don't need GPU)
    device = torch.device('cpu')
    print(f"Using device: {device}\n")

    # Load checkpoint
    print("Loading trained model...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = SpikingNetwork(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_timesteps=config['num_timesteps'],
        threshold=config['threshold'],
        decay=config['decay']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model accuracy: {checkpoint['accuracy']:.2f}%\n")

    # Load test data
    print("Loading MNIST test data...")
    _, test_loader = load_mnist(data_dir=config['data_dir'], batch_size=128)

    # Generate visualizations
    print("\n" + "-"*60)
    print("Generating visualizations...")
    print("-"*60 + "\n")

    print("1. Training curves...")
    plot_training_history()

    print("2. Spike rasters...")
    plot_spike_rasters(model, test_loader, device, num_samples=5)

    print("3. Confusion matrix...")
    plot_confusion_matrix(model, test_loader, device)

    print("4. Weight visualization...")
    plot_weight_visualization(model)

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("\nAll plots saved to ./checkpoints/")
    print("- training_curves.png")
    print("- spike_rasters.png")
    print("- confusion_matrix.png")
    print("- weight_visualization.png")


if __name__ == "__main__":
    main()
