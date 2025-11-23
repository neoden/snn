"""
Training script for spiking neural network with surrogate gradient descent.

This trains an SNN using backpropagation through time (BPTT) with surrogate gradients.
The trained network can be deployed to discrete component hardware for inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

from snn_layers import SpikingNetwork
from data_loader import load_mnist


def train_snn(config):
    """
    Train spiking neural network on MNIST.

    Args:
        config: Dictionary with training configuration
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n" + "="*50)
    print("Loading MNIST dataset...")
    print("="*50)
    train_loader, test_loader = load_mnist(
        data_dir=config['data_dir'],
        batch_size=config['batch_size']
    )

    # Create network
    print("\n" + "="*50)
    print("Creating Spiking Neural Network")
    print("="*50)
    model = SpikingNetwork(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        output_size=config['output_size'],
        num_timesteps=config['num_timesteps'],
        threshold=config['threshold'],
        decay=config['decay']
    ).to(device)

    print(f"Architecture: {config['input_size']} → {config['hidden_size']} → {config['output_size']}")
    print(f"Simulation timesteps: {config['num_timesteps']}")
    print(f"LIF threshold: {config['threshold']}, decay: {config['decay']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Optional: Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'epoch_time': []
    }

    # Test BEFORE training
    print("\n" + "="*50)
    print("INITIAL PERFORMANCE (before training)")
    print("="*50)
    initial_acc = evaluate(model, test_loader, device)
    print(f"Initial test accuracy (random weights): {initial_acc:.2f}%")

    # Training loop
    print("\n" + "="*50)
    print(f"Training for {config['num_epochs']} epochs")
    print("="*50 + "\n")

    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(config['num_epochs']):
        epoch_start = time.time()

        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device, epoch, config['num_epochs'])

        # Evaluate on test set
        test_acc = evaluate(model, test_loader, device)

        # Update learning rate
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epoch_time'].append(epoch_time)

        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{config['num_epochs']} - {epoch_time:.1f}s - "
              f"lr: {current_lr:.6f} - "
              f"loss: {train_loss:.4f} - "
              f"train_acc: {train_acc:.2f}% - "
              f"test_acc: {test_acc:.2f}%")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_model(model, config, test_acc, epoch + 1)

    total_time = time.time() - start_time

    # Final summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Improvement: {test_acc - initial_acc:.2f}% (from {initial_acc:.2f}%)")

    # Save training history
    save_history(history, config)

    return model, history


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        spike_counts, _ = model(data)

        # Loss on spike counts (used as logits)
        loss = criterion(spike_counts, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = spike_counts.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            acc = 100.0 * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            print(f"  Epoch {epoch+1}/{total_epochs} - Batch {batch_idx+1}/{len(train_loader)} - "
                  f"loss: {avg_loss:.4f} - acc: {acc:.2f}%")

    epoch_loss = total_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            spike_counts, _ = model(data)

            # Prediction
            _, predicted = spike_counts.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def save_model(model, config, accuracy, epoch):
    """Save model checkpoint."""
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'accuracy': accuracy,
        'config': config,
        'hardware_params': model.export_for_hardware()
    }

    # Save best model
    save_path = save_dir / 'best_model.pt'
    torch.save(checkpoint, save_path)
    print(f"  → Saved best model (acc: {accuracy:.2f}%) to {save_path}")


def save_history(history, config):
    """Save training history."""
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"  → Saved training history to {history_path}")


def main():
    """Main training function."""

    # Training configuration
    config = {
        # Data
        'data_dir': './data',
        'batch_size': 128,

        # Network architecture
        'input_size': 784,
        'hidden_size': 128,
        'output_size': 10,
        'num_timesteps': 25,  # Temporal simulation steps

        # LIF neuron parameters
        'threshold': 1.0,
        'decay': 0.9,

        # Training
        'num_epochs': 20,
        'learning_rate': 0.001,

        # Saving
        'save_dir': './checkpoints'
    }

    print("\n" + "="*50)
    print("SPIKING NEURAL NETWORK TRAINING")
    print("Surrogate Gradient Descent with BPTT")
    print("="*50)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Train
    model, history = train_snn(config)

    print("\n" + "="*50)
    print("Training complete! Model saved to ./checkpoints/")
    print("="*50)


if __name__ == "__main__":
    main()
