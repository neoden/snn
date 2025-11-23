import numpy as np
import time
from data_loader import load_mnist, MNISTSpikeDataset
from spiking_network import SpikingNetwork


def filter_two_classes(X, y, class_a=0, class_b=1):
    """Filter dataset to only include two classes."""
    mask = (y == class_a) | (y == class_b)
    X_filtered = X[mask]
    y_filtered = y[mask]
    # Remap labels to 0 and 1
    y_binary = np.where(y_filtered == class_a, 0, 1)
    return X_filtered, y_binary


def evaluate_network(network, dataset, n_samples=None):
    """Evaluate network accuracy."""
    if n_samples is None:
        n_samples = len(dataset)

    correct = 0
    n_evaluated = min(n_samples, len(dataset))

    for i in range(n_evaluated):
        spike_train, true_label = dataset[i]
        predicted_label = network.predict(spike_train)

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / n_evaluated
    return accuracy


def train_simple():
    """
    Train on simplified task: binary classification (0 vs 1).
    """
    print("=" * 60)
    print("SIMPLE R-STDP: BINARY MNIST (0 vs 1)")
    print("=" * 60)
    print("\nStarting simple to verify R-STDP can learn at all")
    print("Strategy: Small network, easy task, then scale up")
    print()

    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist()

    # Filter to only classes 0 and 1
    print("Filtering to binary classification (0 vs 1)...")
    X_train_binary, y_train_binary = filter_two_classes(X_train, y_train, 0, 1)
    X_test_binary, y_test_binary = filter_two_classes(X_test, y_test, 0, 1)

    print(f"Train samples: {len(y_train_binary)} (was {len(y_train)})")
    print(f"Test samples: {len(y_test_binary)} (was {len(y_test)})")
    print(f"Class distribution - 0: {np.sum(y_train_binary == 0)}, 1: {np.sum(y_train_binary == 1)}")

    # Create spike datasets
    print("\nCreating spike train datasets...")
    time_steps = 30
    train_dataset = MNISTSpikeDataset(
        X_train_binary, y_train_binary,
        time_steps=time_steps,
        encoding='poisson'
    )
    test_dataset = MNISTSpikeDataset(
        X_test_binary, y_test_binary,
        time_steps=time_steps,
        encoding='poisson'
    )

    # Create SMALL spiking network
    print("\nCreating SMALL spiking network...")
    print("Architecture: 784 input -> 50 hidden -> 2 output neurons")
    print("(Much smaller than before: 50 vs 400 hidden)")

    n_input = 784
    n_hidden = 50  # Reduced from 400
    n_output = 2   # Binary task

    network = SpikingNetwork(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        learning_rate=0.05,  # Even higher for faster learning
        lif_threshold=1.0,
        lif_decay=0.9,
        stdp_params={
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'a_plus': 0.02,  # Higher plasticity
            'a_minus': 0.015,
            'w_min': 0.0,
            'w_max': 1.0,
            'normalize_weights': True
        }
    )

    # Training parameters
    n_epochs = 5
    samples_per_epoch = 500  # Small for fast iteration

    print(f"\nTraining parameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Samples per epoch: {samples_per_epoch}")
    print(f"  Time steps per sample: {time_steps}")
    print(f"  Learning rate: {network.input_synapses.learning_rate}")
    print(f"  Random baseline: 50% (binary task)")

    # Evaluate initial performance
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    initial_accuracy = evaluate_network(network, test_dataset, n_samples=100)
    print(f"Initial test accuracy (random weights): {initial_accuracy:.4f}")
    print(f"Random chance: 0.5000")
    print(f"Need to beat: >0.55 to show learning")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()
    best_test_accuracy = 0.0

    for epoch in range(n_epochs):
        epoch_start = time.time()

        rewards = []
        correct_predictions = 0

        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        # Train on subset of samples
        sample_count = 0
        for spike_train, true_label in train_dataset.iterate_batches(shuffle=True):
            if sample_count >= samples_per_epoch:
                break

            # Train on this sample
            reward, predicted_label = network.train_step(spike_train, true_label)

            rewards.append(reward)
            if predicted_label == true_label:
                correct_predictions += 1

            sample_count += 1

            # Progress update
            if sample_count % 100 == 0:
                current_accuracy = correct_predictions / sample_count
                avg_reward = np.mean(rewards[-100:])
                print(f"  Sample {sample_count}/{samples_per_epoch} - "
                      f"accuracy: {current_accuracy:.4f} - "
                      f"avg_reward: {avg_reward:.3f} - "
                      f"baseline: {network.baseline:.3f}")

        # Epoch statistics
        epoch_time = time.time() - epoch_start
        train_accuracy = correct_predictions / samples_per_epoch
        avg_reward = np.mean(rewards)

        print(f"\n  Epoch summary:")
        print(f"    Time: {epoch_time:.1f}s")
        print(f"    Train accuracy: {train_accuracy:.4f}")
        print(f"    Average reward: {avg_reward:.3f}")
        print(f"    Baseline: {network.baseline:.3f}")

        # Evaluate on test set
        test_accuracy = evaluate_network(network, test_dataset, n_samples=200)
        print(f"    Test accuracy: {test_accuracy:.4f}")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            print(f"    *** New best! ***")

    total_time = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_test_accuracy = evaluate_network(network, test_dataset, n_samples=None)
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print(f"Random baseline: 0.5000")
    print(f"Improvement: {(final_test_accuracy - 0.5) * 100:.1f} percentage points")
    print(f"Total training time: {total_time:.1f}s")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(20):
        spike_train, true_label = test_dataset[i]
        predicted_label = network.predict(spike_train)
        status = "✓" if predicted_label == true_label else "✗"
        print(f"  {status} True: {true_label}, Predicted: {predicted_label}")

    # Verdict
    print("\n" + "=" * 60)
    if final_test_accuracy > 0.55:
        print("SUCCESS: R-STDP is learning!")
        print("Next step: Try 3 classes or more hidden neurons")
    elif final_test_accuracy > 0.52:
        print("WEAK LEARNING: Slight improvement over random")
        print("Next step: Tune hyperparameters or try even simpler task")
    else:
        print("NO LEARNING: Still at random performance")
        print("R-STDP may be fundamentally limited for this task")
    print("=" * 60)


if __name__ == "__main__":
    train_simple()
