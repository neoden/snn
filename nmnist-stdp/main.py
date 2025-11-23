import numpy as np
import time
from data_loader import load_mnist, MNISTSpikeDataset
from spiking_network import SpikingNetwork


def evaluate_network(network, dataset, n_samples=1000):
    """
    Evaluate network accuracy.

    Args:
        network: SpikingNetwork instance
        dataset: MNISTSpikeDataset instance
        n_samples: Number of samples to evaluate

    Returns:
        Accuracy score
    """
    correct = 0
    n_evaluated = min(n_samples, len(dataset))

    for i in range(n_evaluated):
        spike_train, true_label = dataset[i]
        predicted_label = network.predict(spike_train)

        if predicted_label == true_label:
            correct += 1

    accuracy = correct / n_evaluated
    return accuracy


def train_spiking_network():
    """
    Train spiking neural network on MNIST using R-STDP.
    """
    print("=" * 60)
    print("SPIKING NEURAL NETWORK WITH R-STDP")
    print("=" * 60)
    print("\nImprovements:")
    print("  - Winner-take-all competition in output layer")
    print("  - Weight normalization (prevents saturation)")
    print("  - Higher learning rate (0.02)")
    print("  - Poisson encoding (stochastic exploration)")
    print()

    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist()

    # Create spike datasets
    print("\nCreating spike train datasets...")
    print("Encoding: Poisson rate coding")
    time_steps = 30  # Reduced from 100 for speed
    train_dataset = MNISTSpikeDataset(
        X_train, y_train,
        time_steps=time_steps,
        encoding='poisson'  # Stochastic, better for STDP exploration
    )
    test_dataset = MNISTSpikeDataset(
        X_test, y_test,
        time_steps=time_steps,
        encoding='poisson'
    )

    # Create spiking network
    print("\nCreating spiking network...")
    print("Architecture: 784 input -> 400 hidden -> 10 output neurons")
    n_input = 784
    n_hidden = 400
    n_output = 10

    network = SpikingNetwork(
        n_input=n_input,
        n_hidden=n_hidden,
        n_output=n_output,
        learning_rate=0.02,  # Increased from 0.005
        lif_threshold=1.0,
        lif_decay=0.9,
        stdp_params={
            'tau_plus': 20.0,
            'tau_minus': 20.0,
            'a_plus': 0.015,  # Increased LTP amplitude
            'a_minus': 0.012,  # Increased LTD amplitude
            'w_min': 0.0,
            'w_max': 1.0,
            'normalize_weights': True  # Enable weight normalization
        }
    )

    # Training parameters
    n_epochs = 3
    samples_per_epoch = 1000  # Reduced for testing (was 5000)

    print(f"\nTraining parameters:")
    print(f"  Epochs: {n_epochs}")
    print(f"  Samples per epoch: {samples_per_epoch}")
    print(f"  Time steps per sample: {time_steps}")
    print(f"  Learning rate: {network.input_synapses.learning_rate}")

    # Evaluate initial performance (quick check with small sample)
    print("\n" + "=" * 60)
    print("BEFORE TRAINING")
    print("=" * 60)
    initial_accuracy = evaluate_network(network, test_dataset, n_samples=100)  # Reduced from 1000
    print(f"Initial test accuracy (random weights, n=100): {initial_accuracy:.4f}")

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    start_time = time.time()

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

            # Progress update every 250 samples
            if sample_count % 250 == 0:
                current_accuracy = correct_predictions / sample_count
                avg_reward = np.mean(rewards[-250:])
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

        # Evaluate on test set (small sample for speed)
        test_accuracy = evaluate_network(network, test_dataset, n_samples=200)
        print(f"    Test accuracy (n=200): {test_accuracy:.4f}")

    total_time = time.time() - start_time

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    final_test_accuracy = evaluate_network(network, test_dataset, n_samples=len(test_dataset))
    print(f"Final test accuracy: {final_test_accuracy:.4f}")
    print(f"Total training time: {total_time:.1f}s")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(10):
        spike_train, true_label = test_dataset[i]
        predicted_label = network.predict(spike_train)
        status = "✓" if predicted_label == true_label else "✗"
        print(f"  {status} True: {true_label}, Predicted: {predicted_label}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_spiking_network()
