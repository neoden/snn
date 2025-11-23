import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork
import time


def load_mnist():
    """Load and preprocess MNIST dataset"""
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X, y = mnist.data, mnist.target

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # One-hot encode labels
    def one_hot_encode(y, num_classes=10):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot


def train_mnist():
    """Train neural network on MNIST"""
    # Load data
    X_train, y_train, y_train_one_hot, X_test, y_test, y_test_one_hot = load_mnist()

    # Create neural network: 784 inputs -> 128 hidden -> 10 outputs
    print("\nCreating neural network [784 -> 128 -> 10]...")
    nn = NeuralNetwork(
        layer_sizes=[784, 128, 10],
        learning_rate=0.1
    )

    # Training parameters
    epochs = 10
    batch_size = 128
    num_batches = len(X_train) // batch_size

    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    print(f"Batches per epoch: {num_batches}")

    # Test accuracy BEFORE training
    print("\n=== BEFORE TRAINING ===")
    initial_predictions = nn.predict(X_test[:1000])
    initial_accuracy = np.mean(initial_predictions == y_test[:1000])
    print(f"Initial test accuracy (random weights): {initial_accuracy:.4f}")
    print(f"First 20 predictions: {initial_predictions[:20]}")
    print(f"First 20 actual:      {y_test[:20]}")
    print()

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()

        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_one_hot[indices]

        epoch_loss = 0

        # Mini-batch training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size

            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]

            loss = nn.train_step(X_batch, y_batch)
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches

        # Evaluate on training set (sample for speed)
        train_sample_indices = np.random.choice(len(X_train), 1000, replace=False)
        train_predictions = nn.predict(X_train[train_sample_indices])
        train_accuracy = np.mean(train_predictions == y_train[train_sample_indices])

        # Evaluate on test set
        test_predictions = nn.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
              f"loss: {avg_loss:.4f} - "
              f"train_acc: {train_accuracy:.4f} - "
              f"test_acc: {test_accuracy:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Final test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    train_mnist()
