import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from neural_network import NeuralNetwork


def one_hot_encode(y, num_classes):
    """Convert class labels to one-hot encoded vectors"""
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def main():
    print("=" * 60)
    print("Training Neural Network on Iris Dataset")
    print("=" * 60)

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"\nDataset Info:")
    print(f"  - Total samples: {len(X)}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Classes: {len(np.unique(y))}")
    print(f"  - Class names: {iris.target_names.tolist()}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # One-hot encode labels
    num_classes = len(np.unique(y))
    y_train_encoded = one_hot_encode(y_train, num_classes)
    y_test_encoded = one_hot_encode(y_test, num_classes)

    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")

    # Create neural network
    # Architecture: 4 input neurons -> 8 hidden neurons -> 3 output neurons
    layer_sizes = [X_train.shape[1], 8, num_classes]
    learning_rate = 0.1
    epochs = 1000

    print(f"\nNeural Network Architecture:")
    print(f"  - Layer sizes: {layer_sizes}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {epochs}")

    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate)

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(epochs):
        loss = nn.train_step(X_train, y_train_encoded)

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            train_predictions = nn.predict(X_train)
            train_accuracy = np.mean(train_predictions == y_train) * 100

            test_predictions = nn.predict(X_test)
            test_accuracy = np.mean(test_predictions == y_test) * 100

            print(f"Epoch {epoch + 1:4d}/{epochs} | "
                  f"Loss: {loss:.4f} | "
                  f"Train Acc: {train_accuracy:6.2f}% | "
                  f"Test Acc: {test_accuracy:6.2f}%")

    # Final evaluation
    print("-" * 60)
    print("\nFinal Evaluation:")

    train_predictions = nn.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train) * 100
    print(f"  Training Accuracy: {train_accuracy:.2f}%")

    test_predictions = nn.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test) * 100
    print(f"  Test Accuracy: {test_accuracy:.2f}%")

    # Per-class accuracy on test set
    print("\n  Per-class Test Accuracy:")
    for i, class_name in enumerate(iris.target_names):
        class_mask = y_test == i
        class_acc = np.mean(test_predictions[class_mask] == y_test[class_mask]) * 100
        print(f"    {class_name:15s}: {class_acc:6.2f}%")

    # Show some example predictions
    print("\n  Sample Predictions (first 10 test samples):")
    print(f"    {'Predicted':>12s} | {'Actual':>12s} | {'Correct':>7s}")
    print("    " + "-" * 40)
    for i in range(min(10, len(test_predictions))):
        pred_name = iris.target_names[test_predictions[i]]
        true_name = iris.target_names[y_test[i]]
        correct = "✓" if test_predictions[i] == y_test[i] else "✗"
        print(f"    {pred_name:>12s} | {true_name:>12s} | {correct:>7s}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
