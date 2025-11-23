import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from policy_gradient_network import PolicyGradientNetwork


def main():
    print("=" * 60)
    print("Training with Policy Gradients on Iris Dataset")
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

    print(f"\nTrain/Test Split:")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")

    # Create policy gradient network
    # Architecture: 4 input neurons -> 16 hidden neurons -> 3 output neurons
    layer_sizes = [X_train.shape[1], 16, len(np.unique(y))]
    learning_rate = 0.1
    entropy_coef = 0.01
    num_samples = 3
    epochs = 3000

    print(f"\nPolicy Gradient Network Architecture:")
    print(f"  - Layer sizes: {layer_sizes}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Entropy coefficient: {entropy_coef}")
    print(f"  - Samples per update: {num_samples}")
    print(f"  - Epochs: {epochs}")
    print(f"\nKey Differences from Backpropagation:")
    print(f"  - Actions (predictions) are SAMPLED from output probabilities")
    print(f"  - Rewards: +1 for correct, -1 for incorrect")
    print(f"  - Gradients computed from log-probability of sampled actions")
    print(f"  - Scaled by (reward - baseline) for variance reduction")
    print(f"  - Entropy bonus encourages exploration")
    print(f"  - Multiple samples per update reduce gradient variance")

    nn = PolicyGradientNetwork(
        layer_sizes=layer_sizes,
        learning_rate=learning_rate,
        entropy_coef=entropy_coef,
        num_samples=num_samples
    )

    # Training loop
    print("\nTraining...")
    print("-" * 60)

    for epoch in range(epochs):
        avg_reward, baseline = nn.train_step(X_train, y_train)

        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            # Deterministic evaluation (argmax)
            train_predictions = nn.predict(X_train)
            train_accuracy = np.mean(train_predictions == y_train) * 100

            test_predictions = nn.predict(X_test)
            test_accuracy = np.mean(test_predictions == y_test) * 100

            print(f"Epoch {epoch + 1:4d}/{epochs} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Baseline: {baseline:.4f} | "
                  f"Train Acc: {train_accuracy:6.2f}% | "
                  f"Test Acc: {test_accuracy:6.2f}%")

    # Final evaluation
    print("-" * 60)
    print("\nFinal Evaluation (Deterministic - using argmax):")

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
        if np.sum(class_mask) > 0:
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

    # Demonstrate stochastic sampling during training
    print("\n  Stochastic Behavior (sampling same input 10 times):")
    print("  During training, predictions are sampled from probabilities")
    test_sample = X_test[0:1]  # Take first test sample
    true_label = iris.target_names[y_test[0]]
    probabilities = nn.predict_proba(test_sample)[0]

    print(f"    True class: {true_label}")
    print(f"    Probabilities: {probabilities}")
    print(f"    Sampled predictions:")

    sample_counts = {name: 0 for name in iris.target_names}
    num_samples = 100
    for _ in range(num_samples):
        sampled = nn.sample_action(probabilities.reshape(1, -1))[0]
        sample_counts[iris.target_names[sampled]] += 1

    for class_name, count in sample_counts.items():
        percentage = (count / num_samples) * 100
        print(f"      {class_name:15s}: {count:3d}/100 ({percentage:5.1f}%)")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
