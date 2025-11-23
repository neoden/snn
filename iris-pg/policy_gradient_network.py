import numpy as np


class PolicyGradientNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1, entropy_coef=0.01, num_samples=3):
        """
        Initialize a neural network trained with policy gradients.

        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
                        (e.g., [4, 8, 3] for input=4, hidden=8, output=3)
            learning_rate: Learning rate for gradient ascent
            entropy_coef: Coefficient for entropy regularization (encourages exploration)
            num_samples: Number of action samples per update (reduces variance)
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.num_samples = num_samples
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases with small random values
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # He initialization for better convergence
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        # Baseline for variance reduction (running average of rewards)
        self.baseline = 0.0
        self.baseline_decay = 0.95

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def softmax(self, z):
        """Softmax activation for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward propagation through the network.

        Args:
            X: Input data of shape (batch_size, input_size)

        Returns:
            Output probabilities
        """
        activation = X

        # Forward through hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)

        # Output layer with softmax
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        probabilities = self.softmax(z)

        return probabilities

    def sample_action(self, probabilities):
        """
        Sample actions (class predictions) from probability distributions.

        Args:
            probabilities: Array of shape (batch_size, num_classes)

        Returns:
            Sampled class indices for each example
        """
        batch_size = probabilities.shape[0]
        sampled_actions = np.zeros(batch_size, dtype=int)

        for i in range(batch_size):
            # Sample from the probability distribution
            sampled_actions[i] = np.random.choice(
                len(probabilities[i]),
                p=probabilities[i]
            )

        return sampled_actions

    def compute_rewards(self, sampled_actions, true_labels):
        """
        Compute rewards based on whether predictions were correct.

        Args:
            sampled_actions: Sampled class predictions
            true_labels: True class labels

        Returns:
            Rewards (+1 for correct, -1 for incorrect)
        """
        return np.where(sampled_actions == true_labels, 1.0, -1.0)

    def compute_policy_gradient(self, X, sampled_actions, rewards):
        """
        Compute policy gradients using the REINFORCE algorithm.

        The gradient is: ∇J(θ) = E[(R - b) * ∇log π(a|s)]
        where R is reward, b is baseline, π is policy, a is action, s is state

        Args:
            X: Input data
            sampled_actions: Actions that were sampled
            rewards: Rewards received

        Returns:
            Gradients for weights and biases
        """
        batch_size = X.shape[0]

        # Update baseline (running average of rewards)
        current_avg_reward = np.mean(rewards)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * current_avg_reward

        # Compute advantage (reward - baseline) for variance reduction
        advantages = rewards - self.baseline

        # Forward pass to get probabilities and intermediate activations
        activations = [X]
        activation = X

        # Forward through hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            activation = self.sigmoid(z)
            activations.append(activation)

        # Output layer
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]
        probabilities = self.softmax(z)
        activations.append(probabilities)

        # Compute gradient of log-probability of sampled actions
        # For softmax + log-likelihood, the gradient simplifies nicely
        # ∇log π(a|s) = (indicator(a) - π(a|s))

        # Create one-hot encoding of sampled actions
        action_one_hot = np.zeros_like(probabilities)
        action_one_hot[np.arange(batch_size), sampled_actions] = 1

        # Gradient of log-probability: (one_hot - probabilities)
        # Scale by advantages
        delta = advantages[:, np.newaxis] * (action_one_hot - probabilities)

        # Add entropy bonus to encourage exploration
        # Entropy gradient: ∇H = -∇(Σ p log p) = -(log p + 1)
        # We want to maximize entropy, so add positive gradient
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1, keepdims=True)
        entropy_gradient = probabilities * (np.log(probabilities + 1e-8) + 1)
        delta += self.entropy_coef * entropy_gradient

        # Backpropagate to compute gradients for all weights
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Output layer gradients
        dW[-1] = np.dot(activations[-2].T, delta) / batch_size
        db[-1] = np.sum(delta, axis=0, keepdims=True) / batch_size

        # Hidden layer gradients
        for i in range(self.num_layers - 3, -1, -1):
            # Backpropagate delta
            delta = np.dot(delta, self.weights[i + 1].T)
            # Apply derivative of sigmoid
            sigmoid_activation = activations[i + 1]
            delta = delta * sigmoid_activation * (1 - sigmoid_activation)

            # Compute gradients
            dW[i] = np.dot(activations[i].T, delta) / batch_size
            db[i] = np.sum(delta, axis=0, keepdims=True) / batch_size

        return dW, db, advantages

    def update_parameters(self, dW, db):
        """Update weights and biases using gradient ascent"""
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * dW[i]
            self.biases[i] += self.learning_rate * db[i]

    def train_step(self, X, y):
        """
        Perform one training step using policy gradients.
        Takes multiple samples per update to reduce variance.

        Args:
            X: Input data
            y: True labels (NOT one-hot encoded)

        Returns:
            Average reward and baseline
        """
        # Accumulate gradients over multiple samples
        accumulated_dW = [np.zeros_like(w) for w in self.weights]
        accumulated_db = [np.zeros_like(b) for b in self.biases]
        all_rewards = []

        for _ in range(self.num_samples):
            # Forward pass to get probabilities
            probabilities = self.forward(X)

            # Sample actions from the probability distribution
            sampled_actions = self.sample_action(probabilities)

            # Compute rewards (+1 if correct, -1 if wrong)
            rewards = self.compute_rewards(sampled_actions, y)
            all_rewards.append(rewards)

            # Compute policy gradients
            dW, db, advantages = self.compute_policy_gradient(X, sampled_actions, rewards)

            # Accumulate gradients
            for i in range(len(self.weights)):
                accumulated_dW[i] += dW[i]
                accumulated_db[i] += db[i]

        # Average the accumulated gradients
        for i in range(len(self.weights)):
            accumulated_dW[i] /= self.num_samples
            accumulated_db[i] /= self.num_samples

        # Update parameters (gradient ascent to maximize expected reward)
        self.update_parameters(accumulated_dW, accumulated_db)

        return np.mean(all_rewards), self.baseline

    def predict(self, X):
        """
        Make deterministic predictions (argmax) on input data.

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Input data

        Returns:
            Probability distribution over classes
        """
        return self.forward(X)
