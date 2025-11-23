import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        Initialize a feedforward neural network.

        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
                        (e.g., [4, 8, 3] for input=4, hidden=8, output=3)
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
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

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)

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
            Output of the network and intermediate activations
        """
        activations = [X]
        z_values = []

        # Forward through hidden layers
        for i in range(self.num_layers - 2):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.sigmoid(z)
            activations.append(a)

        # Output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        a = self.softmax(z)
        activations.append(a)

        return activations, z_values

    def backward(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients.

        Args:
            X: Input data
            y: True labels (one-hot encoded)
            activations: List of activations from forward pass
            z_values: List of pre-activation values from forward pass

        Returns:
            Gradients for weights and biases
        """
        m = X.shape[0]  # batch size

        # Initialize gradient lists
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Output layer error (cross-entropy + softmax derivative simplifies to this)
        delta = activations[-1] - y

        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Compute gradients
            dW[i] = np.dot(activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m

            # Compute error for previous layer (if not input layer)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(z_values[i - 1])

        return dW, db

    def update_parameters(self, dW, db):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train_step(self, X, y):
        """
        Perform one training step (forward + backward + update).

        Args:
            X: Input data
            y: True labels (one-hot encoded)

        Returns:
            Loss value
        """
        # Forward propagation
        activations, z_values = self.forward(X)

        # Compute loss (cross-entropy)
        m = X.shape[0]
        loss = -np.sum(y * np.log(activations[-1] + 1e-8)) / m

        # Backward propagation
        dW, db = self.backward(X, y, activations, z_values)

        # Update parameters
        self.update_parameters(dW, db)

        return loss

    def predict(self, X):
        """
        Make predictions on input data.

        Args:
            X: Input data

        Returns:
            Predicted class labels
        """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Input data

        Returns:
            Probability distribution over classes
        """
        activations, _ = self.forward(X)
        return activations[-1]
