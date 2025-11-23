import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def load_mnist():
    """
    Load and preprocess MNIST dataset.
    Reuses approach from mnist-nn project.

    Returns:
        X_train, y_train, X_test, y_test (normalized images and labels)
    """
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

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    return X_train, y_train, X_test, y_test


def image_to_poisson_spike_train(image, time_steps=100, max_rate=200.0, dt=1.0):
    """
    Convert static image to spike train using Poisson rate coding.

    Each pixel's intensity determines its firing rate.
    Higher intensity -> higher firing rate -> more spikes

    Args:
        image: Flattened image array (values in [0, 1])
        time_steps: Number of time steps to simulate
        max_rate: Maximum firing rate (Hz)
        dt: Time step duration (ms)

    Returns:
        spike_train: Binary spike train (time_steps x n_pixels)
    """
    n_pixels = len(image)

    # Convert pixel intensities to firing rates
    firing_rates = image * max_rate

    # Probability of spike in each time step (Poisson process)
    # P(spike) = rate * dt / 1000 (convert Hz to probability per ms)
    spike_prob = firing_rates * (dt / 1000.0)

    # Generate random spikes according to Poisson process
    spike_train = np.random.rand(time_steps, n_pixels) < spike_prob

    return spike_train.astype(np.float32)


def image_to_latency_spike_train(image, time_steps=100):
    """
    Convert image to spike train using latency coding.
    Higher intensity pixels spike earlier.

    Args:
        image: Flattened image array (values in [0, 1])
        time_steps: Number of time steps

    Returns:
        spike_train: Binary spike train with one spike per pixel
    """
    n_pixels = len(image)
    spike_train = np.zeros((time_steps, n_pixels), dtype=np.float32)

    for i, intensity in enumerate(image):
        if intensity > 0.01:  # Only encode non-zero pixels
            # Earlier spike for higher intensity
            # Map [0, 1] -> [time_steps-1, 0]
            spike_time = int((1.0 - intensity) * (time_steps - 1))
            spike_time = max(0, min(time_steps - 1, spike_time))
            spike_train[spike_time, i] = 1.0

    return spike_train


class MNISTSpikeDataset:
    """
    Dataset wrapper that converts MNIST images to spike trains on-the-fly.
    """
    def __init__(self, images, labels, time_steps=100, encoding='poisson'):
        """
        Args:
            images: MNIST images (normalized, flattened)
            labels: MNIST labels (integers)
            time_steps: Number of time steps per sample
            encoding: 'poisson' or 'latency'
        """
        self.images = images
        self.labels = labels
        self.time_steps = time_steps
        self.encoding = encoding
        self.n_samples = len(labels)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Get spike train for a single sample."""
        image = self.images[idx]
        label = self.labels[idx]

        if self.encoding == 'poisson':
            spike_train = image_to_poisson_spike_train(image, self.time_steps)
        elif self.encoding == 'latency':
            spike_train = image_to_latency_spike_train(image, self.time_steps)
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")

        return spike_train, label

    def iterate_batches(self, batch_size=1, shuffle=True):
        """
        Iterate through dataset in batches.

        For SNNs, typically process one sample at a time (batch_size=1)
        since each sample needs temporal simulation.

        Args:
            batch_size: Number of samples (typically 1 for SNNs)
            shuffle: Whether to shuffle data

        Yields:
            (spike_train, label) for each sample
        """
        indices = np.arange(self.n_samples)
        if shuffle:
            np.random.shuffle(indices)

        for idx in indices:
            spike_train, label = self.__getitem__(idx)
            yield spike_train, label
