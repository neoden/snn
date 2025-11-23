"""
Spiking Neural Network layers using LIF neurons with surrogate gradients.
"""

import torch
import torch.nn as nn
from lif_neuron import LIFNeuron


class SpikingLinearLayer(nn.Module):
    """
    Fully-connected spiking layer: Linear weights + LIF neurons.

    Architecture:
        input_spikes → Linear(in_features, out_features) → LIF neurons → output_spikes

    This is hardware-deployable:
        - Linear layer = weighted synapses (resistor network or DACs)
        - LIF neurons = RC circuit + comparator + reset switch
    """

    def __init__(self, in_features, out_features, threshold=1.0, decay=0.9,
                 V_reset=0.0, sharpness=10.0):
        """
        Args:
            in_features: Number of input neurons
            out_features: Number of output neurons
            threshold: LIF spike threshold
            decay: LIF membrane decay factor
            V_reset: LIF reset potential
            sharpness: Surrogate gradient sharpness
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Synaptic weights (hardware: resistor network or DACs)
        self.fc = nn.Linear(in_features, out_features, bias=True)

        # LIF neurons (hardware: discrete components)
        self.lif = LIFNeuron(threshold=threshold, decay=decay,
                            V_reset=V_reset, sharpness=sharpness)

    def reset_state(self, batch_size, device='cpu'):
        """Reset neuron states before processing new input."""
        self.lif.reset_state(batch_size, self.out_features, device=device)

    def forward(self, input_spikes):
        """
        Single timestep forward pass.

        Args:
            input_spikes: Input spike train (batch_size, in_features)

        Returns:
            output_spikes: Output spike train (batch_size, out_features)
        """
        # Weighted synaptic input
        I = self.fc(input_spikes)

        # LIF neuron dynamics
        output_spikes = self.lif(I)

        return output_spikes

    def get_weights(self):
        """Export weights for hardware implementation."""
        return {
            'weight_matrix': self.fc.weight.detach().cpu().numpy(),
            'bias': self.fc.bias.detach().cpu().numpy(),
            'lif_params': self.lif.get_parameters_dict()
        }


class SpikingNetwork(nn.Module):
    """
    Multi-layer spiking neural network for MNIST classification.

    Architecture:
        Input (784) → Hidden (128-256 LIF) → Output (10 LIF)

    Training:
        - Surrogate gradient descent with BPTT
        - Cross-entropy loss on spike counts

    Hardware deployment:
        - Export trained weights
        - Implement with discrete components (LIF + weighted synapses)
    """

    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 num_timesteps=20, threshold=1.0, decay=0.9):
        """
        Args:
            input_size: Input dimension (784 for MNIST)
            hidden_size: Hidden layer size
            output_size: Output dimension (10 for MNIST)
            num_timesteps: Number of simulation timesteps per sample
            threshold: LIF spike threshold
            decay: LIF membrane decay
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_timesteps = num_timesteps

        # Layer 1: Input → Hidden
        self.layer1 = SpikingLinearLayer(input_size, hidden_size,
                                        threshold=threshold, decay=decay)

        # Layer 2: Hidden → Output
        self.layer2 = SpikingLinearLayer(hidden_size, output_size,
                                        threshold=threshold, decay=decay)

    def reset_state(self, batch_size, device='cpu'):
        """Reset all neuron states."""
        self.layer1.reset_state(batch_size, device=device)
        self.layer2.reset_state(batch_size, device=device)

    def forward(self, input_data):
        """
        Forward pass with temporal dynamics (BPTT).

        Args:
            input_data: Static input (batch_size, input_size) - will be encoded as spikes

        Returns:
            spike_counts: Total spikes per output neuron (batch_size, output_size)
            spike_trains: Full spike history for analysis (num_timesteps, batch_size, output_size)
        """
        batch_size = input_data.size(0)
        device = input_data.device

        # Reset neuron states
        self.reset_state(batch_size, device=device)

        # Storage for spike trains
        hidden_spikes_history = []
        output_spikes_history = []

        # Temporal simulation (BPTT)
        for t in range(self.num_timesteps):
            # Poisson spike encoding: P(spike) = pixel_intensity / num_timesteps
            # This creates temporal spike trains from static images
            spike_prob = input_data / self.num_timesteps
            input_spikes = torch.rand_like(input_data) < spike_prob
            input_spikes = input_spikes.float()

            # Layer 1: Input → Hidden
            hidden_spikes = self.layer1(input_spikes)
            hidden_spikes_history.append(hidden_spikes)

            # Layer 2: Hidden → Output
            output_spikes = self.layer2(hidden_spikes)
            output_spikes_history.append(output_spikes)

        # Stack spike trains: (num_timesteps, batch_size, num_neurons)
        output_spike_trains = torch.stack(output_spikes_history, dim=0)

        # Sum over time: spike count = total spikes per neuron
        spike_counts = output_spike_trains.sum(dim=0)

        return spike_counts, output_spike_trains

    def export_for_hardware(self):
        """
        Export network parameters for hardware implementation.

        Returns:
            Dictionary containing:
                - Weight matrices for each layer
                - LIF parameters
                - Network architecture details
        """
        return {
            'architecture': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_timesteps': self.num_timesteps
            },
            'layer1': self.layer1.get_weights(),
            'layer2': self.layer2.get_weights()
        }
