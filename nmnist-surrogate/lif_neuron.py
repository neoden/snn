"""
Leaky Integrate-and-Fire (LIF) neuron with surrogate gradient for backpropagation.

This implementation is hardware-compatible - the forward pass uses the exact same
dynamics that will be implemented in discrete component hardware.
"""

import torch
import torch.nn as nn


class SurrogateGradient(torch.autograd.Function):
    """
    Surrogate gradient for the spike function.

    Forward: Binary spike (Heaviside step function)
    Backward: Smooth approximation (fast sigmoid)
    """

    @staticmethod
    def forward(ctx, input, threshold, sharpness=10.0):
        """
        Forward pass: Binary spike based on threshold.

        Args:
            input: Membrane potential (V)
            threshold: Spike threshold (scalar)
            sharpness: Controls steepness of surrogate gradient

        Returns:
            Binary spike (0 or 1)
        """
        # Convert threshold to tensor if it's a scalar
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, device=input.device, dtype=input.dtype)

        ctx.save_for_backward(input, threshold)
        ctx.sharpness = sharpness
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Use smooth surrogate gradient.

        Uses fast sigmoid: g(x) = β / (β|x-θ| + 1)²
        where β is sharpness, θ is threshold
        """
        input, threshold = ctx.saved_tensors
        sharpness = ctx.sharpness

        # Fast sigmoid surrogate gradient
        diff = torch.abs(input - threshold)
        grad_input = sharpness / (sharpness * diff + 1.0) ** 2

        return grad_output * grad_input, None, None


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with surrogate gradient.

    Dynamics (hardware-compatible):
        V(t+1) = decay × V(t) + I(t)
        if V(t) ≥ threshold:
            spike = 1
            V(t) ← V_reset
        else:
            spike = 0

    Parameters:
        threshold: Spike threshold (default: 1.0)
        decay: Membrane potential decay factor (default: 0.9)
        V_reset: Reset potential after spike (default: 0.0)
        sharpness: Surrogate gradient sharpness (default: 10.0)
    """

    def __init__(self, threshold=1.0, decay=0.9, V_reset=0.0, sharpness=10.0):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.V_reset = V_reset
        self.sharpness = sharpness

        # State (will be reset each forward pass)
        self.V = None

    def reset_state(self, batch_size, num_neurons, device='cpu'):
        """Initialize/reset membrane potential."""
        self.V = torch.zeros(batch_size, num_neurons, device=device)

    def forward(self, I):
        """
        Single timestep forward pass.

        Args:
            I: Input current (batch_size, num_neurons)

        Returns:
            spike: Binary spike output (batch_size, num_neurons)
        """
        # Integrate: V(t+1) = decay × V(t) + I(t)
        self.V = self.decay * self.V + I

        # Fire: spike = (V >= threshold) with surrogate gradient
        spike = SurrogateGradient.apply(self.V, self.threshold, self.sharpness)

        # Reset: V ← V_reset where spike occurred
        self.V = torch.where(spike.bool(),
                            torch.full_like(self.V, self.V_reset),
                            self.V)

        return spike

    def get_parameters_dict(self):
        """Export parameters for hardware implementation."""
        return {
            'threshold': self.threshold,
            'decay': self.decay,
            'V_reset': self.V_reset,
            'tau_mem': -1.0 / torch.log(torch.tensor(self.decay)).item()  # Time constant
        }
