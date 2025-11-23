import numpy as np


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model.

    Membrane potential dynamics:
        V(t+1) = decay * V(t) + I(t)
        if V(t) >= threshold: spike and reset to 0
    """
    def __init__(self, n_neurons, threshold=1.0, decay=0.9, refractory_period=1,
                 winner_take_all=False, wta_inhibition=10.0):
        """
        Args:
            n_neurons: Number of neurons in this layer
            threshold: Spike threshold
            decay: Membrane potential decay factor (0 < decay < 1)
            refractory_period: Time steps neuron can't spike after firing
            winner_take_all: If True, only highest membrane potential can spike
            wta_inhibition: Inhibition strength for WTA (subtracts from other neurons)
        """
        self.n_neurons = n_neurons
        self.threshold = threshold
        self.decay = decay
        self.refractory_period = refractory_period
        self.winner_take_all = winner_take_all
        self.wta_inhibition = wta_inhibition

        # State variables
        self.membrane = np.zeros(n_neurons)
        self.refractory_count = np.zeros(n_neurons, dtype=int)

    def step(self, input_current):
        """
        Simulate one time step of neuron dynamics.

        Args:
            input_current: Input current to neurons (shape: n_neurons)

        Returns:
            spikes: Binary spike output (1 = spike, 0 = no spike)
        """
        # Decay membrane potential
        self.membrane = self.decay * self.membrane

        # Add input current (only to non-refractory neurons)
        non_refractory = (self.refractory_count == 0)
        self.membrane += input_current * non_refractory

        # Winner-take-all: only neuron with highest potential can spike
        if self.winner_take_all:
            # Find neuron with max membrane potential
            winner_idx = np.argmax(self.membrane)
            spikes = np.zeros(self.n_neurons)

            # Only winner spikes if above threshold
            if self.membrane[winner_idx] >= self.threshold:
                spikes[winner_idx] = 1.0

                # Inhibit other neurons
                inhibition = np.ones(self.n_neurons) * self.wta_inhibition
                inhibition[winner_idx] = 0
                self.membrane -= inhibition
                self.membrane = np.maximum(0, self.membrane)  # Keep non-negative
        else:
            # Standard: all neurons above threshold spike
            spikes = (self.membrane >= self.threshold).astype(float)

        # Reset membrane potential for neurons that spiked
        self.membrane = self.membrane * (1 - spikes)

        # Set refractory period for neurons that spiked
        self.refractory_count = np.where(spikes > 0, self.refractory_period,
                                        np.maximum(0, self.refractory_count - 1))

        return spikes

    def reset(self):
        """Reset neuron state"""
        self.membrane = np.zeros(self.n_neurons)
        self.refractory_count = np.zeros(self.n_neurons, dtype=int)


class STDPSynapses:
    """
    Synapses with Spike-Timing-Dependent Plasticity (STDP).

    STDP rule:
    - If pre-spike before post-spike: strengthen (LTP - Long Term Potentiation)
    - If post-spike before pre-spike: weaken (LTD - Long Term Depression)

    With reward modulation (R-STDP):
    - Weight updates are scaled by reward signal
    """
    def __init__(self, n_pre, n_post, learning_rate=0.01,
                 tau_plus=20.0, tau_minus=20.0,
                 a_plus=0.01, a_minus=0.01,
                 w_min=0.0, w_max=1.0,
                 normalize_weights=True):
        """
        Args:
            n_pre: Number of pre-synaptic neurons
            n_post: Number of post-synaptic neurons
            learning_rate: Global learning rate multiplier
            tau_plus: Time constant for LTP (ms)
            tau_minus: Time constant for LTD (ms)
            a_plus: LTP amplitude
            a_minus: LTD amplitude
            w_min: Minimum weight value
            w_max: Maximum weight value
            normalize_weights: If True, normalize incoming weights to each neuron
        """
        self.n_pre = n_pre
        self.n_post = n_post
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.normalize_weights = normalize_weights

        # Initialize weights uniformly
        self.weights = np.random.uniform(0.1, 0.5, (n_pre, n_post))

        # Normalize initial weights
        if self.normalize_weights:
            self._normalize()

        # Eligibility traces (track recent spike correlations)
        self.trace_pre = np.zeros(n_pre)
        self.trace_post = np.zeros(n_post)

        # Accumulated weight changes (to be modulated by reward)
        self.weight_eligibility = np.zeros((n_pre, n_post))

    def compute_current(self, pre_spikes):
        """
        Compute input current to post-synaptic neurons.

        Args:
            pre_spikes: Pre-synaptic spikes (shape: n_pre)

        Returns:
            Input current to post-synaptic neurons
        """
        return np.dot(pre_spikes, self.weights)

    def update_traces(self, pre_spikes, post_spikes):
        """
        Update eligibility traces for STDP.

        Args:
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
        """
        # Decay traces
        self.trace_pre *= np.exp(-1.0 / self.tau_plus)
        self.trace_post *= np.exp(-1.0 / self.tau_minus)

        # Increment traces for neurons that spiked
        self.trace_pre += pre_spikes
        self.trace_post += post_spikes

        # Compute STDP weight changes
        # LTP: pre-spike causes post-spike (pre_spike * trace_post)
        # LTD: post-spike causes pre-spike (post_spike * trace_pre)

        # When pre spikes and post has recent trace: strengthen (LTP)
        ltp = self.a_plus * np.outer(pre_spikes, self.trace_post)

        # When post spikes and pre has recent trace: weaken (LTD)
        ltd = -self.a_minus * np.outer(self.trace_pre, post_spikes)

        # Accumulate in eligibility trace (will be modulated by reward later)
        self.weight_eligibility += ltp + ltd

    def _normalize(self):
        """
        Normalize incoming weights to each post-synaptic neuron.
        Each column sums to 1.0 (after accounting for min/max bounds).
        """
        # Normalize each column (incoming weights to each post neuron)
        col_sums = np.sum(self.weights, axis=0, keepdims=True)
        col_sums = np.maximum(col_sums, 1e-8)  # Avoid division by zero
        self.weights = self.weights / col_sums

        # Re-scale to be in valid range
        self.weights = np.clip(self.weights, self.w_min, self.w_max)

    def apply_reward(self, reward):
        """
        Apply reward-modulated weight update (R-STDP).

        Args:
            reward: Reward signal (scalar)
        """
        # Modulate eligibility trace by reward
        dw = self.learning_rate * reward * self.weight_eligibility

        # Update weights
        self.weights += dw

        # Clip weights to valid range
        self.weights = np.clip(self.weights, self.w_min, self.w_max)

        # Normalize weights if enabled
        if self.normalize_weights:
            self._normalize()

        # Reset eligibility trace
        self.weight_eligibility = np.zeros_like(self.weight_eligibility)

    def reset_traces(self):
        """Reset eligibility traces (e.g., at start of new trial)"""
        self.trace_pre = np.zeros(self.n_pre)
        self.trace_post = np.zeros(self.n_post)
        self.weight_eligibility = np.zeros((self.n_pre, self.n_post))


class SpikingNetwork:
    """
    Spiking neural network trained with reward-modulated STDP.
    """
    def __init__(self, n_input, n_hidden, n_output,
                 learning_rate=0.01,
                 lif_threshold=1.0, lif_decay=0.9,
                 stdp_params=None):
        """
        Args:
            n_input: Number of input neurons
            n_hidden: Number of hidden neurons
            n_output: Number of output neurons
            learning_rate: Learning rate for STDP
            lif_threshold: LIF neuron threshold
            lif_decay: LIF neuron decay
            stdp_params: Optional dict of STDP parameters
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Default STDP parameters
        if stdp_params is None:
            stdp_params = {}

        # Create layers
        self.hidden_neurons = LIFNeuron(n_hidden, threshold=lif_threshold, decay=lif_decay)
        # Output layer with winner-take-all competition
        self.output_neurons = LIFNeuron(
            n_output,
            threshold=lif_threshold,
            decay=lif_decay,
            winner_take_all=True,
            wta_inhibition=5.0
        )

        # Create synapses with STDP
        self.input_synapses = STDPSynapses(
            n_input, n_hidden,
            learning_rate=learning_rate,
            **stdp_params
        )
        self.hidden_synapses = STDPSynapses(
            n_hidden, n_output,
            learning_rate=learning_rate,
            **stdp_params
        )

        # Reward baseline for variance reduction (like in REINFORCE)
        self.baseline = 0.0
        self.baseline_decay = 0.95

    def forward(self, spike_trains, record=False):
        """
        Run network forward for a sequence of spike trains.

        Args:
            spike_trains: Input spike trains (shape: time_steps x n_input)
            record: If True, record spikes at each time step

        Returns:
            output_spike_counts: Number of spikes per output neuron
            recordings: (optional) dict of spike recordings
        """
        time_steps = spike_trains.shape[0]

        # Reset neuron states
        self.hidden_neurons.reset()
        self.output_neurons.reset()

        # Reset traces at start of trial
        self.input_synapses.reset_traces()
        self.hidden_synapses.reset_traces()

        # Storage for recordings
        if record:
            hidden_spikes_all = []
            output_spikes_all = []

        # Counters for output spikes
        output_spike_counts = np.zeros(self.n_output)

        # Simulate over time
        for t in range(time_steps):
            input_spikes = spike_trains[t]

            # Input -> Hidden
            hidden_current = self.input_synapses.compute_current(input_spikes)
            hidden_spikes = self.hidden_neurons.step(hidden_current)

            # Hidden -> Output
            output_current = self.hidden_synapses.compute_current(hidden_spikes)
            output_spikes = self.output_neurons.step(output_current)

            # Update STDP traces (but don't apply weight changes yet)
            self.input_synapses.update_traces(input_spikes, hidden_spikes)
            self.hidden_synapses.update_traces(hidden_spikes, output_spikes)

            # Accumulate output spikes
            output_spike_counts += output_spikes

            if record:
                hidden_spikes_all.append(hidden_spikes.copy())
                output_spikes_all.append(output_spikes.copy())

        if record:
            recordings = {
                'hidden_spikes': np.array(hidden_spikes_all),
                'output_spikes': np.array(output_spikes_all)
            }
            return output_spike_counts, recordings

        return output_spike_counts

    def train_step(self, spike_trains, true_label):
        """
        Train on one example using R-STDP.

        Args:
            spike_trains: Input spike trains (time_steps x n_input)
            true_label: True class label (integer)

        Returns:
            reward: Reward received (1 if correct, -1 if wrong)
            predicted_label: Predicted label
        """
        # Forward pass
        output_spike_counts = self.forward(spike_trains)

        # Predict based on which output neuron spiked most
        predicted_label = np.argmax(output_spike_counts)

        # Compute reward (binary: correct=1, incorrect=-1)
        raw_reward = 1.0 if predicted_label == true_label else -1.0

        # Update baseline (exponential moving average)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * raw_reward

        # Compute advantage (reward - baseline) for variance reduction
        advantage = raw_reward - self.baseline

        # Apply reward-modulated weight updates
        self.input_synapses.apply_reward(advantage)
        self.hidden_synapses.apply_reward(advantage)

        return raw_reward, predicted_label

    def predict(self, spike_trains):
        """
        Predict class label for input spike trains.

        Args:
            spike_trains: Input spike trains (time_steps x n_input)

        Returns:
            Predicted class label
        """
        output_spike_counts = self.forward(spike_trains)
        return np.argmax(output_spike_counts)
