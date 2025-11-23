# Policy Gradient Training for Iris Classification

This project implements neural network training using **policy gradients** (REINFORCE algorithm) instead of standard backpropagation. It demonstrates training with a **global reward signal** rather than computing gradients layer-by-layer through the network.

## What is Policy Gradient Training?

Policy gradient methods treat the neural network as a **stochastic policy** that samples actions (predictions) from a probability distribution, rather than always picking the most likely class.

### Key Differences from Backpropagation:

| **Backpropagation** | **Policy Gradient** |
|---------------------|---------------------|
| Deterministic predictions (argmax) | Stochastic sampling during training |
| Computes exact error gradients layer-by-layer | Uses global reward signal |
| Error = true_label - prediction | Reward = +1 (correct) or -1 (wrong) |
| Gradient flows backward through layers | Gradient weighted by reward |
| "You should have predicted X" | "You tried Y and got reward R" |

## How It Works

### 1. Stochastic Action Sampling

Instead of taking `argmax(probabilities)`, we **sample** from the distribution:

```python
# Network outputs: [0.7, 0.2, 0.1]
# Instead of always picking class 0 (deterministic)...
# We sample: 70% chance class 0, 20% class 1, 10% class 2
```

This exploration allows the network to discover which actions lead to good rewards.

### 2. Reward Signal

After sampling a prediction, we get a **global reward**:
- **+1** if the prediction was correct
- **-1** if the prediction was wrong

This reward doesn't tell us "how to fix each layer" - it's just a binary signal about task success.

### 3. Policy Gradient

The REINFORCE algorithm updates weights to increase the probability of actions that led to high rewards:

```
∇J(θ) = E[(R - baseline) * ∇log π(action|input)]
```

Where:
- `R` = reward received
- `baseline` = running average of rewards (variance reduction)
- `∇log π(action|input)` = gradient of log-probability of the sampled action

**Intuition:**
- If reward is positive → increase probability of that action
- If reward is negative → decrease probability of that action

### 4. Gradient Computation

We still use backpropagation, but differently:

```python
# Standard backprop:
loss = cross_entropy(predictions, true_labels)
gradients = backprop(loss)

# Policy gradient:
sampled_action = sample_from(probabilities)
log_prob = log(probabilities[sampled_action])
gradients = reward * backprop(log_prob)  # weighted by reward!
```

## The REINFORCE Algorithm

Original algorithm by Williams (1992):

1. **Forward pass**: Get action probabilities from network
2. **Sample action**: Randomly sample from probability distribution
3. **Get reward**: Environment returns reward for that action
4. **Compute gradient**: `∇ = (R - baseline) * ∇log π(a|s)`
5. **Update weights**: `θ = θ + learning_rate * ∇`
6. **Update baseline**: `baseline = decay * baseline + (1-decay) * R`

## Improvements Beyond Original REINFORCE

Our implementation includes modern improvements for better performance:

### 1. Entropy Regularization ❌ NOT in original REINFORCE

**Source**: A3C (Mnih et al., 2016), PPO (Schulman et al., 2017)

**What it does**: Adds a bonus for maintaining high entropy (uncertainty) in predictions

```python
entropy = -Σ p(a) * log p(a)
gradient += entropy_coef * ∇entropy
```

**Why**: Prevents premature convergence to suboptimal deterministic policies. Encourages exploration.

**Impact**: Significant improvement in learning stability and final accuracy

### 2. Multiple Samples Per Update ❌ NOT in original REINFORCE

**What it does**: For each training step, sample 3 different actions and average gradients

```python
for _ in range(3):
    action = sample(policy)
    reward = get_reward(action)
    gradients += compute_gradient(action, reward)
gradients /= 3  # average
```

**Why**: Reduces variance in gradient estimates (Monte Carlo sampling has high variance)

**Impact**: Much more stable learning, faster convergence

### 3. Reward Structure (+1/-1 instead of 0/1) ❌ Design choice

**Original**: Often use `reward = 1 if correct else 0`

**Our version**: `reward = 1 if correct else -1`

**Why**:
- With 0 reward, wrong predictions give no learning signal (gradient = 0 * ∇log π)
- With -1 reward, wrong predictions actively push away from bad actions

**Impact**: Provides learning signal for both correct and incorrect predictions

### 4. Baseline Variance Reduction ✓ Part of original REINFORCE

**What it does**: Subtract running average of rewards

```python
advantage = reward - baseline
gradient = advantage * ∇log π(action)
```

**Why**: Reduces variance without adding bias (proven by Williams, 1992)

**Impact**: Essential for stable learning

## Hyperparameters

Our final configuration:

```python
layer_sizes = [4, 16, 3]       # Larger hidden layer than basic version
learning_rate = 0.1            # Higher than typical backprop (0.01-0.001)
entropy_coef = 0.01            # Entropy bonus weight
num_samples = 3                # Samples per gradient update
baseline_decay = 0.95          # Running average decay for baseline
epochs = 3000                  # More epochs than backprop needs
```

## Results

### Version Comparison:

| Version | Test Accuracy | Key Features |
|---------|---------------|--------------|
| **Initial** | 66.67% | Basic REINFORCE, lr=0.01, 8 hidden, 0/1 rewards |
| **Improved** | 96.67% | + Entropy reg, 3 samples, 16 hidden, -1/+1 rewards |
| **Backprop** | ~97-98% | Standard cross-entropy training |

### Training Progress:

```
Epoch  100/3000 | Avg Reward: 0.29 | Train Acc:  78.33% | Test Acc:  76.67%
Epoch  500/3000 | Avg Reward: 0.79 | Train Acc:  92.50% | Test Acc:  86.67%
Epoch 1000/3000 | Avg Reward: 0.91 | Train Acc:  96.67% | Test Acc:  93.33%
Epoch 3000/3000 | Avg Reward: 0.95 | Train Acc:  97.50% | Test Acc:  96.67%
```

### Per-class Test Accuracy:
- **Setosa**: 100.00%
- **Versicolor**: 90.00%
- **Virginica**: 100.00%

## Why Use Policy Gradients?

**Advantages:**
- ✓ Works with any reward function (doesn't need to be differentiable)
- ✓ More biologically plausible (global reward signals like dopamine)
- ✓ Can handle discrete/non-differentiable actions
- ✓ Foundation for modern RL (PPO, A3C, etc.)

**Disadvantages:**
- ✗ Higher variance than backprop
- ✗ Slower convergence (needs more epochs)
- ✗ More sensitive to hyperparameters
- ✗ Requires variance reduction techniques

## Running the Code

```bash
# Navigate to project directory
cd iris-pg

# Run training
uv run main.py
```

## Project Structure

```
iris-pg/
├── policy_gradient_network.py   # PolicyGradientNetwork class
├── main.py                       # Training script
├── README.md                     # This file
└── pyproject.toml               # Dependencies (numpy, scikit-learn)
```

## Key Implementation Details

### Forward Pass (same as backprop):
```python
def forward(self, X):
    activation = X
    for i in range(num_hidden_layers):
        z = activation @ weights[i] + biases[i]
        activation = sigmoid(z)
    z = activation @ weights[-1] + biases[-1]
    return softmax(z)
```

### Sampling Actions:
```python
def sample_action(self, probabilities):
    # Sample from categorical distribution
    return np.random.choice(num_classes, p=probabilities)
```

### Policy Gradient Computation:
```python
def compute_policy_gradient(self, X, sampled_actions, rewards):
    # Get probabilities
    probabilities = self.forward(X)

    # Compute advantages
    advantages = rewards - self.baseline

    # Create one-hot encoding of sampled actions
    action_one_hot = np.zeros_like(probabilities)
    action_one_hot[range(batch_size), sampled_actions] = 1

    # Gradient of log-probability weighted by advantage
    delta = advantages * (action_one_hot - probabilities)

    # Add entropy bonus (modern improvement)
    entropy_grad = probabilities * (log(probabilities) + 1)
    delta += entropy_coef * entropy_grad

    # Backpropagate delta through network
    return backprop_gradients(delta)
```

## References

**Original REINFORCE:**
- Williams, R. J. (1992). "Simple statistical gradient-following algorithms for connectionist reinforcement learning." *Machine Learning*, 8(3-4), 229-256.

**Entropy Regularization:**
- Mnih, V., et al. (2016). "Asynchronous methods for deep reinforcement learning." *ICML*.
- Schulman, J., et al. (2017). "Proximal policy optimization algorithms." *arXiv:1707.06347*.

**Policy Gradient Methods:**
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.

## Comparison with Standard Backprop

For reference, the standard backpropagation implementation is in `../iris-nn/`:
- Uses cross-entropy loss
- Deterministic predictions
- Direct error gradients
- Typically achieves 97-98% test accuracy in ~1000 epochs

This policy gradient implementation demonstrates that neural networks can be trained effectively using only global reward signals, without computing exact error gradients through the network.
