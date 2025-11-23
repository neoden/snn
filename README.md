# Neural Network Learning Progression - Project Summary

## Project Goal

**Build a spiking neural network (SNN) that can be deployed to custom discrete component hardware (transistors, resistors, etc.).**

### Key Constraints:
- Hardware will be custom-built from discrete components (not commercial chips like Loihi)
- **No on-chip learning** - train in simulation, deploy weights to hardware
- Hardware only needs to implement: LIF neurons + weighted synapses + spike communication
- Training method can be anything (simulation-based)

---

## What We've Built So Far

### 1. Iris Dataset with Backprop (`iris-nn/`)
- Standard feedforward neural network
- Sigmoid activation, softmax output
- Cross-entropy loss + backpropagation
- **Status**: ✅ Working
- **Learning**: Standard supervised learning works well

### 2. Iris Dataset with REINFORCE (`iris-pg/`)
- Policy gradient network (improved REINFORCE)
- Variance reduction: baseline, entropy regularization, multiple samples
- Reward-based learning (±1 for correct/incorrect)
- **Status**: ✅ Working
- **Learning**: Policy gradients can work on simple tasks

### 3. MNIST with Backprop (`mnist-nn/`)
- Feedforward network [784 → 128 → 10]
- ReLU activation in hidden layers
- Uses `sklearn.datasets.fetch_openml` for data loading
- **Status**: ✅ Working
- **Learning**: Achieves good accuracy on MNIST

### 4. MNIST with R-STDP (`nmnist-stdp/`)
- Spiking neural network with Leaky Integrate-and-Fire (LIF) neurons
- Reward-modulated STDP (R-STDP) learning
- Poisson spike encoding for MNIST images
- **Status**: ❌ **Does NOT work** (critical finding!)

### 5. MNIST with Surrogate Gradients (`nmnist-surrogate/`)
- Spiking neural network with LIF neurons
- **Surrogate gradient descent** with backpropagation through time (BPTT)
- Hardware-compatible LIF dynamics (identical to what hardware will run)
- **Status**: ✅ **WORKING - 97.76% accuracy achieved!**
---

## Critical Findings: Why R-STDP Failed

### Attempted Approach
We tried to train an SNN using **Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP)**:
- Biological learning rule (local, plausible)
- STDP tracks spike timing correlations
- Reward signal (±1) modulates weight updates
- Variance reduction with baseline (like REINFORCE)

### Progressive Simplification
We simplified the problem step-by-step to isolate the issue:

**Attempt 1: Full MNIST (10 classes)**
- Network: 784 → 400 → 10
- Result: ~10% accuracy (random chance)
- Problem: Network always predicted class 0

**Attempt 2: Added improvements**
- Winner-take-all (WTA) competition in output layer
- Weight normalization
- Higher learning rate (0.005 → 0.02)
- Poisson encoding for exploration
- Result: Still ~10% accuracy, same failure mode

**Attempt 3: Binary MNIST (0 vs 1 only)**
- Network: 784 → 50 → 2 (much smaller)
- 50% random baseline (easier than 10%)
- Higher learning rate (0.05)
- Result: **45% accuracy (WORSE than random!)**
- Always predicted class 0

### Root Cause: Fundamental Limitations of R-STDP

**The Sparse Reward Problem:**
```
When network guesses randomly:
  - Average reward ≈ -0.8 (wrong 80% of time)
  - Baseline adapts to -0.8
  - Advantage = reward - baseline ≈ 0
  - Weight updates ≈ learning_rate × 0 × eligibility ≈ 0
  → No learning signal!
```

**The Positive Feedback Loop:**
1. By random chance, one output neuron (e.g., neuron 0) wins slightly more
2. Gets rewarded ~50% of time (binary task)
3. WTA ensures it keeps winning
4. Weights strengthen → wins even more
5. Converges to "always predict class 0"
6. Gets stuck at ≤50% accuracy

**Why R-STDP Cannot Work:**
- ❌ **No gradient information** - only knows "good" or "bad", not "how to improve"
- ❌ **Credit assignment problem** - which of 39,200+ synapses caused the error?
- ❌ **Binary reward** - no partial credit for "almost correct"
- ❌ **No escape from local minima** - once stuck, no mechanism to explore
- ❌ **Exploration problem** - random weights → random spikes → random predictions

### Conclusion
**R-STDP with binary rewards is fundamentally unsuitable for supervised learning on complex visual patterns, even with aggressive simplification.**

---

## SUCCESS: Surrogate Gradient Implementation

### Implementation Details

**Architecture:**
- Network: 784 → 128 → 10 (LIF neurons)
- Training: Adam optimizer with learning rate decay
- Loss: Cross-entropy on spike counts
- Timesteps: 25 per image

**LIF Neuron Model:**
```
V(t+1) = decay × V(t) + I(t)
spike = (V ≥ threshold) ? 1 : 0
V(t) ← 0 if spike
```

**Surrogate Gradient:**
- Forward: Binary spike (Heaviside function)
- Backward: Fast sigmoid approximation for gradients
- Enables backpropagation through non-differentiable spikes

**Key Components:**
- `lif_neuron.py`: LIF with surrogate gradient autograd function
- `snn_layers.py`: Spiking linear layers and full network
- `train.py`: BPTT training loop with evaluation
- `visualize.py`: Spike rasters, confusion matrix, weight viz
- `export_weights.py`: Hardware deployment export

### Results Comparison

**Initial Implementation (Poisson Encoding):**
- Input encoding: Stochastic Poisson spikes
- Best accuracy: 92.82%
- Issue: Random noise from Poisson process
- Training: Slow convergence after 90%

**Improved Implementation (Rate Coding):**
- Input encoding: Deterministic rate coding (pixel intensity → constant current)
- **Best accuracy: 97.76%**
- **Improvement: +4.94% absolute**
- Training: Smooth convergence, fast initial learning
- Epoch 1: 94.65% (vs 85.13% with Poisson)

### Critical Discovery: Input Encoding Matters!

The switch from **Poisson encoding** to **rate coding** was transformative:

| Metric | Poisson | Rate Coding | Delta |
|--------|---------|-------------|-------|
| Best Accuracy | 92.82% | **97.76%** | +4.94% |
| Epoch 1 Accuracy | 85.13% | 94.65% | +9.52% |
| Convergence | Slow | Fast | ✓ |
| Stability | Noisy | Smooth | ✓ |

**Why Rate Coding Wins:**
- ✅ Deterministic (no random variance)
- ✅ Consistent signal across timesteps
- ✅ Stronger input representation
- ✅ Cleaner gradients for backprop
- ✅ Hardware-friendly (constant current per pixel)

### Hardware Deployment Ready

**Network Requirements:**
- Total neurons: 138 (128 hidden + 10 output)
- Total synapses: ~101,770 (100,352 + 1,280 + biases)
- LIF parameters: threshold=1.0, decay=0.9, V_reset=0.0
- Timesteps per inference: 25

**Inference Procedure:**
1. **Input encoding**: Pixel intensity → constant current
2. **Layer 1**: 784→128 LIF neurons with learned weights
3. **Layer 2**: 128→10 LIF neurons with learned weights
4. **Output decoding**: Sum spike counts over time, argmax for class

**Expected Performance:** 97.76% on MNIST

### Visualizations Generated

- Training curves (loss, accuracy)
- Spike raster plots (temporal patterns)
- Confusion matrix (per-digit accuracy)
- Weight visualization (learned receptive fields)

---

## Recommended Next Steps

### Approach: Surrogate Gradient Descent

**Why this works for your hardware:**
1. **Train offline** with surrogate gradients (backprop-compatible)
   - Treat spikes as differentiable during training
   - Get 90%+ accuracy on MNIST, 70-80% on N-MNIST

2. **Deploy to hardware** for inference
   - Export trained weights (simple matrices)
   - Hardware implements: LIF dynamics + weighted synapses
   - No learning circuits needed
   - Pure inference mode

3. **Hardware-friendly**
   - LIF neurons: Simple RC circuit + comparator + reset
   - Synapses: Resistors or DACs with fixed weights
   - Event-driven spike communication

**This is the standard approach in neuromorphic computing research.**

### Implementation Plan

**Phase 1: Build Surrogate Gradient Trainer**
- Implement LIF neuron forward pass (temporal simulation)
- Implement surrogate gradient for spike function
- Use backpropagation through time (BPTT)
- Train on MNIST first (validation)

**Phase 2: Train on N-MNIST**
- Load neuromorphic MNIST data (event-based)
- Train SNN to high accuracy
- Verify temporal spike patterns

**Phase 3: Export for Hardware**
- Extract weight matrices
- Document LIF parameters (threshold, decay, etc.)
- Provide spike encoding scheme
- Create hardware specification

---

## Technical Details

### LIF Neuron Model (for hardware implementation)
```
Membrane potential dynamics:
  V(t+1) = decay × V(t) + I(t)

Spike condition:
  if V(t) ≥ threshold:
    spike = 1
    V(t) ← 0  (reset)
  else:
    spike = 0

Parameters (typical):
  - threshold: 1.0
  - decay: 0.9 (or time constant τ = 10ms)
  - refractory period: 1-2 timesteps
```

### Surrogate Gradient Function
During backprop, replace non-differentiable spike with smooth approximation:
```python
# Forward: binary spike
spike = (V >= threshold).float()

# Backward: use surrogate (e.g., fast sigmoid)
def surrogate_gradient(V, threshold, sharpness=10.0):
    return sharpness / (sharpness * abs(V - threshold) + 1.0)**2
```

### Network Architecture (suggested)
- Input: 784 neurons (MNIST pixels)
- Hidden: 128-256 LIF neurons
- Output: 10 LIF neurons (one per class)
- Time steps: 20-50 per image
- Encoding: Poisson rate coding or temporal contrast

---

### Key Reusable Components

**From `nmnist-stdp/`:**
- `LIFNeuron` class - can reuse for surrogate gradient approach
- `data_loader.py` - MNIST loading and spike encoding (Poisson)
- Network structure - just needs different learning rule

**What to change:**
- Replace STDP learning with surrogate gradient backprop
- Keep LIF neuron dynamics identical (hardware-compatible)
- Add gradient computation through time

---

## Key Lessons Learned

1. ✅ **Backprop works great** for standard neural networks
2. ✅ **REINFORCE/policy gradients** can work on simple tasks
3. ❌ **R-STDP is fundamentally limited** for supervised learning
4. ✅ **Simplification is valuable** - revealed exact failure mode
5. ✅ **Hardware constraints matter** - different from software-only SNNs
6. ➡️ **Surrogate gradients are the practical path** for trained SNNs

---

## References & Resources

**Surrogate Gradient Learning:**
- Neftci et al. (2019) "Surrogate Gradient Learning in Spiking Neural Networks"
- Zenke & Ganguli (2018) "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks"

**Neuromorphic Hardware:**
- Custom discrete component implementation (transistor-level)
- Focus on inference, not on-chip learning
- LIF neurons are standard building block

**N-MNIST Dataset:**
- Neuromorphic version of MNIST
- Event-based spiking data
- Can start with regular MNIST + spike encoding

---

## Questions for Next Session

- [ ] What accuracy target do you need for hardware validation?
- [ ] Any specific constraints on number of neurons/synapses?
- [ ] Preferred spike encoding for input (Poisson, latency, rate)?
- [ ] Digital or analog weight storage on hardware?
- [ ] Power budget considerations?

---

**Document created**: 2024-11-23
**Status**: Ready for surrogate gradient implementation
**Priority**: Implement working SNN trainer for hardware deployment
