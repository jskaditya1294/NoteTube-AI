# Improving Neural Network Performance – Roadmap & Key Concepts


## Overall Goal & Roadmap
- Objective: Given an already trained neural network, learn techniques to improve its performance.

- Two major levers for performance improvement:
  1. Hyperparameter tuning.
  2. Handling core training problems (vanishing/exploding gradients, data scarcity, slow training, overfitting).

---

## Key Hyperparameters to Tune
- Hyperparameters: Values chosen by the deep learning engineer, not learned by the model.
- Important hyperparameters listed:
  - Number of hidden layers.
  - Number of neurons per hidden layer.
  - Learning rate.
  - Optimizer.
  - Batch size.
  - Activation functions.
  - Number of epochs.

### Impact of Good Hyperparameter Choices
- Correctly setting these hyperparameters substantially improves performance even with the same architecture.
  - Then performance will “by default” improve relative to naive choices.

---

## Core Training Problems to Address
- Even with well-chosen hyperparameters, neural networks can suffer from inherent issues:
  1. **Vanishing gradients**:
     - Gradients become very small as backprop moves towards earlier layers (especially with sigmoid).
     - Weight updates become negligible → training effectively stops.
  2. **Data scarcity (not enough data)**:
     - Deep learning is “data-hungry”.
     - Performance degrades when training data is limited.
  3. **Slow training**:
     - Training takes a long time; need techniques to speed up convergence without hurting performance.
  4. **Overfitting**:
     - Too many parameters (e.g., millions of weights in deep networks) relative to data.
     - Model fits training data too well and fails to generalize.


---

## Hyperparameter 1: Number of Hidden Layers
- Neural network layers:
  - Input layer: where data enters.
  - Output layer: where predictions are produced.
  - Hidden layers: intermediate computation layers (core of feature learning).
- Question: How many hidden layers should be used?

### Strategy: Deeper vs. Wider
- You could use:
  - Single hidden layer with many neurons, or
  - Multiple hidden layers with fewer neurons each.
- Empirical insight:
  - Using multiple hidden layers with moderate neurons (deeper networks) tends to work better than a single very wide layer.

### Representation Learning
- Deep learning leverages **representation learning**:
  - Early (lower) layers learn **primitive features**:
    - Edges, simple lines, very basic shapes.
  - Middle layers combine primitive features into **patterns**:
    - Shapes, parts of objects, facial regions.
  - Deeper layers combine patterns into **complex concepts**:
    - Whole faces, objects, and high-level abstractions.
- Deeper architectures (more hidden layers) capture:
  - Hierarchical structure of data.
  - Increasingly complex patterns.

### Practical Guideline for Number of Hidden Layers
- Prefer **more hidden layers** with **fewer neurons each**, rather than one huge layer.
- How many layers is “enough”?
  - Keep adding layers and experimenting until you start seeing **overfitting**.
  - Once overfitting appears:
    - Stop increasing the depth.
- Overfitting acts as a practical stopping criterion for depth.

---

## Transfer Learning (Linked to Depth)
- Transfer learning: reuse knowledge from one trained model on a similar new task.
- Scenario:
  - Model A trained for human face detection.
  - New task: detect monkey faces.
- Approach:
  - Keep **early layers** (which already capture generic primitives: lines, edges, simple shapes).
  - Replace or retrain **later layers** for the new specific task (monkey faces).
- Benefit:
  - Saves training time.
  - Leverages previously learned generic features.
- Connection:
  - Reinforces why we want deeper networks: different layers capture different levels of abstractions, enabling transfer learning.
- Guideline:
  - Maintain **more hidden layers** with moderate neurons, instead of fewer extremely wide layers.

---

## Hyperparameter 2: Number of Neurons per Layer
### Input and Output Layers
- Input layer:
  - Number of neurons fixed by input features.
  - Example: If input has 2 features (e.g., CGPA and IQ), input layer has 2 neurons.
- Output layer:
  - Determined by problem type:
    - Regression: usually 1 neuron.
    - Binary classification: 1 neuron (with appropriate activation).
    - Multiclass classification: `#classes` neurons.

### Hidden Layers – Neurons per Layer
- No strict formula/closed rule.
- Historical heuristic: **pyramid rule**:
  - If you have 3 hidden layers:
    - Layer 1: more neurons (e.g., 64).
    - Layer 2: fewer (e.g., 32).
    - Layer 3: even fewer (e.g., 16).
  - Rationale:
    - Early layers capture many primitive features.
    - Later layers combine them into fewer, more complex features.
- Experimental finding:
  - Pyramid vs. equal neurones-per-layer often gives similar performance.
  - Pyramid pattern is not a strict requirement.

### Key Guideline: Sufficiency
- The most important principle:
  - Number of neurons must be **sufficient**.
- Explanation:
  - If too few neurons are used, some important features may be lost early.
  - Example:
    - 2 inputs, then a hidden layer with 1 neuron:
      - Effectively compresses all features into 1 dimension.
      - Some useful information may be irreversibly discarded.
- Strategy:
  - Start with **more neurons** than you think you need.
  - If you face issues like overfitting or other constraints:
    - Gradually reduce neuron counts.
- Answer to “How many neurons per layer?”:
  - “Sufficient and slightly more than necessary; start large, then prune if needed.”

---

## Hyperparameters to Be Covered Later (Brief Mentions)
- Learning rate:
  - Controls step size in gradient descent.
  - Too small → slow training.
  - Too large → unstable or diverging training.
  - Will be studied in context of **speeding up training**.
- Optimizer:
  - We do not usually use plain vanilla gradient descent.
  - Use advanced optimizers (e.g., Adam) to converge faster and better.
  - Different optimizers will be discussed later.
- Activation functions:
  - Choice affects:
    - Vanishing gradients.
    - Expressiveness and convergence.
  - Will be revisited when discussing vanishing/exploding gradients.

---

## Hyperparameter 3: Batch Size
### Gradient Descent Variants (Context)
- Three types (previously discussed):
  - Batch gradient descent:
    - Use entire dataset to compute one gradient update.
    - Slow but stable.
  - Stochastic gradient descent (SGD):
    - Update after each individual example.
    - Fast but noisy.
  - Mini-batch gradient descent:
    - Compromise: process in chunks of size `batch_size` (e.g., 32, 64, 128).
    - Commonly used in practice.

### Batch Size as a Hyperparameter
- Batch size (`batch_size`) = number of samples used per parameter update.
- Typical choices:
  - “Small” batch sizes, e.g., 32.
  - “Large” batch sizes, e.g., 128 or 192 (often bounded by GPU RAM).

### Trade-offs
- Large batch size:
  - Pros:
    - Faster training (fewer updates per epoch; efficient GPU usage).
  - Cons:
    - Often worse generalization (test performance may degrade).
- Small batch size:
  - Pros:
    - Better generalization (model performs better on unseen data).
  - Cons:
    - Training is slower.

### Two Schools of Thought
1. **Small batch size**:
   - Prefer small `batch_size` (e.g., 32) for better generalization, accept slower training.
2. **Large batch size + Learning Rate Schedule**:
   - Use maximum feasible `batch_size` given GPU memory.
   - Apply **learning rate scheduling** to recover good performance.

### Learning Rate Scheduling: Warmup / Warming-Up
- Idea: Do not keep learning rate constant across epochs.
- Strategy (for large batch sizes):
  - Initially set learning rate **small**.
  - As epochs increase, **increase** learning rate.
- Term: referred to as **warmup / warming with learning rate**.
- Benefit:
  - Achieves both:
    - Fast training (large batch).
    - Good results (due to adaptive learning rate).

### Practical Recommendation
- Prefer:
  - Large batch size + learning rate warmup (fast + accurate) where possible.
  - If this fails or does not work well:
    - Fall back to smaller batch sizes for more stable generalization.

---

## Hyperparameter 4: Number of Epochs
- Epoch: one full pass over the training data.
- Simple view:
  - More epochs → better training (to a point).
- Problem:
  - Too many epochs can cause overfitting.

### Early Stopping
- Technique: **early stopping**.
- Concept:
  - Monitor model performance (often via validation loss/accuracy) after each epoch.
  - If performance stops improving (or worsens) over a certain window:
    - Automatically stop training.
- Implementation:
  - Keras feature via **callbacks** (automatic monitoring and stopping).
- Guideline:
  - Set epochs to a large number.
  - Use early stopping so model itself “decides” when to stop training.
  - You do not need to hand-tune the exact epoch count.

---

## Solutions Overview for Key Problems

### 1. Vanishing / Exploding Gradient Problems
- Vanishing gradient (especially with sigmoid):
  - Gradients near earlier layers become very small.
  - Training stalls.
- Exploding gradient:
  - Gradients become excessively large; weights blow up; instability.

#### Mitigation Techniques
- Better weight initialization:
  - Instead of simple small random values (e.g., uniform around 0.1), use smart initialization strategies.
- Change activation functions:
  - Move away from sigmoid for deeper networks.
  - Use activations like ReLU (to be studied later).
- Batch Normalization:
  - Normalizes activations layer-wise during training.
  - Helps stabilize and speed up training; mitigates gradient issues.
- Gradient clipping:
  - Specific for exploding gradients.
  - Caps gradient values to a certain threshold.

### 2. Not Enough Data
- Deep learning needs large amounts of data.
- Solutions (high-level named):
  - Transfer learning:
    - Reuse models or parts of models trained on other related tasks.
  - Data augmentation / semi-supervised / unsupervised elements (mentioned at high level).
  - Full strategies will be covered further in “transfer learning” and related topics.

### 3. Slow Training
- Training too slow can be mitigated by:
  - Using more advanced optimizers (e.g., Adam instead of vanilla gradient descent).
  - Using learning rate schedulers:
    - Change learning rate as training progresses (e.g., warmup, decay).
  - These improve convergence speed while maintaining or improving accuracy.

### 4. Overfitting
- Overfitting arises due to too many parameters or training too long.
- Mitigation strategies:
  - Regularization:
    - Techniques like L1/L2 regularization (already encountered in previous regression-level context).
  - Dropout:
    - Randomly disable a subset of neurons during training to prevent co-adaptation.
- These methods help models generalize better to unseen data.

---

## Summary of Upcoming Focus Areas
- Each of the following will be studied in detail (with practical demos and Keras code):
  - Choosing number of hidden layers.
  - Choosing number of neurons per layer.
  - Activation functions.
  - Learning rate and optimizers.
  - Batch size selection and learning rate schedules (including warmup).
  - Early stopping and callbacks.
  - Weight initialization techniques.
  - Batch normalization.
  - Gradient clipping.
  - Transfer learning (supervised/unsupervised angles).
  - Regularization (L1/L2).
  - Dropout.
- End goal:
  - Be able to:
    - Construct neural networks for varied data types.
    - Systematically tune them.
    - Solve common training issues.
    - Improve performance beyond baseline implementations.
