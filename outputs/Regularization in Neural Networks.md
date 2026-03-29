# Regularization in Neural Networks

## Context: Improving Neural Network Performance
- Focus is on techniques to improve a poorly performing neural network.
- Previously covered: normalizing inputs, dropout, early stopping.
- Current topic: regularization (L1, L2, and their combination).

## What is Overfitting?
- Overfitting: model performs extremely well on training data but poorly on new/test data.
- Cause: model is too complex and memorizes minor patterns/noise in training data instead of learning general concepts.
- Analogy: a student memorizes a book without understanding; fails when questions change.
- In complex models (e.g., deep neural networks, deep decision trees), decision boundaries become highly irregular to reduce training error to almost zero, but generalization suffers.

## Why Do Neural Networks Overfit?
- Main reason: model complexity (number of parameters/neurons).
- Example with increasing neurons:
  - 1 neuron: simple linear decision boundary.
  - 10 neurons: more complex, slightly curved boundary.
  - 50, 256, 1000 neurons: highly complex, wiggly decision boundaries.
- More neurons → more possible “lines”/segments → model can carve very complex boundaries.
- Consequence: model may capture noise and very small, irrelevant patterns.

## High-Level Idea to Reduce Overfitting
- Make the model simpler by effectively “removing” or weakening some neurons/weights.
- Strategy:
  - Push some weights towards 0 (very small values).
  - Neurons connected only with near-zero weights become effectively non-existent.
  - Decision boundary becomes simpler, reducing the capacity to overfit.

## Methods to Combat Overfitting (Overview)
### 1. Increase Data
- Add more real data (best but often expensive or not possible).
- Use data augmentation:
  - Create synthetic data from existing examples (e.g., for images: rotate, flip, crop).
  - Common in CNNs: e.g., flipping a dog image, rotating, etc.

### 2. Reduce Model Complexity
- Dropout:
  - Randomly turn off a percentage of neurons in each layer during training.
- Early stopping:
  - Monitor where validation performance starts worsening; stop training at that point.
- Regularization (main topic):
  - Add penalty term on weights in the loss/cost function.
  - Variants: L1, L2, and L1+L2.
  - L2 is most commonly used in deep learning; L1 often used in linear/logistic models.

## Core Idea of Regularization
- Original cost function: `J(w)` (e.g., MSE for regression, cross-entropy for classification).
- Goal: find parameter vector `w` that minimizes `J(w)`.
- With regularization, modify cost function to:
  - `J_reg(w) = J(w) + penalty_term`.
- Penalty term discourages large weights, forcing many weights towards smaller values.

## L2 Regularization (Ridge / Weight Decay)
### Definition
- L2 penalty term:
  - For simple form with `n` weights `w1, …, wn`:
  - `Penalty_L2 = (λ / (2n)) × (w1² + w2² + … + wn²)`.
- Full cost:
  - `J_reg(w) = J(w) + (λ / (2n)) × Σ(wi²)`.

### Parameters and Notation
- `λ` (lambda):
  - Hyperparameter controlling strength of regularization.
  - Higher `λ` → stronger penalty → more weight shrinkage → risk of underfitting.
  - `λ = 0` → no regularization; cost reduces to original `J(w)`.
- `n`:
  - Number of training samples when defined as `(λ / (2n))`, or sometimes omitted for convenience.
- `1/2` factor:
  - Used for mathematical convenience (cancels with 2 in derivative).
- Bias terms:
  - Regularization applies only to weights, not biases (biases are not squared-summed).

### Matrix/Layer Form
- For a layer with weight matrix `W`:
  - L2 penalty: `(λ / (2n)) × Σ(W_ij²)` over all elements.
- Alternative writing (neural network notation):
  - `J_reg = J + (λ / (2n)) × Σ ||W^(l)||²` over layers `l`.
- Both scalar sum-of-squares form and matrix norm form are equivalent in meaning.

## How L2 Regularization Shrinks Weights (Intuition via Gradient Descent)
### Original Gradient Descent Update
- For a specific weight `w`:
  - `w_new = w_old - η × (∂J/∂w_old)`,
  - where `η` is learning rate.

### With L2 Regularization
- New cost: `J_reg = J + (λ / (2n)) × Σ(wi²)`.
- For weight `w0`:
  - `∂J_reg/∂w0 = (∂J/∂w0) + (λ / n) × w0`.
- Update rule:
  - `w0_new = w0_old - η × [ (∂J/∂w0_old) + (λ / n) × w0_old ]`.
- Rearranged:
  - `w0_new = w0_old × (1 - η × λ / n) - η × (∂J/∂w0_old)`.
- Interpretation:
  - Extra factor `(1 - η × λ / n)` is a positive number < 1.
  - At every step, before applying the normal gradient term, the weight is multiplied by a factor less than 1.
  - This repeatedly shrinks weights towards 0 (never exactly 0, but arbitrarily close).

### Relation to “Weight Decay”
- Because weights are multiplied by a factor slightly less than 1 at every step, this is often called “weight decay”.
- In practice, L2 regularization is frequently referred to as weight decay in neural networks.

## L1 Regularization (Lasso) – Brief Notes
- L1 penalty term:
  - Replace `Σ(wi²)` with `Σ|wi|`.
  - Cost looks like: `J_reg = J + (λ / (2n)) × Σ|wi|` (up to constant factors).
- Key qualitative difference vs L2:
  - L1 tends to drive many weights exactly to 0.
  - Produces sparse models (many weights become zero).
- In deep learning practice:
  - L1 less commonly used than L2 for standard networks.
  - L1 + L2 (elastic net style) is sometimes used.

## L2 vs L1 Effects
- L2 regularization:
  - Shrinks weights towards 0 but rarely to exactly 0.
  - Model stays dense (most connections remain but weaker).
- L1 regularization:
  - Many weights become exactly 0.
  - Gives sparse models (effectively removes many connections/neurons).

## Practical Use in Keras (Code-Level)
### Base Model (No Regularization)
- Synthetic classification data generated using `make_classification`.
- Simple Keras model:
  - Input layer with 2 features.
  - Hidden layer 1: 128 units (Dense).
  - Hidden layer 2: 128 units (Dense).
  - Output layer: 1 unit (binary classification).
- Training setup:
  - Optimizer: Adam with specific learning rate (e.g., 0.001).
  - Loss: binary cross-entropy.
  - Metric: accuracy.
  - Epochs: large (e.g., 200) to intentionally induce overfitting.
  - Batch size: e.g., 32.
- Observations:
  - Decision boundary: very complex and wiggly around training points.
  - Clear overfitting visible in plots (train/val curves; decision boundary plot).

### Model with L2 Regularization
- Modified hidden layers:
  - `Dense(128, kernel_regularizer=tf.keras.regularizers.l2(λ_value))`.
  - Example: `kernel_regularizer=regularizers.l2(0.01)` or `0.001` etc.
- Keep architecture otherwise same.
- When trained:
  - Decision boundary becomes smoother and simpler.
  - Overfitting reduces; better generalization on unseen/test data.
- Comparing side by side:
  - Without regularization: highly complex decision boundary.
  - With L2: cleaner, smoother, more robust decision boundary.

### Visualizing Weight Distributions
- Procedure:
  - Train `model1` without regularization.
  - Train `model2` with L2 regularization.
  - Extract weights:
    - `weights1 = model1.layers[0].get_weights()[0]` (first layer weights).
    - `weights2 = model2.layers[0].get_weights()[0]`.
  - Reshape to 1D arrays (e.g., `reshape(-1)`).
  - Plot boxplots for both:
    - For model without regularization:
      - Weights spread across a wider range, including larger magnitudes and outliers.
    - For model with L2 regularization:
      - Weights are mostly in a narrow range close to 0 (e.g., between -0.6 and 0.4).
  - Compute stats:
    - For no-regularization model:
      - Max ~1.75, min ~-2.78 (example numbers).
    - For L2-regularized model:
      - Max ~0.3, min ~-0.5 (example).
  - Density plots:
    - Without regularization: distribution wider, more spread.
    - With regularization: distribution concentrated near 0.

## Using L1 Regularization in Keras
- To switch to L1:
  - Use `kernel_regularizer=regularizers.l1(λ_value)` instead of `l2`.
- Behavior:
  - Often shows more weight sparsity; some weights become exactly 0.
  - Decision boundary may differ and can still overfit if λ is too small.
- Need to tune `λ`:
  - Very large `λ` → strong underfitting.
  - Example: adjust `λ` from a high value down (e.g., 0.1 → 0.01 → 0.001) to find suitable point.

## L1+L2 (Elastic Net Style)
- Combination:
  - `regularizers.l1_l2(l1=λ1, l2=λ2)` in Keras.
- Allows leveraging both sparsity (L1) and smooth shrinkage (L2).
- In deep learning practice:
  - Pure L2 is more popular; L1+L2 is used based on specific needs.

## Tuning Lambda (λ) and Effects
- λ too small:
  - Regularization effect almost negligible.
  - Overfitting may persist.
- λ too large:
  - Strong penalty, weights forced to be very small.
  - Model underfits: high bias, poor performance on train and test sets.
- Ideal λ:
  - Determined via validation performance or cross-validation.

## Conceptual Summary
- Overfitting: complex model memorizing training noise.
- Solution framework:
  - Make model view more data (real or augmented).
  - Limit its complexity:
    - Architectural methods (fewer layers/neurons, dropout).
    - Training methods (early stopping).
    - Optimization-level methods (regularization).
- Regularization (L1/L2):
  - Modifies cost function by adding a term dependent on weights.
  - Encourages smaller weights:
    - L2: shrink weights smoothly towards 0 → dense but weaker connections.
    - L1: many weights become exactly 0 → sparse model.
- In neural networks:
  - L2 regularization is commonly used and often referred to as weight decay.
  - Practically implemented by adding `kernel_regularizer` to Dense layers.

## Key images

### "DECISION BOUNDARY PLOTS FOR NEURAL NETWORK EPOCHS"
*Timestamp: 416s*

### "NEURAL NETWORK DECISION BOUNDARY DIAGRAMS"
*Timestamp: 475s*

### "DENSITY PLOT AND MODEL WEIGHTS OUTPUT IN COLAB NOTEBOOK"
*Timestamp: 1819s*

### "DENSITY PLOT AND MODEL WEIGHTS OUTPUT"
*Timestamp: 1832s*

### DECISION BOUNDARY PLOT IN REGULARIZATION EXAMPLE.
*Timestamp: 2027s*

### DENSITY PLOT AND CODE SNIPPET IN JUPYTER NOTEBOOK.
*Timestamp: 2069s*

## Interview question bank

*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*

> **No attributed questions were extracted.** For reliable results, set **`TAVILY_API_KEY`** in `.env` (Tavily search + full-page extract). DuckDuckGo-only mode often returns snippets too short to tie questions to a company.

### Summary table

| Topic | # Companies | # Questions | Top sources |
|-------|-------------|-------------|-------------|
| Overfitting in neural networks | 0 | 0 |  |
| Model complexity and capacity in neural networks | 0 | 0 |  |
| Methods to reduce overfitting | 0 | 0 |  |
| Data augmentation | 0 | 0 |  |
| Dropout | 0 | 0 |  |
| Early stopping | 0 | 0 |  |
| Regularization in neural networks | 0 | 0 |  |
| L2 regularization (weight decay) | 0 | 0 |  |
| L1 regularization | 0 | 0 |  |
| L1+L2 regularization (elastic net style) | 0 | 0 |  |
| Effect of regularization on gradient descent updates | 0 | 0 |  |
| Implementing L1/L2 regularization in Keras | 0 | 0 |  |
| Tuning lambda (λ) in regularization | 0 | 0 |  |
| Visualizing and comparing weight distributions with and without regularization | 0 | 0 |  |

### Coverage report

- Extracted topics: 14
- Questions with attribution: 0
- Excluded (unattributed): 0

**Topic coverage notes:**
Content centers on overfitting in neural networks and how to mitigate it, with a strong focus on regularization (especially L2) and its mathematical effect on gradient descent, plus practical Keras implementation. Secondary topics include other anti-overfitting methods (data augmentation, dropout, early stopping) and qualitative differences between L1 and L2.

### Questions by topic → company
