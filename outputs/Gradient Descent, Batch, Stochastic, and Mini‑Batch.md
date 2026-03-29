# YouTube Notes 7z6yXpYk7sw

## Metadata
- **Created at:** 2026-03-18 16:25:42
- **Video ID:** 7z6yXpYk7sw

## Notes

# Gradient Descent: Batch, Stochastic, and Mini‑Batch

## Gradient Descent Basics
- Gradient descent is an optimization algorithm used to minimize an objective (loss) function and is commonly used to optimize neural networks.
- Goal: minimize loss `L` with respect to model parameters (weights, biases).
- Update rule: `w_new = w_old - η * (∂L/∂w_old)` and similarly for biases; `η` is the learning rate.
- Conceptual view: follow the negative gradient (downhill direction of the loss surface) to reach a minimum.
- In neural networks (via backpropagation):
  - For each data point (or set of points), compute predictions.
  - Compute loss.
  - Compute gradients via derivatives (backprop).
  - Update weights and biases via gradient descent.
- The whole weight-update procedure in backprop is an application of gradient descent.

## Three Variants of Gradient Descent
- Variants differ only in how much data they use to compute the gradient at each update step.
- Three types:
  - Batch gradient descent (also called vanilla gradient descent).
  - Stochastic gradient descent (SGD).
  - Mini‑batch gradient descent.
- Key differentiator: based on the amount of data used to compute the gradient between two successive parameter updates.

---

## Batch Gradient Descent (Vanilla Gradient Descent)

### Core Idea
- Use the entire dataset to compute the gradient, then update parameters once per epoch.
- For dataset with `N` rows (data points):
  - Per epoch, compute predictions for all `N` points at once.
  - Compute a *single* overall loss (e.g., sum or mean over all `N` points).
  - Perform *one* weight & bias update per epoch.

### Vectorization
- Implementation uses vectorized operations (dot products) to avoid explicit inner loops over data points.
- For inputs `X` (shape `N × d`) and weights `W`:
  - Compute predictions: `y_hat = X · W + b` in one shot.
  - You get a vector of `N` predictions.
- Loss is computed over all these predictions (e.g., `Σ loss_i` from `i = 1` to `N`).
- Based on this single aggregated loss, update `W` and `b` once.

### Update Count
- If you run for `E` epochs and dataset has `N` points:
  - Number of weight updates = `E` (independent of `N`).
- Compared to point‑wise methods:
  - In per‑sample updating, one epoch causes `N` updates.
  - In batch gradient descent, one epoch causes 1 update.
- In terms of notation from the transcript:
  - `number_of_updates = number_of_epochs` (for batch gradient descent).
  - `number_of_updates = number_of_epochs × number_of_rows` (for sample‑wise/SGD).

### Example (with code logic)
- Pseudocode for batch gradient descent:
  - Decide `epochs = T`.
  - For `epoch` in `range(T)`:
    - Use *entire* dataset `X` and `y`:
      - `y_hat = X · W + b` (dot product).
      - Compute `loss` over all samples.
      - Compute gradients using this loss.
      - Update `W` and `b` once.

### Performance Comparison (Time)
- With same number of epochs and same dataset size:
  - Batch gradient descent performs fewer updates than SGD:
    - Batch: `E` updates.
    - SGD: `E × N` updates.
- Therefore, in terms of *wall‑clock training time* for a fixed number of epochs, batch gradient descent is typically faster than SGD.
- Demonstrated with Keras example:
  - Using `batch_size = N` ⇒ batch gradient descent.
  - Using `batch_size = 1` ⇒ stochastic gradient descent.
  - Batch (`batch_size = N`) ran much faster (e.g., ~0.5 seconds) compared to SGD (`batch_size = 1`, ~10 seconds) for the same number of epochs.

### Convergence Behavior
- If you compare both algorithms for the *same number of epochs*:
  - Batch is faster in **time** to complete those epochs.
- If you consider **number of parameter updates** instead:
  - SGD does many more updates per epoch.
  - For the *same number of epochs*, SGD may reach a good solution in fewer epochs than batch.
  - But each epoch of SGD is more expensive (more updates) than batch’s single update.

### Loss Curve Stability
- Batch gradient descent:
  - Loss typically decreases smoothly over epochs.
  - Few or no random spikes; trajectory is stable.
- Behavior reason:
  - Each step uses the full dataset, so gradient represents all points and is stable.

### Vectorization Trade‑off
- Vectorization (dot products instead of explicit loops) is much faster computationally than looping in Python.
- But:
  - Full batch gradient descent must load the entire dataset into memory at once.
  - For very large datasets (e.g., millions of rows), this may not fit in RAM.
  - In such cases, pure batch gradient descent is not feasible.

---

## Stochastic Gradient Descent (SGD)

### Core Idea
- Use *one* randomly selected data point at a time to compute the gradient and update parameters.
- Per update:
  - Shuffle data (or otherwise randomize order).
  - Pick 1 sample.
  - Compute its prediction.
  - Compute its loss.
  - Compute gradients based on that single sample.
  - Immediately update weights and biases.

### Update Count
- For dataset with `N` rows and `E` epochs:
  - Each epoch: loop over all `N` rows sample‑by‑sample.
  - Number of updates per epoch = `N`.
  - Total updates = `E × N`.
- In transcript terms:
  - If there are 10 epochs and 50 rows:
    - Updates = `10 × 50 = 500` times.

### Pseudocode Logic
- Decide epochs `T`.
- Outer loop: `for epoch in range(T)`:
  - Optionally shuffle dataset once per epoch.
  - Inner loop: over each row `i` (or pick random indices one by one):
    - Take a single point:
      - `x_i`, `y_i`.
    - Compute `y_hat_i` for that point.
    - Compute `loss_i`.
    - Compute gradient based on `loss_i`.
    - Update `W` and `b` using `loss_i`’s gradient.

### Properties of the Updates
- Update frequency:
  - Much higher than batch gradient descent.
  - Hence often faster convergence (in terms of *number of epochs* needed to get near a good solution).
- Shuffling (randomization):
  - Done to avoid bias from always seeing data in the same order.
  - Random point selection or shuffled order helps reduce systematic biases.

### Convergence Speed vs Time
- **Convergence in terms of epochs / updates:**
  - For the same number of epochs, SGD tends to move towards the solution faster than batch, because:
    - It updates far more often (per epoch).
- **Wall‑clock time per epoch:**
  - SGD takes more time than batch:
    - More individual updates.
    - Demonstrated in Keras example: `batch_size = 1` used ~10 seconds where batch used ~0.5 seconds for the same epochs.

### Loss Curve Shape
- SGD’s loss curve is typically noisy:
  - Shows spikes and oscillations.
  - Loss may go up and down while overall trending downward.
- Explanation:
  - Each update’s direction is determined by a single random sample.
  - That sample’s gradient may not represent the full dataset direction, causing zig‑zagging.

### 2D/3D Intuition (Local vs Global Minimum)
- Imagine a 3D loss surface with multiple minima:
  - Global minimum: best possible.
  - Local minima: suboptimal basins.
- Batch gradient descent:
  - Moves smoothly downhill.
  - If it falls into a local minimum, it may stay there (no inherent randomness to escape).
- SGD:
  - Due to noise and randomness in updates, its path looks jittery, like a drunk walk.
  - This randomness can:
    - Help jump out of local minima.
    - Allow reaching closer to the global minimum.
- Trade‑off:
  - Benefit: ability to escape local minima and explore more of the loss landscape.
  - Drawback: may not converge exactly/stably to the best point; may keep oscillating around it and give an approximate solution rather than a perfectly stable one.

---

## Comparing Batch vs Stochastic Gradient Descent

### Update Frequency
- Batch:
  - Updates per epoch = 1.
- SGD:
  - Updates per epoch = number of rows `N`.
- Consequence:
  - SGD has higher frequency of weight updates; moves parameters more often.

### Time per Epoch vs Convergence per Epoch
- Time to run 1 epoch:
  - Batch is faster; fewer updates.
  - SGD is slower; many more updates.
- Convergence toward good weights (if measured per epoch):
  - SGD usually converges in fewer epochs than batch because of more updates.
- Overall judgment:
  - If you fix *number of epochs*, batch finishes faster in wall‑clock time.
  - If you fix *number of updates* or look at how quickly loss decreases *per update*, SGD can reach good regions faster.

### Stability of Loss
- Batch:
  - Stable, smooth decrease.
- SGD:
  - Noisy, with spikes and oscillations.

### Local Minima and Randomness
- Batch:
  - May get stuck in local minima; low randomness.
- SGD:
  - Randomness can:
    - Help escape local minima.
    - But also hinder precise convergence.

### Memory and Vectorization
- Batch:
  - Uses full vectorization (dot products) over the whole dataset.
  - Requires loading the entire dataset into memory.
  - Not suitable for very large datasets that don’t fit into RAM.
- SGD:
  - Processes one sample at a time; doesn’t need the full dataset in memory simultaneously.
  - But doesn’t fully exploit vectorization on the entire dataset in a single operation.

---

## Mini‑Batch Gradient Descent

### Motivation
- Need a method that:
  - Is faster (like batch, leveraging vectorization).
  - Has better convergence qualities (like SGD, more frequent updates).
  - Avoids huge memory requirements of full batch.
- Mini‑batch gradient descent is the middle ground between batch and SGD.

### Core Idea
- Split the dataset into small batches (mini‑batches) of size `B`.
- For each mini‑batch:
  - Use vectorized operations across that mini‑batch.
  - Compute predictions and loss for those `B` samples.
  - Compute gradient from only those `B` samples.
  - Update weights and biases once per mini‑batch.
- Repeat this across all mini‑batches in an epoch.

### Example of Batch Partitioning
- Dataset has 320 rows; choose `batch_size = 32`:
  - Total number of mini‑batches per epoch: `320 / 32 = 10`.
  - For each inner step:
    - Work on 32 rows.
    - Perform vectorized forward, loss, gradient, and 1 update.
  - Per epoch:
    - Number of updates = 10.

- If dataset has 400 rows and `batch_size = 150`:
  - Theoretical division: `400 / 150 ≈ 2.66`.
  - Actual batches:
    - Batch 1: 150 rows.
    - Batch 2: 150 rows.
    - Batch 3: remaining 100 rows.
  - So 3 batches, 3 updates per epoch.

### General Formulas
- Given:
  - `N` = number of rows in dataset.
  - `B` = batch size.
- Number of mini‑batches per epoch:
  - `num_batches = ceil(N / B)`.
- Number of updates per epoch:
  - Equals `num_batches`.
- Total updates over `E` epochs:
  - `E × num_batches`.

### Relationship to Batch and SGD
- When `batch_size = N`:
  - Mini‑batch becomes batch gradient descent.
- When `batch_size = 1`:
  - Mini‑batch becomes stochastic gradient descent.
- In typical practice:
  - Choose `1 < batch_size < N`, giving behavior between the two extremes.

### Speed and Convergence Ordering
- Speed (per epoch, wall‑clock time), for same epochs:
  - Fastest: batch gradient descent (fewest updates).
  - Middle: mini‑batch gradient descent.
  - Slowest: SGD (most updates).
- Convergence speed (towards solution), for same epochs:
  - Slowest: batch gradient descent (fewest updates per epoch).
  - Middle: mini‑batch gradient descent.
  - Fastest: SGD (most updates per epoch).
- Loss curve smoothness:
  - Batch: smooth.
  - Mini‑batch: moderately noisy.
  - SGD: highly noisy.

### Memory & Vectorization Advantages
- Mini‑batch uses vectorization on batch chunks (dot product on `B` rows at a time).
- Only a subset of data needs to be in memory at a time, so:
  - It scales better to large datasets than full batch.
- Still gets benefits of:
  - Faster computation from vectorization.
  - Some noise from smaller samples, helping with generalization and escaping local minima.

### Practical Default
- In practice, “gradient descent” in deep learning almost always means mini‑batch gradient descent.
- Frameworks (like Keras) use the `batch_size` parameter:
  - `batch_size = N` ⇒ batch gradient descent.
  - `batch_size = 1` ⇒ SGD.
  - Intermediate values ⇒ mini‑batch.

---

## `batch_size` in Keras / Training APIs

### Interpretation of `batch_size`
- `batch_size` controls gradient descent variant:
  - `batch_size = total_number_of_rows`:
    - All rows in one batch ⇒ batch gradient descent.
  - `batch_size = 1`:
    - Each row individually ⇒ stochastic gradient descent.
  - `1 < batch_size < N`:
    - Mini‑batch gradient descent.
- For dataset with 320 rows:
  - `batch_size = 320` ⇒ 1 update/epoch (batch).
  - `batch_size = 1` ⇒ 320 updates/epoch (SGD).
  - `batch_size = 32` ⇒ 10 updates/epoch (mini‑batch).

### Choosing `batch_size` as a Power of Two
- Common practice: choose batch sizes as powers of two (e.g., 8, 16, 32, 64, 128, 256).
- Reason (as described):
  - System memory and hardware architectures are optimized for binary sizes.
  - Powers of 2 often utilize RAM more effectively and can give performance improvements.
- Not strictly required:
  - You *can* set `batch_size` to any integer (e.g., 10, 15).
  - Powers of 2 are a performance optimization, not a hard rule.

### Non‑Divisible `batch_size`
- If `N` is not perfectly divisible by `batch_size`:
  - Final batch may contain fewer rows.
- Example (`N = 40`, `batch_size = 10`):
  - 4 batches of size 10 (perfect division) ⇒ 4 updates.
- Example (`N = 40`, `batch_size = 18`):
  - Batches:
    - Batch 1: 18 rows.
    - Batch 2: 18 rows.
    - Batch 3: 4 rows.
  - Total: 3 updates per epoch.
- General behavior:
  - Last batch is simply smaller; training still uses all remaining rows.

---

## Summary of Trade‑offs

### Batch Gradient Descent
- Pros:
  - Fast per epoch (few updates).
  - Smooth, stable convergence.
  - Clean gradient estimates.
- Cons:
  - Requires full dataset in memory.
  - Can get stuck in local minima.
  - Fewer updates per epoch; slower progress in terms of updates.

### Stochastic Gradient Descent (SGD)
- Pros:
  - Many updates per epoch; can approach solutions with fewer epochs.
  - Randomness helps escape local minima.
- Cons:
  - Noisy loss curve; convergence less stable.
  - More time per epoch due to many updates.
  - May only converge to approximate neighborhood, not a perfectly stable point.

### Mini‑Batch Gradient Descent
- Pros:
  - Middle ground: gets stability + randomness.
  - Leverages vectorization on small chunks.
  - Works on larger datasets without requiring full dataset in memory.
  - Typically used in practice: “best of both worlds.”
- Cons:
  - Not as smooth as full batch, and not as exploratory (random) as pure SGD.
  - Choice of `batch_size` is an extra hyperparameter to tune.

## Key images

### GRADIENT DESCENT EXPLANATION TEXT.
*Timestamp: 43s*

### GRADIENT DESCENT UPDATE EQUATION AND COMPARISON NOTE.
*Timestamp: 1090s*

### COMPARISON OF TRAINING SPEEDS GIVEN SAME NUMBER OF EPOCHS.
*Timestamp: 1165s*

### "TRAINING LOSS PLOT AND ACCURACY METRICS OUTPUT"
*Timestamp: 1423s*

### BATCH SIZE CALCULATION EXAMPLE.
*Timestamp: 2259s*

## Interview question bank

*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*

> **No attributed questions were extracted.** For reliable results, set **`TAVILY_API_KEY`** in `.env` (Tavily search + full-page extract). DuckDuckGo-only mode often returns snippets too short to tie questions to a company.

### Summary table

| Topic | # Companies | # Questions | Top sources |
|-------|-------------|-------------|-------------|
| Gradient descent basics | 0 | 0 |  |
| Loss function and objective minimization | 0 | 0 |  |
| Gradient descent update rule and learning rate | 0 | 0 |  |
| Backpropagation as gradient descent | 0 | 0 |  |
| Gradient descent variants | 0 | 0 |  |
| Batch gradient descent | 0 | 0 |  |
| Stochastic gradient descent | 0 | 0 |  |
| Mini-batch gradient descent | 0 | 0 |  |
| Vectorization in gradient computations | 0 | 0 |  |
| Update frequency and convergence behavior | 0 | 0 |  |
| Loss curve stability and local minima | 0 | 0 |  |
| Batch size hyperparameter and its effects | 0 | 0 |  |
| Choosing batch size as power of two | 0 | 0 |  |

### Coverage report

- Extracted topics: 13
- Questions with attribution: 0
- Excluded (unattributed): 0

**Topic coverage notes:**
Content centers on gradient descent and its three variants (batch, stochastic, mini‑batch), with emphasis on update rules, batch_size behavior in frameworks, vectorization, convergence properties, and practical engineering trade‑offs such as memory and power‑of‑two batch sizes.

### Questions by topic → company
