## Notes

# Nesterov Accelerated Gradient (NAG) vs Momentum – Concept, Math, Geometry & Keras

## Optimizers Overview
- Optimizer: an algorithm used to find optimal weight values `w` that minimize the loss.
- Common baseline optimizer: Gradient Descent (GD) with variants:
  - Batch Gradient Descent
  - Stochastic Gradient Descent (SGD)
  - Mini-batch Gradient Descent
- These vanilla GD methods can be slow, leading to the need for advanced optimizers like Momentum and NAG.

## Linear Regression Demo Setup
- Problem: simple linear regression with Mean Squared Error (MSE) as cost function.
- Visualization:
  - 3D error surface (MSE vs parameters `m` and `b`).
  - 2D contour plot (projection of 3D surface).
  - Data plot with regression line adjusting over iterations.
- Goal: start from random `(m, b)` and reach red point (true optimal `m`, `b`).

## Batch Gradient Descent Behavior
- Starting from a random point in parameter space.
- Takes about 25–30 epochs (or iterations) to converge for the demo data.
- Considered relatively slow, motivating advanced techniques.

## Momentum Optimizer – Intuition & Behavior
- Momentum uses a "velocity" term to accumulate past gradients:
  - Fast directions are emphasized, like a ball rolling downhill.
- Visual behavior:
  - On error surface, path looks like a ball rolling down, overshooting, crossing the minimum, coming back, and gradually settling.
  - On contour plot, path oscillates around the valley and converges eventually.
- With `β = 0.9` (high momentum factor):
  - Old velocity is given high weight.
  - Leads to strong push, larger overshoot, more oscillations.
  - In the shown setup, takes ~50 epochs to settle at minimum (more than plain batch GD).
- However, advantage:
  - Far from the minimum, it moves very quickly towards the minimum region.
  - Example: within ~3 iterations, it crosses the minimum region, versus many more for standard GD.
- Tuning `β` (decay/momentum factor):
  - Reducing `β` lowers the effect of old velocity.
  - Oscillations get smaller; convergence becomes smoother.
  - Example: with a lower `β`, convergence epochs drop to ~38–39 vs ~50.
- Key takeaway:
  - Momentum travels the long initial distance to the minimum much faster.
  - Needs careful tuning of `β` to control oscillations.

## Limitation of Momentum
- Main issue: oscillations around the minimum.
- On simple convex problems, can be mitigated by reducing `β`.
- On non-convex, more complex error surfaces:
  - Oscillations can be more problematic.
  - Can increase convergence time or cause instability.
- This motivates an improved method: Nesterov Accelerated Gradient (NAG/NAG).

## What is Nesterov Accelerated Gradient (NAG)?
- NAG (referred to as NAG or "Nag"): Nesterov Accelerated Gradient.
- Conceptual summary:
  - A way to implement momentum with a small modification.
  - Designed to damp (reduce) oscillations while keeping advantages of momentum.
- Performance:
  - Often converges faster than standard momentum.
  - Reaches minima more quickly with reduced overshooting.
- High-level idea:
  - Instead of computing gradient at current position, compute gradient at a "look-ahead" position (after a momentum step).
  - This anticipatory step helps reduce oscillations.

## Momentum – Mathematical Formulation
- Update rules (Momentum with SGD):
  - Velocity update:
    - `v_t = β * v_(t-1) + η * ∂L/∂w_t`
      - `β`: momentum factor (decay).
      - `η`: learning rate.
      - `∂L/∂w_t`: gradient of loss wrt weights at time `t`.
  - Weight update:
    - `w_(t+1) = w_t - v_t`
- If you expand `w_(t+1)`:
  - `w_(t+1) = w_t - (β * v_(t-1) + η * ∂L/∂w_t)`
  - The update is a combination of:
    - Old velocity term: `β * v_(t-1)`
    - Current gradient term: `η * ∂L/∂w_t`
- Comparison with vanilla GD:
  - Vanilla GD update: `w_(t+1) = w_t - η * ∂L/∂w_t`
  - Momentum adds an extra push from the accumulated velocity.
- Geometric intuition (1D loss `L(w)`):
  - At a position on the curve:
    - Vanilla GD moves based only on current gradient.
    - Momentum moves based on:
      - Past velocity direction and magnitude.
      - Current gradient direction.
  - Result: larger, faster jumps in the descent direction, but can overshoot.

## NAG – Mathematical Formulation from Momentum
- Key conceptual difference:
  - Momentum: compute momentum and gradient at the same current position, then update once.
  - NAG: first move using momentum, then compute gradient at the new ("look-ahead") position, and then update.
- Formal steps (NAG):
  1. Compute look-ahead position `w_lookahead` using momentum:
     - `w_lookahead = w_t - β * v_(t-1)`
       - This is "where we would go" if we moved purely by momentum.
  2. Compute new velocity using gradient at the look-ahead position:
     - `v_t = β * v_(t-1) + η * ∂L/∂w_lookahead`
       - Gradient is evaluated at `w_lookahead` (not at `w_t`).
  3. Update weights:
     - `w_(t+1) = w_t - v_t`
- Intuition of the decomposition:
  - Step 1: apply only the momentum part to get the look-ahead point.
  - Step 2: at that look-ahead point, compute gradient and adjust.
  - Combined effect: update from original `w_t` using:
    - The momentum "push" plus
    - A gradient correction at the anticipated future location.

## Geometric Intuition – Momentum vs NAG (1D Error Surface)

### Momentum Geometry
- Consider loss surface `L(w)` with a minimum.
- At each point:
  - Compute momentum (from past steps).
  - Compute gradient at that same point.
  - Take one combined step based on both.
- Example behavior:
  - Starting from left side of minimum with negative slope:
    - Gradient is negative, so update direction is positive.
    - Momentum also points positive due to accumulated movement.
  - You repeatedly move in positive direction, building strong positive momentum.
  - When you overshoot past the minimum:
    - Gradient flips sign (becomes positive), pushing you back.
    - But momentum may still be large positive.
    - Net effect:
      - Still move slightly further to the right (overshoot more).
  - Eventually:
    - Momentum decreases, changes sign, combines with gradient to pull you back.
    - You oscillate around the minimum with decreasing amplitude.
- Path:
  - Start left of minimum, overshoot to the right, U-turn back, several smaller U-turns until settling at the minimum.

### NAG Geometry
- Same loss surface and starting point.
- At each step:
  1. Use only momentum to compute a look-ahead point:
     - Move from current position to `w_lookahead`.
  2. Compute gradient at `w_lookahead`.
  3. Use that gradient to decide the corrective move.
- Crucial difference:
  - Momentum:
    - At a point near the minimum, momentum and gradient are combined at that point, possibly pushing you further past the minimum before correcting.
  - NAG:
    - You first move to where momentum alone would take you.
    - At that new point, the gradient may already indicate that you should go in the opposite direction.
    - So the next correction step is smaller or even backwards, effectively taking a "shallower" U-turn.
- Result:
  - When you would overshoot with momentum, NAG "anticipates" and reduces the overshoot.
  - Oscillations are damped earlier, leading to a quicker and smoother convergence.
- Visual summary:
  - Momentum path: longer overshoot, larger U-turn arcs around the minimum.
  - NAG path: smaller overshoot, early correction, more direct approach to the minimum.

## Why NAG Reduces Oscillations
- By using gradient at the look-ahead position:
  - You are effectively "seeing" future loss landscape where momentum will take you.
  - If that future position lies beyond the minimum, the gradient there will point back, so the net correction is reduced or reversed.
- This anticipatory step:
  - Decreases the effective step size when close to the minimum on steep sides.
  - Leads to smaller, better-aligned updates.
  - Reduces oscillations relative to classical momentum.

## Potential Disadvantage of NAG
- NAG damps momentum around minima.
- In non-convex surfaces with multiple local minima:
  - Momentum might help you "jump over" shallow local minima due to strong velocity (ball crosses valley).
  - NAG, by damping oscillations and effective momentum, may not gather enough energy to cross such barriers.
  - You can end up stuck in a local minimum, oscillating very close and settling there instead of reaching a better global minimum.
- So:
  - NAG’s damping of oscillations can improve convergence stability.
  - But it may reduce the ability to escape some local minima compared to strong momentum.

## Keras Implementation of SGD, Momentum, and NAG
- In Keras, all three are controlled using the `SGD` optimizer class.
- Key parameters:
  - `momentum`
  - `nesterov`
- Configurations:
  - Plain SGD:
    - `momentum = 0.0`
    - `nesterov = False`
  - SGD with Momentum:
    - `momentum = β` (e.g., 0.9, 0.8, etc.)
    - `nesterov = False`
  - Nesterov Accelerated Gradient (NAG):
    - `momentum = β` (non-zero)
    - `nesterov = True`
- Using these flags, you can switch between:
  - Standard SGD
  - Momentum SGD
  - NAG SGD, all with simple parameter changes in Keras.

## Key images

### "BATCH GRADIENT DESCENT VISUALIZATION WITH GROUND TRUTH AND LOSS PLOTS"
*Timestamp: 134s*

### "MOMENTUM OPTIMIZER VISUALIZATIONS: GROUND TRUTH, WEIGHTS & LOSS, 3D LOSS SURFACE."
*Timestamp: 404s*

### MOMENTUM EQUATION AND DECAY FACTOR EXPLANATION.
*Timestamp: 584s*

### "NESTEROV ACCELERATED GRADIENT EQUATIONS AND DIAGRAM"
*Timestamp: 764s*

### NESTEROV ACCELERATED GRADIENT EXPLANATION WITH EQUATIONS AND DIAGRAMS.
*Timestamp: 944s*

### NESTEROV ACCELERATED GRADIENT EXPLANATION WITH EQUATIONS AND DIAGRAMS.
*Timestamp: 1034s*

## Interview question bank

*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*

### Summary table

| Topic | # Companies | # Questions | Top sources |
|-------|-------------|-------------|-------------|
| Gradient descent variants | 11 | 14 | tryexponent.com, interviewnode.com, linkedin.com, machinelearningmastery.com, an |
| Momentum optimization | 0 | 0 | rohan-paul.com, geeksforgeeks.org |
| Nesterov Accelerated Gradient | 0 | 0 | machinelearningmastery.com, jlmelville.github.io |
| Mathematical formulation of Momentum and NAG | 0 | 0 |  |
| Geometric intuition of optimization methods | 0 | 0 |  |
| Linear regression error surface visualization | 0 | 0 |  |
| Convergence speed and oscillations in optimizers | 2 | 2 |  |
| Convex vs non-convex optimization behavior | 0 | 0 |  |
| Advantages and disadvantages of NAG | 0 | 0 |  |
| Hyperparameter tuning of momentum factor | 0 | 0 |  |
| Keras implementation of SGD, Momentum, and NAG | 0 | 0 |  |
| Optimizers as algorithms to minimize loss | 0 | 0 |  |
| **Total (all in bank)** | 11 | 16 | See sections below |

### Coverage report

- Extracted topics: 12
- Questions with attribution: 16
- Excluded (unattributed): 3

**Topic coverage notes:**
Content focuses on optimization in deep learning: baseline gradient descent variants, Momentum, and Nesterov Accelerated Gradient, with emphasis on their math, geometric intuition, convergence behavior (oscillations), and practical implementation in Keras for a linear regression demo. Non-convex landscapes and NAG’s trade-offs are briefly discussed.

### Questions by topic → company

#### Convergence speed and oscillations in optimizers

**Nvidia**
- **Q:** How do neural networks learn from data?
  - *Role/level:* Data Scientist
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* ### 1. How do neural networks learn from data?
...
ℹ️

This interview question was asked at Nvidia.

**Snapchat**
- **Q:** Explain mini-batch and stochastic gradient descent
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* Watch an MLE from Snapchat answer this interview question: "Explain mini-batch and stochastic gradient descent".


#### Gradient descent variants

**Amazon**
- **Q:** Describe linear regression
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* - Watch Amazon MLE answer, "Describe linear regression".

**Apple**
- **Q:** Can you implement batch normalization using NumPy?
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* ℹ️

This interview question was asked at Apple. "Implement batch normalization with NumPy".

**FAANG**
- **Q:** Implement Logistic Regression from scratch
  - *Role/level:* Mixed
  - *Source:* https://www.interviewnode.com/post/ace-your-ml-interview-50-commonly-asked-questions-at-faang-companies
  - *Evidence:* ##### **1. Implement Logistic Regression from scratch**

Problem: Write a Python function to implement logistic regression using gradient descent.

- **Q:** Implement k-means clustering
  - *Role/level:* Mixed
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* - Watch Snap MLE answer, "Implement k-means clustering".

**Meta**
- **Q:** Design Instagram ranking model
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* Sneak peek:  
- Watch Meta MLE answer, "Design Instagram ranking model".

- **Q:** Design an evaluation framework
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* 3. Design an evaluation framework. (Meta)

**Nielsen**
- **Q:** What is the difference between batch gradient descent and stochastic gradient descent (SGD)?
  - *Role/level:* Data Scientist
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
8. Difference between batch gradient descent and stochastic gradient descent (SGD)?

**Nvidia**
- **Q:** Explain deep learning to non-technical stakeholders
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* ℹ️

This interview question was asked at Nvidia. "Explain deep learning to non-technical stakeholders".

**OpenAI**
- **Q:** Explain gradient descent
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* 1. Explain gradient descent. (OpenAI)

- **Q:** Explain gradient descent and model optimization
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* 1. Explain gradient descent. (OpenAI)
...
https://www.tryexponent.com/questions/4359/gradient-descent-and-model-optimization

**Pinterest**
- **Q:** Explain overfitting
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* ### 1. What is overfitting? How can you avoid it?
...
ℹ️

This question was asked at Pinterest. "Explain overfitting".

**Snap**
- **Q:** Implement k-means clustering
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* - Watch Snap MLE answer, "Implement k-means clustering".

**Snapchat**
- **Q:** Explain mini-batch and stochastic gradient descent
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* ℹ️

Watch an MLE from Snapchat answer this interview question: "Explain mini-batch and stochastic gradient descent".

**Spotify**
- **Q:** Design Spotify recommendation system
  - *Role/level:* MLE
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* 4. Design a recommendation system. (Spotify)


### Unattributed (excluded from main list)

- **Topic:** Gradient descent variants — YouTube page mentions 'ML Interview Questions: Gradient Descent' as a video title but does not show any explicit question sentences in the visible transcript/description. — _ML Interview Questions: Gradient Descent_
- **Topic:** Momentum optimization — Rohan's article clearly states this as an interview question, but no specific company is mentioned in the same block; cannot attribute to a real tech employer as required. — _What is momentum in the context of gradient-based optimization? Write the update equations for gradient descent with momentum and explain how momentum helps accelerate convergence and can help escape _
- **Topic:** Gradient descent variants — Devinterview.io gradient-descent article is generic and not tied to a specific company name; per rules, questions without company context must be skipped. — _What are the main variants of gradient descent algorithms?_
