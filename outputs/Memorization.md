
# Memorization, Dynamic Programming, and Backpropagation in Deep Neural Networks

## Definition of Memorization (Memoization)
- In computing, memoization is an optimization technique used to speed up programs by:
  - Storing the results of expensive function calls
  - Returning the stored result when the same input occurs again
- Trade-off:
  - Reduces computation time
  - Increases memory/space usage to store intermediate results
- Widely used in:
  - Dynamic Programming (DP)
  - Internal implementations of optimization/learning libraries (e.g., in gradient-based algorithms)


## General Concept of Memoization
- Abstract pattern:
  - Identify repeated subcomputations
  - Compute each such subcomputation once
  - Store its result
  - Reuse the stored result whenever needed again
- Essence:
  - “Spend space to save time”
  - Typical in dynamic programming and other optimization-oriented code

## Deep Neural Networks and Increasing Complexity
- Previously considered:
  - Neural networks with a single hidden layer
- Now considering:
  - Networks with more than one hidden layer (deeper networks)
- Example architecture:
  - 4 layers total:
    - Input layer
    - Hidden layer 1
    - Hidden layer 2
    - Output layer
- Consequence of deeper networks:
  - Backpropagation derivatives become more complex
  - Derivative formulas involve many chain rule multiplications
  - More weights → more paths from each weight to the loss → more complex expressions

## Backpropagation in Multi-Layer Networks
- Aim:
  - Compute ∂L/∂w for various weights w in a multi-layer network
- For a weight in a deeper layer (e.g., closer to output):
  - Loss L depends on a node’s output via a single path
  - Derivative structure is relatively simple:
    - Use standard chain rule along one chain
- For a weight in an earlier (first) hidden layer:
  - The output of that node fans out into multiple downstream nodes and then to the loss
  - Example:
    - A node’s output goes:
      - Path 1: through one node → then to output → to loss
      - Path 2: through another node → then to output → to loss
  - ∂L/∂w must aggregate influence via all such paths

## Multi-Path Chain Rule Structure
- Generic math situation:
  - A variable x influences y and z
  - y and z both influence L
- Then:
  - L = L(y, z)
  - y = y(x), z = z(x)
- Derivative:
  - dL/dx = (∂L/∂y)(dy/dx) + (∂L/∂z)(dz/dx)
- Neural network analogy:
  - A single weight (or node output) affects multiple downstream nodes
  - Need to sum contributions from all paths to the loss

## Applying to First-Layer Weights
- Consider weight w₁₁ in the first hidden layer:
  - Affects node a₁ in hidden layer 1
  - a₁ affects:
    - Node h₂₁ in hidden layer 2
    - Node h₂₂ in hidden layer 2
  - These in turn affect:
    - Final output y-hat
    - Then loss L
- ∂L/∂w₁₁ has two major path contributions:
  - Path through h₂₁
  - Path through h₂₂
- Symbolically:
  - ∂L/∂w₁₁ = (∂L/∂… via first path)(∂…/∂w₁₁) + (∂L/∂… via second path)(∂…/∂w₁₁)
- Each path includes:
  - Product of derivatives along that path (chain rule)
  - Leads to long, complex expressions

## Complexity Growth With More Hidden Layers
- With just 2 hidden layers:
  - Formulas for ∂L/∂w in earlier layers already become lengthy and tedious
- If more hidden layers are added:
  - Number of intermediate derivatives grows
  - Number of paths from a weight to output increases
  - Formulas become:
    - Hard to write
    - Hard to compute directly without structure
- Observed issue:
  - To compute some derivatives, you must recompute many sub-derivatives multiple times if not careful

## Role of Memorization in Backpropagation
- Observation:
  - In the manual derivation, certain derivative terms appear multiple times in expressions for different weights
  - Example:
    - A derivative like ∂L/∂(some intermediate activation) is needed:
      - For one weight’s gradient
      - Again for another weight’s gradient
- Idea:
  - Once such a derivative is computed for one purpose, store it
  - Reuse it when needed to compute other gradients
- Equivalent to:
  - Applying memoization to the derivatives in the chain rule graph

## Concrete Memoization Pattern in Backprop
- In a backpropagation pass:
  - Start from the output and move backward layer by layer
  - At each node/layer:
    - Compute a derivative like ∂L/∂(activation or pre-activation)
    - Store this result
- Later, when computing ∂L/∂w for weights “behind” this node:
  - Do not recompute ∂L/∂(that node)
  - Just reuse the stored derivative
- Effect:
  - Prevents recalculating the same derivative repeatedly for different upstream weights
  - Reduces repetitive work analogous to naive Fibonacci recursion

## Backpropagation as Chain Rule + Memorization
- Conceptual decomposition:
  - Mathematics part:
    - Chain rule from calculus
    - Managing multiple paths and summing contributions
  - Computer science part:
    - Memoization (from dynamic programming)
    - Storing intermediate results to reuse
- Backpropagation can be seen as:
  - Chain rule applied systematically over a computation graph
  - Enhanced by memoization of intermediate derivatives to keep it efficient

## Benefits of Memorization in Deep Networks
- Without memoization-like reuse:
  - Gradient computations in deep networks would:
    - Be extremely slow
    - Involve exponential-like blow-up in repeated work
- With memoization:
  - Time to compute all gradients is greatly reduced
  - Training becomes practically feasible for deep networks
  - Memory usage slightly increases (to store intermediate derivatives), but:
    - Time savings far outweigh memory cost

## Key Takeaways
- As networks get deeper:
  - Derivative formulas (∂L/∂w) become more complex
  - Many intermediate derivatives are reused across multiple weights
- If the same derivative expressions are recalculated many times:
  - Time is wasted
  - Overall training slowdowns occur
- Using memoization:
  - Store intermediate derivatives during backpropagation
  - Reuse them when needed
  - This is the same optimization concept as in dynamic programming
- Backpropagation core idea:
  - Chain rule (math)
  - Implemented with memoization (computer science)
  - Their combination enables efficient training in libraries like TensorFlow / PyTorch (implied as “such libraries”)
