# Activation Functions in Neural Networks

## Metadata
- **Created at:** 2026-03-29 07:15:22
- **Video ID:** 7LcUkgzx3AY

## Notes

# Activation Functions in Neural Networks

## Overview of Activation Functions
- Activation functions (also called **transfer functions**) are core components of artificial neural networks.
- In an artificial neuron:
  - Compute weighted sum + bias:
    $$z = \sum_i w_i x_i + b$$
  - Pass $z$ through an activation function $g(\cdot)$:
    $$a = g(z)$$
- $g(\cdot)$ acts as a **mathematical gate** between:
  - Net input $z$ (weighted sum + bias).
  - Final output $a = g(z)$ passed to the next layer.
- The activation function decides:
  - Whether a neuron will be “activated” or not.
  - If activated, **how strongly** it responds.

## Computation Inside a Neuron (Forward Pass)
- Scalar form for one neuron:
  - Net input:
    $$z = \sum_i w_i x_i + b$$
  - Activation:
    $$a = g(z)$$
- Layer-wise notation (clarification):
  - $a^{(l)}$ denotes the **activations of layer $l$**.
  - $z^{(l)}$ denotes the **pre-activations** (weighted sum + bias) of layer $l$.
- Vector/matrix form for a layer:
  - Inputs: $\mathbf{x}$, weights: $W$, biases: $\mathbf{b}$.
  - Net input and activation:
    $$
    \mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad
    \mathbf{a} = g(\mathbf{z})
    $$

---

# Need for Activation Functions

## Linear vs Non-Linear Behavior (Conceptual)
- If **no activation function** is used, or $g$ is **linear** (e.g., identity $g(z) = z$):
  - Each layer performs an affine transformation:
    $$\mathbf{y} = W\mathbf{x} + \mathbf{b}$$
  - Composition of affine transformations is still affine.
  - The full network behaves like a **linear model**:
    - With a linear output: like **linear regression**.
    - With a sigmoid output: like **logistic regression**.
  - It **cannot capture non-linear patterns** or non-linearly separable data.
- If a **non-linear activation function** is used in hidden layers (e.g., sigmoid, $\tanh$, ReLU):
  - Overall input–output relationship becomes **non-linear**.
  - The network can model **non-linearly separable** data and complex decision boundaries.
  - This is what the instructor calls **“non-linear pattern / non-linear editing capture power”**.

## Explicit Scalar Derivation (Mirroring Transcript Style)
- Consider a small feedforward network (one hidden layer, identity activations) with scalar inputs:
  - Inputs: $x_1, x_2$.
  - Hidden neuron:
    - Weights: $w_{11}, w_{12}$, bias: $b_1$.
    - Pre-activation:
      $$z_1 = w_{11} x_1 + w_{12} x_2 + b_1$$
    - Activation (identity $g(z) = z$):
      $$a_1 = g(z_1) = z_1$$
  - Output neuron:
    - Weights: $w_{21}$, bias: $b_2$.
    - Output (again identity for illustration):
      $$y = w_{21} a_1 + b_2$$
- Substitute $a_1$:
  $$
  y = w_{21} (w_{11} x_1 + w_{12} x_2 + b_1) + b_2
  $$
  $$
  y = (w_{21} w_{11}) x_1 + (w_{21} w_{12}) x_2 + (w_{21} b_1 + b_2)
  $$
- This is a **first-degree polynomial** (degree 1) in $(x_1, x_2)$:
  - A simple **linear relationship** between input and output.
  - Matches the instructor’s statement: input–output relation is **linear (degree-1 polynomial)** when hidden layers are linear.

## General Matrix Derivation: Linear Network Collapses to Linear Map
- Consider a feedforward network with:
  - Input $\mathbf{x}$.
  - Hidden layer with linear activation (identity).
  - Output layer with linear activation (identity).
- Let:
  - Hidden layer: weights $W_1$, biases $\mathbf{b}_1$.
  - Output layer: weights $W_2$, biases $\mathbf{b}_2$.
- Forward propagation:
  - Hidden activations (identity activation $g(z) = z$):
    $$
    \mathbf{a}^{(1)} = W_1 \mathbf{x} + \mathbf{b}_1
    $$
  - Output:
    $$
    \mathbf{a}^{(2)} = W_2 \mathbf{a}^{(1)} + \mathbf{b}_2
    $$
- Substitute:
  $$
  \mathbf{a}^{(2)} = W_2 (W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
  $$
  $$
  \mathbf{a}^{(2)} = (W_2 W_1)\mathbf{x} + (W_2 \mathbf{b}_1 + \mathbf{b}_2)
  $$
- Again a single affine transformation:
  $$
  \mathbf{a}^{(2)} = W' \mathbf{x} + \mathbf{b}', \quad
  W' = W_2 W_1, \ \mathbf{b}' = W_2 \mathbf{b}_1 + \mathbf{b}_2
  $$
- Conclusion:
  - Stacking linear layers **without non-linear activations** is equivalent to **one linear layer**.
  - With a linear output → behaves like **linear regression**.
  - With a sigmoid output → behaves like **logistic regression**.

## Concrete Experimental Illustration (2D Non-Linear Data)
- Dataset:
  - 2D data that is **non-linearly separable** (no single straight line can separate the classes).
- Model 1 (all-linear hidden layers):
  - Two hidden layers with 128 neurons each.
  - Hidden activations: **linear** (identity; effectively no activation).
  - Output layer: 1 neuron with **sigmoid** activation.
- Behavior after training (as described):
  - Decision boundary is essentially a **single straight line**.
  - Despite multiple layers, the model behaves like a **logistic regression** classifier:
    - Because hidden layers collapse into one affine transformation.
- Model 2 (non-linear hidden activations):
  - Same architecture, but hidden layers use **ReLU** activation.
  - Output layer: 1 neuron with **sigmoid** activation.
- Behavior after training:
  - Learns **complex, non-linear** decision boundaries.
  - Correctly captures the **non-linearly separable** structure of the data.
- Takeaways:
  - In this specific experiment: linear hidden layers + sigmoid output $\Rightarrow$ behavior like **logistic regression**.
  - In general: without non-linear activations, any deep ANN collapses to a **single affine map** and cannot capture non-linear data.

---

# Desirable Properties of an “Ideal” Activation Function

## 1. Non-Linearity
- Requirement:
  - Activation function $g(x)$ should be **non-linear**.
  - Relationship between input and output should **not** be globally of the form $ax + b$.
- Reason (as per instructor):
  - Non-linear activation adds **non-linear data capture power** to the network.
  - Enables capturing **non-linear patterns/relationships** in data.
- Related concept explicitly mentioned:
  - **Universal Approximation** idea:
    - With a **non-linear activation function** and **enough neurons/hidden units**, a network can capture **any non-linear relationship** (universal approximation theorem intuition).

## 2. Differentiability (Important but Not Strictly Mandatory)
- Preferred property:
  - $g(x)$ should be **differentiable**:
    - We can compute $\frac{dg(x)}{dx}$ (or $\frac{\partial g}{\partial x}$) at all relevant points.
- Reason (lecture context):
  - Training uses **gradient descent** and **backpropagation**.
  - These algorithms:
    - Rely on **derivatives of activation functions** to update weights.
    - If $g$ is not differentiable, standard gradient descent cannot be applied directly.
- Nuance emphasized:
  - Differentiability is **important** but **not strictly mandatory**.
  - Example given: **ReLU**:
    - Not differentiable at $x = 0$.
    - Still widely used in practice.
    - Frameworks handle the non-differentiable point with a convention; training still works.

## 3. Computational Efficiency
- Desired property:
  - Activation $g(x)$ and its derivative $g'(x)$ should be:
    - **Simple to compute**.
    - **Fast** to evaluate.
- Reason:
  - Activations and derivatives are computed repeatedly during training.
  - If computation is expensive:
    - Overall training becomes **slow**.
- Hence:
  - Prefer **computationally inexpensive** functions (e.g., simple algebraic / piecewise forms).

## 4. Zero-Centered / Mean-Near-Zero Outputs
- Desired property:
  - Activations should be **zero-centered** or have **mean approximately 0**.
  - That is, outputs of $g(x)$ over typical inputs are centered around $0$.
- Reason (as stated):
  - If outputs/activations are around **zero mean**, then:
    - Inputs to subsequent layers are more “balanced”.
    - Training tends to **converge faster**.
- Example explicitly mentioned:
  - **$\tanh$ activation**:
    - Output range: $(-1, 1)$.
    - Can produce **zero-centered activations**.
- Instructor’s emphasis:
  - When data (and thus activations) are more **normal / centered**, convergence is usually **faster**.
  - The key requirement is **mean near zero** (not necessarily strictly Gaussian-shaped).

## 5. Non-Saturating Behavior
- Definition (lecture framing):
  - **Saturating activation functions**:
    - “Squeeze” inputs into a **fixed, bounded interval**.
    - Examples:
      - Sigmoid: outputs in $(0,1)$.
      - $\tanh$: outputs in $(-1,1)$.
  - **Non-saturating activation functions**:
    - Do **not** compress inputs into a narrow fixed range.
    - Example: **ReLU**, which is unbounded on the positive side.
- Reason (as stated in the segment):
  - Saturating functions (e.g., sigmoid, $\tanh$):
    - Create issues like **vanishing gradient problems** during backpropagation.
    - Especially when outputs are squeezed into small intervals (e.g., $(0,1)$, $(-1,1)$).
  - Non-saturating functions:
    - Operate over a **wider range**.
    - Help avoid/mitigate **vanishing gradient** issues.
- Key point:
  - When choosing an activation function, prefer one that is **non-saturating in nature** so that it does **not squash inputs into a tiny range**.

---

# Saturating vs Non-Saturating Activation Functions

## Saturating Activation Functions (As in Lecture)
- Concept:
  - Functions that **squeeze** their input values into a **fixed, bounded range**.
  - When input is very large (positive or negative), output:
    - Becomes **almost constant** within that range.
- Examples (from transcript):
  - **Sigmoid**:
    - Any input is mapped to the interval $(0, 1)$.
  - **$\tanh$**:
    - Any input is mapped to the interval $(-1, 1)$.
- Consequence (as discussed):
  - These functions are called **saturating functions**.
  - In saturated regions:
    - Backpropagation experiences **vanishing gradient problems**.
    - Gradients get smaller as they propagate backwards, leading to:
      - **Very small updates** to earlier-layer weights.
      - Training may effectively **stall** (“weights will not update”).

## Non-Saturating Activation Functions (As in Lecture)
- Concept:
  - Functions that **do not squash** inputs into a narrow fixed interval.
  - Maintain useful behavior over a **larger input range**.
- Example:
  - **ReLU** (Rectified Linear Unit):
    - Defined as:
      $$\text{ReLU}(x) = \max(0, x)$$
    - For $x > 0$, outputs can grow without being clamped to a small interval.
    - Therefore, considered **non-saturating** (on the positive side).
- Practical implication (as per instructor):
  - With ReLU and similar non-saturating functions:
    - **Vanishing gradient** problem is much less severe.
    - Training of deeper networks is more feasible.

## Supplemental Mathematical Details (Standard, Not Fully Spelled Out in Transcript)
- Sigmoid:
  - Function:
    $$
    \sigma(x) = \frac{1}{1 + e^{-x}}
    $$
  - Derivative:
    $$
    \sigma'(x) = \sigma(x) \bigl(1 - \sigma(x)\bigr)
    $$
- Hyperbolic tangent:
  - Function:
    $$
    \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
    $$
  - Derivative:
    $$
    \tanh'(x) = 1 - \tanh^2(x)
    $$
- ReLU:
  - Function:
    $$
    \text{ReLU}(x) = \max(0, x)
    $$
  - Derivative (almost everywhere):
    $$
    \frac{d}{dx}\text{ReLU}(x) =
    \begin{cases}
    0, & x < 0 \\
    1, & x > 0
    \end{cases}
    $$
    (not defined at $x=0$; handled by convention in practice).
- These formulas are **standard** and help understand:
  - Why sigmoid/$\tanh$ have **small derivatives** in tails (saturation).
  - Why ReLU has a **constant derivative 1** on $x>0$ (non-saturating region).

---

# Sigmoid Activation Function (As Introduced Here)

## Definition and Basic Properties
- Sigmoid (logistic) activation function:
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  $$
- Range and limits (as described with the graph):
  - Output range: $(0, 1)$.
  - For very large positive $x$:
    - $\sigma(x) \approx 1$.
  - For very large negative $x$:
    - $\sigma(x) \approx 0$.
  - At $x = 0$:
    - $\sigma(0) = 0.5$.
- Graph (qualitative description):
  - S-shaped, monotonically increasing curve.
  - Steepest near $x = 0$.
  - Flattens near 0 and 1 at extremes.

## Derivative Behavior (Qualitative, Per Slides)
- The derivative curve (shown in lecture):
  - Maximum derivative at $x = 0$:
    - Approximately $0.25$ (peak of the derivative curve).
  - For $\lvert x \rvert$ large:
    - Derivative $\approx 0$.
  - Non-negligible derivative mainly in a central band (roughly between some negative and positive value).
- Interpretation:
  - Sigmoid is **most sensitive** near $x = 0$.
  - In the far tails:
    - Gradient becomes very small.
    - Contributes to **vanishing gradient** when stacked across many layers.

## Advantages Highlighted
- Output in $(0,1)$:
  - Can be interpreted as a **probability** (e.g., probability of class “1”).
- Use in binary classification (as mentioned):
  - **Output layer** activation for **binary classification problems**:
    - Example scenario: predicting **Yes/No** decisions.
    - Output close to 1 → likely “Yes”.
    - Output close to 0 → likely “No”.
- Non-linear:
  - Provides **non-linear decision boundaries** when used in hidden layers.
- Differentiable:
  - Smooth and differentiable for all real $x$ (helpful for gradient-based training).

## Disadvantages (Implied / Connected to Earlier Discussion)
- Saturating:
  - Squashes inputs into the range $(0,1)$.
  - In extreme regions:
    - Output is almost constant.
    - Derivative is very small.
  - Leads to **vanishing gradient** issues in deep networks.
- Not zero-centered:
  - Outputs are strictly positive $(0,1)$.
  - Mean activation is **not around 0**.
  - This can **slow convergence** compared to zero-centered activations (like $\tanh$).

---

# Tanh and ReLU (As Mentioned in This Segment)

## Hyperbolic Tangent ($\tanh$)
- Activation function (standard formula):
  $$
  \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
  $$
- Properties emphasized:
  - Output range: $(-1, 1)$.
  - **Zero-centered**:
    - Can have activations with mean near 0.
- Role in this lecture:
  - Example of an activation whose outputs can be **zero-centered**.
  - Also an example of a **saturating** activation (since it maps to $(-1,1)$), so:
    - Can contribute to **vanishing gradient** problems similar to sigmoid.

## ReLU (Rectified Linear Unit)
- Activation function (standard formula):
  $$
  \text{ReLU}(x) = \max(0, x)
  $$
- Key properties emphasized:
  - **Non-linear**.
  - **Non-saturating** for $x > 0$:
    - Outputs can grow without being squashed into a small interval.
  - Not differentiable at $x=0$, yet:
    - Still widely used in practice (supports the point that strict differentiability is not mandatory).
- Role in this lecture:
  - Used in the experimental example:
    - ReLU in hidden layers enabled learning **non-linear decision boundaries**.
  - Given as a primary example of a **non-saturating** activation function.

---

# Ideal Activation Function Summary (From This Lecture)

## Consolidated List of Five Properties
- An “ideal” activation function (as described here) should:
  1. **Be non-linear**:
     - So the network has **non-linear data capture power**.
     - Supports the **universal approximation** idea: with enough neurons and non-linear activation, can approximate complex non-linear relationships.
  2. **Be differentiable** (or almost everywhere differentiable):
     - So gradients can be computed for backpropagation and gradient descent.
     - Minor non-differentiable points (e.g., ReLU at 0) are acceptable.
  3. **Be computationally inexpensive**:
     - So forward $g(x)$ and backward $g'(x)$ computations are fast.
  4. **Produce zero-centered / mean-near-zero outputs**:
     - Activations should have mean approximately 0 (zero-centered), which:
       - Helps make the input to each layer more “normal/centered”.
       - Typically leads to **faster convergence**.
     - Example: $\tanh$ with range $(-1,1)$.
  5. **Be non-saturating**:
     - Should not squeeze inputs into a narrow bounded range with flat extremes.
     - Helps avoid **vanishing gradient** problems.
     - Example: ReLU (non-saturating on $x>0$).

---

# Scope Note (Adjusted to This Segment)

- These notes are based on the provided transcript segment and include:
  - Definitions and roles of activation functions.
  - The conceptual and algebraic demonstration that purely linear networks collapse to a single **linear/affine** model.
  - The **five ideal properties** of activation functions as listed by the instructor.
  - Discussion of:
    - **Saturating vs non-saturating** functions.
    - Examples: sigmoid, $\tanh$, ReLU.
    - Mention of the **universal approximation** idea.
- Some explicit formulas and derivative expressions (for sigmoid, $\tanh$, ReLU) are:
  - **Standard supplemental details**.
  - Included for mathematical clarity, even when not fully written out in the transcript.
- More advanced topics (e.g., full formal backpropagation derivations beyond what was verbally hinted) are **not expanded here**, to keep alignment with the content and level of the given lecture segment.

## Key images

### Slide on activation functions with diagram and equation
*Timestamp: 51s*

### Partial equation and notes on activation functions
*Timestamp: 170s*

### Wikipedia page on activation functions
*Timestamp: 212s*

### Scatter plot showing data points
*Timestamp: 310s*

### Code snippet for neural network model
*Timestamp: 323s*

### Decision boundary plot with data points
*Timestamp: 346s*

### Sequential model code snippet with activation functions
*Timestamp: 364s*

### Code snippet for plotting decision regions
*Timestamp: 383s*

### Diagram illustrating activation functions
*Timestamp: 403s*

### Partial equations and diagram on activation functions
*Timestamp: 505s*

### More complete equations on activation functions
*Timestamp: 554s*

### Equations with additional graph
*Timestamp: 605s*

### Detailed equations with graph and annotations
*Timestamp: 630s*

### Complete equations with graph and sigmoid function
*Timestamp: 646s*

### Detailed notes on activation functions with diagram
*Timestamp: 864s*

### Continuation of notes on zero-centered activation functions
*Timestamp: 978s*

### Complete notes on non-saturating activation functions
*Timestamp: 1036s*

### Graphs of sigmoid activation function
*Timestamp: 1126s*

### Equation for sigmoid function with partial notes
*Timestamp: 1239s*

### Sigmoid function graphs and equation
*Timestamp: 1314s*

### Advantages of sigmoid function
*Timestamp: 1321s*

### Complete advantages of sigmoid function
*Timestamp: 1365s*

### Disadvantages of sigmoid function
*Timestamp: 1405s*
