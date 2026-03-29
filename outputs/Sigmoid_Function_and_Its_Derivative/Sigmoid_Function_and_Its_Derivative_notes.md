# Sigmoid Function and Its Derivative

## Metadata
- **Created at:** 2026-03-27 17:01:49
- **Video ID:** awjXaFR1jOM

## Notes

# Sigmoid Function and Its Derivative

## Sigmoid Function: Definition and Properties
- Sigmoid (logistic) function:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Maps any real number $x \in (-\infty, \infty)$ to a value in $(0, 1)$.
- “Squashes” large positive and negative inputs into a bounded range.
- Widely used in:
  - Logistic regression (as activation / link function)
  - Neural networks (e.g., multilayer perceptron) as activation
  - Loss functions and backpropagation require its derivative

## Goal
- Compute the derivative of the sigmoid function $\sigma(x)$.
- Final result (to be derived):
  $$\boxed{\frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)}$$

## Derivative Rules Used
- Reciprocal rule:
  $$\frac{d}{dx}\left(\frac{1}{u(x)}\right) = -\frac{u'(x)}{[u(x)]^2}$$
- Chain rule:
  $$\frac{d}{dx}f\bigl(g(x)\bigr) = f'\bigl(g(x)\bigr) \cdot g'(x)$$
- Exponential derivative:
  $$\frac{d}{dx}e^{x} = e^{x}, \quad \frac{d}{dx}e^{-x} = -e^{-x}$$

## Step-by-Step Derivation

### 1. Start from the definition
- Given:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- Let:
  - $u(x) = 1 + e^{-x}$  
  - So $\sigma(x) = \frac{1}{u(x)}$

### 2. Apply reciprocal rule
- Using $\frac{d}{dx}\bigl(\frac{1}{u}\bigr) = -\frac{u'}{u^2}$:
  $$\frac{d}{dx}\sigma(x) = -\frac{u'(x)}{[u(x)]^2}$$

### 3. Differentiate $u(x) = 1 + e^{-x}$
- Derivatives:
  - $\frac{d}{dx}(1) = 0$
  - $\frac{d}{dx}(e^{-x}) = -e^{-x}$ (by chain rule)
- So:
  $$u'(x) = 0 + (-e^{-x}) = -e^{-x}$$

### 4. Substitute back into reciprocal rule
- Substitute $u'(x) = -e^{-x}$ and $u(x) = 1 + e^{-x}$:
  $$
  \frac{d}{dx}\sigma(x)
  = -\frac{-e^{-x}}{(1 + e^{-x})^2}
  = \frac{e^{-x}}{(1 + e^{-x})^2}
  $$

### 5. Express derivative in terms of $\sigma(x)$
- Observe:
  $$\sigma(x) = \frac{1}{1 + e^{-x}}$$
  $$1 - \sigma(x) = 1 - \frac{1}{1 + e^{-x}} 
  = \frac{1 + e^{-x} - 1}{1 + e^{-x}}
  = \frac{e^{-x}}{1 + e^{-x}}$$
- Multiply:
  $$
  \sigma(x)\bigl(1 - \sigma(x)\bigr)
  = \left(\frac{1}{1 + e^{-x}}\right)\left(\frac{e^{-x}}{1 + e^{-x}}\right)
  = \frac{e^{-x}}{(1 + e^{-x})^2}
  $$

- Therefore:
  $$
  \frac{d}{dx}\sigma(x)
  = \sigma(x)\bigl(1 - \sigma(x)\bigr)
  $$

## Final Result and Interpretation
- Derivative of sigmoid:
  $$\boxed{\frac{d}{dx}\sigma(x) = \sigma(x)\bigl(1 - \sigma(x)\bigr)}$$
- Key characteristics:
  - Always positive (since $0 < \sigma(x) < 1$).
  - Maximum at $x = 0$ where $\sigma(0) = 0.5$:
    $$\sigma'(0) = 0.5 \cdot (1 - 0.5) = 0.25$$
  - Symmetric around $x = 0$.
- Very useful in machine learning:
  - Appears directly in backpropagation when sigmoid is used as activation.
  - Gradient computations simplify because the derivative is expressed in terms of $\sigma(x)$ itself.

## Key images

### Sigmoid function graph with menu
*Timestamp: 0s*

### Partial derivative equation with graph
*Timestamp: 121s*

### Complete derivative equation derivation
*Timestamp: 218s*

### Continuation of derivative equation
*Timestamp: 291s*

### Graph comparing sigmoid and its derivative
*Timestamp: 342s*
