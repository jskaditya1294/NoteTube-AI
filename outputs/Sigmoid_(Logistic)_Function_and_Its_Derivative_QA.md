# Sigmoid (Logistic) Function and Its Derivative — Interview Q&A

*Questions generated from study notes. Answers are based solely on the transcript content.*

## Q1. Why is the sigmoid (logistic) function a useful activation in ML models like logistic regression and neural networks?
- It maps any real-valued input to the range (0, 1), making outputs interpretable as probabilities.
- It “squashes” large positive and negative values into a bounded interval, stabilizing outputs.
- Its smooth, differentiable shape is suitable for gradient-based optimization and backpropagation.

## Q2. Derive the derivative of the sigmoid function σ(x) = 1 / (1 + e^(−x)) using the reciprocal rule.
- Let `g(x) = 1 + e^(−x)` so `σ(x) = 1 / g(x)`.
- By the reciprocal rule, `σ'(x) = −g'(x) / (g(x))^2`.
- `g'(x) = 0 + (−e^(−x)) = −e^(−x)`, so `σ'(x) = −(−e^(−x)) / (1 + e^(−x))^2 = e^(−x) / (1 + e^(−x))^2`.

## Q3. Show how to express the sigmoid derivative in terms of σ(x) itself and explain why that is convenient.
- From the definition, `σ(x) = 1 / (1 + e^(−x))`.
- One can show `1 − σ(x) = e^(−x) / (1 + e^(−x))`.
- Then `σ(x) · (1 − σ(x)) = [1 / (1 + e^(−x))] · [e^(−x) / (1 + e^(−x))] = e^(−x) / (1 + e^(−x))^2`, which equals `σ'(x)`.
- So `σ'(x) = σ(x) · (1 − σ(x))`, which is convenient in backpropagation because it reuses the already-computed activation σ(x).

## Q4. Using the chain rule, how do you differentiate e^(−x) and why is this step important in deriving σ'(x)?
- For `h(x) = e^(−x)`, the outer derivative is `e^(−x)` and the inner derivative of `−x` is `−1`.
- Thus `h'(x) = e^(−x) · (−1) = −e^(−x)`.
- This result is used to compute `g'(x)` when `g(x) = 1 + e^(−x)`, which is crucial for applying the reciprocal rule to find `σ'(x)`.

## Q5. What is the qualitative shape of the sigmoid function and its derivative, and what does this imply for learning?
- σ(x) is S-shaped, approaching 0 as x → −∞ and 1 as x → +∞.
- σ'(x) is bell-shaped, centered at 0, with maximum at x = 0 where σ(0) = 0.5 and σ'(0) = 0.5 · 0.5 = 0.25.
- As x → ±∞, σ'(x) → 0, implying gradients vanish in the saturated regions, which can slow learning for very large positive or negative inputs.

## Q6. Why does the identity σ'(x) = σ(x) · (1 − σ(x)) make implementations of backpropagation more efficient?
- During forward pass, σ(x) is already computed and stored.
- The derivative can then be obtained using only σ(x) with simple multiplications and a subtraction.
- This avoids recomputing exponentials, making gradient computation cheaper and numerically straightforward.

## Q7. Explain the role of the bounded output range (0, 1) of the sigmoid function in logistic regression.
- It ensures model outputs can be interpreted as probabilities of the positive class.
- It guarantees outputs remain within a stable numeric range, which suits loss functions like cross-entropy.
- It naturally connects linear scores on ℝ to probability space (0, 1) via the logistic function.

## Q8. At what input x is the derivative of the sigmoid function maximal, and how do you compute that value?
- The derivative is `σ'(x) = σ(x) · (1 − σ(x))`, which is a quadratic in σ(x).
- This expression is maximized when σ(x) = 0.5, since p(1 − p) is maximal at p = 0.5.
- Solving σ(x) = 0.5 for the standard sigmoid gives x = 0, so the maximum derivative is at x = 0.
- The maximum value is `σ'(0) = 0.5 · 0.5 = 0.25`.

## Q9. How does the expression `1 − σ(x)` relate algebraically to the original sigmoid definition, and why is that relationship useful?
- Starting from `σ(x) = 1 / (1 + e^(−x))`, we get:
  - `1 − σ(x) = 1 − 1 / (1 + e^(−x)) = (1 + e^(−x) − 1) / (1 + e^(−x)) = e^(−x) / (1 + e^(−x))`.
- This shows `1 − σ(x)` is just the normalized e^(−x) term.
- It is useful because it enables rewriting the derivative as `σ(x) · (1 − σ(x))` without exponentials, simplifying gradient formulas in ML models.
