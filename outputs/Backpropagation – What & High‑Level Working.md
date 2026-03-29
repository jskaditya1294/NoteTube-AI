# Backpropagation – What & High‑Level Working

## Video Series Structure (What / How / Why)
- Topic: Backpropagation in deep learning (training neural networks).
- Series split into 3 parts:
  - Part 1: “What” – conceptual understanding and basic math of backpropagation.
  - Part 2: “How” – full derivations + implement from scratch in code on:
    - 1 regression dataset.
    - 1 classification dataset.
  - Part 3: “Why” – conceptual questions:
    - Why certain behaviors occur.
    - Why specific formulae/steps make sense.
- Teaching format: For any tough topic, understand `What`, `How`, and `Why` to gain full clarity.

## Prerequisites
- To properly understand backpropagation, you should already know:
  - Gradient Descent:
    - *An optimization algorithm to minimize a loss function.*
  - Forward Propagation:
    - Technique by which a neural network computes predictions.
- Both topics are covered in separate videos; recommended to watch them first if not confident.

## Official vs Simplified Definition of Backpropagation
### Official‑style Definition (simplified paraphrase)
- Backpropagation (short for backward propagation of errors):
  - An *algorithm for supervised learning* in artificial neural networks.
  - Uses gradient descent.
  - Given a network and a loss (error) function, it computes the gradient of the error w.r.t. activations/weights.

### Simplified Working Definition
- Backpropagation:
  - A *training algorithm* used to train neural networks.
  - Given data, it finds (assigns/updates) the values of weights and biases to optimal values.
  - Goal: Find weights and biases that yield best predictions (minimize loss) on given data.

## Example Problem Setup (Regression)
- Dataset:
  - Input features:
    - Student’s CGPA.
    - Student’s IQ.
  - Target:
    - Salary package (in LPA).
- Example records:
  - Student A: CGPA 8, IQ 90, Package 8 LPA.
  - Student B: CGPA 7, IQ 70, Package 7 LPA.
  - (More such points; 4 students total in the toy example.)
- Task:
  - Predict package given CGPA and IQ.

## Neural Network Architecture (Toy Model)
- Very simple feedforward neural network:
  - Input layer:
    - 2 inputs: CGPA, IQ.
  - Hidden layer:
    - 2 neurons.
  - Output layer:
    - 1 neuron (regression output = predicted package).
- Parameters:
  - Weights (`w`) and biases (`b`) on all connections/nodes.
  - All activation functions used are *linear* for this example (including output since it’s regression).

### Parameter Labeling
- Inputs:
  - `x1` = IQ.
  - `x2` = CGPA.  
- Hidden layer:
  - Neuron 1 output: `o1`.
  - Neuron 2 output: `o2`.
- Output layer:
  - Network prediction: `ŷ` (also referred to as `y_hat` or output of some node like `o_21`).
- Example weight/bias notation (as used):
  - `w111, w121, w221, w221` etc. (names are illustrative; actual pattern shown in video as indices).
  - Biases: `b11, b12, b21` etc.
- Reason for explicit labeling:
  - Needed for writing and differentiating the exact expressions during backpropagation.

## Forward Propagation (Single Data Point)
1. Pick one student’s data (e.g., first row: IQ, CGPA, true package `y`).
2. Feed `IQ` and `CGPA` into the input layer.
3. Compute hidden layer outputs (`o1`, `o2`) via:
   - `o1 = IQ * w111 + CGPA * w121 + b11`
   - `o2 = IQ * w211 + CGPA * w221 + b12`
   - (Linear activation → no extra nonlinearity applied.)
4. Compute final output:
   - `ŷ = o1 * w2_1 + o2 * w2_2 + b21`
5. Because initial weights are arbitrary (e.g., all 1) and biases 0, prediction `ŷ` will usually be wrong.
   - Example: network predicts `18` LPA while true value `y` is `3` LPA → clear error.

## Loss Function (Error)
- For this regression example:
  - Loss function: Mean Squared Error (MSE).
  - For a single data point:
    - `L = (y - ŷ)²`
- Example:
  - True `y = 3`, prediction `ŷ = 18`.
  - `L = (3 - 18)² = 225`.
  - This scalar `L` is the error with respect to this one student.

## Intuition: What Needs to Change?
- Loss `L` depends on:
  - True label `y` (from data; cannot change).
  - Prediction `ŷ` (depends on network parameters; can change).
- To reduce `L`, we must change `ŷ`.
  - Which direction?
    - If `ŷ` is greater than `y`, we need to decrease `ŷ`.
    - If `ŷ` is smaller than `y`, we need to increase `ŷ`.
- `ŷ` depends on:
  - Output layer weights and biases (e.g., `w221`, `w222`, `b21`).
  - Hidden outputs `o1`, `o2`.
- `o1`, `o2` in turn depend on:
  - Input features (`IQ`, `CGPA`) — fixed, cannot change.
  - Earlier weights and biases (e.g., `w111, w121, w211, w221, b11, b12`) — can change.
- Therefore:
  - To reduce `L`, we must adjust *all* trainable weights and biases in the network.
  - This adjustment is done “backwards” from loss to earlier layers → “backward propagation of errors”.

## Why “Backpropagation” / Backward View
- We have a dependency chain:
  - `L` → depends on `ŷ`.
  - `ŷ` → depends on output layer parameters and hidden outputs.
  - Hidden outputs → depend on hidden layer parameters and inputs.
- Changing a weight in an earlier layer:
  - Indirectly influences `L` via its effect on `o1`, `o2`, then `ŷ`, then `L`.
- Backpropagation:
  - Systematic method to compute:
    - ∂L/∂(each weight and bias),
  - By propagating derivatives from the output layer backwards through layers using the chain rule.

## Gradient Descent Update Rule
- Core update formula for any weight `w`:
  - `w_new = w_old - η · (∂L/∂w_old)`
- For any bias `b`:
  - `b_new = b_old - η · (∂L/∂b_old)`
- Here:
  - `η` = learning rate (small positive scalar, e.g., 0.1).
  - `∂L/∂w` is the gradient computed by backpropagation.
- In the toy network:
  - There are 9 trainable parameters (6 weights + 3 biases, by his count).
  - Need to compute 9 partial derivatives:
    - `∂L/∂w_...` for all weights.
    - `∂L/∂b_...` for all biases.

## Step 0: Initialization
- Before training begins:
  - Initialize all weights and biases.
- Possible initialization strategies:
  - Random initialization.
  - All weights = 1 and all biases = 0 (simplified assumption for this example).
- For the example:
  - All weights `w` start at 1.
  - All biases `b` start at 0.
- Note: Initialization affects convergence behavior, but kept simple here for clarity.

## Formalizing the Gradient Computation Goal
- From the formal definition:
  - Backpropagation “calculates the gradient of the error function with respect to the network’s weights”.
- Our concrete objective:
  - For every trainable parameter `θ` (a weight or bias):
    - Compute `∂L/∂θ`.
- Once we have these:
  - Plug into gradient descent update to get new parameter values.

## Derivative Intuition (Why Derivatives?)
- Derivative meaning:
  - `∂y/∂x` ≈ “how much does `y` change for a small change in `x`”.
- Applied here:
  - `∂L/∂w_221`:
    - “If I slightly change weight `w_221`, how much (and in what direction) does the loss `L` change?”
- This tells:
  - Direction:
    - Positive derivative → increasing weight increases loss → we should decrease weight.
    - Negative derivative → increasing weight decreases loss → we should increase weight.
  - Magnitude:
    - How strongly `L` is affected by this parameter.
- That’s why all training focuses on computing these derivatives.

## Chain Rule in Backpropagation
### Core Chain Rule Idea
- Suppose weight `w_221` is not directly in the loss formula; it affects `L` via multiple intermediate variables:
  - `w_221 → ŷ → L`.
- Chain rule:
  - `∂L/∂w_221 = (∂L/∂ŷ) · (∂ŷ/∂w_221)`
- For earlier weights (e.g., a first‑layer weight like `w_121`), dependency is longer:
  - `w_121 → o1 → ŷ → L`.
- Chain rule then:
  - `∂L/∂w_121 = (∂L/∂ŷ) · (∂ŷ/∂o1) · (∂o1/∂w_121)`

### Loss Gradient w.r.t Prediction
- Loss: `L = (y - ŷ)²`
- Derivative:
  - `∂L/∂ŷ = 2 · (ŷ - y)` or equivalently `-2 · (y - ŷ)` (sign consistent with expansion).
- This term appears in all parameter derivatives (common factor).

## Output Layer Derivatives (Example)
- Prediction formula (simplified view):
  - `ŷ = o1 · w2_1 + o2 · w2_2 + b21`
- Derivatives:
  - `∂ŷ/∂w2_1 = o1`
  - `∂ŷ/∂w2_2 = o2`
  - `∂ŷ/∂b21 = 1`
- Using chain rule:
  - `∂L/∂w2_1 = (∂L/∂ŷ) · (∂ŷ/∂w2_1) = 2 · (ŷ - y) · o1`
  - `∂L/∂w2_2 = 2 · (ŷ - y) · o2`
  - `∂L/∂b21 = 2 · (ŷ - y) · 1`

## Hidden Layer Derivatives (Example)
- Hidden output `o1` (for first hidden neuron):
  - `o1 = IQ · w111 + CGPA · w121 + b11`
- Derivatives:
  - `∂o1/∂w111 = IQ = x1`
  - `∂o1/∂w121 = CGPA = x2`
  - `∂o1/∂b11 = 1`
- Because:
  - `L` depends on `o1` only through `ŷ`:
    - `∂L/∂o1 = (∂L/∂ŷ) · (∂ŷ/∂o1)`
    - Here, `∂ŷ/∂o1 = w2_1`.
- Then:
  - `∂L/∂w111 = (∂L/∂o1) · (∂o1/∂w111) = (∂L/∂ŷ · w2_1) · x1`
  - `∂L/∂w121 = (∂L/∂ŷ · w2_1) · x2`
  - `∂L/∂b11  = (∂L/∂ŷ · w2_1) · 1`

- Similarly for second hidden neuron and its weights/bias:
  - Use:
    - `o2 = IQ · w211 + CGPA · w221 + b12`
    - `∂o2/∂w211 = x1`, `∂o2/∂w221 = x2`, `∂o2/∂b12 = 1`
    - `∂Ŷ/∂o2 = w2_2`
    - `∂L/∂o2 = (∂L/∂Ŷ) · w2_2`
  - Then:
    - `∂L/∂w211 = (∂L/∂Ŷ · w2_2) · x1`
    - `∂L/∂w221 = (∂L/∂Ŷ · w2_2) · x2`
    - `∂L/∂b12  = (∂L/∂Ŷ · w2_2) · 1`

*(The transcript goes through these derivations in detail with his index naming and chain rule steps.)*

## Summary of All Needed Derivatives
- For this small network, you ultimately need:
  - `∂L/∂w2_1`, `∂L/∂w2_2`, `∂L/∂b21`
  - `∂L/∂w111`, `∂L/∂w121`, `∂L/∂b11`
  - `∂L/∂w211`, `∂L/∂w221`, `∂L/∂b12`
- All can be expressed in terms of:
  - `x1` (IQ), `x2` (CGPA),
  - current outputs `o1`, `o2`,
  - current prediction `ŷ`,
  - true value `y`,
  - current weights of later layers.
- Once expressions are known, for each data point:
  - Plug in numerical values to compute gradients.
  - Apply gradient descent updates.

## Overall Backpropagation Algorithm (Single Epoch)
1. **Initialize parameters**
   - Set all `w = 1`, all `b = 0` (for this example).
2. **Loop over all training examples (students)**
   - For each student:
     1. Forward propagation:
        - Feed `IQ` and `CGPA`.
        - Compute hidden layer outputs.
        - Compute prediction `ŷ`.
     2. Compute loss:
        - `L = (y - ŷ)²`.
     3. Backward pass:
        - Compute all required partial derivatives `∂L/∂w` and `∂L/∂b` using chain rule.
     4. Parameter update:
        - For every weight: `w_new = w_old - η · (∂L/∂w_old)`.
        - For every bias: `b_new = b_old - η · (∂L/∂b_old)`.
   - After processing each student, weights and biases have been slightly adjusted.

## Multiple Epochs & Convergence
- One pass over all data points:
  - Called an epoch.
- Typically:
  - Run multiple epochs (e.g., hundreds or thousands).
  - Outer loop: epochs.
  - Inner loop: iterate over every training sample and perform:
    - Forward pass.
    - Loss computation.
    - Backprop + parameter update.
- Convergence:
  - When loss is “minimized enough” or stops decreasing significantly.
  - For example:
    - In the earlier example, prediction moves from 18 towards 3; once close enough, we say network has converged for that point.

## Essence of Backpropagation (Conceptual Recap)
- Backpropagation:
  - A systematic application of chain rule to compute gradients of the loss w.r.t. all network parameters.
  - Uses these gradients with gradient descent to iteratively adjust weights and biases.
- Direction:
  - Forward pass: compute outputs and loss.
  - Backward pass: propagate error signals backward layer by layer, computing gradients.
- Naming:
  - “Backward propagation of errors” because:
    - Error at output layer is propagated backwards to update earlier layers.

## Next Steps (Future Parts)
- In the next videos:
  - Use an actual dataset (regression + classification).
  - Perform full, step‑by‑step calculations with real numbers.
  - Convert derived math expressions into actual code.
  - Verify empirically that loss decreases over epochs.
