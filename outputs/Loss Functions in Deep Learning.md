# Loss Functions in Deep Learning

## Definition and Role of Loss Function
- A loss function is a mathematical function to evaluate how well an algorithm/model performs on your dataset.
- It measures the performance of the algorithm: 
  - Large loss value → model is performing poorly.
  - Small loss value → model is performing well.
- It is a function of model parameters (weights, biases). Changing parameters changes the loss value.
- Training = searching parameter values for which the loss is minimized (typically via gradient descent).
- Conceptually, loss is the "eye" or "guiding signal" that tells the algorithm which direction to adjust parameters.

## Example: Linear Regression and Loss
- Setup: Simple linear regression with data points and a line `y = m x + b`.
- Common loss: squared error `loss = (y_true - y_pred)²`.
- Changing `m` and/or `b` changes the line → changes loss.
- Goal: find `m, b` such that total loss over data is minimized.
- Gradient descent:
  - Compute loss for current line.
  - Compute gradients and update `m, b`.
  - Repeat until loss is minimized.

## Importance of Loss Functions
- Quote capturing importance: “You can’t improve what you can’t measure.”
- Loss measures “how wrong” the model is → enables improvement.
- In machine learning:
  - Start with random model.
  - Predict on data, compute loss.
  - Adjust parameters to reduce loss.
  - Iterate until (approximately) minimal loss → trained model.
- In deep learning:
  - Same principle applied to neural networks via backpropagation + gradient descent.
  - Loss is crucial for backprop to compute gradients and update weights.

## Loss Function in Deep Learning Workflow
- Example dataset: input `(CGPA, IQ)` → output `package in LPA`.
- Neural network:
  - Inputs fed through layers with initial random weights.
  - Forward propagation produces an output `ŷ`.
- Choose a loss function (e.g., mean squared error for regression).
- For each data point (or batch):
  - Compute loss from true value `y` and prediction `ŷ`.
  - Use gradient descent/backpropagation to update weights and biases.
- Iterate over dataset multiple epochs until the loss is minimized.
- The parameter values at minimal loss are treated as the "right" weights/biases.

## Types of Loss Functions (Overview)
- Regression:
  - Mean Squared Error (MSE) / Squared Loss / L2 Loss.
  - Mean Absolute Error (MAE) / L1 Loss.
  - Huber Loss.
- Classification:
  - Binary Cross Entropy (a.k.a. Log Loss).
  - Categorical Cross Entropy.
  - Hinge Loss (e.g., SVM).
- Other architectures/problems:
  - Autoencoders: Kullback-Leibler (KL) divergence (e.g., variational autoencoders).
  - GANs: discriminator losses (e.g., max-margin losses).
  - Object detection: Focal Loss.
  - Embeddings: Triplet Loss.
- You can design custom loss functions (e.g., in Keras) according to problem requirements.

## Choosing a Loss Function
- Choice depends on problem type:
  - Regression vs classification vs specialized tasks.
- Each loss has its own advantages and disadvantages.
- Wrong choice of loss ⇒ suboptimal/poor solutions even if training converges.
- Research often involves designing or tuning loss functions for better performance.

## Loss Function vs Cost Function
- Loss function:
  - Defined per single training example.
  - Example for one sample: `loss_i = (y_i - ŷ_i)²`.
- Cost function:
  - Aggregate loss over a batch or entire training set (often average).
  - Example: `cost = (1/N) Σ (y_i - ŷ_i)²` over all `N` samples.
- In practice:
  - Sometimes used interchangeably, but strictly they are different:
    - “Loss” = per example.
    - “Cost” (or “error”) = aggregated over many examples.

## Mean Squared Error (MSE) / Squared Loss / L2 Loss
### Definition
- Per-sample loss: `loss_i = (y_i - ŷ_i)²`.
- Cost over dataset: `J = (1/N) Σ (y_i - ŷ_i)²`.

### Why the Square?
- If using simple difference `y - ŷ`:
  - Positive and negative errors cancel when summed.
- Squaring:
  - Makes all contributions non-negative.
  - Emphasizes larger errors.
- After squaring, loss becomes quadratic in error:
  - Error 1 → contribution 1.
  - Error 2 → contribution 4.
  - Error 4 → contribution 16.
  - Points far from the true value have disproportionately large influence.

### Effect on Regression (Geometric Intuition)
- Data points vs fitted line:
  - Points close to the line → small squared errors, small impact on updates.
  - Points far from the line → large squared errors, strong influence on updates.
- During optimization:
  - Outlying points (large error) cause larger parameter updates.
  - Near points cause small updates.

### Advantages of MSE
- Intuitive and simple to understand.
- Always differentiable:
  - Its loss surface is smooth, easy for gradient-based methods.
- Has a single global minimum (convex in linear models):
  - Only one set of parameters gives minimal loss; no multiple local minima issue (in linear models).

### Disadvantages of MSE
- Unit mismatch:
  - If target `y` is in LPA, loss is in `(LPA)²`.
  - To interpret in original units, you need to take square root.
- Not robust to outliers:
  - Outliers give huge squared errors → dominate training.
  - Model can shift to fit outliers and misrepresent the bulk of data.

### MSE in Deep Learning Regression
- To use MSE in a neural network regression:
  - Final (output) neuron’s activation should be linear.
  - Hidden layers may use ReLU, sigmoid, etc., but last layer is linear.
  - In code, compile the model with loss set to MSE (e.g., `"mse"`).
- Training loop:
  - Forward pass on each row.
  - Compute MSE for that prediction.
  - Backprop gradients and update weights.
  - Repeat over all rows and epochs.

## Mean Absolute Error (MAE) / L1 Loss
### Definition
- Per-sample loss: `loss_i = |y_i - ŷ_i|`.
- Cost over dataset: `J = (1/N) Σ |y_i - ŷ_i|`.

### Relationship to MSE
- Same structure as MSE but:
  - Square replaced by absolute value.
- Many problems of MSE are mitigated because:
  - No squaring, so outliers are not penalized quadratically.

### Advantages of MAE
- Intuitive and easy to understand:
  - Direct magnitude of average error in original units.
- Same units as target:
  - If `y` in LPA, MAE also in LPA.
- More robust to outliers than MSE:
  - Outliers do not blow up errors as squares.
  - Model is less distorted by a few extreme points.

### Disadvantages of MAE
- Not differentiable at 0:
  - Loss vs parameter graph has a kink at 0.
  - Gradient descent needs subgradients; computation is more complex and heavier.
- If no outliers:
  - MSE is often preferred because it is smoother and more convenient for optimization.

### When to Use MSE vs MAE
- If data has no (or very few) outliers:
  - Prefer MSE for smooth optimization.
- If data has strong outliers:
  - Prefer MAE for robustness.

## Huber Loss
### Motivation
- MSE:
  - Highly sensitive to outliers.
- MAE:
  - Robust to outliers but less smooth (non-differentiable at 0).
- Huber loss:
  - Combines benefits:
    - Behaves like MSE near small errors.
    - Behaves like MAE for large errors.
  - Controlled by a parameter `δ`:
    - Determines threshold between “small” and “large” error.

### Behavior
- For error magnitude less than `δ`:
  - Loss behaves like squared error (MSE-style).
- For error magnitude greater than `δ`:
  - Loss behaves like absolute error (MAE-style).
- Effective when there is a noticeable fraction of outliers (e.g., around 25% of points).
  - Neither pure MSE nor pure MAE is ideal.
  - Huber gives a balanced trade-off.

## Binary Cross Entropy (BCE) / Log Loss
### Use Case
- Used for binary classification problems (2 classes, e.g., Yes/No, 0/1).
- Same loss as used in logistic regression.

### Neural Network Requirements
- Output layer:
  - Single neuron.
  - Activation must be sigmoid.
- Output `ŷ` is in `[0, 1]` interpreted as probability of class 1.

### Formulas
- Per-sample loss:
  - `L = - [ y * log(ŷ) + (1 - y) * log(1 - ŷ) ]`
  - `y` ∈ {0, 1}, `ŷ` ∈ (0, 1).
- Cost over dataset:
  - `J = (1/N) Σ - [ y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i) ]`.

### Example Computation
- For a sample with:
  - `y = 1`, `ŷ = 0.738`:
    - `L = - log(0.738) ≈ 0.133`.
  - `y = 0`, `ŷ = 0.75`:
    - `L = - log(1 - 0.75) = - log(0.25) ≈ 1.386` (illustrative).
- Process:
  - Feed sample through network → get `ŷ`.
  - Compute BCE loss.
  - Backpropagate and update weights.
  - Repeat for each sample/batch.

### Properties
- Advantages:
  - Differentiable; works well with gradient descent.
- Disadvantages:
  - Multiple local minima can exist (esp. in deep nets).
  - Not very intuitive at a glance; loss mechanism is less obvious than simple difference/square.
- Also called:
  - Log loss.
  - Logistic loss (in logistic regression context).

## Categorical Cross Entropy (CCE)
### Use Case
- Multi-class classification where there are more than two classes.
  - Example: output categories {Yes, No, Maybe}.

### Neural Network Requirements
- Output layer:
  - Number of neurons = number of classes `K`.
  - Each neuron corresponds to one class.
- Activation for output layer:
  - Softmax:
    - For class `j`: `ŷ_j = exp(z_j) / Σ_k exp(z_k)`.
    - All `ŷ_j` in `[0, 1]`, and `Σ_j ŷ_j = 1`.

### Targets and One-Hot Encoding
- True labels `y` must be one-hot encoded:
  - Example with 3 classes (Yes, No, Maybe):
    - Yes → `[1, 0, 0]`.
    - No → `[0, 1, 0]`.
    - Maybe → `[0, 0, 1]`.

### Formula
- Per-sample loss:
  - `L = - Σ_j y_j * log(ŷ_j)`.
  - Since one-hot, this simplifies to `- log(ŷ_true_class)`.
- Cost over dataset:
  - `J = (1/N) Σ_i - Σ_j y_ij * log(ŷ_ij)`.

### Example Computation
- Suppose:
  - True label for a sample: Yes → `[1, 0, 0]`.
  - Network output (softmax): `[0.2, 0.3, 0.5]`.
  - Loss: `L = - [1 * log(0.2) + 0 * log(0.3) + 0 * log(0.5)] = - log(0.2)`.
- For another sample:
  - True label: No → `[0, 1, 0]`.
  - Prediction: `[0.3, 0.6, 0.1]`.
  - Loss: `L = - log(0.6)`.

### Training Loop
- For each sample:
  - Encode target as one-hot.
  - Forward pass → softmax probabilities `ŷ_j`.
  - Compute `L = - Σ y_j log(ŷ_j)`.
  - Backpropagate loss to update weights.
  - Repeat over dataset/epochs.

## Sparse Categorical Cross Entropy (Sparse CCE)
### Motivation and Setup
- Same underlying loss as Categorical Cross Entropy.
- Difference is in how labels are represented:
  - CCE: labels are one-hot vectors.
  - Sparse CCE: labels are integer indices.
    - Example for 3 classes:
      - Yes → 1.
      - No → 2.
      - Maybe → 3.

### Behavior
- Architecture and outputs:
  - Same as CCE: `K`-neuron softmax output.
- Loss computation:
  - Only uses the predicted probability of the true class index:
    - If label is `2`, use only `ŷ_2`: `L = - log(ŷ_2)`.
- Efficiency:
  - Slightly faster than CCE because:
    - Does not need to one-hot encode labels.
    - Does not need to evaluate log for all classes when aggregating.

### When to Use
- Large number of classes:
  - Prefer Sparse CCE for efficiency.
- Semantics:
  - CCE and Sparse CCE are mathematically equivalent; difference is representation and computational speed.

## Summary of When to Use Which Loss
- Regression:
  - No outliers: Mean Squared Error (MSE).
  - Significant outliers: Mean Absolute Error (MAE).
  - Mixed with many outliers: Huber Loss.
- Binary Classification:
  - Use Binary Cross Entropy.
  - Architecture: 1 sigmoid output neuron.
- Multi-class Classification:
  - Use Categorical Cross Entropy:
    - Many classes: prefer Sparse Categorical Cross Entropy.
  - Architecture: `K` softmax outputs (one per class).
- Mixed / Robust:
  - When dataset contains a combination of normal points and strong outliers, consider Huber loss.
- Always ensure:
  - Output layer activation and loss function are compatible (e.g., linear+MSE for regression, sigmoid+BCE for binary classification, softmax+CCE/Sparse CCE for multi-class).

## Key images

### KERAS MODEL CODE SNIPPET.
*Timestamp: 1814s*

### "SEQUENTIAL MODEL CODE SNIPPET FOR MNIST CLASSIFICATION"
*Timestamp: 2998s*

### SPARSE CATEGORICAL CROSS ENTROPY EQUATION AND DIAGRAM.
*Timestamp: 3205s*

## Interview question bank

*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*

### Summary table

| Topic | # Companies | # Questions | Top sources |
|-------|-------------|-------------|-------------|
| Loss functions in deep learning | 1 | 2 | analyticsvidhya.com, wecreateproblems.com |
| Loss vs cost function | 1 | 2 | wecreateproblems.com, stats.stackexchange.com |
| Gradient descent and loss minimization | 3 | 10 | tryexponent.com, medium.com, analyticsarora.com, wecreateproblems.com, linkedin. |
| Loss functions in linear regression | 0 | 0 |  |
| Backpropagation and role of loss in neural networks | 1 | 13 | analyticsarora.com, medium.com, linkedin.com, analyticsvidhya.com, wecreateprobl |
| Mean Squared Error (MSE) loss | 0 | 0 |  |
| Properties, advantages and disadvantages of MSE | 0 | 0 |  |
| Mean Absolute Error (MAE) loss | 0 | 0 |  |
| Huber loss | 0 | 0 |  |
| Binary Cross Entropy loss | 0 | 0 |  |
| Categorical Cross Entropy and Sparse Categorical Cross Entropy | 0 | 0 |  |
| Loss functions for specialized architectures | 0 | 0 |  |
| **Total (all in bank)** | 3 | 27 | See sections below |

### Coverage report

- Extracted topics: 12
- Questions with attribution: 27
- Excluded (unattributed): 3

**Topic coverage notes:**
Content centers on defining loss functions, their role in optimization and backprop, and detailing key regression (MSE, MAE, Huber) and classification (binary/categorical cross-entropy, sparse CCE) losses, plus the loss vs cost distinction. A few other specialized losses are briefly named without deep treatment.

### Questions by topic → company

#### Backpropagation and role of loss in neural networks

**Unspecified**
- **Q:** What is backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/
  - *Evidence:* ## **Backpropagation ML Interview Questions**

#### **What is backpropagation?**

Backpropagation is very much a reason why neural network training works.

- **Q:** Why do we need backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/
  - *Evidence:* #### **Why do we need backpropagation?**

The reasons why we need backpropagation are several.

- **Q:** What are the advantages of backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/
  - *Evidence:* #### **What are the advantages of backpropagation?**

* The advantages of backpropagation are as follows:

- **Q:** What are the disadvantages of backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/
  - *Evidence:* #### **What are the disadvantages of backpropagation?**

The disadvantages of backpropagation are as stated below:

- **Q:** What are the types of backpropagation? Also, how are they different?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-on-backpropagation/
  - *Evidence:* #### **What are the types of backpropagation? Also, how are they different?**

There are two types of backpropagation:

- **Q:** How do you understand Backpropagation? Explain the mechanism of action?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://medium.com/jp-tech/12-deep-learning-interview-questions-you-should-not-be-missed-part-2-8f42deeb4483
  - *Evidence:* ## 5. How do you understand Backpropagation? Explain the mechanism of action?

- **Q:** Explain backpropagation in simple terms.
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
5. What is forward propagation in a neural network?
6. Explain backpropagation in simple terms.

- **Q:** What is the role of loss functions in training a neural network?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
7. What is the role of loss functions in training a neural network?

- **Q:** What is forward propagation in a neural network?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
5. What is forward propagation in a neural network?

- **Q:** What are epochs, batches, and iterations in DL training?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
9. What are epochs, batches, and iterations in DL training?

- **Q:** How do you prevent a neural network from overfitting?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
10. How do you prevent a neural network from overfitting?

- **Q:** What is backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.analyticsvidhya.com/blog/2022/11/advance-guide-on-interview-questions-of-deep-learning/
  - *Evidence:* ### Q8. What is backward propagation?

- **Q:** What is backpropagation?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.wecreateproblems.com/interview-questions/deep-learning-interview-questions
  - *Evidence:* 7. What is forward propagation?
8. What is backpropagation?


#### Gradient descent and loss minimization

**OpenAI**
- **Q:** Explain gradient descent
  - *Role/level:* ML Engineer
  - *Source:* https://www.tryexponent.com/blog/top-machine-learning-interview-questions
  - *Evidence:* 1. Explain gradient descent. (OpenAI)

**Snapchat**
- **Q:** Explain mini-batch and stochastic gradient descent
  - *Role/level:* ML Engineer
  - *Source:* https://www.tryexponent.com/blog/top-deep-learning-interview-questions
  - *Evidence:* Watch an MLE from Snapchat answer this interview question: "Explain mini-batch and stochastic gradient descent"

**Unspecified**
- **Q:** What is gradient descent?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://medium.com/grabngoinfo/gradient-descent-vs-616ba269de8d
  - *Evidence:* Some example interview questions are

*   What is gradient descent?
*   What are the pros and cons of stochastic gradient descent?
*   What are the differences between batch gradient descent and mini-batch gradient descent?

- **Q:** What are the pros and cons of stochastic gradient descent?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://medium.com/grabngoinfo/gradient-descent-vs-616ba269de8d
  - *Evidence:* Some example interview questions are

*   What is gradient descent?
*   What are the pros and cons of stochastic gradient descent?
*   What are the differences between batch gradient descent and mini-batch gradient descent?

- **Q:** What are the differences between batch gradient descent and mini-batch gradient descent?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://medium.com/grabngoinfo/gradient-descent-vs-616ba269de8d
  - *Evidence:* Some example interview questions are

*   What is gradient descent?
*   What are the pros and cons of stochastic gradient descent?
*   What are the differences between batch gradient descent and mini-batch gradient descent?

- **Q:** What is the difference between the Gradient Descent method and the Ordinary Least Squares method? Which is better?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-gradient-descent/
  - *Evidence:* #### **What is the difference between the Gradient Descent method and the Ordinary Least Squares method? Which is better?**

- **Q:** What are the different types of Gradient Descent methods?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-gradient-descent/
  - *Evidence:* #### **What are the different types of Gradient Descent methods?**

- **Q:** Difference between batch gradient descent and stochastic gradient descent (SGD)?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.linkedin.com/posts/nihar-penchala_datascience-deeplearning-interviewquestions-activity-7371013068417458176-r2K_
  - *Evidence:* Here are today’s 10 interview questions:
...
8. Difference between batch gradient descent and stochastic gradient descent (SGD)?

- **Q:** Define gradient descent.
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.wecreateproblems.com/interview-questions/deep-learning-interview-questions
  - *Evidence:* 9. Define gradient descent.

- **Q:** Differentiate between stochastic gradient descent (SGD) and batch gradient descent.
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.wecreateproblems.com/interview-questions/deep-learning-interview-questions
  - *Evidence:* 6. Differentiate between stochastic gradient descent (SGD) and batch gradient descent.


#### Loss functions in deep learning

**Unspecified**
- **Q:** What are loss functions?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.analyticsvidhya.com/blog/2022/11/advance-guide-on-interview-questions-of-deep-learning/
  - *Evidence:* ### Q4. What are loss functions?

**Mean squared error:** This loss function is used for regression problems...

- **Q:** What is a loss function in Deep Learning?
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.wecreateproblems.com/interview-questions/deep-learning-interview-questions
  - *Evidence:* ### **Deep Learning – Beginner (1–40)**
...
13. What is a loss function in Deep Learning?


#### Loss vs cost function

**Unspecified**
- **Q:** What is the difference between Gradient Descent and Cost Function?
  - *Role/level:* ML / Data Science (general)
  - *Source:* https://analyticsarora.com/8-unique-machine-learning-interview-questions-about-gradient-descent/
  - *Evidence:* #### **What is the difference between Gradient Descent and Cost Function?**

The Cost Function is something that we intend to minimize...

- **Q:** Differentiate between cost function and loss function.
  - *Role/level:* Deep Learning (general)
  - *Source:* https://www.wecreateproblems.com/interview-questions/deep-learning-interview-questions
  - *Evidence:* 13. What is a loss function in Deep Learning?
14. Differentiate between cost function and loss function.


### Unattributed (excluded from main list)

- **Topic:** Loss functions in deep learning — This is a StackExchange question itself, not tied to a specific ML employer interview, and the pipeline requires tech employer attribution. — _Objective function, cost function, loss function: are they the same thing?_
- **Topic:** Gradient descent and loss minimization — Appears in devinterview.io content but without explicit company attribution; site is a general prep resource, not a specific tech employer. — _How does gradient descent help in finding the local minimum of a function?_
- **Topic:** Gradient descent and loss minimization — Same as above: general prep site with no company context in the provided block. — _What are the main variants of gradient descent algorithms?_
