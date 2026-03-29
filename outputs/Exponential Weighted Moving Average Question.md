# Interview question bank

*MLE / Data Scientist / AI Engineer — sourced from web; not invented.*

## Interview question bank

*Target roles: ML Engineer / Data Scientist / AI Engineer. Questions are sourced from web pages only (not invented).*

### Summary table

| Topic | # Companies | # Questions | Top sources |
|-------|-------------|-------------|-------------|
| Exponential Weighted Moving Average | 2 | 2 | aryanupadhyay.com, corporatefinanceinstitute.com, towardsdatascience.com, medium |
| Time series forecasting with EWMA | 1 | 1 | corporatefinanceinstitute.com, aryanupadhyay.com |
| EWMA mathematical formulation | 3 | 3 | aryanupadhyay.com, corporatefinanceinstitute.com, towardsdatascience.com, medium |
| Effect of beta parameter in EWMA | 3 | 3 | aryanupadhyay.com, corporatefinanceinstitute.com, towardsdatascience.com |
| Exponential weight decay interpretation | 1 | 1 |  |
| Initialization choices for EWMA | 0 | 0 |  |
| EWMA in deep learning optimizers | 1 | 1 | medium.com |
| EWMA implementation in Python (Pandas) | 0 | 0 |  |
| Time series smoothing and trend extraction | 1 | 1 |  |
| **Total (all in bank)** | 4 | 12 | See sections below |

### Coverage report

- Extracted topics: 9
- Questions with attribution: 12
- Excluded (unattributed): 3

**Topic coverage notes:**
Content focuses on Exponential Weighted Moving Average: definition, properties, β parameter behavior, mathematical expansion, and practical implementation (Pandas) with mentions of its role in time series forecasting and deep learning optimizers. No unrelated ML topics appear in this excerpt.

### Questions by topic → company

#### EWMA in deep learning optimizers

**Medium (Piyush Kashyap)**
- **Q:** In deep learning, EWMA is used extensively in optimizers like **Adam**.
  - *Role/level:* General ML / DL
  - *Source:* https://medium.com/@piyushkashyap045/exponential-weighted-moving-average-ewma-in-deep-learning-a-simple-guide-with-examples-ef247b3f964b
  - *Evidence:* ## EWMA in Deep Learning Optimizers

In deep learning, EWMA is used extensively in optimizers like **Adam**. During training, Adam uses an exponential moving average to adjust the learning rate, making it more stable and allowing the model to converge faster.


#### EWMA mathematical formulation

**Aryan Upadhyay (blog)**
- **Q:** To understand how the curve is generated, we look at the recursive formula:
  - *Role/level:* General ML / DS
  - *Source:* https://www.aryanupadhyay.com/post/exponential-weighted-moving-average-ewma-theory-formula-example-intuition
  - *Evidence:* **Mathematical Formulation**

To understand how the curve is generated, we look at the recursive formula:

Vₜ = β Vₜ₋₁ + (1 − β)θₜ

**Corporate Finance Institute**
- **Q:** The EWMA’s simple mathematical formulation described below:
  - *Role/level:* Quant / Finance
  - *Source:* https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
  - *Evidence:* ### EWMA Formula

The EWMA’s simple mathematical formulation described below:

Where:

* **Alpha** = The weight decided by the user
* **r** = Value of the series in the current period

**Towards Data Science**
- **Q:** To understand how the exponential moving average works, let us look at its recursive equation:
  - *Role/level:* General ML / DS
  - *Source:* https://towardsdatascience.com/intuitive-explanation-of-exponential-moving-average-2eb9693ea4dc/
  - *Evidence:* ## Formula

To understand how the exponential moving average works, let us look at its recursive equation:

![Exponential moving average formula](...)


#### Effect of beta parameter in EWMA

**Aryan Upadhyay (blog)**
- **Q:** The value of β dictates how far back into the past the model looks.
  - *Role/level:* General ML / DS
  - *Source:* https://www.aryanupadhyay.com/post/exponential-weighted-moving-average-ewma-theory-formula-example-intuition
  - *Evidence:* **The Impact of Beta (**β**)**

The value of β dictates how far back into the past the model looks. A simple way to visualize this is :

* If β = 0.9 : We are averaging over roughly the last 10 days (1 / 0.1).

**Corporate Finance Institute**
- **Q:** For example, a 15-day moving average’s alpha is given by 2/(15+1), which means alpha is 0.125.
  - *Role/level:* Quant / Finance
  - *Source:* https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
  - *Evidence:* #### N-Day EWMA

To compute the moving average, we first need to find the corresponding alpha, which is given by the formula below:

...

For example, a 15-day moving average’s alpha is given by 2/(15+1), which means alpha is 0.125.

**Towards Data Science**
- **Q:** In reality, weights lower than 1 / e make a tiny impact on the exponentially weighted average. That is why it is said that **for a given value of β, the exponential weighted average takes into consideration the last t = 1 / (1 – β) observations**.
  - *Role/level:* General ML / DS
  - *Source:* https://towardsdatascience.com/intuitive-explanation-of-exponential-moving-average-2eb9693ea4dc/
  - *Evidence:* In reality, weights lower than 1 / e make a tiny impact on the exponentially weighted average. That is why it is said that **for a given value of β, the exponential weighted average takes into consideration the last t = 1 / (1 – β) observations**.


#### Exponential Weighted Moving Average

**Medium (Piyush Kashyap)**
- **Q:** What is EWMA?
  - *Role/level:* General ML / DL
  - *Source:* https://medium.com/@piyushkashyap045/exponential-weighted-moving-average-ewma-in-deep-learning-a-simple-guide-with-examples-ef247b3f964b
  - *Evidence:* ## What is EWMA?

The Exponential Weighted Moving Average (EWMA) is a statistical technique used to find trends in time-series data.

**Towards Data Science**
- **Q:** What is Exponential Moving Average (EMA)?
  - *Role/level:* General ML / DS
  - *Source:* https://towardsdatascience.com/intuitive-explanation-of-exponential-moving-average-2eb9693ea4dc/
  - *Evidence:* An **exponential (weighted) moving average** is a robust trade-off between these two methods.

This article covers the motivation behind the method, a description of its workflow and bias correction – an effective technique to overcome a bias obstacle in approximation.


#### Exponential weight decay interpretation

**Aryan Upadhyay (blog)**
- **Q:** We previously stated that EWMA gives more importance to new data and less to old data. But can we prove this mathematically ?
  - *Role/level:* General ML / DS
  - *Source:* https://www.aryanupadhyay.com/post/exponential-weighted-moving-average-ewma-theory-formula-example-intuition
  - *Evidence:* **Mathematical Intuition: The Proof**

We previously stated that EWMA gives more importance to new data and less to old data. But can we prove this mathematically ?

Let's expand the recursive formula step-by-step to see exactly how the weights are assigned.


#### Time series forecasting with EWMA

**Corporate Finance Institute**
- **Q:** What is the Exponentially Weighted Moving Average (EWMA)?
  - *Role/level:* Quant / Finance
  - *Source:* https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/exponentially-weighted-moving-average-ewma/
  - *Evidence:* ## What is the Exponentially Weighted Moving Average (EWMA)?

The Exponentially Weighted Moving Average (EWMA) is a quantitative or statistical measure used to model or describe a time series.


#### Time series smoothing and trend extraction

**Aryan Upadhyay (blog)**
- **Q:** What is Exponential Weighted Moving Average (EWMA) ?
  - *Role/level:* General ML / DS
  - *Source:* https://www.aryanupadhyay.com/post/exponential-weighted-moving-average-ewma-theory-formula-example-intuition
  - *Evidence:* Exponential Weighted Moving Average (EWMA): Theory, Formula, Example & Intuition

**What is Exponential Weighted Moving Average (EWMA) ?**

Let's look at the time-series data in the graph above.


### Unattributed (excluded from main list)

- **Topic:** Any — This exact question string does not appear verbatim or as a substring in any provided source; only descriptive prose exists. — _Explain EWMA in the context of Adam optimizer._
- **Topic:** EWMA implementation in Python (Pandas) — No provided block mentions pandas or Series.ewm(), so cannot attribute this as a real interview question. — _How do you implement EWMA using pandas.Series.ewm().mean()?_
- **Topic:** Initialization choices for EWMA — Initialization trade-offs are discussed conceptually but not posed as an explicit question in the text. — _How should you initialize V0 for EWMA (e.g., 0 vs first observation) and what are the trade-offs?_
