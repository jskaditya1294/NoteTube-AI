# YouTube Notes jAqVuYJ8TP8

## Metadata
- **Created at:** 2026-03-23 20:24:14
- **Video ID:** jAqVuYJ8TP8

## Notes

# Exponential Weighted Moving Average (EWMA)

## Time Series Context
- Time series data: values recorded over time (e.g., daily stock prices, daily city temperatures).
- Axes example:
  - X-axis: `date`
  - Y-axis: `temperature`
- Raw data plotted as a jagged line; a smooth line overlaid can represent an average/trend.

## What is Exponential Weighted Moving Average
- EWMA is a technique to find trends/patterns in time series–based data.
- Used to:
  - Extract trend from noisy time series.
  - Capture how the series is moving over time.
- Intuition:
  - Produces a smoothed curve (trend line) that follows the data but is less noisy.
  - Newer points influence the average more than older points.

## Where EWMA is Used
- Time series forecasting.
- Business forecasting (e.g., companies forecasting metrics).
- Signal processing.
- Deep learning:
  - Used while building optimizer algorithms.
  - Specifically relevant to momentum-based optimization methods.

## Two Key Properties of EWMA
- Property 1 (more weight to recent points):
  - For time series `x₁, x₂, x₃, …`, later points have higher weight than earlier ones.
  - Example: `x₃` has higher weight than `x₁`, `x₄` higher than `x₃`, etc.
- Property 2 (weights decay over time):
  - As time passes, the weight of any fixed point decreases.
  - A specific past data point is given less and less influence as we move further forward in time.

## EWMA Formula
- EWMA at time `t`: `v_t`
- Data value at time `t`: `x_t`
- Decay/weight parameter: `β` with `0 ≤ β < 1`
- Recursive formula:
  - `v_t = β * v_(t-1) + (1 - β) * x_t`
- Interpretation:
  - `β * v_(t-1)`: contribution from past (previous EWMA).
  - `(1 - β) * x_t`: contribution from current data point.

## Initialization of EWMA
- `v_0` (initial EWMA) is treated inconsistently in different resources:
  - Some take `v_0 = 0`.
  - Others take `v_0 = x_0` (first data point).
- Both are seen in practice; there is no single strict standard.

## Step-by-Step EWMA Computation (Example Pattern)
- Given time series values `x_0, x_1, x_2, …` and chosen `β`:
  - Choose `v_0` (e.g., `0` or `x_0`).
  - `v_1 = β * v_0 + (1 - β) * x_1`
  - `v_2 = β * v_1 + (1 - β) * x_2`
  - Continue similarly for `v_3, v_4, …`.
- Each `v_t` is then plotted to obtain the EWMA curve.

## Effect of β: Intuition via Window Length
- Approximate effective window length of EWMA:
  - EWMA acts roughly like an average over the last `1 / (1 - β)` points.
- Examples:
  - If `β = 0.9`:
    - `1 / (1 - 0.9) = 10`
    - EWMA behaves like an average of roughly last 10 points.
  - If `β = 0.5`:
    - `1 / (1 - 0.5) = 2`
    - EWMA behaves like an average of roughly last 2 points.

## Graphical Behavior for Different β
- Same data, EWMA computed with different β values:
  - Large `β` (e.g., `β = 0.98`):
    - Very smooth, slow to react.
    - Strong influence from long history; curve far from rapid fluctuations of raw data.
  - Medium `β` (e.g., `β = 0.8`, `β = 0.5`):
    - Intermediate smoothing; balances current vs. past.
  - Small `β` (e.g., `β = 0.1`):
    - EWMA follows data very closely.
    - Very “reactive” to recent changes; less smoothing.

## Interpretation of β in the Formula
- Formula: `v_t = β * v_(t-1) + (1 - β) * x_t`
- `β` controls how much weight is given to past vs current:
  - `β * v_(t-1)`: past component (older points).
  - `(1 - β) * x_t`: present component (current point).
- High `β`:
  - Past component is large; older points get more cumulative weight.
  - System is less sensitive to sudden recent changes.
- Low `β`:
  - Present component dominates; current point heavily influences `v_t`.
  - System is more “moody,” quickly updating based on recent events.

## β and Weight Decay (Mathematical Expansion)
- Start with:
  - `v_t = β * v_(t-1) + (1 - β) * x_t`
- Expand recursively:
  - `v_1 = β * v_0 + (1 - β) * x_1`
  - `v_2 = β * v_1 + (1 - β) * x_2`
    - Substitute `v_1`:
    - `v_2 = β * (β * v_0 + (1 - β) * x_1) + (1 - β) * x_2`
    - `v_2 = β² * v_0 + β * (1 - β) * x_1 + (1 - β) * x_2`
  - `v_3 = β * v_2 + (1 - β) * x_3`
    - Substitute `v_2`:
    - `v_3 = β³ * v_0 + β² * (1 - β) * x_1 + β * (1 - β) * x_2 + (1 - β) * x_3`
  - Similarly for `v_4`, etc.
- General pattern:
  - Coefficients on older `x` values are powers of `β` multiplied by `(1 - β)`.
  - Example up to `v_4`:
    - `v_4` includes terms like:
      - `(1 - β) * x_4`
      - `β * (1 - β) * x_3`
      - `β² * (1 - β) * x_2`
      - `β³ * (1 - β) * x_1`
- Since `0 ≤ β < 1`:
  - Higher powers `β², β³, …` get smaller.
  - Thus older points are multiplied by smaller factors.
  - This proves mathematically: older data points have exponentially decaying weight.

## Intuitive Analogy for β
- Very small β (high focus on present):
  - Like a person whose mood changes instantly with today’s events.
- Very large β (high focus on past):
  - Like a person whose current state is determined mainly by long-term history, not by today’s event.
- Choice of β:
  - Need a “sweet spot” between being too reactive and too sluggish.
  - In many deep learning optimizers, values like `β ≈ 0.9` are common.

## EWMA in Python (Pandas)
- Example dataset:
  - Daily climate data (Delhi).
  - Columns:
    - `date`
    - `mean_temperature`
- Pandas function:
  - `Series.ewm()` is used to compute EWMA.
  - Key parameter: `alpha`
    - Relation to β: `alpha = 1 - β`
    - If `alpha = 0.9` then `β = 0.1`, etc.
- Basic usage pattern:
  - Compute EWMA for each day using:
    - `ewm(alpha=alpha_value).mean()` (implicitly applying EWMA).
  - Result:
    - A new column with EWMA values for each date.
- Plotting:
  - Plot raw temperature and EWMA together to see smoothing.
- Suggested practice:
  - Try different `alpha` (or equivalently `β`) values.
  - Observe how the EWMA curve changes.
  - As an exercise, implement EWMA manually from the formula instead of relying solely on `ewm()`.

## Key images

### "EFFECT OF BETA IN EWMA WITH EQUATIONS AND GRAPHS"
*Timestamp: 695s*

### EXPONENTIAL WEIGHTED MOVING AVERAGE (EWMA) GRAPHS WITH DIFFERENT BETA VALUES.
*Timestamp: 756s*

### "MATHEMATICAL INTUITION AND EWMA EQUATIONS WITH GRAPHS"
*Timestamp: 1002s*

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
