Technical Deep Dive · Customer Lifetime Value (CLV) · bp

# How to Calculate Customer Lifetime Value: From Linked Transactions to a 5-Year Value per Customer

A step-by-step walkthrough of the CLV method: defining churn, building the customer record, training survival models, and turning survival probabilities into a discounted lifetime value.

Agnivesh Das · Senior Data Analytics Engineer · bp

> **Note:** this describes the method and model logic. All dates, thresholds, parameters, code, customers and figures are illustrative, not production values or production code.

---

## The Formula

```
CLV  =  HCM  +  NPV

HCM  = Historical Customer Margin: actual margin earned from the customer over the last 24 months

NPV  = Σ (m = 1 … 60)   S(m) × E[margin(m)] × (1 + inflation)^(m/12)
                        ─────────────────────────────────────────────
                                   (1 + discount rate)^(m/12)

S(m)           = predicted probability the customer is still active in month m
E[margin(m)]   = expected monthly margin if the customer is active
```

Everything below exists to produce three inputs reliably: **HCM**, **S(m)** and **E[margin(m)]**.

---

## The Pipeline at a Glance

| Step | What happens | Output |
| ---- | ------------ | ------ |
| 1. Link | Take the unified customer view from SCI | One record per customer across loyalty, payment and till data |
| 2. Define windows | Split history into observation and performance periods | Time frames for features and labels |
| 3. Define churn | Turn "stopped visiting" into an event with a date | Churn flag + time to churn per customer |
| 4. Build the CAR | Engineer features in staged SQL | Customer Analytic Record (~580 features) |
| 5. Segment | Assign each customer to a behavioural segment | One modelling population per segment |
| 6. Prepare features | Clean, cap, scale, de-correlate, select | 60–130 features per segment |
| 7. Train | Survival model per segment (GBM with Cox loss) | Survival curve per customer |
| 8. Evaluate | C-index, AUC, rank order, calibration | Trust in the predictions |
| 9. Value | Margin × survival × inflation ÷ discount | NPV per customer |
| 10. Score & monitor | Monthly re-scoring, periodic retraining | Live CLV per customer |

---

## Step 1: Start From a Linked Customer View

CLV needs every purchase attributed to the right customer. That comes from the **Single Customer Identifier (SCI)**:

- Till (EPOS) baskets tell you *what* was bought and the margin it earned.
- Payment tokens tell you *how* it was paid.
- Loyalty accounts tell you *who* bought it.

SCI links the three, so a loyalty customer's card-paid visits (where they didn't swipe) still count towards their value. Without this, CLV would undervalue exactly the customers who are less engaged with loyalty, often the ones most at risk.

**Unit of analysis:** the loyalty account (`account_id`), with linked card spend attributed to it.

---

## Step 2: Define the Time Windows

Split a rolling 24-month history into two halves:

```
|<──── Observation period (12 months) ────>|<──── Performance period (12 months) ────>|
   Nov Y1 ............................ Oct Y2   Nov Y2 ............................ Oct Y3
        build features from here                      measure churn here
```

- **Observation period:** everything the model is allowed to *see* (features).
- **Performance period:** what the model has to *predict* (did they churn, and when).

Keeping these strictly separate prevents leakage: no feature may use data from the performance period.

---

## Step 3: Define Churn and Time-to-Churn

Fuel customers don't cancel; they just stop coming. So churn has to be defined:

**Rule:** a customer's churn date = **last transaction date + 6 months** of inactivity.

For each customer:

| Field | Definition |
| ----- | ---------- |
| `last_txn_date` | Date of last transaction up to the end of the performance period |
| `churn_date` | `last_txn_date` + 6 months |
| `churn_flag` | 1 if `churn_date` falls within the performance period, otherwise 0 |
| `time_to_churn` | Months from the start of the performance period to `churn_date` (if churned); otherwise 12 (censored) |

**Exclusions:** customers whose churn date falls *before* the performance period starts had already left during observation. They are removed (they would otherwise be left-censored noise).

**Censoring:** customers with `churn_flag = 0` haven't churned *yet*. We know they survived at least 12 months, but not how long beyond that. This is why we use survival models: they learn from censored customers rather than discarding them or treating them as permanent.

### Illustrative SQL

```sql
WITH last_txn AS (
    SELECT account_id,
           MAX(txn_date) AS last_txn_date
    FROM   sci_linked_transactions
    WHERE  txn_date < DATE '2024-11-01'          -- end of performance period
    GROUP  BY account_id
),
labels AS (
    SELECT account_id,
           last_txn_date,
           DATEADD(month, 6, last_txn_date)    AS churn_date
    FROM   last_txn
)
SELECT account_id,
       CASE WHEN churn_date < DATE '2024-11-01' THEN 1 ELSE 0 END          AS churn_flag,
       CASE WHEN churn_date < DATE '2024-11-01'
            THEN DATEDIFF(month, DATE '2023-11-01', churn_date)
            ELSE 12 END                                                      AS time_to_churn
FROM   labels
WHERE  churn_date >= DATE '2023-11-01';           -- exclude customers already churned in observation
```

---

## Step 4: Build the Customer Analytic Record (CAR)

The CAR is one row per customer, holding everything the model learns from. It is built in SQL in stages, then joined:

| Stage | Feature family | Examples |
| ----- | -------------- | -------- |
| 1 | **Transactions** | Fuel quantity and spend, shop spend, units, transaction counts, by month, quarter and trend |
| 2 | **Loyalty events & marketing** | Points issued and redeemed, virtual money, coupons, gift redemptions, CRM campaign responses |
| 3 | **Products** | Fuel grade mix (regular vs. premium), shop category mix (convenience, food service, impulse, tobacco) |
| 4 | **Customer behaviour flags** | Tenure, first swipe, active / sliding / gone-away flags, days since last visit, visit regularity |
| 5 | **Pricing** | Prices paid vs. competitor prices near the customer's usual sites |
| 6 | **Combine** | Join stages 1–5 on `account_id` into the final CAR |

### Feature patterns that matter most for churn

- **Recency:** days since last visit.
- **Frequency trend:** visits in the last 3 months vs. the 3 months before (a falling ratio signals a sliding customer).
- **Monetary trend:** spend and margin, recent vs. earlier.
- **Breadth:** fuel only, shop only, or both (customers using both tend to stay longer).
- **Engagement:** loyalty redemptions and campaign responses.
- **Price exposure:** how often the customer paid above nearby competitor prices.

### Illustrative SQL (Stage 1 fragment)

```sql
SELECT account_id,
       SUM(CASE WHEN item_type = 'FUEL' THEN quantity END)                          AS fuel_litres_12m,
       SUM(CASE WHEN item_type = 'SHOP' THEN sales_value END)                       AS shop_spend_12m,
       COUNT(DISTINCT txn_id)                                                       AS txns_12m,
       COUNT(DISTINCT CASE WHEN txn_date >= DATE '2023-08-01' THEN txn_id END)      AS txns_last_3m,
       COUNT(DISTINCT CASE WHEN txn_date BETWEEN DATE '2023-05-01'
                                             AND DATE '2023-07-31' THEN txn_id END) AS txns_prev_3m,
       DATEDIFF(day, MAX(txn_date), DATE '2023-10-31')                              AS days_since_last_txn
FROM   sci_linked_transactions
WHERE  txn_date BETWEEN DATE '2022-11-01' AND DATE '2023-10-31'   -- observation period only
GROUP  BY account_id;
```

### Validate the CAR before modelling

- Aggregate key KPIs (fuel quantity, shop spend, transaction share) by segment and reconcile them to existing dashboards.
- Profile every feature: min, mean, median, 95th / 99th / 99.5th percentiles, to catch broken joins and extreme values.

---

## Step 5: Segment the Customers

Customers behave very differently, so one model per segment works better than one model for all. Segments are based on transaction volume, value and margin, for example:

| Segment | Group | Margin |
| ------- | ----- | ------ |
| Ultimate Fans | B2C | High |
| Saving Regulars | B2C | High |
| Partner Focused Loyals | B2C | High |
| Fuel & Rewards Frequents | B2C | High |
| Saving Occasionals | B2C | Low |
| Low Fuel Occasionals | B2C | Low |
| Partner Focused Occasionals | B2C | Low |
| Fuel Card Professionals | B2B | High |

Churn rates varied several-fold across segments, which is the main reason for modelling them separately.

---

## Step 6: Prepare the Features

Starting from the full CAR, each segment's feature set is reduced to the 60–130 most useful features:

1. **Shortlist** business-relevant variables (~370 of ~580).
2. **Missing values:** fill with 0 (a missing activity count means no activity).
3. **Duplicates:** remove duplicate rows.
4. **Near-constant columns:** drop any feature whose most frequent value appears in more than 95% of rows.
5. **Outliers:** cap ratios, shares and tenure; clip other features at high quantiles.
6. **Scaling:** standardise numerical features to mean 0, standard deviation 1.
7. **Correlation:** drop one of each pair of highly correlated features.
8. **Multicollinearity:** iteratively remove features with high Variance Inflation Factor (VIF).
9. **Balanced selection:** keep a spread of features across categories (spend, quantity, transactions, marketing, price, behaviour) so no single family dominates.

### Illustrative Python

```python
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def prepare(car: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = car[features].fillna(0).drop_duplicates()

    # Drop near-constant columns
    keep = [c for c in X if X[c].value_counts(normalize=True).iloc[0] <= 0.95]
    X = X[keep]

    # Clip outliers
    X = X.clip(lower=X.quantile(0.005), upper=X.quantile(0.995), axis=1)

    # Standardise
    X = (X - X.mean()) / X.std()

    # Drop highly correlated pairs
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper if (upper[c] > 0.8).any()])

    # Iteratively remove high-VIF features
    while True:
        vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
                        index=X.columns)
        if vif.max() < 10:
            break
        X = X.drop(columns=vif.idxmax())
    return X
```

---

## Step 7: Train a Survival Model per Segment

### Why survival analysis

A standard classifier answers "will this customer churn within 12 months?" A survival model answers "**what is the probability this customer is still active at month 1, 2, 3 … 60?**" That full curve, S(m), is exactly what the NPV formula needs, and survival models use censored customers correctly.

### Models compared

| Model | Strength | Limitation |
| ----- | -------- | ---------- |
| Cox Proportional Hazards | Simple, interpretable hazard ratios | Assumes linear, constant effects over time |
| Random Survival Forest | Captures non-linear patterns | Heavier, harder to explain |
| **Gradient Boosting Survival (Cox loss)** | **Best accuracy, handles censoring and non-linearity, explainable via feature importance** | Needs hyperparameter tuning |

**Selected:** Gradient Boosting Machine trained with a Cox Proportional Hazards loss.

### Illustrative Python (scikit-survival)

```python
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

y = Surv.from_arrays(event=car["churn_flag"].astype(bool),
                     time=car["time_to_churn"])
X = prepare(car, feature_list)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=car["churn_flag"])

model = GradientBoostingSurvivalAnalysis(
    loss="coxph",
    learning_rate=0.05,
    n_estimators=400,
    max_depth=3,
    min_samples_split=50,
    min_samples_leaf=25,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)
```

**Tuning:** grid or random search over learning rate, number of trees, depth and minimum samples per split and leaf, optimising C-index on a validation fold.

**Extending beyond 12 months:** the model is trained on a 12-month performance window, but CLV needs 60 months. The baseline survival curve is extrapolated beyond the observed window (for example with a fitted parametric tail such as a Weibull or exponential decay), and each customer's risk score scales that baseline.

---

## Step 8: Evaluate the Model

| Metric | What it tells you | Good looks like |
| ------ | ----------------- | --------------- |
| **C-index** | How well the model *ranks* customers by time to churn | 0.5 = random; closer to 1 is better |
| **Time-dependent AUC (12 months)** | How well it separates churners from non-churners at 12 months | Well above 0.5 |
| **Accuracy / recall at 12 months** | Simplified confusion matrix using a probability cut-off | Recall high enough to catch most churners |
| **Decile rank order** | Do the highest-risk deciles really contain the most churners? | Churn rate rising steadily from lowest to highest decile |
| **Survival over time plot** | Mean predicted curve vs. baseline, with 10th–90th percentile band | Sensible, monotonic, not over-confident |
| **Feature importance** | Which features drive risk | Business-plausible drivers (recency, frequency trend, engagement) |

### Illustrative Python

```python
import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

risk = model.predict(X_test)
c_index = concordance_index_censored(y_test["event"], y_test["time"], risk)[0]

auc_12m, _ = cumulative_dynamic_auc(y_train, y_test, risk, times=[11.9])

# Decile rank order
deciles = pd.qcut(risk, 10, labels=False) + 1
rank_table = (pd.DataFrame({"decile": deciles, "churned": y_test["event"]})
                .groupby("decile")["churned"].agg(["count", "mean"]))
```

---

## Step 9: Turn Survival Into Value (NPV)

### 9a. Expected monthly margin

For each customer, estimate margin per active month from the observation period, using **margin rates by product**:

```
E[margin] = fuel_litres × fuel margin per litre (by grade)
          + Σ shop spend in category c × margin rate of category c
```

Using category-level shop margins (food service and hot drinks earn far more per euro than tobacco) means two customers with the same spend can have very different value.

### 9b. Inflation

Apply a yearly inflation rate to future margins, converted to a monthly factor:

```
inflation_factor(m) = (1 + yearly inflation)^(m / 12)
```

### 9c. Discounting

Discount future margin to today's money:

```
discount_factor(m) = 1 / (1 + discount rate)^(m / 12)
```

### 9d. Combine

```python
import numpy as np

months = np.arange(1, 61)                           # 5 years

def npv(surv_curve, monthly_margin, inflation=0.02, discount=0.08):
    s = surv_curve(months)                          # S(m) for m = 1..60
    infl = (1 + inflation) ** (months / 12)
    disc = (1 + discount) ** (-months / 12)
    return float(np.sum(s * monthly_margin * infl * disc))

surv_fns = model.predict_survival_function(X_score)   # one curve per customer
car_scored["npv"] = [npv(fn, m) for fn, m in zip(surv_fns, car_scored["monthly_margin"])]
car_scored["clv"] = car_scored["hcm_24m"] + car_scored["npv"]
```

*(In production, the 60-month curve uses the extrapolated tail from Step 7, and the NPV calculation is done in SQL against the scored survival tables.)*

### Worked example (yearly, simplified)

An *Ultimate Fans* customer with average margin of €18 per month (€216 a year), 2% inflation, 8% discount rate:

| Year | S(year end) | Expected margin | Discounted value |
| ---- | ----------- | --------------- | ---------------- |
| 1 | 0.80 | €216.0 | €160.0 |
| 2 | 0.66 | €220.3 | €124.7 |
| 3 | 0.55 | €224.7 | €98.1 |
| 4 | 0.46 | €229.2 | €77.5 |
| 5 | 0.39 | €233.8 | €62.1 |
| **NPV** | | | **≈ €522** |

```
HCM (24 months × €18) = €432
CLV = €432 + €522     ≈ €954
```

---

## Step 10: Score Monthly, Retrain Periodically

**Monthly scoring:**

1. Refresh the CAR for each segment with the latest rolling window.
2. Apply the same preprocessing (using the *training* parameters for scaling and caps).
3. Load each segment's trained model.
4. Predict survival curves for every `account_id`.
5. Calculate NPV and CLV, and write results to the output tables.
6. Log errors per segment without stopping the whole run.

**Retraining:** every 12–18 months, or sooner if monitoring shows drift.

**Monitoring:**

- **Data drift:** feature distributions vs. training (for example, population stability index).
- **Performance drift:** compare predicted vs. actual churn as new months arrive.
- **Value drift:** track the CLV distribution by segment month on month.

**Architecture (illustrative):** data in Amazon S3, exploration in Amazon Athena, model training, validation and scheduled scoring in Amazon SageMaker, with outputs written back to S3 for reporting and activation.

---

## Step 11: Use the Output

| Use case | How CLV is used |
| -------- | --------------- |
| **Loyalty rewards** | Reward customers by predicted future value, not just past points |
| **Retention** | Target high-CLV customers whose survival probability is falling |
| **Marketing** | Personalise offers and spend by value tier |
| **Value distribution** | Understand which segments hold most of the future margin |
| **Investment planning** | Estimate the return of long-term marketing programmes; model inflation's effect on value |

A useful view combines **value** and **risk**:

|                     | Low churn risk        | High churn risk                 |
| ------------------- | --------------------- | ------------------------------- |
| **High CLV**        | Protect and reward    | **Priority: retention offers**  |
| **Low CLV**         | Grow: cross-sell shop | Low-cost automated engagement   |

---

## Possible Extensions

- **Spend model alongside survival:** predict future margin per active month (for example with a Gamma-Gamma or gradient-boosted regression model) instead of assuming last year's margin continues.
- **Probabilistic CLV benchmarks:** compare against BG/NBD + Gamma-Gamma models as a sanity check.
- **Uncertainty bands:** report a range for each customer's CLV, not just a point estimate.
- **Uplift modelling:** estimate how much a retention offer actually changes survival, to target customers who will respond.

---

Created by Agnivesh Das · bp Customer & Product Data · All code, parameters and examples illustrative.
