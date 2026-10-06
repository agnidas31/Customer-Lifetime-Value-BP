Field-Tested Innovation · Fuel & Convenience Retail · bp

# Creative Solutions: Customer Lifetime Value (CLV)

How a single, linked view of every customer was turned into a forward-looking number: what each customer has been worth to bp, and what they are likely to be worth over the next five years.

Agnivesh Das · Senior Data Analytics Engineer · bp · 2023–2025

> **Draft note:** this version is built from the Spain CLV documentation. The Germany use cases (OneNote) will be folded in once they can be read. Items marked *[Confirm]* need your check before publishing.

---

## My Role

### Building the data foundation the model stands on

A CLV model is only as good as its view of the customer. CLV was built directly on top of the Single Customer Identifier (SCI), which I helped build to link EPOS, Loyalty and Payment transactions into one customer record. SCI became the single source of customer truth that CLV was trained on.

As Senior Data Analytics Engineer on CLV, I:

- **Fed SCI into CLV.** Built and maintained the pipelines that delivered linked customer and transaction data into the CLV inputs. Without that link, the model could only have valued loyalty swipes. With it, the model could also see the visits customers paid for by card.
- **Engineered the Customer Analytic Record (CAR).** Wrote the SQL for the staged build of the per-customer record of roughly 580 features: transaction KPIs, loyalty events and campaigns, product usage, customer behaviour flags and competitor pricing, combined into one modelling dataset.
- **Partnered with the data scientists.** Worked alongside the modelling team on the rest of the project, so that the data the model learned from was accurate, complete and fit for purpose.

---

## The Problem

### Every customer looked the same size, because only the past was visible

Before CLV, the business could see what customers *had* spent, and only in fragments. Loyalty, marketing, pricing and store teams each had their own partial view, which led to fragmented decisions.

What nobody could answer was the question that actually drives investment:

- **Which customers are worth the most over the long run**, not just last month?
- **Who is about to leave**, while there is still time to keep them?
- **How much should we spend** to acquire, reward or win back a given customer?

A customer who spent heavily last year but is drifting away, and a modest spender who visits every week without fail, looked the same in a historical report. Their future value to bp is very different.

---

## The Constraint Nobody Was Naming

### In fuel retail, customers never tell you they've left

A mobile phone customer cancels a contract. A fuel customer just stops coming. That made "lifetime" hard to measure, for three reasons:

**1. There is no churn event to observe.** A customer who hasn't visited for two months might be on holiday, or might be gone for good. Churn had to be *defined* before it could be predicted.

**2. Most customers haven't churned yet.** At any point in time, most customers are still active, so their true lifetime is unknown. A standard model either ignores them or treats them as if they will never leave. Both are wrong. This is called **censored data**, and it needs survival analysis rather than ordinary regression.

**3. Customers are not one population.** A high-margin "Ultimate Fan" and a low-fuel occasional visitor churn at very different rates, buy different products, and respond to different offers. One model for everyone would blur exactly the differences the business needed to see.

The business didn't need a better report of last year's spend. It needed a model that could estimate, customer by customer, how long they would stay and what that time was worth.

---

## The Fix

### Past margin + predicted survival + discounting = lifetime value

CLV combines two parts:

```
CLV = Historical Customer Margin (HCM)  +  Net Present Value of future margin (NPV)
```

- **HCM** is what the customer has actually contributed in margin over the last 24 months.
- **NPV** is what they are expected to contribute over the next 5 years, weighted by the probability they are still a customer at each point, and discounted back to today's money.

Three settings define the model:

**X: Observation & Performance Windows**
12 months to observe each customer's behaviour, followed by 12 months to see whether they churned. The model is built on a rolling 24-month history.

**Y: Churn Definition**
A customer is treated as churned six months after their last transaction. Customers who had already churned before the observation window are excluded.

**Z: Forecast Horizon & Discounting**
Future margin is projected for up to 5 years, adjusted for inflation and discounted to present value.

### How it works

1. **Link:** Take the unified customer view from SCI: loyalty accounts linked to their card payments and till baskets.
2. **Segment:** Assign each customer to one of eight behavioural segments (seven B2C, one B2B), from high-margin "Ultimate Fans" to "Low Fuel Occasionals" and "Fuel Card Professionals".
3. **Build the CAR:** Create a Customer Analytic Record with roughly 580 features per customer, built in SQL in five stages: transactions, loyalty events and campaigns, products, customer behaviour flags and competitor pricing.
4. **Predict survival:** Train a survival model per segment that predicts, for each customer, the probability they are still active at each month over the next 5 years.
5. **Value the future:** Multiply expected survival by expected margin (category-level fuel and shop margins), apply inflation, and discount to present value.
6. **Score monthly:** Re-score every customer each month to track movement in value and churn risk; retrain the models every 12–18 months.

---

## Choosing the Model

### Three survival models tested, one clear winner

Three survival techniques were tested on the same segment:

| Model | What it does | Result |
| ----- | ------------ | ------ |
| **Cox Proportional Hazards** | Classic statistical survival model; assumes each factor's effect on churn risk is constant over time | Interpretable, but too rigid for non-linear behaviour |
| **Random Survival Forest** | Many decision trees, each predicting a survival curve | Handles complexity, harder to tune and explain |
| **Gradient Boosting Machine (survival)** | Boosted trees trained with a Cox loss function, so it handles censored data | **Selected:** best predictive accuracy, handles censoring, and still explainable through feature importance |

### Getting from 580 features to the ones that matter

For each segment, the feature set was cut down to the most informative 60–130 features:

- Removed duplicates and near-constant columns (one value more than 95% of the time).
- Capped outliers on ratios, shares and tenure.
- Standardised numerical features.
- Dropped highly correlated features and used the Variance Inflation Factor (VIF) to remove multicollinearity.
- Kept a balanced mix across categories: spend, quantity, transactions, marketing, price and customer behaviour.

### Checking it works

Each segment's model was tested on a held-out test set using:

- **Concordance index (C-index):** how well the model ranks customers by how long they will stay.
- **AUC-ROC at 12 months:** how well it separates customers who churn within a year from those who don't.
- **Accuracy and recall:** using a simplified confusion matrix at the 12-month point.
- **Decile rank order:** confirming that customers ranked as highest-risk really did churn most.

Across the eight segments, test C-index and AUC were roughly in the **0.75–0.85** range. *[Confirm OK to publish]*

---

## Worked Example

### What a CLV score actually looks like

> **Note:** the method reflects how the model actually worked. The customer, margins, survival probabilities and rates below are illustrative, not real production figures.

**Customer:** loyalty account in the *Ultimate Fans* segment (high-margin B2C, buys premium fuel and shop items).

**Step 1: Historical Customer Margin (last 24 months)**
Average margin of €18 a month across fuel and shop → **HCM = €432**

**Step 2: Predicted survival (from the segment model)**

| Year | Probability still active | Expected annual margin (2% inflation) | Discounted at 8% |
| ---- | ------------------------ | ------------------------------------- | ---------------- |
| 1 | 0.80 | €216.0 | €160.0 |
| 2 | 0.66 | €220.3 | €124.7 |
| 3 | 0.55 | €224.7 | €98.1 |
| 4 | 0.46 | €229.2 | €77.5 |
| 5 | 0.39 | €233.8 | €62.1 |

**Step 3: Net Present Value of future margin** → **NPV ≈ €522**

**Step 4: Customer Lifetime Value**
**CLV = €432 + €522 ≈ €954**

Now compare a second customer with the same past margin but a steeper survival curve (for example 0.55 after year 1). Their HCM is identical, but their NPV, and therefore their CLV, is far lower. That is the difference a historical report could never show, and it is exactly where retention spend should go.

---

## Built to Scale

### A repeatable, monthly CLV product

- **Monthly scoring.** Every customer is re-scored each month with the pre-trained segment models, so changes in value and churn risk are tracked over time.
- **Periodic retraining.** Models are retrained every 12–18 months to keep up with changing customer behaviour and inflation.
- **Granular margins.** Fuel margins by grade (Regular and Ultimate) and shop margins by category (Convenience, Food Service, Impulse, Tobacco and their subcategories), so the value reflects *what* a customer buys, not just how much.
- **Market-specific design.** Each market's loyalty mechanics are reflected in the features. In Spain, for example, the record captures points, virtual money, coupons ("chollos"), gift redemptions and partner-store activity.
- **Cloud pipeline.** Data exploration in Amazon Athena, model building, validation and production scoring in Amazon SageMaker, with data in Amazon S3.
- **Multi-market.** The same approach, on the same SCI foundation, was applied to Spain and Germany. *[Confirm and add Germany detail]*

---

## Why This Beat The Old Approach

### Same customers, fundamentally different view of their value

|                                 | Before CLV                                  | With CLV                                                      |
| ------------------------------- | ------------------------------------------- | ------------------------------------------------------------- |
| View of value                   | Past spend only                             | Past margin + 5-year predicted margin                         |
| Churn                           | Noticed after the customer was already gone | Predicted per customer, month by month                        |
| Measure used                    | Revenue                                     | Margin, by fuel grade and shop category                       |
| Customers treated as            | One population                              | Eight behavioural segments, each with its own model           |
| Inflation and time value        | Ignored                                     | Inflation-adjusted and discounted to present value            |
| Freshness                       | Ad hoc reports                              | Re-scored monthly, retrained every 12–18 months               |

---

## Impact

### What changed

**24 months → 5 years**
From looking back at two years of history to looking ahead five years per customer.

**8 segment models**
A dedicated survival model for each customer segment.

**~580 features**
Per customer, across transactions, loyalty events, products, behaviour and pricing.

**Monthly**
Every customer re-scored each month.

**What the business could now do:**

- **Reward loyalty based on value:** align loyalty rewards with each customer's predicted future value, not just past points.
- **Target retention:** focus offers on valuable customers whose survival is falling, before they leave.
- **Personalise marketing:** tailor promotions to projected value instead of sending the same offer to everyone.
- **Plan long-term investment:** evaluate the return of marketing programmes and model the impact of inflation on customer value.

The bigger shift wasn't building a churn model. It was changing the question from "what did this customer spend?" to "what is this customer worth, and how is that changing?" That gave loyalty, marketing and pricing teams one shared number to make decisions with.

---

Created by Agnivesh Das · bp Customer & Product Data · Illustrative example: customer, margins and rates fictionalised. CAR = Customer Analytic Record. SCI = Single Customer Identifier.
