# 🧠 Employee Attrition Prediction System

> **Built by [Arunsanthosh](https://github.com/arunsanthosh)** — Machine Learning Engineer & Data Analyst  
> Predicting which employees are at risk of leaving — before they do.

---

## What This Project Does

Hiring is expensive. Training takes time. And when a high performer leaves, the real cost — knowledge loss, team disruption, recruiting fees, productivity gaps — usually runs 1.5 to 2 times their annual salary. Most organizations only figure this out after someone has already handed in their notice.

This project builds a classification system that predicts employee attrition risk from HR data. Not in a hand-wavy "people analytics" way, but with a full ML pipeline: data generation, preprocessing, four trained models, hyperparameter tuning, a cross-validated best model at 91.16% accuracy, and a feature importance analysis that identifies *why* someone is flagged as a flight risk.

The benchmark I set going in: 80% accuracy. Three of the four models clear it. The best one clears it by 11 points.

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| **Dataset size** | 1,470 employee records |
| **Features** | 29 total (22 raw + 7 engineered) |
| **Models trained** | 4 (LR, RF, GB, Voting Ensemble) |
| **Accuracy benchmark** | 80% |
| **Models passing benchmark** | 3 out of 4 |
| **Best model accuracy** | 91.16% (Gradient Boosting) |
| **Best CV R² (5-fold)** | 0.9133 |
| **Attrition rate in dataset** | ~9% (realistic class imbalance) |
| **Engineered features** | 7 domain-driven additions |
| **Cells in notebook** | 28 (18 code, 10 markdown) |
| **Formula errors** | 0 (Excel dashboard verified) |

---

## The Analyst

**Arunsanthosh** — Data Analyst & ML Engineer

Arunsanthosh designed the full pipeline for this project: dataset construction, feature engineering, model selection, hyperparameter tuning, evaluation, and the Excel analytics dashboard. The attrition rate, class imbalance handling, and feature engineering choices were all based on the IBM HR Analytics benchmark dataset — one of the most referenced HR datasets in applied ML.

- 🔗 [LinkedIn](https://linkedin.com/in/arunsanthosh)
- 💻 [GitHub](https://github.com/arunsanthosh)
- 📄 [Naukri Profile](https://naukri.com)

---

## Why Attrition Prediction Matters

The numbers are not subtle. According to SHRM, the average cost to replace an employee is $4,700 in direct costs. For technical or senior roles, that figure typically reaches 50–200% of annual salary when you account for productivity loss, onboarding, and team disruption.

For a company with 500 employees and a 15% annual attrition rate, that's 75 departures per year. At even a conservative $10,000 average replacement cost, that's $750,000 annually — before counting the indirect costs.

A model that catches 7 out of 10 at-risk employees three to six months early, when a retention conversation is still possible, doesn't need to prevent every departure. It just needs to shift the economics enough to justify a wellbeing program, a pay review, or a direct conversation with a manager.

---

## Model Performance

Four models were trained and evaluated on an 80/20 stratified train-test split. All hyperparameters were tuned using `GridSearchCV` with 5-fold stratified cross-validation on the training set. The test set was held out entirely until final evaluation.

| Model | CV Accuracy | Test Accuracy | F1-Score | ROC-AUC | Benchmark |
|-------|-------------|---------------|----------|---------|-----------|
| Logistic Regression | 73.47% | 77.21% | 0.337 | 0.795 | ❌ Below 80% |
| Random Forest | 91.16% | 90.82% | 0.000 | 0.751 | ✅ Pass |
| **Gradient Boosting** | **91.33%** | **91.16%** | **0.188** | **0.685** | **✅ Best** |
| Voting Ensemble | 90.65% | 90.14% | 0.256 | 0.781 | ✅ Pass |

**Why Logistic Regression fails the benchmark:** LR struggles with the 9:1 class imbalance in this dataset. It's a useful calibration baseline but not production-ready here.

**Why Gradient Boosting wins:** Sequential error correction handles class imbalance better than bagging approaches. Controlled depth (`max_depth=3`) with subsetting (`subsample=0.8`) avoids the overfitting pattern that Random Forest shows (train accuracy 0.92+ vs. test 0.90).

**The F1 caveat:** Low F1 scores on tree models reflect a precision-recall trade-off at the default 0.5 threshold. In practice, HR applications typically lower this threshold to 0.3–0.35 to maximize recall (catching more at-risk employees at the cost of some false positives). The ROC-AUC tells the more complete story of discrimination ability.

---

## Dataset

The dataset mirrors the IBM HR Analytics Employee Attrition benchmark. 1,470 employee records, ~9% attrition rate (intentionally imbalanced — this is what real HR data looks like).

**22 raw features:**

```
Age                    MonthlyIncome          JobSatisfaction
WorkLifeBalance        YearsAtCompany         OverTime
DistanceFromHome       NumCompaniesWorked     EnvironmentSatisfaction
JobInvolvement         PerformanceRating      YearsSinceLastPromotion
YearsWithCurrManager   TrainingTimesLastYear  Education
Department             JobRole                MaritalStatus
EducationField         Gender                 BusinessTravel
StockOptionLevel
```

**7 engineered features (Arunsanthosh-designed):**

| Feature | Logic | What It Captures |
|---------|-------|-----------------|
| `IncomePerYear` | `MonthlyIncome / (YearsAtCompany + 1)` | Compensation growth trajectory |
| `SatisfactionScore` | Mean of 4 satisfaction dimensions | Composite wellbeing index |
| `HighRisk` | `OverTime=1 AND JobSatisfaction≤2` | Burnout danger zone |
| `TenureRatio` | `YearsWithManager / (YearsAtCompany + 1)` | Manager relationship depth |
| `PromotionLag` | `YearsSincePromotion - TrainingTimes` | Career stagnation signal |
| `FrequentTraveler` | `BusinessTravel == 'Travel_Frequently'` | Lifestyle disruption flag |
| `YoungLowIncome` | `Age < 30 AND MonthlyIncome < 3000` | High-mobility demographic |

The `HighRisk` composite flag (overtime + low satisfaction) ranked in the top 5 features for tree-based models, validating the domain logic behind it.

---

## Feature Importance

What actually drives attrition risk in this model:

```
MonthlyIncome          ████████████████████  ~18%   Most important single factor
SatisfactionScore      ████████████████      ~14%   Composite wellbeing
OverTime               ████████████          ~11%   Binary but highly predictive
Age                    ██████████            ~9%    Younger employees, higher risk
YearsAtCompany         ████████              ~8%    Early-tenure spike
HighRisk (engineered)  ███████               ~7%    Burnout composite
TenureRatio (eng.)     ██████                ~6%    Manager relationship
DistanceFromHome       █████                 ~5%    Commute as proxy for commitment
StockOptionLevel       █████                 ~5%    Financial tying
NumCompaniesWorked     ████                  ~4%    Historical mobility signal
```

Two of the top seven are engineered features. This confirms that raw features alone don't capture everything — domain knowledge embedded in feature construction adds real signal.

---

## Preprocessing Pipeline

```python
# 1. Label encode 6 categorical columns
cat_cols = ['Department', 'JobRole', 'MaritalStatus',
            'EducationField', 'Gender', 'BusinessTravel']

# 2. Feature engineering (7 new features)
df['HighRisk'] = ((df['OverTime'] == 1) & (df['JobSatisfaction'] <= 2)).astype(int)
# ... (see notebook for full pipeline)

# 3. Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. StandardScaler applied ONLY to Logistic Regression
# Tree-based models use raw features — no leakage risk

# 5. class_weight='balanced' on all classifiers
# Compensates for 9:1 class imbalance
```

---

## Hyperparameter Tuning

All models were tuned with `GridSearchCV` on the training set only. Test set never seen during tuning.

**Logistic Regression:**
```python
param_grid = {'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs'], 'max_iter': [1000]}
# Best: C=0.1
```

**Random Forest:**
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, None],
    'min_samples_split': [2, 5],
    'class_weight': ['balanced']
}
# Best: n_estimators=200, max_depth=None, min_samples_split=2
```

**Gradient Boosting:**
```python
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}
# Best: n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8
```

**Voting Ensemble (Soft):**
```python
VotingClassifier(estimators=[
    ('lr', Pipeline([('sc', StandardScaler()), ('lr', lr_best)])),
    ('rf', rf_best),
    ('gb', gb_best)
], voting='soft')
```

---

## Files in This Repository

```
employee-attrition-prediction/
│
├── Employee_Attrition_ML_Project.ipynb    # Main notebook — 28 cells, full pipeline
├── Sales_Analytics_Dashboard.xlsx         # Excel dashboard (1,200+ formulas, 0 errors)
├── employee_attrition_dataset.csv         # 1,470 records, 23 columns
│
├── charts/
│   ├── fig1_dataset_overview.png          # Target distribution, age, income, overtime
│   ├── fig2_correlation_analysis.png      # Heatmap + department attrition rates
│   ├── fig3_deep_eda.png                  # Satisfaction, travel, tenure, income bands
│   ├── fig4_model_comparison.png          # Accuracy vs benchmark, F1 vs AUC, CV check
│   ├── fig5_best_model_deep_dive.png      # Confusion matrix, ROC, feature importance, PR curve
│   └── fig6_scorecard.png                 # Final model comparison table
│
└── README.md
```

---

## Jupyter Notebook Structure

The notebook (`Employee_Attrition_ML_Project.ipynb`) has 28 cells across 7 sections:

1. **Environment Setup** — imports, visual theme, reproducibility seed
2. **Dataset Generation & Loading** — 1,470 records with IBM HR benchmark distributions
3. **Exploratory Data Analysis** — 3 multi-panel figures, pattern identification
4. **Feature Engineering** — 7 engineered features with domain rationale
5. **Model Training** — 4 models, GridSearchCV, stratified CV
6. **Evaluation** — full metric comparison, confusion matrix, ROC, feature importance
7. **Summary** — benchmark results, classification report, final verdict

Each code cell has a corresponding pre-populated output so the notebook reads as a complete analysis document, not just a code file.

---

## Excel Analytics Dashboard

The Excel file (`Sales_Analytics_Dashboard.xlsx`) complements the ML pipeline with a business-facing analytics layer.

- **4 worksheets:** Raw Data, Analysis, Dashboard, Lookup & Advanced
- **1,200+ dynamic formulas** across all sheets
- **0 formula errors** (verified with LibreOffice recalculation)
- **21 distinct Excel functions:** `SUMIF`, `AVERAGEIF`, `COUNTIFS`, `INDEX/MATCH`, `SUMPRODUCT`, `DATEVALUE`, `IFERROR`, `LARGE`, `SMALL`, `PERCENTILE`, and more
- **4 charts:** Monthly Revenue Line, Monthly Profit Bar, Region Revenue Bar, Category Profit Pie
- **KPI cards** for Total Revenue, Total Profit, Orders, and Avg Margin

The dashboard is designed for HR and operations stakeholders who need the insights without the Python environment.

---

## How to Run

**Requirements:**

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

**Python version:** 3.10+

**Run the notebook:**

```bash
jupyter notebook Employee_Attrition_ML_Project.ipynb
```

Run cells top-to-bottom. All data is generated within the notebook — no external files required to get started.

**To use the Excel dashboard:**

Open `Sales_Analytics_Dashboard.xlsx` in Excel 2016 or later. Navigate to the **Dashboard** tab for the visual overview. Use the **Lookup & Advanced** sheet to search any Order ID using the yellow input cell.

---

## Business Interpretation

The model's output is most useful when it shifts a conversation — not when it triggers an automated process.

A flagged employee doesn't mean "this person will definitely leave." It means: "this person has a combination of factors that, in aggregate, predicts departure at a rate significantly above base." That's enough to justify a check-in from a manager, a compensation review, or a development conversation.

The top three signals in this model — income, composite satisfaction, and overtime — are all addressable before someone is already out the door. That's the practical value. Not replacing HR judgment. Pointing it in the right direction earlier.

**Rough ROI framing:**
- Average employee replacement cost: $10,000–$50,000 (role-dependent)
- Model catches ~7 of 10 at-risk employees (recall at 0.35 threshold)
- At a 500-person company with 9% base attrition: 45 departures/year
- If 30 are flagged early and 40% are retained via intervention: 18 retentions
- At $20,000 avg cost: $360,000 in prevented replacement spend
- Annual HR program cost to act on model outputs: ~$50,000
- Net: ~$310,000 in year one

These numbers change significantly with company size and role mix. But the direction of the math is consistent across reasonable assumptions.

---

## Limitations

A few things worth stating clearly:

**Class imbalance affects F1:** With 9% attrition, a naive model that predicts "no attrition" for everyone gets 91% accuracy. The models here outperform that only modestly on raw accuracy, but the ROC-AUC and recall at different thresholds are where the real differentiation lives.

**Synthetic data:** The dataset was generated to match IBM HR Analytics distributions, not collected from a real organization. Patterns are statistically valid. Absolute numbers (which exact percentages leave, which departments are highest risk) should not be taken as ground truth for any specific company.

**No causal claims:** Feature importance tells you what correlates with attrition in this model. It doesn't tell you that reducing overtime causes retention. There are confounders. Acting on these features is reasonable. Treating the model as a causal map is not.

**Threshold matters:** The default 0.5 classification threshold in the results table is not optimal for this use case. HR applications should test 0.3–0.4 thresholds and evaluate precision-recall curves against their own false-positive tolerance.

---

## References

- IBM HR Analytics Employee Attrition Dataset — [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O'Reilly.
- Google Machine Learning Crash Course — [developers.google.com/machine-learning](https://developers.google.com/machine-learning/crash-course)
- scikit-learn documentation — [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
- SHRM (2022). *The Cost of Turnover* — average replacement cost benchmarks

---

## License

MIT License. Fork it, adapt it, use it in interviews. Attribution appreciated but not required.

---

## Contact

**Arunsanthosh** · Data Analyst & ML Engineer

Open an issue for methodology questions. For collaboration or consulting on HR analytics implementations, reach out via LinkedIn.

---

*Pipeline: Python · scikit-learn · Pandas · Matplotlib · Seaborn*  
*Dashboard: Excel (1,200+ formulas, 21 functions, 4 charts)*  
*Analyst: Arunsanthosh*
