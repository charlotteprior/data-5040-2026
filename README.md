# Predicting Diabetes Risk from Health Indicators and Lifestyle Factors

CAPS-DATASTUDIES 5040, Applied Machine Learning (Fall 2026). Final project, binary logistic regression / screening-risk classification.

## Project summary

**Analytical question.** Can self-reported health, lifestyle, and demographic survey answers predict whether a respondent *reports* a diabetes diagnosis?

**Stakeholders and decision.** Public health programs and primary care providers deciding whom to flag for follow-up testing or targeted health education. The output supports prioritizing screening. It is not a diagnosis.

**Observation.** One respondent in the CDC Behavioral Risk Factor Surveillance System (BRFSS) survey.

**Target.** `Diabetes_binary` (0 = no diabetes, 1 = prediabetes or diabetes per UCI). The model learns *reported diabetes status at survey time*. It does not estimate future risk, and it does not detect undiagnosed diabetes. See "Scope of claims" below.

**Mathematical contract.** z_i = (x_i, y_i) ∈ 𝒵 = 𝒳 × 𝒴, with x_i ∈ ℝ^21 the survey answers for respondent i, y_i ∈ {0, 1} the reported diabetes indicator, and S_n = {z_i}, i = 1..n, n = 253,680.

## Scope of claims (from Milestone 1 feedback)

| Question | Supported by this data? |
|---|---|
| Does this respondent report having diabetes? | Yes. This is what the label records. |
| Does this respondent have *undiagnosed* diabetes? | No. Undiagnosed cases are labeled 0 and cannot be told apart. |
| Will this respondent develop diabetes later? | No. The data is cross-sectional (one survey wave, no follow-up). |

A random train/test split shows performance on held-out respondents from the same 2014 survey population. It does not show performance in current clinical screening.

## Goals

1. Audit the data: provenance, structure, missingness, duplicates, class balance, and the real coding of the target.
2. Resolve the target definition (binary vs. three-class; what the positive class actually contains).
3. Assess feature availability and leakage (e.g. `HighBP`, `HighChol`, `CholCheck`, `AnyHealthcare`, `GenHlth` as proxies for prior medical contact).
4. Decide whether BRFSS survey weights and design matter for the intended claim.
5. Build a leakage-safe pipeline (all fitted preprocessing learned on training folds only).
6. Fit a simple baseline (dummy, then logistic regression) with cross-validated results on the development set.
7. Choose metrics tied to screening costs: a missed case costs more than an unnecessary follow-up, so recall, precision, and PR-AUC lead, with calibration for probability outputs.
8. Keep the final test set untouched until a later milestone.

Milestone requirements are tracked in [MILESTONE2_REQUIREMENTS.md](MILESTONE2_REQUIREMENTS.md).

## Data

- **Source:** UCI Machine Learning Repository, [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (DOI 10.24432/C53919). Derived from CDC BRFSS annual data; the Kaggle copy is [alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).
- **Size:** 253,680 rows, 21 features plus the target. UCI reports no missing values. UCI's own page is inconsistent (it says 21 features in the metadata and 35 in the text), so confirm in the notebook.
- **Collection:** telephone survey, all self-reported. Age is pre-bucketed. Gender, income, and education are flagged as sensitive.
- **UCI recommended evaluation:** cross-validation or a fixed train/test split.
- **Load in Python:**

```python
from ucimlrepo import fetch_ucirepo
ds = fetch_ucirepo(id=891)
X, y = ds.data.features, ds.data.targets
```

### Items to verify in the notebook (not yet checked)

- **Survey year.** UCI says 2014; the Kaggle file names say BRFSS 2015. Confirm which year the rows come from before stating it.
- **What `Diabetes_binary = 1` contains.** UCI describes it as "prediabetes or diabetes". As I recall, the Kaggle three-class file (`Diabetes_012`: 0 none, 1 prediabetes, 2 diabetes) has about 4,631 prediabetes rows, and the binary file's positive count matches the diabetes-only count. If that holds, prediabetes sits in class 0, not class 1. Check the counts against the three-class file.
- **Duplicate rows.** Expect a substantial number of exact duplicates; duplicates across a train/test split are a leakage path.
- **Survey weights.** The file has no weight, strata, or PSU columns, so population-level conclusions cannot be weighted from this data alone.

## Data dictionary

Coding follows the UCI/Kaggle documentation. The UCI page lists only the first ten rows in full; the remainder is from the dataset's Kaggle documentation and should be confirmed against the BRFSS codebook.

| Variable | Role | Type | Meaning / coding |
|---|---|---|---|
| `ID` | ID | Integer | Respondent identifier (not a feature; drop before modeling). |
| `Diabetes_binary` | Target | Binary | 0 = no diabetes; 1 = prediabetes or diabetes (see verification note above). |
| `HighBP` | Feature | Binary | 1 = told they have high blood pressure. |
| `HighChol` | Feature | Binary | 1 = told they have high cholesterol. |
| `CholCheck` | Feature | Binary | 1 = cholesterol checked in the past 5 years. |
| `BMI` | Feature | Integer | Body mass index (self-reported height and weight). |
| `Smoker` | Feature | Binary | 1 = has smoked at least 100 cigarettes (5 packs) in lifetime. |
| `Stroke` | Feature | Binary | 1 = ever told they had a stroke. |
| `HeartDiseaseorAttack` | Feature | Binary | 1 = coronary heart disease or myocardial infarction. |
| `PhysActivity` | Feature | Binary | 1 = physical activity in past 30 days, excluding job. |
| `Fruits` | Feature | Binary | 1 = consumes fruit 1 or more times per day. |
| `Veggies` | Feature | Binary | 1 = consumes vegetables 1 or more times per day. |
| `HvyAlcoholConsump` | Feature | Binary | 1 = heavy drinker (men over 14 drinks/week, women over 7). |
| `AnyHealthcare` | Feature | Binary | 1 = has any health care coverage. |
| `NoDocbcCost` | Feature | Binary | 1 = needed a doctor in past 12 months but could not because of cost. |
| `GenHlth` | Feature | Ordinal 1-5 | Self-rated general health: 1 excellent, 2 very good, 3 good, 4 fair, 5 poor. |
| `MentHlth` | Feature | Integer 0-30 | Days of poor mental health in past 30 days. |
| `PhysHlth` | Feature | Integer 0-30 | Days of poor physical health in past 30 days. |
| `DiffWalk` | Feature | Binary | 1 = serious difficulty walking or climbing stairs. |
| `Sex` | Feature | Binary | 0 = female, 1 = male. |
| `Age` | Feature | Ordinal 1-13 | 13-level age category: 1 = 18-24, 2 = 25-29, ... 12 = 75-79, 13 = 80 or older. |
| `Education` | Feature | Ordinal 1-6 | 1 never attended or kindergarten only ... 6 college graduate. |
| `Income` | Feature | Ordinal 1-8 | 1 = under $10,000 ... 8 = $75,000 or more. |

## Reproducibility

Environment, data access, and rerun steps will be added once `milestone2.ipynb` is built (Python version, package versions, `random_state`, and the exact cell order).

## Repository contents

- `milestone2.ipynb`: Milestone 2 notebook (data audit, baseline, evaluation design).
- `MILESTONE2_REQUIREMENTS.md`: requirements checklist and Milestone 1 feedback matrix.
- `README.md`: this file.
