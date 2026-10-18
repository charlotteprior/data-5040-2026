# Which Survey Questions Help Anticipate Diabetes Status? Health Indicators and Lifestyle Factors

CAPS-DATASTUDIES 5040, Applied Machine Learning (Fall 2026). Final project, binary logistic regression / screening-risk classification.

## Project summary

**Analytical question.** Which self-reported health, lifestyle, demographic, and care-history answers are associated with a respondent's *reported diabetes diagnosis*, and how well do they jointly discriminate it?

**Stakeholders and decision (revised after Milestone 1).** Primary care providers (PCPs) and public-health researchers. The project is a research guide for which questions to ask and which tests to consider when estimating diabetes status without directly testing glucose. It is not a diagnosis tool and not an individual patient decision tool. Prior medical contact (known blood pressure, cholesterol, care access) is accepted as part of the inputs.

**Observation.** One respondent in the CDC Behavioral Risk Factor Surveillance System (BRFSS) survey.

**Target.** `diabetes_reported`, engineered in the notebook from UCI's `Diabetes_binary`: 1 = reported diabetes, 0 = otherwise. Prediabetes cannot be separated using UCI data: a one-time external cross-check against the original three-class copy of this dataset showed that UCI's `Diabetes_binary = 1` is diabetes only and that prediabetes (4,631 respondents) is coded 0, contrary to UCI's description ("prediabetes or diabetes"). The project therefore models **binary reported diabetes**, with prediabetes in the reference class.

**Mathematical contract.** z_i = (x_i, y_i) in Z = X x Y, with x_i in R^21 the survey answers, y_i in {0, 1}, S_n = {z_i}, i = 1..n, n = 253,680 raw (229,474 after dropping exact duplicates).

## Scope of claims

| Question | Supported by this data? |
|---|---|
| Does this respondent report having diabetes? | Yes. This is what the label records. |
| Does this respondent have *undiagnosed* diabetes or prediabetes? | No. They are in the reference class and cannot be told apart. |
| Will this respondent develop diabetes later? | No. The data are cross-sectional (one survey wave, no follow-up). |
| Does asking a question or running a test *cause* better detection? | No. Results are associations in predictive accuracy. |

A random stratified split shows performance on held-out respondents from the same survey wave. It does not show performance in a later year, in clinical populations, within subgroups, or for the US adult population (no survey weights in the file).

## Milestone 2 headline results (development set, 5-fold CV)

- Reported diabetes is discriminated well above chance: PR-AUC 0.41 versus 0.15 prevalence, ROC-AUC 0.81. At 80% recall precision is 0.30 and about 40% of respondents are flagged.
- BMI, general health and functioning, and known conditions (high blood pressure, cholesterol, stroke, heart disease) carry most of the signal. Lifestyle and care-contact answers add little once those are known.
- Class weighting did not improve ranking and badly distorted probabilities.
- The final test set (45,895 rows) is locked and unused.

## Goals

1. Audit the data: provenance, structure, missingness, duplicates, class balance, and the real contents of the target. (Done in Milestone 2.)
2. Resolve the target definition. (Done: binary reported diabetes; prediabetes not separable from UCI data.)
3. Assess feature availability and leakage under the revised intended use. (Done; care-contact and comorbidity variables kept and measured.)
4. Decide whether BRFSS survey weights matter. (Open: the file has no design variables.)
5. Build a leakage-safe pipeline with all fitted preprocessing learned on training folds only. (Done.)
6. Fit a simple baseline with cross-validated results. (Done: dummy and logistic regression, plus a feature-group analysis.)
7. Choose metrics tied to the use: PR-AUC and precision at 80% recall, with log loss, Brier score and calibration. (Done.)
8. Later milestones: stronger models and threshold selection inside CV, subgroup checks, and a single evaluation on the locked test set.

Milestone requirements are tracked in [MILESTONE2_REQUIREMENTS.md](MILESTONE2_REQUIREMENTS.md).

## Data

- **Source:** UCI Machine Learning Repository, [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (DOI 10.24432/C53919). Derived from CDC BRFSS annual data. **UCI is the only data source the notebook uses.**
- **Size:** 253,680 rows, 21 features plus `ID` and the target. UCI reports no missing values. UCI's own page is inconsistent (it says 21 features in the metadata and 35 in the text), so the notebook confirms the structure.
- **Collection:** telephone survey, all self-reported. Age is pre-bucketed. Gender, income, and education are flagged as sensitive.
- **UCI recommended evaluation:** cross-validation or a fixed train/test split.
- **Load in Python:** `fetch_ucirepo(id=891)` from `ucimlrepo` (done by the notebook).

### Findings from the audit (computed in `milestone2.ipynb`)

- **Target contents.** Inside UCI data, `Diabetes_binary` has 218,334 zeros and 35,346 ones. A one-time cross-check outside the notebook against the original three-class copy found the ones are exactly the diabetes rows and 4,631 prediabetes respondents are zeros. The notebook does not use that copy.
- **Duplicates.** 24,206 exact duplicate rows (9.5%), concentrated in the reference class; 1,566 answer profiles occur with both labels.
- **Survey weights.** No weight, strata, or PSU column exists, so population-level conclusions cannot be weighted from this file.
- **Survey year (open).** UCI says 2014; other copies of the file say 2015. Not resolvable from the UCI data.

## Data dictionary

Coding follows the UCI documentation. The UCI page lists only the first ten variables in full; the remainder follows the widely published description of this dataset and should be confirmed against the BRFSS codebook.

| Variable | Role | Type | Meaning / coding |
|---|---|---|---|
| `ID` | ID | Integer | Respondent identifier (not a feature; drop before modeling). |
| `diabetes_reported` | Target (engineered in the notebook) | Binary | Copy of `Diabetes_binary`: 1 = reported diabetes; 0 = otherwise (including prediabetes). |
| `Diabetes_binary` | UCI label (kept for audit, not a feature) | Binary | UCI text says prediabetes or diabetes; external check shows diabetes only. |
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

- Install: `pip install -r requirements.txt` (versions the notebook was executed with are pinned there; Python 3.8.5).
- Run: open `milestone2.ipynb`, *Restart kernel and Run All* (about 1-2 minutes). The first run downloads the data from UCI via `ucimlrepo` into `data/` (git-ignored).
- Randomness: `SEED = 5040` for the split, the folds, and all models.
- Do not commit `data/`.

## Repository contents

- `milestone2.ipynb`: executed Milestone 2 notebook (data audit, baseline, evaluation design, Workflow Canvas v2, feedback matrix).
- `MILESTONE2_REQUIREMENTS.md`: requirements checklist.
- `requirements.txt`: package versions.
- `README.md`: this file.
