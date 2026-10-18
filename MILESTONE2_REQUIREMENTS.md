# Milestone 2 Requirements: Data Audit, Baseline, and Evaluation Design

**Status: all sections implemented and executed in `milestone2.ipynb`.** The checklist below is the original requirement list; items marked done were checked against the notebook, not auto-verified. Project changes since this file was first written: the target is binary reported diabetes (the UCI binary label excludes prediabetes and UCI data cannot separate it), intended use reframed as a PCP research guide, and a feature-group analysis added (Section 7.6). Duplicates: rows are not deduplicated on answers (identical answers do not imply the same person); an `ID` uniqueness check is used instead. Wherever the checklist below mentions prediabetes, class balance, or dropping duplicates, read it for the binary target and this ID-based approach.

100 points, due Sunday Oct 18, 2026, 11:59pm (Canvas shows availability until Oct 22). Submit a **fully executed `.ipynb`**, **Workflow Canvas v2**, and **reproducibility instructions**. GitHub link encouraged. Never upload restricted data.

Guiding question: *What evidence shows the data and workflow can support credible learning and evaluation?* A strong score is not required; a reproducible, defensible evaluation is. Grading favors evidence, correct information boundaries, reproducibility, and thoughtful revision over model complexity.

## Notebook outline

0. Setup, versions, seed, data access
1. Project continuity + feedback matrix (10)
2. Provenance and structural audit (10)
3. Data quality and missingness audit (12)
4. EDA (8)
5. Leakage and feature availability (10)
6. Preprocessing strategy (10)
7. Baseline model (10)
8. Metrics (10)
9. Validation design (15)
10. Workflow Canvas v2 + reproducibility (5)
11. Final reflection

Work order matters: the split (section 9) must be created **before** any EDA or fitting that informs decisions (sections 4, 6, 7). Create the held-out test set first, lock it away, and do all development on the remainder.

## 1. Project continuity and feedback integration (10)

- [x] Restate: analytical question, stakeholder/decision, observation, target, use time, generalization objective.
- [x] Define the notation for this project: 𝒵 = 𝒳 × 𝒴, Z = (X, Y), z_i = (x_i, y_i), S_n = {z_i}. Define index i, x (21 features), y, and say there is no group/time/location index (or justify if one exists). Distinguish random Z, realized z, finite sample S_n.
- [x] Feedback response matrix (table below) with a status of Resolved / Partially resolved / Open. For each open item, say what remains uncertain and the concrete next step. Acknowledgment alone does not count; defending an original choice with evidence is allowed.

### Feedback matrix (seed)

| # | Instructor feedback | Planned action | Evidence in notebook | Status |
|---|---|---|---|---|
| 1 | Clarify: diagnosed diabetes vs. undiagnosed vs. future risk | State the target as *reported diagnosis at survey time*; say what the data cannot support | Sec. 1, 2 | |
| 2 | Justify (or change) combining prediabetes and diabetes | Check the true contents of `Diabetes_binary = 1` against the 3-class file; decide binary vs. multiclass and justify | Sec. 2, 3 | |
| 3 | Formalize with z_i = (x_i, y_i) | Notation block | Sec. 1 | |
| 4 | Exact feature definitions, coding, availability at screening time; HighBP/HighChol reflect prior evaluation or healthcare access | Dictionary table plus availability/leakage table | Sec. 2, 5 | |
| 5 | Investigate variable definitions and coding | Verify codes and ranges against the data (value counts) | Sec. 2, 3 | |
| 6 | Examine how risks affect interpretation; no causal claims | Association-vs-causation statements beside each EDA finding | Sec. 4 | |
| 7 | Random split does not show contemporary generalization (2014/2015 data) | Define "unseen" precisely; list what is not demonstrated | Sec. 9 | |
| 8 | BRFSS complex sampling; are survey weights needed? | Check for weight/strata columns (none expected); state the consequence for population-level claims | Sec. 2, 9 | |
| 9 | Define which preprocessing is learned from training data; prevent leakage | Pipeline/ColumnTransformer, fit on train folds only | Sec. 6, 9 | |
| 10 | Expand Canvas: baseline, candidate methods, preprocessing, metrics | Canvas v2 | Sec. 10 | |
| 11 | Error costs: a missed case differs from unnecessary follow-up; metrics must reflect it | Recall/precision/PR-AUC, threshold discussion | Sec. 8 | |

## 2. Data provenance and structural audit (10)

- [x] Source, collection period (resolve 2014 vs. 2015), measurement and labeling process (self-report, telephone survey), actual unit of observation, n, variable types, target, identifiers (`ID`), group/time/spatial structure.
- [x] Executed dataset inventory: shape, dtypes, memory, unique values per column.
- [x] Data dictionary table: variable, meaning, type, source, measurement/availability time, coding notes (starting point is in [README.md](README.md)).
- [x] Confirm the raw data support the target and observation from section 1: what does class 1 contain, is a row one respondent, any repeated respondents, any weight/strata/PSU columns.
- [x] State that no merge is performed (single table), so no keys or join checks apply.

## 3. Data quality and missingness audit (12)

- [x] Missingness per feature, m̂_j = (1/n) Σ 1{x_ij missing}; table plus a visual. Also check for sentinel codes (e.g. 7/9/77/99 "don't know/refused") hiding as valid values.
- [x] Exact duplicates (count and share), and whether duplicates carry conflicting labels.
- [x] Invalid values and implausible ranges: binary columns outside {0,1}; ordinals outside their range (Age 1-13, GenHlth 1-5, Education 1-6, Income 1-8); `MentHlth`/`PhysHlth` outside 0-30; BMI extremes.
- [x] Outliers (BMI especially).
- [x] Label availability and class balance (counts and prevalence); consequence for the metrics.
- [x] Interpret each important issue and justify any exclusion or cleaning. Descriptive audit only here; any fitted imputer goes inside the training fold.
- [x] Proposed treatments and limitations.

## 4. Exploratory data analysis (8)

- [x] At least **three** analyses or visualizations, each with analytical question, finding, implication, and a statement separating association from causation.
- [x] Candidates: class distribution; target rate by Age, BMI, GenHlth, Income, Education; correlation or association of binary features with the target (HighBP, HighChol, DiffWalk); feature-feature correlation.
- [x] **Use only the development/training portion** for any EDA that drives decisions. The held-out test set stays untouched.

## 5. Leakage and feature availability assessment (10)

- [x] Table with one row per feature or feature group: availability at use time (before/after/uncertain), leakage or dependency risk, decision (retain/exclude/investigate) with evidence.
- [x] Specific items: `HighBP`, `HighChol`, `CholCheck`, `AnyHealthcare`, `NoDocbcCost` (prior medical contact or access); `GenHlth`, `PhysHlth`, `DiffWalk`, `HeartDiseaseorAttack`, `Stroke` (could be downstream of diabetes rather than predictors); `ID` (drop); `Income`/`Education`/`Sex` (sensitive).
- [x] Define the information boundary for a non-temporal task: what a screener would know at intake versus what is a consequence of the diagnosis.
- [x] Duplicates and repeated entities across splits; fitted transformations across the split.
- [x] List what still needs verification.

## 6. Preprocessing and feature preparation (10)

- [x] Separate **deterministic** operations (drop `ID`, type casting, any code cleaning) from **fitted** operations (scaling BMI/MentHlth/PhysHlth, imputation if used, one-hot encoding of ordinals if chosen).
- [x] For each fitted step: the state learned (means, scales, categories, medians) and the permitted fitting set.
- [x] Write the stage formally: η̂ = Fit_T(S_tr), x′ = T_η̂(x). Show standardization with training-fold μ̂_j, σ̂_j and state the zero-variance rule.
- [x] Implement with scikit-learn `Pipeline` + `ColumnTransformer`. Every fold: `fit` on training partition, `transform` only on validation. Never refit on validation/test.
- [x] Decide treatment of ordinal columns (numeric vs. categorical) and justify.

## 7. Baseline model and initial results (10)

- [x] A trivial reference (`DummyClassifier`) plus one simple model (logistic regression, matching the project theme).
- [x] Formal statement: θ̂ = Fit_f({(T_η̂(x_i), y_i) : i ∈ I_tr}), M̂(x) = f_θ̂(T_η̂(x)). Distinguish preprocessing state η̂ from model parameters θ̂.
- [x] State assumptions (independent rows, linearity in log-odds, no weights), handling of class imbalance (e.g. `class_weight`, to be compared, not tuned heavily), and output type (probabilities versus labels).
- [x] Executed code, cross-validated validation results, short interpretation. No extensive tuning.
- [x] Not applicable: a three-class formulation is not possible from UCI data (see README).

## 8. Primary evaluation metrics (10)

- [x] Define and justify primary metric(s) against the decision: recall of the positive class (a missed case is costly), precision (follow-up workload), PR-AUC (imbalance, threshold-free ranking). Add log loss / Brier / calibration if reporting probabilities.
- [x] Formulas: Precision = TP/(TP+FP), Recall = TP/(TP+FN).
- [x] State whether each metric evaluates labels, rankings, or calibrated probabilities.
- [x] Explain error consequences (false negative versus false positive), the effect of imbalance (why accuracy misleads), and limitations.
- [x] Compute the chosen metrics for the dummy and the logistic baseline. Show the confusion matrix at a stated threshold.

## 9. Validation design and generalization (15)

- [x] State exactly what "unseen" means (new respondents from the same survey wave) and what is **not** demonstrated (contemporary clinical populations, other years, undiagnosed or future cases, population-weighted estimates).
- [x] Partition: stratified random, since rows are treated as independent respondents and there is no group/time index. Justify, and address duplicates (dedupe before splitting or show they don't cross partitions).
- [x] 𝒟 = 𝒟_train ∪ 𝒟_val ∪ 𝒟_test with pairwise-disjoint parts. Reproducible procedure with fixed seed; report sizes and class prevalence in each partition.
- [x] Hold out a **final test set now and do not touch it**. Use stratified K-fold CV on the development portion for baseline evaluation.
- [x] In each fold k: learn (η̂_k, θ̂_k) from training rows only, evaluate on the fold's validation rows, carry no state across folds. Report mean and spread across folds.
- [x] Discuss BRFSS complex sampling and whether weights are needed (they are absent from this file; say what that means for the claim).

## 10. Workflow Canvas v2 and reproducibility (5)

- [x] Canvas chain: decision/problem → observation → measurement and features → preparation → baseline learning → model output → validation → metric(s) → intended use.
- [x] Mark fixed vs. learned operations, their stored state and fitting sample, the final baseline output, and the claim the validation tests. Mark risks, evidence from this audit, and outstanding decisions. Express as M = M_K ∘ ... ∘ M_1.
- [x] Notebook is executed top to bottom, with Markdown explanation, readable tables and figures.
- [x] Dependency and version info (Python, pandas, numpy, scikit-learn, matplotlib, ucimlrepo), data access instructions, rerun steps, `random_state`.
- [x] Feedback matrix and Canvas inside the notebook or as an identified supplement.

## Final reflection

- [x] What changed after Milestone 1 feedback.
- [x] What the data audit revealed.
- [x] Which assumptions were challenged.
- [x] What the baseline and validation do and do not show.
- [x] The next evidence needed.

## Out of scope for this milestone

Hyperparameter search, many algorithms, production deployment, and uniformly high performance. Pipeline optimization belongs to the next milestone.

## Submission checklist

- [x] `milestone2.ipynb` executed, outputs saved, restarts cleanly with Run All
- [x] Workflow Canvas v2 included
- [x] Reproducibility instructions included
- [x] Test set untouched
- [x] No restricted data uploaded
