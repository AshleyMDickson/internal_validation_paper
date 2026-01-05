# Calibration Slope Investigation

## Problem Discovery

During analysis of the internal validation simulation results, we discovered a **negative correlation** between internal validation calibration slope estimates and external validation calibration slope estimates. This was unexpected and concerning because:

1. Other metrics (AUC, MAPE) showed positive correlations as expected
2. The negative correlation was strongest for bootstrap (-0.806), followed by CV (-0.713), and weakest for sample splitting (-0.169)
3. A negative correlation implies that higher internal calibration slope predicts lower external performance, which seems fundamentally wrong

## Initial Hypothesis (INCORRECT)

Initially suspected the problem was comparing models trained on different sample sizes:
- External validation: model trained on 100% of dev_data (n=858)
- Sample split: model trained on 70% of dev_data (~600 obs)
- Cross-validation: models trained on 90% of dev_data (~772 obs)

However, this hypothesis was **rejected** because:
- Bootstrap also uses 100% of data but showed the STRONGEST negative correlation
- If sample size was the issue, bootstrap should have shown the best (most positive) correlation

## Observed Pattern

Across all methods, when internal calibration slope is high, external calibration slope tends to be low, and vice versa:

### Bootstrap Examples (from main simulation):
**High internal cal slope:**
- Internal: 0.92-0.93 → External: 0.64-0.76

**Low internal cal slope:**
- Internal: 0.80-0.82 → External: 1.01-1.35

Note: External calibration slopes >1.0 indicate the model appears *underfit* on that external sample.

## Diagnostic Simulations

Created minimal reproduction (`debug_calibration_slope.R`):
- 100 simulations with n_dev=858, n_ext=100,000
- Clean implementation of CV
- **Result: Reproduced the negative correlation (r = -0.675)**

### Critical Test (`test_hypothesis.R`)

Tested whether the issue is about different models vs. different data:

**Key Findings:**
1. **Same model, dev vs ext data: r = -0.06** (essentially zero)
2. **CV vs External: r = -0.699** (strongly negative - the problem!)
3. **CV vs Full model on same dev data: r = 0.033** (essentially zero)

**Calibration slope statistics:**
- Full model on dev_data: Mean = 1.000, SD = 0.000 (always exactly 1.0)
- Full model on ext_data: Mean = 0.900, SD = 0.118 (varies)
- CV on dev_data: Mean = 0.804, SD = 0.050 (varies, but less than external)

**Key observation:** The full model evaluated on its own training data (apparent calibration) is always exactly 1.0 with zero variation.

## Current Theory

### Why Calibration Slope Behaves Differently

Calibration slope is calculated by **fitting a second logistic regression model** on the evaluation dataset:

```r
logit_pred <- qlogis(predictions)
cal_model <- glm(y_true ~ logit_pred, family = binomial)
cal_slope <- coef(cal_model)[2]
```

This is fundamentally different from other metrics:
- **AUC**: Ranks predictions vs outcomes (no model fitting)
- **Brier Score**: Mean squared error of predictions
- **MAPE**: Mean absolute error of predictions

### The Proposed Mechanism

Calibration slope measures **the interaction between predictions and the specific random realization of outcomes** in the evaluation sample.

When `dev_data` happens to be an "easy" sample (random outcomes align well with predictors):
- Models trained on it (CV folds) generalize well within dev_data
- CV shows relatively good calibration (higher slope, e.g., 0.84)
- But `ext_data` is an independent sample - could be a "harder" sample
- External calibration appears worse (lower slope, e.g., 0.76)

When `dev_data` is a "hard" sample (random outcomes don't align as well):
- CV shows relatively poor calibration (lower slope, e.g., 0.70)
- But `ext_data` could be an "easier" sample
- External calibration appears better (higher slope, e.g., 1.04)

### Why Bootstrap Shows the Strongest Correlation

Bootstrap has the lowest variance in internal estimates (most precise), so it shows the clearest signal of this underlying pattern. Sample splitting has the highest variance (lots of noise), which dilutes/obscures the correlation.

This is actually diagnostic: **increasingly strong spurious correlations with increasingly precise methods** suggests a fundamental design issue rather than random noise.

## Implications

1. **Calibration slope may not be directly comparable** across independent samples from the same DGP when using internal vs external validation
2. The metric appears to capture something about the **specific dataset's characteristics** in addition to model performance
3. This could explain why calibration slopes >1.0 appear on external validation (model appears underfit) even though the same model shows slope ~0.80-0.90 on internal validation

## Questions Remaining

1. Is this a known property of calibration slope in the literature?
2. Is there a different/better way to calculate calibration slope that avoids this issue?
3. Should we be using a different calibration metric entirely?
4. Does this invalidate our comparison of internal validation methods using calibration slope?
5. Is the negative correlation a problem with our simulation design, or a real property of how calibration slope behaves?

## Files Created

- `debug_calibration_slope.R`: Minimal reproduction of the negative correlation
- `test_hypothesis.R`: Tests whether issue is models vs data
- `debug_calibration_results.csv`: Results from minimal simulation
- `hypothesis_test_results.csv`: Results from hypothesis test
- `debug_calibration_scatter.png`: Visualization of negative correlation

## Next Steps

- [ ] Review calibration slope literature to understand if this is expected behavior
- [ ] Consider alternative calibration metrics (e.g., calibration intercept, E/O ratio)
- [ ] Investigate whether the "correct" implementation should use a different approach
- [ ] Decide whether to keep calibration slope in the analysis or remove it
- [ ] If keeping it, document this limitation clearly in the paper
