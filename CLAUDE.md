# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Quarto-based research project that simulates and compares internal validation methods (sample splitting, cross-validation, and bootstrap) for clinical risk prediction models. The project generates a static website with interactive results and visualizations.

## Building and Rendering

### Render the entire website
```bash
quarto render
```

This generates the static site in the `docs/` directory, which includes:
- `index.html` - Main simulation results and analysis
- `sample_splitting.html` - Sample splitting cost analysis
- All associated plots and assets

### Render a single document
```bash
quarto render index.qmd
quarto render sample_splitting.qmd
```

### Preview the website locally
```bash
quarto preview
```

## Running Simulations

The simulation code is in `simulation_code.R`. This script:
1. Calculates sample sizes using the `samplesizedev` package
2. Runs Monte Carlo simulations (default: 500 iterations) comparing validation methods
3. Generates performance metrics (AUC, calibration slope, Brier score, MAPE)
4. Creates visualization plots
5. Outputs `validation_results.csv` and `setup_parameters.csv`

To run the simulation:
```bash
Rscript simulation_code.R
```

Note: This is computationally intensive and uses parallel processing (`parallel::mclapply`). It will use all available CPU cores minus one.

## Code Architecture

### Data Generation Process
The simulation uses a realistic clinical prediction scenario:
- Binary outcome with 15% prevalence
- 10 continuous predictors (X₁...X₁₀) drawn from N(0,1)
- Logistic regression with mixed effect sizes
- Intercept (α) calculated to achieve exactly 15% prevalence using numerical root-finding

Key functions:
- `generate_data(n, prevalence)` - Generates synthetic datasets with specified characteristics
- `calculate_performance_metrics(y_true, y_pred)` - Computes AUC, calibration slope, Brier score, and MAPE

### Validation Methods Implementation
Located in `simulation_code.R`:
- `sample_split_validation(data, split_ratio)` - 70/30 split validation
- `cross_validation(data, k)` - k-fold CV with stratified folds and pooled predictions
- `bootstrap_validation(data, B)` - Harrell's optimism correction method

### Simulation Workflow
1. `run_single_simulation(sim, ...)` - Executes one complete simulation iteration:
   - Generates development data
   - Fits model and gets apparent performance
   - Runs all three internal validation methods
   - Generates external validation dataset (n=100,000)
   - Returns all performance metrics
2. `run_simulation(...)` - Orchestrates parallel execution of multiple simulations

### Analysis and Visualization
The `index.qmd` file:
- Loads `validation_results.csv` and `setup_parameters.csv`
- Calculates summary statistics (mean, SD) for each method
- Computes bias and RMSE relative to external validation
- Generates tables and includes pre-generated plots
- Uses inline R code extensively to embed dynamic values in prose

## Important Files

- `simulation_code.R` - Core simulation logic and plotting
- `index.qmd` - Main analysis document with results
- `sample_splitting.qmd` - Analysis of sample splitting costs
- `_quarto.yml` - Quarto configuration (output directory, theme, navigation)
- `validation_results.csv` - Simulation output (generated)
- `setup_parameters.csv` - Simulation parameters (generated)
- `*.png` files - Pre-generated plots included in the rendered documents

## R Dependencies

Required packages:
- `pROC` - ROC curve analysis and AUC calculation
- `rms` - Regression modeling strategies
- `ggplot2` - Visualization
- `parallel` - Parallel computation
- `samplesizedev` - Sample size calculation for prediction models
- `tidyr`, `dplyr` - Data manipulation
- `knitr`, `gt` - Table generation in Quarto documents

## Key Design Decisions

1. **External validation uses n=100,000**: Large sample provides stable gold standard estimates with minimal sampling variability

2. **Bootstrap uses 200 resamples**: Follows Harrell's recommendations for reliable optimism correction

3. **10-fold CV**: Standard choice balancing bias-variance tradeoff

4. **Parallel processing**: Simulations are embarrassingly parallel; code automatically detects available cores

5. **Calibration slope calculation**: Uses logit transformation of predictions to fit calibration model (`glm(y_true ~ logit_pred)`)

6. **Results stored in long format**: Makes it easier to group and analyze by validation method

## Workflow for Changes

1. Modify simulation parameters in `simulation_code.R` (lines 10-27)
2. Run simulation: `Rscript simulation_code.R`
3. Update analysis if needed in `index.qmd`
4. Render website: `quarto render`
5. Commit generated CSV files but plots are tracked in git

## Git and Publishing

The `docs/` directory contains the rendered website and is configured for GitHub Pages deployment. After rendering, commit both source files and the `docs/` directory to publish changes.
