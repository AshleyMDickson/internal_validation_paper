# internal_validation_paper

This repository contains code and analysis for a simulation study comparing internal validation methods for clinical prediction models.

## Overview

The project evaluates three common internal validation approaches:
- Sample splitting (70/30 split)
- 10-fold cross-validation
- Bootstrap validation (200 resamples)

Simulations are based on a realistic clinical prediction scenario with 15% outcome prevalence and 10 continuous predictors. Performance metrics (AUC, calibration slope, Brier score, MAPE) are compared against external validation in a large independent sample (n=100,000).

## Requirements

- R (version 4.0 or higher recommended)
- Quarto
- R packages: `pROC`, `rms`, `ggplot2`, `parallel`, `samplesizedev`, `tidyr`, `dplyr`, `knitr`, `gt`

## Usage

Run the simulation:
```bash
Rscript simulation_code.R
```

Render the analysis website:
```bash
quarto render
```

Preview locally:
```bash
quarto preview
```

## Output

The rendered website is generated in the `docs/` directory and includes:
- Main simulation results and analysis (`index.html`)
- Sample splitting cost analysis (`sample_splitting.html`)
- Performance comparison plots and summary tables

## Repository Structure

- `simulation_code.R` - Core simulation logic and data generation
- `index.qmd` - Main analysis document
- `sample_splitting.qmd` - Sample splitting analysis
- `_quarto.yml` - Quarto configuration
- `validation_results.csv` - Simulation output (generated)
- `setup_parameters.csv` - Simulation parameters (generated)
