library(pROC)      # For AUC calculation
library(rms)       # For validated clinical prediction models (Frank Harrell)
library(ggplot2)   # For visualization
library(parallel)  # For parallel processing
library(samplesizedev)

set.seed(123)

# Calculate required development sample size

sample_size_params <- list(
  p = 10,
  phi = 0.15,
  c = 0.75,
  S = 0.9
)

res <- samplesizedev(
  outcome = "Binary",
  S = sample_size_params$S,
  phi = sample_size_params$phi,
  c = sample_size_params$c,
  p = sample_size_params$p
)

N_DEV_RILEY <- ceiling(res$rvs)
N_DEV_PAVLOU <- ceiling(res$sim)

N_DEV <- N_DEV_PAVLOU
N_EVENTS <- ceiling(N_DEV * sample_size_params$phi)

# HELPER FUNCTIONS FOR PERFORMANCE

calculate_performance_metrics <- function(y_true, y_pred) {

  auc_val <- as.numeric(auc(y_true, y_pred, quiet = TRUE))
  
  # Calibration slope (logistic regression of outcomes on logit predictions)
  logit_pred <- qlogis(pmax(pmin(y_pred, 0.9999), 0.0001))  # Avoid infinity
  cal_model <- glm(y_true ~ logit_pred, family = binomial)
  cal_slope <- coef(cal_model)[2]
  
  # Brier score
  brier <- mean((y_pred - y_true)^2)
  
  # MAPE (Mean Absolute Prediction Error)
  mape <- mean(abs(y_pred - y_true))
  
  return(list(
    auc = auc_val,
    cal_slope = as.numeric(cal_slope),
    brier = brier,
    mape = mape
  ))
}

# 1. DATA GENERATING PROCESS

expit <- function(x) 1 / (1 + exp(-x))

# Generate a large sample to estimate the intercept
n_sample <- 10000
X_sample <- replicate(10, rnorm(n_sample))
colnames(X_sample) <- paste0("X", 1:10)
beta <- c(0.45,   # X1: Moderate effect
          0.40,   # X2: Moderate effect
         -0.35,   # X3: Moderate negative effect
          0.30,   # X4: Small-moderate effect
         -0.25,   # X5: Small-moderate negative effect
          0.20,   # X6: Small effect
          0.15,   # X7: Small effect
          0.10,   # X8: Small effect
          0.08,   # X9: Small effect
          0.05)   # X10: Small effect

target_prev <- 0.15
find_alpha <- function(a) {
  mean(expit(a + as.vector(X_sample %*% beta))) - target_prev
}

# Solve for alpha using uniroot
alpha <- uniroot(find_alpha, c(-10, 10))$root
cat(sprintf("Calculated intercept (alpha): %.4f\n", alpha))
cat(sprintf("Target prevalence: %.1f%%\n\n", target_prev * 100))

# Verify the prevalence with the calculated alpha
pi_verify <- expit(alpha + as.vector(X_sample %*% beta))
y_verify <- rbinom(n_sample, 1, pi_verify)
observed_prev_verify <- sum(y_verify) / length(y_verify)

# Main data generation function
generate_data <- function(n, prevalence = 0.15) {
  X1 <- rnorm(n, mean = 0, sd = 1)
  X2 <- rnorm(n, mean = 0, sd = 1)
  X3 <- rnorm(n, mean = 0, sd = 1)
  X4 <- rnorm(n, mean = 0, sd = 1)
  X5 <- rnorm(n, mean = 0, sd = 1)
  X6 <- rnorm(n, mean = 0, sd = 1)
  X7 <- rnorm(n, mean = 0, sd = 1)
  X8 <- rnorm(n, mean = 0, sd = 1)
  X9 <- rnorm(n, mean = 0, sd = 1)
  X10 <- rnorm(n, mean = 0, sd = 1)
  
  # True logistic model
  logit_p <- alpha + 
             beta[1] * X1 + beta[2] * X2 + beta[3] * X3 + beta[4] * X4 + 
             beta[5] * X5 + beta[6] * X6 + beta[7] * X7 + beta[8] * X8 + 
             beta[9] * X9 + beta[10] * X10
  
  prob <- plogis(logit_p)
  
  # Generate outcome
  outcome <- rbinom(n, size = 1, prob = prob)
  
  data.frame(
    outcome = outcome,
    X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5,
    X6 = X6, X7 = X7, X8 = X8, X9 = X9, X10 = X10
  )
}

# 2. VALIDATION METHODS

# Apparent validation (resubstitution)
apparent_validation <- function(data) {
  model <- glm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
               data = data, family = binomial)
  pred <- predict(model, type = "response")
  
  # Calculate all performance metrics
  metrics <- calculate_performance_metrics(data$outcome, pred)
  
  list(
    auc = metrics$auc,
    cal_slope = metrics$cal_slope,
    brier = metrics$brier,
    mape = metrics$mape,
    model = model
  )
}

# Sample splitting validation
sample_split_validation <- function(data, split_ratio = 0.7) {
  n <- nrow(data)
  n_train <- floor(n * split_ratio)
  
  # Random split
  train_idx <- sample(1:n, n_train)
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  
  # Fit model on training data
  model <- glm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
               data = train_data, family = binomial)
  
  # Evaluate on test set
  pred <- predict(model, newdata = test_data, type = "response")
  
  # Calculate all performance metrics
  metrics <- calculate_performance_metrics(test_data$outcome, pred)
  
  return(metrics)
}

# Cross-validation (pooled predictions approach)
cv_validation <- function(data, k = 10) {
  n <- nrow(data)
  
  # Create folds
  fold_ids <- sample(rep(1:k, length.out = n))
  
  # Store predictions
  cv_predictions <- numeric(n)
  
  for (fold in 1:k) {
    # Training and test sets
    train_data <- data[fold_ids != fold, ]
    test_data <- data[fold_ids == fold, ]
    
    # Fit model on training data
    cv_model <- glm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
                    data = train_data, family = binomial)
    
    # Predict on test fold
    cv_predictions[fold_ids == fold] <- predict(cv_model, 
                                                 newdata = test_data, 
                                                 type = "response")
  }
  
  # Calculate all performance metrics on pooled predictions
  metrics <- calculate_performance_metrics(data$outcome, cv_predictions)
  
  return(metrics)
}

# Bootstrap optimism correction (Harrell, rms)
bootstrap_validation <- function(data, B = 200) {
  # Fit model using lrm for proper integration with validate()
  model <- lrm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
               data = data, x = TRUE, y = TRUE)
  
  # Perform bootstrap validation for discrimination and calibration
  val_results <- validate(model, B = B)
  
  # Extract optimism-corrected discrimination (Dxy -> C-statistic)
  dxy_corrected <- val_results["Dxy", "index.corrected"]
  c_corrected <- 0.5 + (dxy_corrected / 2)
  
  # Extract optimism-corrected calibration slope
  slope_corrected <- val_results["Slope", "index.corrected"]
  
  # For Brier score and MAPE, calculate optimism via bootstrap, (not directly available from rms::validate)
  n <- nrow(data)
  brier_optimism <- numeric(B)
  mape_optimism <- numeric(B)
  
  for (b in 1:B) {
    # Bootstrap sample
    boot_idx <- sample(1:n, n, replace = TRUE)
    boot_data <- data[boot_idx, ]
    
    # Fit model on bootstrap sample
    boot_model <- glm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
                      data = boot_data, family = binomial)
    
    # Performance on bootstrap sample
    pred_boot <- predict(boot_model, newdata = boot_data, type = "response")
    brier_boot <- mean((pred_boot - boot_data$outcome)^2)
    mape_boot <- mean(abs(pred_boot - boot_data$outcome))
    
    # Performance on original sample
    pred_orig <- predict(boot_model, newdata = data, type = "response")
    brier_orig <- mean((pred_orig - data$outcome)^2)
    mape_orig <- mean(abs(pred_orig - data$outcome))
    
    # Optimism
    brier_optimism[b] <- brier_boot - brier_orig
    mape_optimism[b] <- mape_boot - mape_orig
  }
  
  # Get apparent Brier and MAPE
  model_glm <- glm(outcome ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10, 
                   data = data, family = binomial)
  pred_apparent <- predict(model_glm, type = "response")
  brier_apparent <- mean((pred_apparent - data$outcome)^2)
  mape_apparent <- mean(abs(pred_apparent - data$outcome))
  
  # Apply optimism correction
  brier_corrected <- brier_apparent - mean(brier_optimism)
  mape_corrected <- mape_apparent - mean(mape_optimism)
  
  return(list(
    auc = as.numeric(c_corrected),
    cal_slope = as.numeric(slope_corrected),
    brier = brier_corrected,
    mape = mape_corrected
  ))
}

# External validation
external_validation <- function(model, external_data) {
  pred <- predict(model, newdata = external_data, type = "response")
  
  # Calculate all performance metrics
  metrics <- calculate_performance_metrics(external_data$outcome, pred)
  
  return(metrics)
}

# 3. SIMULATION STUDY (WITH PARALLEL PROCESSING)

# Single simulation iteration (to be parallelized)
run_single_simulation <- function(sim, n_dev, n_ext, B_boot, k_cv, seed_base) {
  set.seed(seed_base + sim)
  dev_data <- generate_data(n_dev)

  apparent_result <- apparent_validation(dev_data)
  dev_model <- apparent_result$model
  
  # external
  ext_data <- generate_data(n_ext)
  ext_metrics <- external_validation(dev_model, ext_data)
  
  # Apply internal validation 
  split_metrics <- sample_split_validation(dev_data)
  cv_metrics <- cv_validation(dev_data, k = k_cv)
  bootstrap_metrics <- bootstrap_validation(dev_data, B = B_boot)
  
  # Store results
  sim_results <- data.frame()
  
  for (method in c("Sample Split", "Cross-validation", "Bootstrap")) {
    
    # Select appropriate internal metrics
    if (method == "Sample Split") {
      int_metrics <- split_metrics
    } else if (method == "Cross-validation") {
      int_metrics <- cv_metrics
    } else {  # Bootstrap
      int_metrics <- bootstrap_metrics
    }
    
    sim_results <- rbind(sim_results, data.frame(
      simulation = sim,
      method = method,
      # Apparent metrics
      apparent_auc = apparent_result$auc,
      apparent_cal_slope = apparent_result$cal_slope,
      apparent_brier = apparent_result$brier,
      apparent_mape = apparent_result$mape,
      # Internal validated metrics
      internal_auc = int_metrics$auc,
      internal_cal_slope = int_metrics$cal_slope,
      internal_brier = int_metrics$brier,
      internal_mape = int_metrics$mape,
      # External metrics
      external_auc = ext_metrics$auc,
      external_cal_slope = ext_metrics$cal_slope,
      external_brier = ext_metrics$brier,
      external_mape = ext_metrics$mape
    ))
  }
  
  return(sim_results)
}

run_simulation <- function(n_dev = N_DEV, n_ext = 100000, n_sim = 200, 
                          B_boot = 200, k_cv = 10, n_cores = NULL) {
  
  cat("Running simulation with parallel processing...\n")
  
  # Determine number of cores to use
  if (is.null(n_cores)) {
    n_cores <- max(1, detectCores() - 1)  # Leave one core free
  }
  
  cat(sprintf("Using %d cores for parallel processing\n", n_cores))
  cat(sprintf("Parameters: n_dev=%d, n_ext=%d, n_sim=%d, B_boot=%d, k_cv=%d\n\n", 
              n_dev, n_ext, n_sim, B_boot, k_cv))
  
  # Run simulations in parallel
  seed_base <- 12300
  
  if (.Platform$OS.type == "windows") {
    # Windows: use parLapply with cluster
    cat("Detected Windows - using cluster parallelization\n")
    cl <- makeCluster(n_cores)
    clusterExport(cl, c("generate_data", "apparent_validation", "external_validation",
                       "sample_split_validation", "cv_validation", "bootstrap_validation",
                       "calculate_performance_metrics", "n_dev", "n_ext", "B_boot", 
                       "k_cv", "seed_base"))
    clusterEvalQ(cl, {
      library(pROC)
      library(rms)
    })
    
    # Run with progress tracking
    cat("Starting parallel simulations...\n")
    results_list <- parLapply(cl, 1:n_sim, function(sim) {
      if (sim %% 20 == 0) cat(sprintf("  Progress: %d/%d simulations\n", sim, n_sim))
      run_single_simulation(sim, n_dev, n_ext, B_boot, k_cv, seed_base)
    })
    
    stopCluster(cl)
  } else {
    # Unix/Linux/Mac: use mclapply (fork-based, more efficient)
    cat("Detected Unix/Linux/Mac - using fork-based parallelization\n")
    cat("Starting parallel simulations...\n")
    
    # Track progress manually (mclapply runs all at once)
    start_time <- Sys.time()
    results_list <- mclapply(1:n_sim, function(sim) {
      run_single_simulation(sim, n_dev, n_ext, B_boot, k_cv, seed_base)
    }, mc.cores = n_cores, mc.set.seed = TRUE)
    end_time <- Sys.time()
    
    cat(sprintf("Completed %d simulations in %.1f seconds\n", 
                n_sim, as.numeric(difftime(end_time, start_time, units = "secs"))))
  }
  
  # Combine results
  results <- do.call(rbind, results_list)
  
  cat("\nSimulation complete!\n")
  
  return(results)
}

# 4. RUN SIMULATION

results <- run_simulation(n_dev = N_DEV, n_ext = 100000, n_sim = 200, 
                         B_boot = 200, k_cv = 10)

# Calculate bias for all metrics
results$bias_auc <- results$internal_auc - results$external_auc
results$bias_cal_slope <- results$internal_cal_slope - results$external_cal_slope
results$bias_brier <- results$internal_brier - results$external_brier
results$bias_mape <- results$internal_mape - results$external_mape

# Summary statistics
methods_list <- unique(results$method)

cat("AUC (C-statistic) - Higher is Better:\n")
for (m in methods_list) {
  method_data <- results[results$method == m, ]
  cat(sprintf("  %s:\n", m))
  cat(sprintf("    Apparent: %.3f (SD %.3f)\n", 
              mean(method_data$apparent_auc), sd(method_data$apparent_auc)))
  cat(sprintf("    Internal: %.3f (SD %.3f)\n", 
              mean(method_data$internal_auc), sd(method_data$internal_auc)))
  cat(sprintf("    External: %.3f (SD %.3f)\n", 
              mean(method_data$external_auc), sd(method_data$external_auc)))
  cat(sprintf("    Bias: %.4f (RMSE %.4f)\n\n", 
              mean(method_data$bias_auc), sqrt(mean(method_data$bias_auc^2))))
}

cat("\nCalibration Slope - Closer to 1.0 is Better:\n")
for (m in methods_list) {
  method_data <- results[results$method == m, ]
  cat(sprintf("  %s:\n", m))
  cat(sprintf("    Apparent: %.3f (SD %.3f)\n", 
              mean(method_data$apparent_cal_slope), sd(method_data$apparent_cal_slope)))
  cat(sprintf("    Internal: %.3f (SD %.3f)\n", 
              mean(method_data$internal_cal_slope), sd(method_data$internal_cal_slope)))
  cat(sprintf("    External: %.3f (SD %.3f)\n", 
              mean(method_data$external_cal_slope), sd(method_data$external_cal_slope)))
  cat(sprintf("    Bias: %.4f (RMSE %.4f)\n\n", 
              mean(method_data$bias_cal_slope), sqrt(mean(method_data$bias_cal_slope^2))))
}

cat("\nBrier Score - Lower is Better:\n")
for (m in methods_list) {
  method_data <- results[results$method == m, ]
  cat(sprintf("  %s:\n", m))
  cat(sprintf("    Apparent: %.4f (SD %.4f)\n", 
              mean(method_data$apparent_brier), sd(method_data$apparent_brier)))
  cat(sprintf("    Internal: %.4f (SD %.4f)\n", 
              mean(method_data$internal_brier), sd(method_data$internal_brier)))
  cat(sprintf("    External: %.4f (SD %.4f)\n", 
              mean(method_data$external_brier), sd(method_data$external_brier)))
  cat(sprintf("    Bias: %.5f (RMSE %.5f)\n\n", 
              mean(method_data$bias_brier), sqrt(mean(method_data$bias_brier^2))))
}

cat("\nMAPE (Mean Absolute Prediction Error) - Lower is Better:\n")
for (m in methods_list) {
  method_data <- results[results$method == m, ]
  cat(sprintf("  %s:\n", m))
  cat(sprintf("    Apparent: %.4f (SD %.4f)\n", 
              mean(method_data$apparent_mape), sd(method_data$apparent_mape)))
  cat(sprintf("    Internal: %.4f (SD %.4f)\n", 
              mean(method_data$internal_mape), sd(method_data$internal_mape)))
  cat(sprintf("    External: %.4f (SD %.4f)\n", 
              mean(method_data$external_mape), sd(method_data$external_mape)))
  cat(sprintf("    Bias: %.5f (RMSE %.5f)\n\n", 
              mean(method_data$bias_mape), sqrt(mean(method_data$bias_mape^2))))
}

# 5. VISUALIZATION

# 5 boxplots per metric:
#   1. Apparent (baseline) - performance on development data
#   2. Sample Split (internal)
#   3. Cross-validation (internal)
#   4. Bootstrap (internal)
#   5. External (gold standard) - performance on large external data
#
# NOTE: Apparent and External values are the SAME across all three method rows
# for each simulation (since they come from the same dev and external data).
# We arbitrarily extract them from the "Sample Split" rows, but could use any method.
# Only the "internal" values differ between methods.

# Define estimate types and colors
estimate_colors <- c(
  "Apparent" = "#95A5A6",           # Gray (overfitted baseline)
  "Sample Split" = "#E74C3C",       # Red
  "Cross-validation" = "#3498DB",   # Blue
  "Bootstrap" = "#2ECC71",          # Green
  "External" = "#F39C12"            # Orange (gold standard)
)

estimate_levels <- c("Apparent", "Sample Split", "Cross-validation", 
                    "Bootstrap", "External")

# --- Plot 1: AUC Comparison ---
auc_plot_data <- data.frame()
apparent_data <- results[results$method == "Sample Split", ]
auc_plot_data <- rbind(auc_plot_data, data.frame(
  estimate_type = "Apparent",
  value = apparent_data$apparent_auc
))

# Internal validation methods
for (method_name in c("Sample Split", "Cross-validation", "Bootstrap")) {
  method_data <- results[results$method == method_name, ]
  auc_plot_data <- rbind(auc_plot_data, data.frame(
    estimate_type = method_name,
    value = method_data$internal_auc
  ))
}

external_data <- results[results$method == "Sample Split", ]
auc_plot_data <- rbind(auc_plot_data, data.frame(
  estimate_type = "External",
  value = external_data$external_auc
))

# Convert to factor with proper order
auc_plot_data$estimate_type <- factor(auc_plot_data$estimate_type, 
                                      levels = estimate_levels)

p_auc <- ggplot(auc_plot_data, aes(x = estimate_type, y = value, fill = estimate_type)) +
  geom_boxplot(alpha = 0.75, width = 0.7) +
  scale_fill_manual(values = estimate_colors) +
  labs(title = "AUC (C-statistic): Performance Across Validation Approaches",
       x = "",
       y = "AUC (C-statistic)") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = 0.3, color = "gray85"),
        panel.grid.minor = element_blank())

ggsave("comparison_auc.png", p_auc, 
       width = 10, height = 6, dpi = 300)

# --- Plot 2: Calibration Slope Comparison ---
cal_plot_data <- data.frame()

apparent_data <- results[results$method == "Sample Split", ]
cal_plot_data <- rbind(cal_plot_data, data.frame(
  estimate_type = "Apparent",
  value = apparent_data$apparent_cal_slope
))

# Internal validation methods
for (method_name in c("Sample Split", "Cross-validation", "Bootstrap")) {
  method_data <- results[results$method == method_name, ]
  cal_plot_data <- rbind(cal_plot_data, data.frame(
    estimate_type = method_name,
    value = method_data$internal_cal_slope
  ))
}

# External
external_data <- results[results$method == "Sample Split", ]
cal_plot_data <- rbind(cal_plot_data, data.frame(
  estimate_type = "External",
  value = external_data$external_cal_slope
))

cal_plot_data$estimate_type <- factor(cal_plot_data$estimate_type, 
                                      levels = estimate_levels)

p_cal <- ggplot(cal_plot_data, aes(x = estimate_type, y = value, fill = estimate_type)) +
  geom_boxplot(alpha = 0.75, width = 0.7) +
  geom_hline(yintercept = 1.0, linetype = "dashed", color = "black", 
             linewidth = 0.6, alpha = 0.7) +
  scale_fill_manual(values = estimate_colors) +
  labs(title = "Calibration Slope: Performance Across Validation Approaches",
       x = "",
       y = "Calibration Slope") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = 0.3, color = "gray85"),
        panel.grid.minor = element_blank())

ggsave("comparison_calibration.png", p_cal, 
       width = 10, height = 6, dpi = 300)

# --- Plot 3: Brier Score Comparison ---
brier_plot_data <- data.frame()

# Apparent
apparent_data <- results[results$method == "Sample Split", ]
brier_plot_data <- rbind(brier_plot_data, data.frame(
  estimate_type = "Apparent",
  value = apparent_data$apparent_brier
))

# Internal validation methods
for (method_name in c("Sample Split", "Cross-validation", "Bootstrap")) {
  method_data <- results[results$method == method_name, ]
  brier_plot_data <- rbind(brier_plot_data, data.frame(
    estimate_type = method_name,
    value = method_data$internal_brier
  ))
}

# External
external_data <- results[results$method == "Sample Split", ]
brier_plot_data <- rbind(brier_plot_data, data.frame(
  estimate_type = "External",
  value = external_data$external_brier
))

brier_plot_data$estimate_type <- factor(brier_plot_data$estimate_type, 
                                        levels = estimate_levels)

p_brier <- ggplot(brier_plot_data, aes(x = estimate_type, y = value, fill = estimate_type)) +
  geom_boxplot(alpha = 0.75, width = 0.7) +
  scale_fill_manual(values = estimate_colors) +
  labs(title = "Brier Score: Performance Across Validation Approaches",
       x = "",
       y = "Brier Score") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = 0.3, color = "gray85"),
        panel.grid.minor = element_blank())

ggsave("comparison_brier.png", p_brier, 
       width = 10, height = 6, dpi = 300)

# --- Plot 4: MAPE Comparison ---
mape_plot_data <- data.frame()

# Apparent
apparent_data <- results[results$method == "Sample Split", ]
mape_plot_data <- rbind(mape_plot_data, data.frame(
  estimate_type = "Apparent",
  value = apparent_data$apparent_mape
))

# Internal validation methods
for (method_name in c("Sample Split", "Cross-validation", "Bootstrap")) {
  method_data <- results[results$method == method_name, ]
  mape_plot_data <- rbind(mape_plot_data, data.frame(
    estimate_type = method_name,
    value = method_data$internal_mape
  ))
}

# External
external_data <- results[results$method == "Sample Split", ]
mape_plot_data <- rbind(mape_plot_data, data.frame(
  estimate_type = "External",
  value = external_data$external_mape
))

mape_plot_data$estimate_type <- factor(mape_plot_data$estimate_type, 
                                       levels = estimate_levels)

p_mape <- ggplot(mape_plot_data, aes(x = estimate_type, y = value, fill = estimate_type)) +
  geom_boxplot(alpha = 0.75, width = 0.7) +
  scale_fill_manual(values = estimate_colors) +
  labs(title = "MAPE (Mean Absolute Prediction Error): Performance Across Validation Approaches",
       x = "",
       y = "Mean Absolute Prediction Error") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.major.y = element_line(linewidth = 0.3, color = "gray85"),
        panel.grid.minor = element_blank())

ggsave("comparison_mape.png", p_mape, 
       width = 10, height = 6, dpi = 300)

cat("All condensed plots saved!\n")

# Save results
write.csv(results, "validation_results.csv", row.names = FALSE)
