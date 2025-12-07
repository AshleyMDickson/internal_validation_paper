library(pROC)
library(rms)
library(ggplot2)
library(parallel)
library(tidyr)
library(dplyr)

set.seed(123)

# Load the Pavlou sample size from the previous simulation
setup_params <- read.csv("setup_parameters.csv")
N_DEV_PAVLOU <- setup_params$value[setup_params$parameter == "pavlou_n"]

# Define validation sizes: n, 2n, 4n, 8n, ..., up to 100,000
validation_sizes <- c(858, 1716, 3432, 6864, 13728, 27456, 54912, 100000)

# Use the same data generating parameters from the original simulation
calculate_performance_metrics <- function(y_true, y_pred) {
  auc_val <- as.numeric(auc(y_true, y_pred, quiet = TRUE))
  logit_pred <- qlogis(pmax(pmin(y_pred, 0.9999), 0.0001))
  cal_model <- glm(y_true ~ logit_pred, family = binomial)
  cal_slope <- coef(cal_model)[2]
  brier <- mean((y_pred - y_true)^2)
  mape <- mean(abs(y_pred - y_true))

  return(list(
    auc = auc_val,
    cal_slope = cal_slope,
    brier = brier,
    mape = mape
  ))
}

expit <- function(x) 1 / (1 + exp(-x))

# Calculate alpha (same as original simulation)
n_sample <- 100000
X_sample <- replicate(10, rnorm(n_sample))
beta <- c(0.45, 0.40, -0.35, 0.30, -0.25, 0.20, 0.15, 0.10, 0.08, 0.05)
target_prev <- 0.15

find_alpha <- function(a) {
  mean(expit(a + as.vector(X_sample %*% beta))) - target_prev
}
alpha <- uniroot(find_alpha, c(-10, 10))$root

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

  logit_p <- alpha +
    beta[1] * X1 + beta[2] * X2 + beta[3] * X3 + beta[4] * X4 +
    beta[5] * X5 + beta[6] * X6 + beta[7] * X7 + beta[8] * X8 +
    beta[9] * X9 + beta[10] * X10

  prob <- plogis(logit_p)
  outcome <- rbinom(n, size = 1, prob = prob)

  data.frame(
    outcome = outcome,
    X1 = X1, X2 = X2, X3 = X3, X4 = X4, X5 = X5,
    X6 = X6, X7 = X7, X8 = X8, X9 = X9, X10 = X10
  )
}

run_single_simulation <- function(sim, n_dev, validation_sizes, seed_base) {
  set.seed(seed_base + sim)

  # Generate training data (fixed size)
  dev_data <- generate_data(n_dev)

  # Fit model on development data
  model_dev <- glm(outcome ~ ., data = dev_data, family = binomial)

  # Validate on each external validation size
  results_list <- list()

  for (n_val in validation_sizes) {
    ext_data <- generate_data(n_val)
    pred_ext <- predict(model_dev, newdata = ext_data, type = "response")
    perf <- calculate_performance_metrics(ext_data$outcome, pred_ext)

    results_list[[as.character(n_val)]] <- data.frame(
      simulation = sim,
      val_size = n_val,
      auc = perf$auc,
      cal_slope = perf$cal_slope,
      brier = perf$brier,
      mape = perf$mape
    )
  }

  do.call(rbind, results_list)
}

run_simulation <- function(n_dev = N_DEV_PAVLOU, val_sizes = NULL,
                           n_sim = 1000, seed_base = 54321) {

  if (is.null(val_sizes)) {
    val_sizes <- c(858, 1716, 3432, 6864, 13728, 27456, 54912, 100000)
  }

  n_cores <- max(1, detectCores() - 1)
  cat(sprintf("Running %d simulations across %d validation sizes using %d cores...\n",
              n_sim, length(val_sizes), n_cores))

  results_list <- mclapply(1:n_sim, function(i) {
    if (i %% 100 == 0) cat(sprintf("  Completed %d/%d simulations\n", i, n_sim))
    run_single_simulation(i, n_dev, val_sizes, seed_base)
  }, mc.cores = n_cores)

  results <- do.call(rbind, results_list)

  cat("Simulation complete!\n")
  results
}

# Run simulation
cat("Starting external validation size simulation...\n")
results <- run_simulation()
write.csv(results, "external_validation_size_results.csv", row.names = FALSE)

# Save simulation parameters
ext_val_params <- data.frame(
  parameter = c("n_dev", "validation_sizes", "n_simulations", "alpha"),
  value = c(N_DEV_PAVLOU,
            paste(validation_sizes, collapse = ", "),
            1000,
            alpha)
)
write.csv(ext_val_params, "external_validation_size_parameters.csv", row.names = FALSE)

# Create boxplots for each metric
cat("Creating boxplots...\n")

# Convert val_size to factor with proper ordering
results$val_size_label <- factor(
  format(results$val_size, big.mark = ",", scientific = FALSE, trim = TRUE),
  levels = c("858", "1,716", "3,432", "6,864", "13,728", "27,456", "54,912", "100,000")
)

# AUC boxplot
p_auc <- ggplot(results, aes(x = val_size_label, y = auc)) +
  geom_boxplot(aes(fill = val_size_label), alpha = 0.7, show.legend = FALSE) +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "AUC by External Validation Dataset Size",
       x = "External Validation Sample Size",
       y = "AUC (C-statistic)") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("external_val_size_auc.png", p_auc, width = 10, height = 6, dpi = 300)

# Calibration slope boxplot
p_cal <- ggplot(results, aes(x = val_size_label, y = cal_slope)) +
  geom_boxplot(aes(fill = val_size_label), alpha = 0.7, show.legend = FALSE) +
  geom_hline(yintercept = 1.0, linetype = "dashed", color = "red", size = 0.8) +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "Calibration Slope by External Validation Dataset Size",
       x = "External Validation Sample Size",
       y = "Calibration Slope") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("external_val_size_calibration.png", p_cal, width = 10, height = 6, dpi = 300)

# Brier score boxplot
p_brier <- ggplot(results, aes(x = val_size_label, y = brier)) +
  geom_boxplot(aes(fill = val_size_label), alpha = 0.7, show.legend = FALSE) +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "Brier Score by External Validation Dataset Size",
       x = "External Validation Sample Size",
       y = "Brier Score") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("external_val_size_brier.png", p_brier, width = 10, height = 6, dpi = 300)

# MAPE boxplot
p_mape <- ggplot(results, aes(x = val_size_label, y = mape)) +
  geom_boxplot(aes(fill = val_size_label), alpha = 0.7, show.legend = FALSE) +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "MAPE by External Validation Dataset Size",
       x = "External Validation Sample Size",
       y = "Mean Absolute Prediction Error") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 10, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("external_val_size_mape.png", p_mape, width = 10, height = 6, dpi = 300)

# Calculate correlation matrices for each metric
cat("Creating correlation matrices...\n")

# Reshape data for correlation analysis
results_wide <- results %>%
  select(simulation, val_size, auc, cal_slope, brier, mape) %>%
  pivot_wider(
    id_cols = simulation,
    names_from = val_size,
    values_from = c(auc, cal_slope, brier, mape)
  )

# Calculate correlations for each metric
metrics <- c("auc", "cal_slope", "brier", "mape")
correlation_matrices <- list()

for (metric in metrics) {
  # Select columns for this metric
  cols <- grep(paste0("^", metric, "_"), names(results_wide), value = TRUE)
  metric_data <- results_wide[, cols]

  # Calculate correlation matrix
  cor_matrix <- cor(metric_data, use = "complete.obs")

  # Clean up column/row names (remove metric prefix)
  colnames(cor_matrix) <- gsub(paste0(metric, "_"), "", colnames(cor_matrix))
  rownames(cor_matrix) <- gsub(paste0(metric, "_"), "", rownames(cor_matrix))

  correlation_matrices[[metric]] <- cor_matrix

  # Create heatmap
  cor_melted <- as.data.frame(as.table(cor_matrix))
  names(cor_melted) <- c("Var1", "Var2", "value")

  p_heatmap <- ggplot(cor_melted, aes(x = Var1, y = Var2, fill = value)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.3f", value)), size = 3, color = "white") +
    scale_fill_gradient2(low = "#2166ac", mid = "#f7f7f7", high = "#b2182b",
                        midpoint = 0.5, limit = c(0, 1),
                        name = "Correlation") +
    labs(title = sprintf("Correlation Matrix: %s",
                        switch(metric,
                               "auc" = "AUC",
                               "cal_slope" = "Calibration Slope",
                               "brier" = "Brier Score",
                               "mape" = "MAPE")),
         subtitle = "Pairwise correlations between external validation sample sizes",
         x = "Validation Sample Size",
         y = "Validation Sample Size") +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", size = 14),
          plot.subtitle = element_text(size = 10, color = "gray40"),
          axis.text.x = element_text(angle = 45, hjust = 1),
          axis.title = element_text(size = 12, face = "bold"))

  filename <- sprintf("external_val_size_correlation_%s.png", metric)
  ggsave(filename, p_heatmap, width = 10, height = 8, dpi = 300)
}

# Create scatter plots comparing each size to the largest (truth)
cat("Creating scatter plots vs. truth...\n")

truth_size <- max(validation_sizes)

for (metric in metrics) {
  # Get data for this metric
  results_metric <- results %>%
    select(simulation, val_size, value = all_of(metric)) %>%
    pivot_wider(
      id_cols = simulation,
      names_from = val_size,
      values_from = value,
      names_prefix = "n_"
    )

  # Prepare data for plotting (all sizes vs truth, including truth vs itself)
  # First add truth column, then pivot
  results_metric_with_truth <- results_metric %>%
    mutate(truth = .data[[paste0("n_", truth_size)]])

  plot_data <- results_metric_with_truth %>%
    pivot_longer(
      cols = starts_with("n_"),
      names_to = "val_size",
      values_to = "value",
      names_prefix = "n_"
    ) %>%
    mutate(
      val_size = as.numeric(val_size),
      val_size_label = factor(
        format(val_size, big.mark = ",", scientific = FALSE, trim = TRUE),
        levels = c("858", "1,716", "3,432", "6,864", "13,728", "27,456", "54,912", "100,000")
      )
    )

  # Calculate correlations for annotation
  correlations <- plot_data %>%
    group_by(val_size_label) %>%
    summarise(
      cor = cor(value, truth, use = "complete.obs"),
      .groups = "drop"
    )

  # Create faceted scatter plot
  p_scatter <- ggplot(plot_data, aes(x = truth, y = value)) +
    geom_point(alpha = 0.3, size = 1, color = "#2A9D8F") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", size = 0.8) +
    geom_smooth(method = "lm", se = TRUE, color = "#264653", size = 1, alpha = 0.2) +
    geom_text(data = correlations,
              aes(label = sprintf("r = %.3f", cor)),
              x = -Inf, y = Inf, hjust = -0.1, vjust = 1.5,
              size = 3, fontface = "italic", color = "gray20") +
    facet_wrap(~ val_size_label, scales = "free", ncol = 4) +
    labs(
      title = sprintf("Correlation with Truth (n=100k): %s",
                     switch(metric,
                            "auc" = "AUC",
                            "cal_slope" = "Calibration Slope",
                            "brier" = "Brier Score",
                            "mape" = "MAPE")),
      subtitle = "Dashed red line shows perfect agreement (y = x)",
      x = sprintf("%s at n=100k (Truth)",
                 switch(metric,
                        "auc" = "AUC",
                        "cal_slope" = "Calibration Slope",
                        "brier" = "Brier Score",
                        "mape" = "MAPE")),
      y = sprintf("%s at Smaller Sample Sizes",
                 switch(metric,
                        "auc" = "AUC",
                        "cal_slope" = "Calibration Slope",
                        "brier" = "Brier Score",
                        "mape" = "MAPE"))
    ) +
    theme_bw() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10, color = "gray40"),
      axis.title = element_text(size = 11, face = "bold"),
      strip.text = element_text(size = 9, face = "bold"),
      strip.background = element_rect(fill = "gray90")
    )

  filename <- sprintf("external_val_size_scatter_%s.png", metric)
  ggsave(filename, p_scatter, width = 12, height = 10, dpi = 300)
}

cat("All plots created successfully!\n")
