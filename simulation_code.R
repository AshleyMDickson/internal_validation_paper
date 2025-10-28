library(pROC)
library(rms)
library(ggplot2)
library(parallel)
library(samplesizedev)
library(tidyr)

set.seed(123)

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

n_sample <- 100000
X_sample <- replicate(10, rnorm(n_sample))
beta <- c(0.45, 0.40, -0.35, 0.30, -0.25, 0.20, 0.15, 0.10, 0.08, 0.05)
target_prev <- 0.15

find_alpha <- function(a) {
  mean(expit(a + as.vector(X_sample %*% beta))) - target_prev
}
alpha <- uniroot(find_alpha, c(-10, 10))$root

pi_verify <- expit(alpha + as.vector(X_sample %*% beta))
y_verify <- rbinom(n_sample, 1, pi_verify)
observed_prev_verify <- sum(y_verify) / length(y_verify)

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

sample_split_validation <- function(data, split_ratio = 0.7) {
  n <- nrow(data)
  train_idx <- sample(1:n, size = floor(split_ratio * n))
  
  train_data <- data[train_idx, ]
  test_data <- data[-train_idx, ]
  
  model <- glm(outcome ~ ., data = train_data, family = binomial)
  pred_test <- predict(model, newdata = test_data, type = "response")
  
  calculate_performance_metrics(test_data$outcome, pred_test)
}

cross_validation <- function(data, k = 10) {
  n <- nrow(data)
  fold_size <- floor(n / k)
  indices <- sample(1:n)
  
  all_preds <- numeric(n)
  all_outcomes <- numeric(n)
  
  for (i in 1:k) {
    test_idx <- indices[((i-1) * fold_size + 1):min(i * fold_size, n)]
    if (i == k) test_idx <- indices[((i-1) * fold_size + 1):n]
    
    train_idx <- setdiff(1:n, test_idx)
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]
    
    model <- glm(outcome ~ ., data = train_data, family = binomial)
    pred_fold <- predict(model, newdata = test_data, type = "response")
    
    all_preds[test_idx] <- pred_fold
    all_outcomes[test_idx] <- test_data$outcome
  }
  
  calculate_performance_metrics(all_outcomes, all_preds)
}

bootstrap_validation <- function(data, B = 200) {
  n <- nrow(data)
  optimism_auc <- numeric(B)
  optimism_cal <- numeric(B)
  optimism_brier <- numeric(B)
  optimism_mape <- numeric(B)
  
  for (b in 1:B) {
    boot_idx <- sample(1:n, n, replace = TRUE)
    boot_data <- data[boot_idx, ]
    
    model <- glm(outcome ~ ., data = boot_data, family = binomial)
    
    pred_boot <- predict(model, newdata = boot_data, type = "response")
    perf_boot <- calculate_performance_metrics(boot_data$outcome, pred_boot)
    
    pred_orig <- predict(model, newdata = data, type = "response")
    perf_orig <- calculate_performance_metrics(data$outcome, pred_orig)
    
    optimism_auc[b] <- perf_boot$auc - perf_orig$auc
    optimism_cal[b] <- perf_boot$cal_slope - perf_orig$cal_slope
    optimism_brier[b] <- perf_boot$brier - perf_orig$brier
    optimism_mape[b] <- perf_boot$mape - perf_orig$mape
  }
  
  model_full <- glm(outcome ~ ., data = data, family = binomial)
  pred_full <- predict(model_full, newdata = data, type = "response")
  apparent_perf <- calculate_performance_metrics(data$outcome, pred_full)
  
  list(
    auc = apparent_perf$auc - mean(optimism_auc),
    cal_slope = apparent_perf$cal_slope - mean(optimism_cal),
    brier = apparent_perf$brier - mean(optimism_brier),
    mape = apparent_perf$mape - mean(optimism_mape)
  )
}

run_single_simulation <- function(sim, n_dev, n_ext, B_boot, k_cv, seed_base) {
  set.seed(seed_base + sim)
  
  dev_data <- generate_data(n_dev)
  
  model_dev <- glm(outcome ~ ., data = dev_data, family = binomial)
  pred_dev <- predict(model_dev, newdata = dev_data, type = "response")
  apparent <- calculate_performance_metrics(dev_data$outcome, pred_dev)
  
  split_perf <- sample_split_validation(dev_data)
  cv_perf <- cross_validation(dev_data, k = k_cv)
  boot_perf <- bootstrap_validation(dev_data, B = B_boot)
  
  ext_data <- generate_data(n_ext)
  pred_ext <- predict(model_dev, newdata = ext_data, type = "response")
  external <- calculate_performance_metrics(ext_data$outcome, pred_ext)
  
  data.frame(
    simulation = sim,
    
    apparent_auc = apparent$auc,
    apparent_cal_slope = apparent$cal_slope,
    apparent_brier = apparent$brier,
    apparent_mape = apparent$mape,
    
    split_auc = split_perf$auc,
    split_cal_slope = split_perf$cal_slope,
    split_brier = split_perf$brier,
    split_mape = split_perf$mape,
    
    cv_auc = cv_perf$auc,
    cv_cal_slope = cv_perf$cal_slope,
    cv_brier = cv_perf$brier,
    cv_mape = cv_perf$mape,
    
    boot_auc = boot_perf$auc,
    boot_cal_slope = boot_perf$cal_slope,
    boot_brier = boot_perf$brier,
    boot_mape = boot_perf$mape,
    
    external_auc = external$auc,
    external_cal_slope = external$cal_slope,
    external_brier = external$brier,
    external_mape = external$mape
  )
}

run_simulation <- function(n_dev = N_DEV, n_ext = 100000, n_sim = 500, 
                           B_boot = 200, k_cv = 10, seed_base = 12345) {
  
  n_cores <- max(1, detectCores() - 1)
  results_list <- mclapply(1:n_sim, function(i) {
    run_single_simulation(i, n_dev, n_ext, B_boot, k_cv, seed_base)
  }, mc.cores = n_cores)
  
  results <- do.call(rbind, results_list)
  
  results_long <- data.frame(
    simulation = rep(results$simulation, 3),
    method = rep(c("Sample Split", "Cross-validation", "Bootstrap"), each = nrow(results)),
    
    apparent_auc = rep(results$apparent_auc, 3),
    apparent_cal_slope = rep(results$apparent_cal_slope, 3),
    apparent_brier = rep(results$apparent_brier, 3),
    apparent_mape = rep(results$apparent_mape, 3),
    
    internal_auc = c(results$split_auc, results$cv_auc, results$boot_auc),
    internal_cal_slope = c(results$split_cal_slope, results$cv_cal_slope, results$boot_cal_slope),
    internal_brier = c(results$split_brier, results$cv_brier, results$boot_brier),
    internal_mape = c(results$split_mape, results$cv_mape, results$boot_mape),
    
    external_auc = rep(results$external_auc, 3),
    external_cal_slope = rep(results$external_cal_slope, 3),
    external_brier = rep(results$external_brier, 3),
    external_mape = rep(results$external_mape, 3)
  )
  
  results_long
}

results <- run_simulation()
write.csv(results, "validation_results.csv", row.names = FALSE)

setup_params <- data.frame(
  parameter = c("alpha", "n_dev", "n_events", "epv", "n_ext", "target_prev", 
                "observed_prev", "n_predictors", "n_simulations", "n_bootstrap", 
                "n_cv_folds", "riley_n", "pavlou_n"),
  value = c(alpha, N_DEV, N_EVENTS, N_EVENTS/10, 100000, target_prev, 
            observed_prev_verify, 10, 500, 200, 10, N_DEV_RILEY, N_DEV_PAVLOU)
)
write.csv(setup_params, "setup_parameters.csv", row.names = FALSE)

p_auc <- ggplot(results, aes(x = method, y = internal_auc)) +
  geom_boxplot(data = results, aes(x = "Apparent", y = apparent_auc), 
               fill = "#E76F51", alpha = 0.7) +
  geom_boxplot(aes(fill = method), alpha = 0.7) +
  geom_boxplot(data = results, aes(x = "External", y = external_auc), 
               fill = "#264653", alpha = 0.7) +
  scale_fill_manual(values = c("#F4A261", "#E9C46A", "#2A9D8F")) +
  scale_x_discrete(limits = c("Apparent", "Sample Split", "Cross-validation", 
                              "Bootstrap", "External")) +
  labs(title = "AUC (C-statistic) Comparison Across Validation Methods",
       subtitle = paste0(nrow(results)/3, " simulations per method"),
       x = "Validation Method",
       y = "AUC (C-statistic)") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("comparison_auc.png", p_auc, width = 10, height = 6, dpi = 300)

p_cal <- ggplot(results, aes(x = method, y = internal_cal_slope)) +
  geom_boxplot(data = results, aes(x = "Apparent", y = apparent_cal_slope), 
               fill = "#E76F51", alpha = 0.7) +
  geom_boxplot(aes(fill = method), alpha = 0.7) +
  geom_boxplot(data = results, aes(x = "External", y = external_cal_slope), 
               fill = "#264653", alpha = 0.7) +
  scale_fill_manual(values = c("#F4A261", "#E9C46A", "#2A9D8F")) +
  scale_x_discrete(limits = c("Apparent", "Sample Split", "Cross-validation", 
                              "Bootstrap", "External")) +
  labs(title = "Calibration Slope Comparison Across Validation Methods",
       subtitle = paste0(nrow(results)/3, " simulations per method"),
       x = "Validation Method",
       y = "Calibration Slope") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("comparison_calibration.png", p_cal, width = 10, height = 6, dpi = 300)

p_brier <- ggplot(results, aes(x = method, y = internal_brier)) +
  geom_boxplot(data = results, aes(x = "Apparent", y = apparent_brier), 
               fill = "#E76F51", alpha = 0.7) +
  geom_boxplot(aes(fill = method), alpha = 0.7) +
  geom_boxplot(data = results, aes(x = "External", y = external_brier), 
               fill = "#264653", alpha = 0.7) +
  scale_fill_manual(values = c("#F4A261", "#E9C46A", "#2A9D8F")) +
  scale_x_discrete(limits = c("Apparent", "Sample Split", "Cross-validation", 
                              "Bootstrap", "External")) +
  labs(title = "Brier Score Comparison Across Validation Methods",
       subtitle = paste0(nrow(results)/3, " simulations per method"),
       x = "Validation Method",
       y = "Brier Score") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("comparison_brier.png", p_brier, width = 10, height = 6, dpi = 300)

p_mape <- ggplot(results, aes(x = method, y = internal_mape)) +
  geom_boxplot(data = results, aes(x = "Apparent", y = apparent_mape), 
               fill = "#E76F51", alpha = 0.7) +
  geom_boxplot(aes(fill = method), alpha = 0.7) +
  geom_boxplot(data = results, aes(x = "External", y = external_mape), 
               fill = "#264653", alpha = 0.7) +
  scale_fill_manual(values = c("#F4A261", "#E9C46A", "#2A9D8F")) +
  scale_x_discrete(limits = c("Apparent", "Sample Split", "Cross-validation", 
                              "Bootstrap", "External")) +
  labs(title = "MAPE Comparison Across Validation Methods",
       subtitle = paste0(nrow(results)/3, " simulations per method"),
       x = "Validation Method",
       y = "Mean Absolute Prediction Error") +
  theme_bw() +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title.y = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(size = 11, angle = 25, hjust = 1),
        axis.text.y = element_text(size = 10))

ggsave("comparison_mape.png", p_mape, width = 10, height = 6, dpi = 300)

density_data_auc <- data.frame(
  value = c(results$apparent_auc[results$method == "Sample Split"],
            results$internal_auc[results$method == "Sample Split"],
            results$internal_auc[results$method == "Cross-validation"],
            results$internal_auc[results$method == "Bootstrap"],
            results$external_auc[results$method == "Sample Split"]),
  method = rep(c("Apparent", "Sample Split", "Cross-validation", "Bootstrap", "External"),
               each = nrow(results)/3)
)

density_data_cal <- data.frame(
  value = c(results$apparent_cal_slope[results$method == "Sample Split"],
            results$internal_cal_slope[results$method == "Sample Split"],
            results$internal_cal_slope[results$method == "Cross-validation"],
            results$internal_cal_slope[results$method == "Bootstrap"],
            results$external_cal_slope[results$method == "Sample Split"]),
  method = rep(c("Apparent", "Sample Split", "Cross-validation", "Bootstrap", "External"),
               each = nrow(results)/3)
)

density_data_brier <- data.frame(
  value = c(results$apparent_brier[results$method == "Sample Split"],
            results$internal_brier[results$method == "Sample Split"],
            results$internal_brier[results$method == "Cross-validation"],
            results$internal_brier[results$method == "Bootstrap"],
            results$external_brier[results$method == "Sample Split"]),
  method = rep(c("Apparent", "Sample Split", "Cross-validation", "Bootstrap", "External"),
               each = nrow(results)/3)
)

density_data_mape <- data.frame(
  value = c(results$apparent_mape[results$method == "Sample Split"],
            results$internal_mape[results$method == "Sample Split"],
            results$internal_mape[results$method == "Cross-validation"],
            results$internal_mape[results$method == "Bootstrap"],
            results$external_mape[results$method == "Sample Split"]),
  method = rep(c("Apparent", "Sample Split", "Cross-validation", "Bootstrap", "External"),
               each = nrow(results)/3)
)

method_levels <- c("Apparent", "Sample Split", "Cross-validation", "Bootstrap", "External")
density_data_auc$method <- factor(density_data_auc$method, levels = method_levels)
density_data_cal$method <- factor(density_data_cal$method, levels = method_levels)
density_data_brier$method <- factor(density_data_brier$method, levels = method_levels)
density_data_mape$method <- factor(density_data_mape$method, levels = method_levels)

method_colors <- c("Apparent" = "#E76F51", 
                   "Sample Split" = "#F4A261", 
                   "Cross-validation" = "#E9C46A", 
                   "Bootstrap" = "#2A9D8F", 
                   "External" = "#264653")

p_density_auc <- ggplot(density_data_auc, aes(x = value, fill = method, color = method)) +
  geom_density(alpha = 0.4, size = 1) +
  scale_fill_manual(values = method_colors) +
  scale_color_manual(values = method_colors) +
  labs(title = "AUC Distribution Across Validation Methods",
       x = "AUC (C-statistic)",
       y = "Density",
       fill = "Method",
       color = "Method") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        legend.position = "right")

ggsave("density_auc.png", p_density_auc, width = 10, height = 6, dpi = 300)

# Calibration slope density plot (exclude Apparent, add reference line)
density_data_cal_filtered <- density_data_cal[density_data_cal$method != "Apparent", ]

p_density_cal <- ggplot(density_data_cal_filtered, aes(x = value, fill = method, color = method)) +
  geom_density(alpha = 0.4, size = 1) +
  geom_vline(xintercept = 1.0, linetype = "dashed", color = "#E76F51", size = 1) +
  annotate("text", x = 1.0, y = Inf, label = "Perfect calibration (Apparent = 1.0)", 
           vjust = 1.5, hjust = -0.05, color = "#E76F51", size = 3.5, fontface = "italic") +
  scale_fill_manual(values = method_colors[names(method_colors) != "Apparent"]) +
  scale_color_manual(values = method_colors[names(method_colors) != "Apparent"]) +
  labs(title = "Calibration Slope Distribution Across Validation Methods",
       subtitle = "Dashed line shows perfect calibration (where Apparent validation = 1.0)",
       x = "Calibration Slope",
       y = "Density",
       fill = "Method",
       color = "Method") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 10, color = "gray40"),
        axis.title = element_text(size = 12, face = "bold"),
        legend.position = "right")

ggsave("density_calibration.png", p_density_cal, width = 10, height = 6, dpi = 300)

p_density_brier <- ggplot(density_data_brier, aes(x = value, fill = method, color = method)) +
  geom_density(alpha = 0.4, size = 1) +
  scale_fill_manual(values = method_colors) +
  scale_color_manual(values = method_colors) +
  labs(title = "Brier Score Distribution Across Validation Methods",
       x = "Brier Score",
       y = "Density",
       fill = "Method",
       color = "Method") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        legend.position = "right")

ggsave("density_brier.png", p_density_brier, width = 10, height = 6, dpi = 300)

p_density_mape <- ggplot(density_data_mape, aes(x = value, fill = method, color = method)) +
  geom_density(alpha = 0.4, size = 1) +
  scale_fill_manual(values = method_colors) +
  scale_color_manual(values = method_colors) +
  labs(title = "MAPE Distribution Across Validation Methods",
       x = "Mean Absolute Prediction Error",
       y = "Density",
       fill = "Method",
       color = "Method") +
  theme_bw() +
  theme(plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12, face = "bold"),
        legend.position = "right")

ggsave("density_mape.png", p_density_mape, width = 10, height = 6, dpi = 300)

cat("Simulation complete.")