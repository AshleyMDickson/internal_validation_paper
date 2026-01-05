library(pROC)

set.seed(42)

# Use same DGP parameters as main simulation
expit <- function(x) 1 / (1 + exp(-x))
beta <- c(0.45, 0.40, -0.35, 0.30, -0.25, 0.20, 0.15, 0.10, 0.08, 0.05)
target_prev <- 0.15

# Calculate alpha (same as main simulation)
n_sample <- 100000
X_sample <- replicate(10, rnorm(n_sample))
find_alpha <- function(a) {
  mean(expit(a + as.vector(X_sample %*% beta))) - target_prev
}
alpha <- uniroot(find_alpha, c(-10, 10))$root

generate_data <- function(n) {
  X <- matrix(rnorm(n * 10), ncol = 10)
  logit_p <- alpha + as.vector(X %*% beta)
  prob <- plogis(logit_p)
  outcome <- rbinom(n, size = 1, prob = prob)

  data.frame(outcome = outcome, X)
}

calculate_cal_slope <- function(y_true, y_pred) {
  logit_pred <- qlogis(pmax(pmin(y_pred, 0.9999), 0.0001))
  cal_model <- glm(y_true ~ logit_pred, family = binomial)
  coef(cal_model)[2]
}

# Simple cross-validation implementation
cv_predictions <- function(data, k = 10) {
  n <- nrow(data)
  fold_size <- floor(n / k)
  indices <- sample(1:n)

  all_preds <- numeric(n)

  for (i in 1:k) {
    test_idx <- indices[((i-1) * fold_size + 1):min(i * fold_size, n)]
    if (i == k) test_idx <- indices[((i-1) * fold_size + 1):n]

    train_idx <- setdiff(1:n, test_idx)
    train_data <- data[train_idx, ]
    test_data <- data[test_idx, ]

    model <- glm(outcome ~ ., data = train_data, family = binomial)
    all_preds[test_idx] <- predict(model, newdata = test_data, type = "response")
  }

  all_preds
}

# Run simple simulation
n_sim <- 100
n_dev <- 858
n_ext <- 100000

results <- data.frame(
  sim = integer(),
  cv_cal_slope = numeric(),
  ext_cal_slope = numeric(),
  apparent_cal_slope = numeric()
)

cat("Running", n_sim, "simulations...\n")
for (i in 1:n_sim) {
  if (i %% 20 == 0) cat("  Simulation", i, "\n")

  # Generate dev data
  dev_data <- generate_data(n_dev)

  # Fit model on FULL dev data (for external validation)
  model_full <- glm(outcome ~ ., data = dev_data, family = binomial)

  # Apparent performance on dev data
  pred_dev <- predict(model_full, newdata = dev_data, type = "response")
  apparent_cal <- calculate_cal_slope(dev_data$outcome, pred_dev)

  # CV predictions (uses different models - 90% trained)
  cv_preds <- cv_predictions(dev_data, k = 10)
  cv_cal <- calculate_cal_slope(dev_data$outcome, cv_preds)

  # External validation (uses full model)
  ext_data <- generate_data(n_ext)
  pred_ext <- predict(model_full, newdata = ext_data, type = "response")
  ext_cal <- calculate_cal_slope(ext_data$outcome, pred_ext)

  results <- rbind(results, data.frame(
    sim = i,
    cv_cal_slope = cv_cal,
    ext_cal_slope = ext_cal,
    apparent_cal_slope = apparent_cal
  ))
}

# Calculate correlation
cor_cv_ext <- cor(results$cv_cal_slope, results$ext_cal_slope)

cat("\n=== RESULTS ===\n")
cat("Correlation (CV vs External):", round(cor_cv_ext, 3), "\n\n")

cat("CV calibration slope:\n")
cat("  Mean:", round(mean(results$cv_cal_slope), 3), "\n")
cat("  SD:", round(sd(results$cv_cal_slope), 3), "\n\n")

cat("External calibration slope:\n")
cat("  Mean:", round(mean(results$ext_cal_slope), 3), "\n")
cat("  SD:", round(sd(results$ext_cal_slope), 3), "\n\n")

cat("Apparent calibration slope:\n")
cat("  Mean:", round(mean(results$apparent_cal_slope), 3), "\n")
cat("  SD:", round(sd(results$apparent_cal_slope), 3), "\n\n")

# Show extreme cases
cat("Cases with HIGHEST CV cal slope:\n")
extreme_high <- results[order(-results$cv_cal_slope)[1:5], ]
print(extreme_high)

cat("\nCases with LOWEST CV cal slope:\n")
extreme_low <- results[order(results$cv_cal_slope)[1:5], ]
print(extreme_low)

# Save results
write.csv(results, "debug_calibration_results.csv", row.names = FALSE)
cat("\nResults saved to debug_calibration_results.csv\n")

# Create simple scatter plot
png("debug_calibration_scatter.png", width = 800, height = 600)
plot(results$ext_cal_slope, results$cv_cal_slope,
     xlab = "External Calibration Slope",
     ylab = "CV Calibration Slope",
     main = paste0("CV vs External Calibration Slope\n(correlation = ",
                   round(cor_cv_ext, 3), ")"),
     pch = 16, col = rgb(0, 0, 0, 0.3))
abline(a = 0, b = 1, col = "red", lty = 2, lwd = 2)
abline(lm(cv_cal_slope ~ ext_cal_slope, data = results), col = "blue", lwd = 2)
legend("topleft",
       c("Perfect agreement (y=x)", "Fitted line"),
       col = c("red", "blue"), lty = c(2, 1), lwd = 2)
dev.off()
cat("Plot saved to debug_calibration_scatter.png\n")
