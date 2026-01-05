library(pROC)

# Load the debug results
results <- read.csv("debug_calibration_results.csv")

# Re-run a few simulations but this time also calculate:
# 1. "True" calibration slope: how well does the SAME model perform on dev_data vs ext_data
# 2. Check if dev_data's inherent "predictability" explains the pattern

set.seed(42)

expit <- function(x) 1 / (1 + exp(-x))
beta <- c(0.45, 0.40, -0.35, 0.30, -0.25, 0.20, 0.15, 0.10, 0.08, 0.05)
target_prev <- 0.15

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

# Key test: Use the SAME model on both dev_data and ext_data
n_sim <- 100
n_dev <- 858
n_ext <- 100000

test_results <- data.frame(
  sim = integer(),
  full_model_on_dev_cal = numeric(),  # Full model evaluated on dev_data
  full_model_on_ext_cal = numeric(),  # SAME model evaluated on ext_data
  cv_cal_slope = numeric()            # CV (different models)
)

cat("Running hypothesis test...\n")
for (i in 1:n_sim) {
  if (i %% 20 == 0) cat("  Simulation", i, "\n")

  dev_data <- generate_data(n_dev)
  ext_data <- generate_data(n_ext)

  # Fit ONE model on full dev_data
  model_full <- glm(outcome ~ ., data = dev_data, family = binomial)

  # Evaluate this SAME model on dev_data
  pred_dev <- predict(model_full, newdata = dev_data, type = "response")
  cal_on_dev <- calculate_cal_slope(dev_data$outcome, pred_dev)

  # Evaluate this SAME model on ext_data
  pred_ext <- predict(model_full, newdata = ext_data, type = "response")
  cal_on_ext <- calculate_cal_slope(ext_data$outcome, pred_ext)

  # CV (for comparison - uses different models)
  indices <- sample(1:n_dev)
  fold_size <- floor(n_dev / 10)
  cv_preds <- numeric(n_dev)

  for (k in 1:10) {
    test_idx <- indices[((k-1) * fold_size + 1):min(k * fold_size, n_dev)]
    if (k == 10) test_idx <- indices[((k-1) * fold_size + 1):n_dev]

    train_idx <- setdiff(1:n_dev, test_idx)
    train_data <- dev_data[train_idx, ]
    test_data <- dev_data[test_idx, ]

    model_cv <- glm(outcome ~ ., data = train_data, family = binomial)
    cv_preds[test_idx] <- predict(model_cv, newdata = test_data, type = "response")
  }
  cv_cal <- calculate_cal_slope(dev_data$outcome, cv_preds)

  test_results <- rbind(test_results, data.frame(
    sim = i,
    full_model_on_dev_cal = cal_on_dev,
    full_model_on_ext_cal = cal_on_ext,
    cv_cal_slope = cv_cal
  ))
}

cat("\n=== CRITICAL TEST ===\n")
cat("Correlation between SAME model evaluated on dev vs ext data:\n")
cor_same_model <- cor(test_results$full_model_on_dev_cal,
                       test_results$full_model_on_ext_cal)
cat("  r =", round(cor_same_model, 3), "\n\n")

cat("Correlation between CV (on dev) and full model on ext:\n")
cor_cv_ext <- cor(test_results$cv_cal_slope,
                  test_results$full_model_on_ext_cal)
cat("  r =", round(cor_cv_ext, 3), "\n\n")

cat("Correlation between CV and full model evaluated on SAME dev data:\n")
cor_cv_dev <- cor(test_results$cv_cal_slope,
                  test_results$full_model_on_dev_cal)
cat("  r =", round(cor_cv_dev, 3), "\n\n")

cat("Full model on dev_data:\n")
cat("  Mean:", round(mean(test_results$full_model_on_dev_cal), 3),
    " SD:", round(sd(test_results$full_model_on_dev_cal), 3), "\n")

cat("Full model on ext_data:\n")
cat("  Mean:", round(mean(test_results$full_model_on_ext_cal), 3),
    " SD:", round(sd(test_results$full_model_on_ext_cal), 3), "\n")

cat("CV on dev_data:\n")
cat("  Mean:", round(mean(test_results$cv_cal_slope), 3),
    " SD:", round(sd(test_results$cv_cal_slope), 3), "\n")

# Save
write.csv(test_results, "hypothesis_test_results.csv", row.names = FALSE)
cat("\nResults saved to hypothesis_test_results.csv\n")

# Create diagnostic plot
png("hypothesis_test_plot.png", width = 1200, height = 400)
par(mfrow = c(1, 3))

# Plot 1: Same model, different data
plot(test_results$full_model_on_ext_cal, test_results$full_model_on_dev_cal,
     xlab = "Full Model on ext_data",
     ylab = "Full Model on dev_data",
     main = paste0("SAME model, different data\n(r = ",
                   round(cor_same_model, 3), ")"),
     pch = 16, col = rgb(0, 0, 0, 0.3))
abline(0, 1, col = "red", lty = 2, lwd = 2)
abline(lm(full_model_on_dev_cal ~ full_model_on_ext_cal, data = test_results),
       col = "blue", lwd = 2)

# Plot 2: CV vs external (different models)
plot(test_results$full_model_on_ext_cal, test_results$cv_cal_slope,
     xlab = "Full Model on ext_data",
     ylab = "CV on dev_data",
     main = paste0("Different models, different data\n(r = ",
                   round(cor_cv_ext, 3), ")"),
     pch = 16, col = rgb(0, 0, 0, 0.3))
abline(0, 1, col = "red", lty = 2, lwd = 2)
abline(lm(cv_cal_slope ~ full_model_on_ext_cal, data = test_results),
       col = "blue", lwd = 2)

# Plot 3: CV vs full model on SAME data (dev_data)
plot(test_results$full_model_on_dev_cal, test_results$cv_cal_slope,
     xlab = "Full Model on dev_data",
     ylab = "CV on dev_data",
     main = paste0("Different models, SAME data\n(r = ",
                   round(cor_cv_dev, 3), ")"),
     pch = 16, col = rgb(0, 0, 0, 0.3))
abline(0, 1, col = "red", lty = 2, lwd = 2)
abline(lm(cv_cal_slope ~ full_model_on_dev_cal, data = test_results),
       col = "blue", lwd = 2)

dev.off()
cat("Plot saved to hypothesis_test_plot.png\n")
