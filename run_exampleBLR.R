library(rstan)
options(mc.cores = 2)

## ============================================================
## Example 3: Logistic regression, multivariate normal prior
## and variational family. Compared against Stan
## NUTS (MCMC reference) and Stan ADVI (meanfield VI baseline).
## ============================================================
set.seed(123)
n <- 1000
p <- 9
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta_true <- c(0.5, -1, 0.2, 0.8, -0.5, 1.2, -0.3, 0.6, -0.9)

sigmoid <- function(z) 1 / (1 + exp(-z))
y <- rbinom(n, 1, sigmoid(X %*% beta_true))

Sigma0_inv <- diag(0.01, p)     # prior N(0, 100 I)
mu0        <- rep(0, p)

## BLR update
##   Lambda_{t+1} = (1-rho) Lambda_t + rho (Sigma0^{-1} + X'W_tX),  W_t = diag{p_i(1-p_i)}
##   mu_{t+1}     = mu_t - rho * Sigma_{t+1} { Sigma0^{-1}(mu_t-mu0) - X'(y-p_t) + tg_t }
## where tg_t is the exact third-moment term (eq. 3-gradmu)
run_blr <- function(X, y, mu0, Sigma0_inv, rho = 0.7, max_iters = 200, tol = 1e-6) {
  p_total <- ncol(X)
  Lambda  <- Sigma0_inv
  Sigma   <- solve(Lambda)
  mu      <- mu0
  for (iter in 1:max_iters) {
    mu_old  <- mu
    p_i     <- sigmoid(as.vector(X %*% mu))
    W       <- p_i * (1 - p_i)
    Lambda  <- (1 - rho) * Lambda + rho * (Sigma0_inv + crossprod(X, W * X))
    Sigma   <- solve(Lambda)
    xSx     <- rowSums((X %*% Sigma) * X)
    tg      <- 0.5 * crossprod(X, p_i * (1 - p_i) * (1 - 2 * p_i) * xSx)
    grad_mu <- Sigma0_inv %*% (mu - mu0) - crossprod(X, y - p_i) + tg
    mu      <- mu - rho * as.vector(Sigma %*% grad_mu)
    if (max(abs(mu - mu_old)) < tol) break
  }
  list(mean = mu, Sigma = Sigma, sd = sqrt(diag(Sigma)), iters = iter)
}

t_qvmp <- system.time({
  fit_blr <- run_blr(X, y, mu0, Sigma0_inv)
})

beta_est <- fit_blr$mean
Cov      <- fit_blr$Sigma
sd_qvmp  <- fit_blr$sd

stan_code_logistic <- "
data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  int<lower=0, upper=1> y[N];
}
parameters { vector[K] beta; }
model { beta ~ normal(0, 10); y ~ bernoulli_logit(X * beta); }
"
stan_data   <- list(N = n, K = p, X = X, y = y)
sm_logistic <- stan_model(model_code = stan_code_logistic)

t_mcmc <- system.time({
  fit_stan <- sampling(sm_logistic, data = stan_data, iter = 10000, chains = 2, seed = 123)
})
samples   <- extract(fit_stan)
mean_mcmc <- colMeans(samples$beta)
sd_mcmc   <- apply(samples$beta, 2, sd)

## Stan ADVI
t_advi <- system.time({
  fit_advi <- vb(sm_logistic, data = stan_data, algorithm = "fullrank",
                 output_samples = 10000, seed = 2)
})
samples_advi <- extract(fit_advi)
mean_advi    <- colMeans(samples_advi$beta)
sd_advi      <- apply(samples_advi$beta, 2, sd)

## Mean-field ADVI
fit_advi_mf   <- vb(sm_logistic, data = stan_data, algorithm = "meanfield",
                     output_samples = 10000, seed = 2)
samples_advi_mf <- extract(fit_advi_mf)
mean_advi_mf    <- colMeans(samples_advi_mf$beta)
sd_advi_mf      <- apply(samples_advi_mf$beta, 2, sd)


summary_tab <- data.frame(
  beta      = sprintf("beta_%d", 0:(p - 1)),
  true      = beta_true,
  qvmp_mean = beta_est,  qvmp_sd = sd_qvmp,
  mcmc_mean = mean_mcmc, mcmc_sd = sd_mcmc
)

cat(sprintf("\nQ-VMP (BLR) wall-clock time: %.3f s (%d iterations)\n",
            t_qvmp["elapsed"], fit_blr$iters))
cat(sprintf("Stan ADVI wall-clock time: %.3f s\n", t_advi["elapsed"]))
cat(sprintf("Stan/MCMC wall-clock time: %.3f s\n", t_mcmc["elapsed"]))
cat(sprintf("Speed-up (MCMC / Q-VMP): %.1fx\n", t_mcmc["elapsed"] / t_qvmp["elapsed"]))
cat(sprintf("Speed-up (MCMC / ADVI): %.1fx\n", t_mcmc["elapsed"] / t_advi["elapsed"]))

#pdf("fig_logistic.pdf", width = 8, height = 8)
tiff("fig_logistic.tiff", width = 8, height = 8, units = "in", res = 600, compression = "lzw")
par(mfrow = c(3, 3))

col_qvmp    <- "#222222" 
col_full    <- "#0072B2" 
col_mf      <- "#E69F00" 
col_mcmc_bg <- "#F2F2F2" 
col_mcmc_fg <- "#888888" 

for (i in 1:p) {
  h <- hist(samples$beta[, i], breaks = 30, plot = FALSE)

  peak <- max(h$density, dnorm(0, 0, c(sd_qvmp[i], sd_advi[i], sd_advi_mf[i])))
  
  
  hist(samples$beta[, i], breaks = 30, probability = TRUE,
       main = bquote(Posterior ~ beta[.(i - 1)]),  ylim = c(0, peak * 1.6), 
       xlab = bquote(beta[.(i - 1)]), col = col_mcmc_bg, border = col_mcmc_fg)
  
  # Proposed Q-VMP
  curve(dnorm(x, mean = beta_est[i], sd = sd_qvmp[i]),
        col = col_qvmp, lty = 1, lwd = 2.5, add = TRUE)
  
  # ADVI Full-Rank
  curve(dnorm(x, mean = mean_advi[i], sd = sd_advi[i]),
        col = col_full, lty = 5, lwd = 2, add = TRUE)
  
  # ADVI Mean-Field
  curve(dnorm(x, mean = mean_advi_mf[i], sd = sd_advi_mf[i]),
        col = col_mf, lty = 2, lwd = 2, add = TRUE)
  
 
  legend("topright", legend = c("Q-VMP (BLR)", "ADVI (full-rank)", "ADVI (mean-field)", "MCMC"),
         col = c(col_qvmp, col_full, col_mf, col_mcmc_fg), 
         lty = c(1, 5, 2, NA),     # 1=Solid, 5=Long-dash, 2=Dashed
         lwd = c(2.5, 2, 2, NA),
         pch = c(NA, NA, NA, 15),  # Square matching the MCMC histogram texture
         bty = "n", cex = 0.7)
}
dev.off()

# saveRDS(list(summary_tab = summary_tab, t_qvmp = t_qvmp, t_mcmc = t_mcmc, t_advi = t_advi),
#         "sim_logistic_results.rds")

## ============================================================
## Example 4: Robust spatial binomial regression, heavy-tailed
## (Student-t) field; the range phi is held FIXED at phi_true.
## ============================================================
set.seed(2)

library(mvtnorm)

sigmoid <- function(z) 1 / (1 + exp(-z))

##(5x5 grid, n = 25 locations)
g        <- 5
coords   <- as.matrix(expand.grid(x = 1:g, y = 1:g))
n        <- nrow(coords)

Dm       <- as.matrix(dist(coords))
phi_true <- 1.2
alpha    <- 4
R_true   <- exp(-Dm / phi_true) + diag(1e-6, n) # For stability

p_beta    <- 5
Xb        <- cbind(1, matrix(rnorm(n * (p_beta - 1)), n, p_beta - 1))
beta_true <- c(-0.3, 0.6, -0.4, 0.2, -0.5)

m <- sample(10:40, n, replace = TRUE)

## u ~ t_alpha(0, R_true)
u_true <- as.vector(
  rmvt(1, sigma = R_true, df = alpha)
)

## y_i ~ Binomial(m_i, sigmoid(X_i beta + u_i))
eta <- as.vector(Xb %*% beta_true) + u_true
y   <- rbinom(n, m, sigmoid(eta))
X   <- cbind(Xb, diag(n)); d <- p_beta + n; ui <- (p_beta + 1):d

## beta gets a flat (improper) prior 

Rpi_c <- solve(R_true)

nparam_total <- d + d * (d + 1) / 2 + 2

##   (1) Lambda_{t+1} = (1-rho) Lambda_t + rho (gamma_t B + X'W_tX),  W_t = diag{m_i p_i(1-p_i)}
##   (2) mu_{t+1}     = mu_t - rho * Sigma_{t+1} { gamma_t B mu_t - X'(y - m*p_t) + tg_t }
##   (3) a_{t+1} = alpha/2 + n/2,  b_{t+1} = alpha/2 + (1/2)(mu_{t+1}'B mu_{t+1} + tr{B Sigma_{t+1}})
## using the freshly updated mu_{t+1}, Sigma_{t+1} from steps (1)-(2); 
run_blr_spatial <- function(X, y, m, B, alpha, ui, Rpi_c, rho = 0.5, max_iters = 2000, tol = 1e-6) {
  d      <- ncol(X)
  n      <- length(y)
  Lambda <- diag(d)
  Sigma  <- solve(Lambda)
  mu     <- rep(0, d)
  a      <- (alpha + n) / 2
  b      <- alpha / 2
  for (iter in 1:max_iters) {
    mu_old <- mu
    gam    <- a / b
    eta    <- as.vector(X %*% mu)
    p_i    <- sigmoid(eta)
    W      <- m * p_i * (1 - p_i)
    Lambda <- (1 - rho) * Lambda + rho * (gam * B + crossprod(X, W * X))
    Sigma  <- chol2inv(chol(Lambda))
    xSx    <- rowSums((X %*% Sigma) * X)
    tg     <- 0.5 * crossprod(X, m * p_i * (1 - p_i) * (1 - 2 * p_i) * xSx)
    grad   <- gam * as.vector(B %*% mu) - crossprod(X, y - m * p_i) + tg
    mu     <- mu - rho * as.vector(Sigma %*% grad)
    a      <- (alpha + n) / 2
    b      <- (alpha + as.numeric(t(mu[ui]) %*% Rpi_c %*% mu[ui]) + sum(Rpi_c * Sigma[ui, ui])) / 2
    if (max(abs(mu - mu_old)) < tol) break
  }
  list(mu = mu, Sigma = Sigma, a = a, b = b, iters = iter)
}

B <- matrix(0, d, d)
B[ui, ui] <- Rpi_c

t_qvmp <- system.time({
  fit_blr <- run_blr_spatial(X, y, m, B, alpha, ui, Rpi_c)
})

mu_hat   <- fit_blr$mu
Sig_hat  <- fit_blr$Sigma
beta_hat <- mu_hat[1:p_beta]
beta_sd  <- sqrt(diag(Sig_hat)[1:p_beta])
u_hat    <- mu_hat[ui]
a_hat    <- fit_blr$a; b_hat <- fit_blr$b
gam_hat  <- a_hat / b_hat

## MCMC of the exact same model, phi fixed at phi_true
stan_data <- list(n = n, p = p_beta, Xb = Xb, Rc = R_true, m = m, y = y, alpha = alpha)
t_mcmc <- system.time({
  fit_stan <- stan(model_code = "
data{ int<lower=0> n; int<lower=0> p; matrix[n,p] Xb; matrix[n,n] Rc;
      int<lower=0> m[n]; int<lower=0> y[n]; real<lower=0> alpha; }
parameters{ vector[p] beta; vector[n] u; }
model{
  // beta: no prior statement -> implicit flat/improper prior
  u ~ multi_student_t(alpha, rep_vector(0, n), Rc);
  y ~ binomial_logit(m, Xb * beta + u);
}",
  data = stan_data, iter = 10000, chains = 2, seed = 1)
})

samples <- extract(fit_stan)
u_mcmc  <- colMeans(samples$u)

## Full-rank ADVI baseline (reuses the model already compiled inside fit_stan).
t_advi <- system.time({
  fit_advi <- vb(get_stanmodel(fit_stan), data = stan_data, algorithm = "fullrank",
                 output_samples = 10000, seed = 2)
})
samples_advi   <- extract(fit_advi)
mean_advi_beta <- colMeans(samples_advi$beta)
sd_advi_beta   <- apply(samples_advi$beta, 2, sd)
mean_advi_u    <- colMeans(samples_advi$u)
cat(sprintf("Stan full-rank ADVI wall-clock: %.3f s\n", t_advi["elapsed"]))

## Mean-field ADVI, (independence assumption across beta_j, versus full-rank ADVI above).
fit_advi_mf      <- vb(get_stanmodel(fit_stan), data = stan_data, algorithm = "meanfield",
                        output_samples = 10000, seed = 2)
samples_advi_mf  <- extract(fit_advi_mf)
mean_advi_mf_beta <- colMeans(samples_advi_mf$beta)
sd_advi_mf_beta   <- apply(samples_advi_mf$beta, 2, sd)
mean_advi_mf_u    <- colMeans(samples_advi_mf$u)

summary_tab <- data.frame(
  param     = sprintf("beta_%d", 0:(p_beta - 1)),
  true      = beta_true,
  qvmp_mean = beta_hat,
  qvmp_sd   = beta_sd,
  mcmc_mean = colMeans(samples$beta),
  mcmc_sd   = apply(samples$beta, 2, sd)
)
print(summary_tab, digits = 3, row.names = FALSE)

field_rmse <- c(qvmp_vs_mcmc  = sqrt(mean((u_hat  - u_mcmc)^2)),
                qvmp_vs_truth = sqrt(mean((u_hat  - u_true)^2)),
                mcmc_vs_truth = sqrt(mean((u_mcmc - u_true)^2)))
cat("\nfield RMSE:\n"); print(round(field_rmse, 3))
cat(sprintf("Q-VMP (BLR) wall-clock : %.3f s  (%d parameters, %d BLR iterations)\n",
            t_qvmp["elapsed"], nparam_total, fit_blr$iters))
cat(sprintf("MCMC  wall-clock : %.1f s\n", t_mcmc["elapsed"]))
cat(sprintf("Speed-up (MCMC / Q-VMP): %.1f x\n", t_mcmc["elapsed"] / t_qvmp["elapsed"]))

#pdf("fig_spatial_joint.pdf", width = 9, height = 6)
tiff("fig_spatial_joint.tiff", width = 9, height = 6, units = "in", res = 600, compression = "lzw")
par(mfrow = c(2, 3))

for (j in 1:p_beta) {
  h <- hist(samples$beta[, j], breaks = 30, plot = FALSE)
  peak <- max(h$density, dnorm(0, 0, c(beta_sd[j], sd_advi_beta[j], sd_advi_mf_beta[j])))
  
  hist(samples$beta[, j], breaks = 30, probability = TRUE,
       main = bquote(Posterior ~ beta[.(j - 1)]), ylim = c(0, peak * 1.6), 
       xlab = bquote(beta[.(j - 1)]), col = col_mcmc_bg, border = col_mcmc_fg)
  
  curve(dnorm(x, beta_hat[j], beta_sd[j]), 
        col = col_qvmp, lty = 1, lwd = 2.5, add = TRUE)
  
  curve(dnorm(x, mean_advi_beta[j], sd_advi_beta[j]), 
        col = col_full, lty = 5, lwd = 2, add = TRUE)
  
  curve(dnorm(x, mean_advi_mf_beta[j], sd_advi_mf_beta[j]), 
        col = col_mf, lty = 2, lwd = 2, add = TRUE)
  
  legend("topright", legend = c("Q-VMP (BLR)", "ADVI (full-rank)", "ADVI (mean-field)", "MCMC"),
         col = c(col_qvmp, col_full, col_mf, col_mcmc_fg),
         lty = c(1, 5, 2, NA), lwd = c(2.5, 2, 2, NA), 
         pch = c(NA, NA, NA, 15), bty = "n", cex = 0.7)
}

plot(u_mcmc, mean_advi_u, type = "n", 
     xlab = "MCMC posterior mean  u", ylab = "Posterior mean  u",
     main = "Spatial field", ylim = range(c(u_hat, mean_advi_u, mean_advi_mf_u)))
abline(0, 1, col = "gray40", lwd = 1, lty = 3) 

points(u_mcmc, mean_advi_u, pch = 17, cex = 0.9, col = col_full)
points(u_mcmc, u_hat, pch = 16, cex = 0.9, col = col_qvmp)
points(u_mcmc, mean_advi_mf_u, pch = 1, cex = 0.9, col = col_mf, lwd = 1.5)

legend("topleft", 
       legend = c("Q-VMP (BLR)", "ADVI (full-rank)", "ADVI (mean-field)"), 
       col = c(col_qvmp, col_full, col_mf),
       pch = c(16, 17, 1), 
       pt.lwd = c(1, 1, 1.5),
       bty = "n", cex = 0.7)

dev.off()

# saveRDS(list(summary_tab = summary_tab, beta_hat = beta_hat, beta_sd = beta_sd,
#              u_hat = u_hat, u_true = u_true, u_mcmc = u_mcmc, gam_hat = gam_hat,
#              mean_advi_u = mean_advi_u, phi_true = phi_true, field_rmse = field_rmse,
#              mean_advi_beta = mean_advi_beta, sd_advi_beta = sd_advi_beta,
#              t_qvmp = t_qvmp, t_mcmc = t_mcmc, t_advi = t_advi, n = n, nparam = nparam_total),
#         "sim_spatial_joint_results.rds")

## ============================================================
## Bad approximation example
## ============================================================
p_ex3          <- 9
mu0_ex3        <- rep(0, p_ex3)
Sigma0_inv_ex3 <- diag(0.01, p_ex3)
beta_true_ex3  <- c(0.5, -1, 0.2, 0.8, -0.5, 1.2, -0.3, 0.6, -0.9)

n_stress    <- 25
chosen_seed <- 3
set.seed(chosen_seed)
X_stress <- cbind(1, matrix(rnorm(n_stress * (p_ex3 - 1)), n_stress, p_ex3 - 1))
y_stress <- rbinom(n_stress, 1, sigmoid(X_stress %*% beta_true_ex3))

fit_blr_stress <- run_blr(X_stress, y_stress, mu0_ex3, Sigma0_inv_ex3)

## MCMC ground truth on the same dataset
fit_stan_stress <- sampling(sm_logistic,
                             data = list(N = n_stress, K = p_ex3, X = X_stress, y = y_stress),
                             iter = 10000, chains = 2, seed = chosen_seed)
samp_stress <- extract(fit_stan_stress)$beta
mcmc_mean_stress <- colMeans(samp_stress); mcmc_sd_stress <- apply(samp_stress, 2, sd)

## Stan ADVI, full-rank and mean-field
fit_advi_fr_stress <- vb(sm_logistic,
                          data = list(N = n_stress, K = p_ex3, X = X_stress, y = y_stress),
                          algorithm = "fullrank", output_samples = 10000, seed = 2)
samp_advi_fr_stress <- extract(fit_advi_fr_stress)$beta
advi_fr_mean_stress <- colMeans(samp_advi_fr_stress); advi_fr_sd_stress <- apply(samp_advi_fr_stress, 2, sd)

fit_advi_mf_stress <- vb(sm_logistic,
                          data = list(N = n_stress, K = p_ex3, X = X_stress, y = y_stress),
                          algorithm = "meanfield", output_samples = 10000, seed = 2)
samp_advi_mf_stress <- extract(fit_advi_mf_stress)$beta
advi_mf_mean_stress <- colMeans(samp_advi_mf_stress); advi_mf_sd_stress <- apply(samp_advi_mf_stress, 2, sd)


#pdf("fig_logistic_stress.pdf", width = 8, height = 8)
tiff("fig_logistic_stress.tiff", width = 8, height = 8, units = "in", res = 600, compression = "lzw")
par(mfrow = c(3, 3))

for (i in 1:p_ex3) {
  h <- hist(samp_stress[, i], breaks = 30, plot = FALSE)
  peak <- max(h$density, dnorm(0, 0, c(fit_blr_stress$sd[i], advi_fr_sd_stress[i], advi_mf_sd_stress[i])))
  
  hist(samp_stress[, i], breaks = 30, probability = TRUE,
       main = bquote(Posterior ~ beta[.(i - 1)]), ylim = c(0, peak * 1.3),
       xlab = bquote(beta[.(i - 1)]), col = col_mcmc_bg, border = col_mcmc_fg)
  
  curve(dnorm(x, mean = fit_blr_stress$mean[i], sd = fit_blr_stress$sd[i]), 
        col = col_qvmp, lty = 1, lwd = 2.5, add = TRUE)
  
  curve(dnorm(x, mean = advi_fr_mean_stress[i], sd = advi_fr_sd_stress[i]), 
        col = col_full, lty = 5, lwd = 2, add = TRUE)
  
  curve(dnorm(x, mean = advi_mf_mean_stress[i], sd = advi_mf_sd_stress[i]), 
        col = col_mf, lty = 2, lwd = 2, add = TRUE)
  
  legend("topright", legend = c("Q-VMP (BLR)", "ADVI (full-rank)", "ADVI (mean-field)", "MCMC"),
         col = c(col_qvmp, col_full, col_mf, col_mcmc_fg), 
         lty = c(1, 5, 2, NA), lwd = c(2.5, 2, 2, NA), 
         pch = c(NA, NA, NA, 15), bty = "n", cex = 0.6)
}
dev.off()

