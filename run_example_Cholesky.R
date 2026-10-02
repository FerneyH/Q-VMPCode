library(rstan)
options(mc.cores = 2)

## ============================================================
## Example 1: Normal--Normal
## ============================================================
set.seed(123)
n1       <- 1000
true_mu1 <- 1.5
sigma1   <- 1
x1    <- rnorm(n1, mean = true_mu1, sd = sigma1)
xbar1 <- mean(x1)

## F_hat_normal(mu, sigma2): the free energy of eq. (7),
##   \hat F(mu,sigma2) = (n+1)/2 (mu^2+sigma2) - n*xbar*mu - (1/2) log(sigma2) - 1/2
## Optimized over (mu, log(sigma2)) for stability
F_hat_normal <- function(par) {
  mu         <- par[1]
  log_sigma2 <- par[2]
  sigma2     <- exp(log_sigma2)
  ((n1 + 1) / 2) * (mu^2 + sigma2) - n1 * xbar1 * mu - 0.5 * log_sigma2 - 0.5
}

t_qvmp_normal <- system.time({
  fit_normal <- optim(par = c(1, 0), fn = F_hat_normal, method = "BFGS")
})
mu_opt_normal     <- fit_normal$par[1]
sigma2_opt_normal <- exp(fit_normal$par[2])

stan_data_normal <- list(N = n1, x = x1)
t_mcmc_normal <- system.time({
  fit_stan_normal <- stan(model_code = "
data {
  int<lower=0> N;
  real x[N];
}
parameters {
  real mu;
}
model {
  mu ~ normal(0, 1);
  x ~ normal(mu, 1);
}
", data = stan_data_normal, iter = 10000, chains = 2, seed = 123)
})
mu_samples_normal <- extract(fit_stan_normal)$mu
mean_mcmc_normal  <- mean(mu_samples_normal)
sd_mcmc_normal    <- sd(mu_samples_normal)

cat(sprintf("Q-VMP:\n", mu_opt_normal, sqrt(sigma2_opt_normal)))
cat(sprintf("MCMC:\n", mean_mcmc_normal, sd_mcmc_normal))
cat(sprintf("Q-VMP wall-clock time\n", t_qvmp_normal["elapsed"]))
cat(sprintf("Stan/MCMC wall-clock time\n", t_mcmc_normal["elapsed"]))
cat(sprintf("Speed-up (MCMC / Q-VMP)\n", t_mcmc_normal["elapsed"] / t_qvmp_normal["elapsed"]))

#pdf("fig_normal.pdf", width = 6, height = 5)
tiff("fig_normal.tiff", width = 6, height = 5, units = "in", res = 600, compression = "lzw")
col_qvmp    <- "#222222"
col_mcmc_bg <- "#F2F2F2"
col_mcmc_fg <- "#888888"

hist(mu_samples_normal, breaks = 40, probability = TRUE, main = "", 
     xlab = expression(theta), col = col_mcmc_bg, border = col_mcmc_fg)

curve(dnorm(x, mean = mu_opt_normal, sd = sqrt(sigma2_opt_normal)),
      col = col_qvmp, lty = 1, lwd = 2.5, add = TRUE)

legend("topright", legend = c("Q-VMP", "MCMC"), col = c(col_qvmp, col_mcmc_fg),
       lty = c(1, NA), lwd = c(2.5, NA), pch = c(NA, 15), bty = "n", cex = 0.8)

dev.off()

# saveRDS(list(mu_opt = mu_opt_normal, sigma2_opt = sigma2_opt_normal,
#              mean_mcmc = mean_mcmc_normal, sd_mcmc = sd_mcmc_normal,
#              t_qvmp = t_qvmp_normal, t_mcmc = t_mcmc_normal),
#         "sim_normal_results.rds")

## ============================================================
## Example 2: Normal likelihood, beta prior and variational
## ============================================================
set.seed(123)
n2       <- 1000
true_mu2 <- 0.6
x2    <- rnorm(n2, mean = true_mu2, sd = 1)
xbar2 <- mean(x2)

## F_hat_beta(eta1, eta2): the free energy of eq. (8),
##   \hat F(eta1,eta2) = eta1(psi(eta1+1)-psi(eta1+eta2+2)) + eta2(psi(eta2+1)-psi(eta1+eta2+2))
##                       + logGamma(eta1+eta2+2) - logGamma(eta1+1) - logGamma(eta2+1)
##                       - n[ (eta1+1)/(eta1+eta2+2) xbar
##                            - (1/2)(eta1+1)(eta1+2) / ((eta1+eta2+2)(eta1+eta2+3)) ]
F_hat_beta <- function(eta) {
  eta1 <- eta[1]
  eta2 <- eta[2]

  psi_sum  <- digamma(eta1 + eta2 + 2)
  psi_eta1 <- digamma(eta1 + 1)
  psi_eta2 <- digamma(eta2 + 1)

  kl_term      <- eta1 * (psi_eta1 - psi_sum) + eta2 * (psi_eta2 - psi_sum)        # KL (prior/q) part
  logpart_term <- lgamma(eta1 + eta2 + 2) - lgamma(eta1 + 1) - lgamma(eta2 + 1)    # log-partition part
  loglik_term  <- n2 * (
    (eta1 + 1) / (eta1 + eta2 + 2) * xbar2 -
      0.5 * (eta1 + 1) * (eta1 + 2) / ((eta1 + eta2 + 2) * (eta1 + eta2 + 3))
  )                                                                                # sum_i E_q[ell(theta;y_i)]

  kl_term + logpart_term - loglik_term
}

t_qvmp_beta <- system.time({
  fit_beta <- optim(c(3, 4), F_hat_beta, method = "L-BFGS-B", lower = c(-0.9, -0.9))
})
eta_opt   <- fit_beta$par
alpha_opt <- eta_opt[1] + 1
beta_opt  <- eta_opt[2] + 1

Mean_qvmp_beta <- alpha_opt / (alpha_opt + beta_opt)
Var_qvmp_beta  <- (alpha_opt * beta_opt) /
  ((alpha_opt + beta_opt)^2 * (alpha_opt + beta_opt + 1))

stan_data_beta <- list(N = n2, x = x2)
t_mcmc_beta <- system.time({
  fit_stan_beta <- stan(model_code = "
data {
  int<lower=0> N;
  real x[N];
}
parameters {
  real<lower=0, upper=1> mu;
}
model {
  // Uniform prior on [0,1] is Beta(1,1)
  x ~ normal(mu, 1);
}
", data = stan_data_beta, iter = 10000, chains = 2, seed = 123)
})
mu_samples_beta <- extract(fit_stan_beta)$mu
mean_mcmc_beta  <- mean(mu_samples_beta)
sd_mcmc_beta    <- sd(mu_samples_beta)


cat(sprintf("Q-VMP: mean\n",Mean_qvmp_beta, sqrt(Var_qvmp_beta), alpha_opt, beta_opt))
cat(sprintf("MCMC:  mean\n", mean_mcmc_beta, sd_mcmc_beta))
cat(sprintf("Q-VMP wall-clock time\n", t_qvmp_beta["elapsed"]))
cat(sprintf("Stan/MCMC wall-clock time\n", t_mcmc_beta["elapsed"]))
cat(sprintf("Speed-up (MCMC / Q-VMP)\n", t_mcmc_beta["elapsed"] / t_qvmp_beta["elapsed"]))

#pdf("fig_beta.pdf", width = 6, height = 5)
tiff("fig_beta.tiff", width = 9, height = 6, units = "in", res = 600, compression = "lzw")
hist(mu_samples_beta, breaks = 40, probability = TRUE, main = "", 
     xlab = expression(theta), col = col_mcmc_bg, border = col_mcmc_fg)

curve(dbeta(x, alpha_opt, beta_opt), col = col_qvmp, lty = 1, lwd = 2.5, add = TRUE)

legend("topright", legend = c("Q-VMP", "MCMC"), col = c(col_qvmp, col_mcmc_fg),
       lty = c(1, NA), lwd = c(2.5, NA), pch = c(NA, 15), bty = "n", cex = 0.8)

dev.off()

saveRDS(list(alpha_opt = alpha_opt, beta_opt = beta_opt,
             mean_mcmc = mean_mcmc_beta, sd_mcmc = sd_mcmc_beta,
             t_qvmp = t_qvmp_beta, t_mcmc = t_mcmc_beta),
        "sim_beta_results.rds")

## ============================================================
## Example 3: Logistic regression, multivariate normal prior
## and variational family 
## ============================================================
set.seed(123)
n <- 1000
p <- 9
X <- cbind(1, matrix(rnorm(n * (p - 1)), n, p - 1))
beta_true <- c(0.5, -1, 0.2, 0.8, -0.5, 1.2, -0.3, 0.6, -0.9)

sigmoid <- function(z) 1 / (1 + exp(-z))
y <- rbinom(n, 1, sigmoid(X %*% beta_true))

Sigma0_inv <- diag(0.01, p)
mu0        <- rep(0, p)


F_hat <- function(par) {
  mu <- par[1:p]                                    # mu

  L   <- matrix(0, p, p)                            # Cholesky factor: Sigma = L L'
  idx <- p + 1
  for (i in 1:p) {
    for (j in 1:i) {
      L[i, j] <- par[idx]
      idx <- idx + 1
    }
  }
  diag(L) <- exp(diag(L))
  Sigma        <- L %*% t(L)                        # Sigma
  logdet_Sigma <- 2 * sum(log(diag(L)))             # log|Sigma|

  z_mu <- as.vector(X %*% mu)                       # x_i' mu
  s2   <- rowSums((X %*% Sigma) * X)                # x_i' Sigma x_i
  p_mu <- sigmoid(z_mu)                             # p_i = sigmoid(x_i' mu)

  # Example 3 free energy, additive constants in (mu,Sigma) dropped, term by term:
  trace_term  <- 0.5 * sum(Sigma0_inv * Sigma)                       #  (1/2) tr(Sigma0^{-1} Sigma)
  quad_term   <- 0.5 * as.numeric(t(mu) %*% Sigma0_inv %*% mu) -
                 as.numeric(t(mu0) %*% Sigma0_inv %*% mu)            #  (1/2) mu' Sigma0^{-1} mu - mu0' Sigma0^{-1} mu
  logdet_term <- -0.5 * logdet_Sigma                                 # -(1/2) log|Sigma|
  loglik_term <- sum(y * z_mu - log(1 + exp(z_mu)) - 0.5 * s2 * p_mu * (1 - p_mu))  # sum_i E_q[ell(theta*; y_i)]

  trace_term + quad_term + logdet_term - loglik_term
}

L_flat_init <- rep(0, p * (p + 1) / 2)
init_par <- c(rep(0, p), L_flat_init)

t_qvmp <- system.time({
  fit <- optim(par = init_par, fn = F_hat, method = "BFGS",
               control = list(maxit = 10000, reltol = 1e-12))
})

mu_opt <- fit$par[1:p]
L_opt  <- matrix(0, p, p)
idx    <- p + 1
for (i in 1:p) {
  for (j in 1:i) {
    L_opt[i, j] <- fit$par[idx]
    idx <- idx + 1
  }
}
diag(L_opt) <- exp(diag(L_opt))
Cov <- L_opt %*% t(L_opt)

beta_est <- mu_opt
sd_qvmp  <- sqrt(diag(Cov))

## No MCMC here: the MCMC (Stan NUTS) reference for this model is run once,
## in run_exampleBLR.R, and used for both the BLR and this quasi-Newton
## cross-check comparison, so it is not duplicated in this file.
summary_tab <- data.frame(
  beta      = sprintf("beta_%d", 0:(p - 1)),
  true      = beta_true,
  qvmp_mean = beta_est,
  qvmp_sd   = sd_qvmp
)
print(summary_tab, digits = 3, row.names = FALSE)

cat(sprintf("\nQ-VMP (quasi-Newton) wall-clock time: %.3f s\n", t_qvmp["elapsed"]))

saveRDS(list(summary_tab = summary_tab, t_qvmp = t_qvmp),
        "sim_logistic_resultsChole.rds")

## ============================================================
## Example 4: Robust spatial binomial regression, heavy-tailed
## (Student-t) field; the range phi is held FIXED at phi_true.
## Direct quasi-Newton minimization of the free
## energy F(mu, Sigma, a, b) of eq. (Fjoint) over the Cholesky
## factor of Sigma. 
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


## phi is FIXED at phi_true (not estimated), R(phi) and its inverse are constants.
Rpi_c     <- solve(R_true)
logdetR_c <- as.numeric(determinant(R_true, logarithm = TRUE)$modulus)

nL <- d * (d + 1) / 2

build_L <- function(Lvec) {
  L <- matrix(0, d, d)
  L[lower.tri(L, diag = TRUE)] <- Lvec
  diag(L) <- exp(diag(L))
  L
}

## eq. (Fjoint) with phi fixed; additive constants dropped:
##   F = -1/2 log|Sigma|                                            (T1)
##       + gamma/2 ( mu' B mu + tr{B Sigma} )                       (T2)
##       + 1/2 log|R|  +  n/2 ( log b - psi(a) )                    (T3, T4)
##       - sum_i [ y_i xt_i'mu - m_i log(1+e^{xt_i'mu})
##                 - 1/2 m_i p_i(1-p_i) xt_i' Sigma xt_i ]          (T5)
##       + KL{IG(a,b) || IG(alpha/2, alpha/2)}                      (T6)
##   gamma = a/b;  B = blkdiag(0_{p_beta}, R^{-1});  xt_i = i-th row of X = [Xb | I_n]
negF_full <- function(par) {
  mu      <- par[1:d]
  Lvec    <- par[(d + 1):(d + nL)]
  log_a   <- par[d + nL + 1]; log_b <- par[d + nL + 2]
  a <- exp(log_a); b <- exp(log_b); gam <- a / b

  L   <- build_L(Lvec)
  Sig <- tcrossprod(L)
  S0i <- matrix(0, d, d)                       # gamma * B: beta block 0 (flat prior),
  S0i[ui, ui] <- gam * Rpi_c                   # u block gamma * R^{-1}  

  eta <- as.vector(X %*% mu); p_i <- sigmoid(eta)
  W   <- pmax(m * p_i * (1 - p_i), 1e-8)
  qd  <- rowSums((X %*% Sig) * X)              # xt_i' Sigma xt_i
  ll  <- sum(y * eta - m * log1p(exp(eta))) - 0.5 * sum(W * qd)

  prior       <- 0.5 * sum(S0i * Sig) + 0.5 * sum(mu * (S0i %*% mu)) -
    0.5 * as.numeric(determinant(Sig, logarithm = TRUE)$modulus)   # = T2 + T1
  leftover    <- 0.5 * n * (log(b) - digamma(a))                   # = T4
  logdet_term <- 0.5 * logdetR_c                                   # = T3

  kl <- a * log(b) - (alpha / 2) * log(alpha / 2) - lgamma(a) + lgamma(alpha / 2) +
    (alpha / 2 - a) * (log(b) - digamma(a)) + (alpha / 2 - b) * (a / b)   # = T6

  prior + leftover + logdet_term - ll + kl
}

## starting point: mu = 0, Sigma = I (L = I), a = b = alpha/2 (gamma = 1)
par0 <- rep(0, d + nL + 2)
par0[d + nL + 1] <- log(alpha / 2)
par0[d + nL + 2] <- log(alpha / 2)

t_qvmp <- system.time({
  fit <- optim(par0, negF_full, method = "BFGS",
               control = list(maxit = 500, reltol = 1e-6))
})

mu_hat   <- fit$par[1:d]
L_hat    <- build_L(fit$par[(d + 1):(d + nL)])
Sig_hat  <- tcrossprod(L_hat)
beta_hat <- mu_hat[1:p_beta]
beta_sd  <- sqrt(diag(Sig_hat)[1:p_beta])
u_hat    <- mu_hat[ui]
a_hat    <- exp(fit$par[d + nL + 1]); b_hat <- exp(fit$par[d + nL + 2])
gam_hat  <- a_hat / b_hat

## No MCMC here: the MCMC (Stan NUTS) reference for this model is run once,
## in run_exampleBLR.R, and used for both the BLR and this quasi-Newton
## cross-check comparison, so it is not duplicated in this file.
summary_tab <- data.frame(
  param     = sprintf("beta_%d", 0:(p_beta - 1)),
  true      = beta_true,
  qvmp_mean = beta_hat,
  qvmp_sd   = beta_sd
)
print(summary_tab, digits = 3, row.names = FALSE)

cat(sprintf("\nQ-VMP (quasi-Newton) wall-clock : %.3f s  (%d parameters)\n",
            t_qvmp["elapsed"], length(par0)))

saveRDS(list(summary_tab = summary_tab, beta_hat = beta_hat, beta_sd = beta_sd,
             u_hat = u_hat, u_true = u_true, gam_hat = gam_hat,
             phi_true = phi_true, t_qvmp = t_qvmp, n = n, nparam = length(par0)),
        "sim_spatial_joint_resultsChole.rds")
