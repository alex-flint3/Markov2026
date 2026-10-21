clear;
clc;
close all;

rng(42); % Set random seed for reproducibility

% Given data
x = [-1.77, 2.47, 1.34, 2.18, 13.05, 1.60, 1.50, -1.97, 0.79, 1.50, 2.00, 2.63, 1.19, -2.31, 0.87];
n = length(x);

% Log target density log g(alpha, b) where b = log(beta)
log_g = @(a, b) n*b - sum(log(exp(b)^2 + (x - a).^2));

T = 50000;
steps = [0.02, 0.5, 5.0];

chains = cell(length(steps), 1);
acc_rates = zeros(length(steps), 1);
iats = zeros(length(steps), 1);

% Run MCMC for each step size
for k = 1:length(steps)
    s = steps(k);
    chain = zeros(T, 2); % [alpha, b]
    z_curr = [0; 0];
    log_g_curr = log_g(z_curr(1), z_curr(2));
    n_accept = 0;
    
    for t = 1:T
        prop = z_curr + s * randn(2, 1);
        log_g_prop = log_g(prop(1), prop(2));
        
        if log(rand()) < (log_g_prop - log_g_curr)
            z_curr = prop;
            log_g_curr = log_g_prop;
            n_accept = n_accept + 1;
        end
        chain(t, :) = z_curr';
    end
    
    chains{k} = chain;
    acc_rates(k) = n_accept / T;
    iats(k) = compute_iat(chain(:, 1));
end

% Display Acceptance Rates and IAT
fprintf('--- MCMC Diagnostics ---\n');
for k = 1:length(steps)
    fprintf('Step size s = %.2f: Acceptance Rate = %.1f%%, IAT(alpha) = %.2f\n', ...
        steps(k), acc_rates(k)*100, iats(k));
end

%% Plot 1: First 400 values of alpha for each s
figure;
for k = 1:length(steps)
    subplot(3, 1, k);
    plot(1:400, chains{k}(1:400, 1), 'LineWidth', 1.2);
    ylabel('\alpha');
    title(sprintf('Trace of First 400 Steps for s = %.2f (Acc Rate: %.1f%%)', ...
        steps(k), acc_rates(k)*100));
    grid on;
end
xlabel('Iteration (t)');

%% Best s analysis (s = 0.5), discarding first 5000 burn-in steps
best_idx = 2; % s = 0.5
burnin = 5000;
post_chain = chains{best_idx}(burnin+1:end, :);

alpha_mcmc = post_chain(:, 1);
b_mcmc = post_chain(:, 2);
beta_mcmc = exp(b_mcmc);

% MCMC Summary Stats
mean_alpha_mcmc = mean(alpha_mcmc);
ci_alpha_mcmc = quantile(alpha_mcmc, [0.05, 0.95]);

mean_b_mcmc = mean(b_mcmc);
ci_b_mcmc = quantile(b_mcmc, [0.05, 0.95]);

mean_beta_mcmc = mean(beta_mcmc);
ci_beta_mcmc = quantile(beta_mcmc, [0.05, 0.95]);

%% Numerical Grid Computation
a_grid = linspace(0.0, 3.0, 300);
b_grid = linspace(-1.0, 1.5, 300);
[A, B] = meshgrid(a_grid, b_grid);

log_G = zeros(size(A));
for i = 1:size(A, 1)
    for j = 1:size(A, 2)
        log_G(i, j) = log_g(A(i, j), B(i, j));
    end
end

G = exp(log_G - max(log_G(:)));
da = a_grid(2) - a_grid(1);
db = b_grid(2) - b_grid(1);
Z = sum(G(:)) * da * db;
pdf_grid = G / Z;

% Grid Marginals
pdf_a = sum(pdf_grid, 1) * db;
pdf_b = sum(pdf_grid, 2)' * da;

mean_a_grid = sum(a_grid .* pdf_a) * da;
cdf_a = cumsum(pdf_a) * da;
ci_a_grid = [a_grid(find(cdf_a >= 0.05, 1)), a_grid(find(cdf_a >= 0.95, 1))];

mean_b_grid = sum(b_grid .* pdf_b) * db;
cdf_b = cumsum(pdf_b) * db;
ci_b_grid = [b_grid(find(cdf_b >= 0.05, 1)), b_grid(find(cdf_b >= 0.95, 1))];

mean_beta_grid = sum(exp(b_grid) .* pdf_b) * db;

%% Display Parameter Estimates Comparison
fprintf('\n--- Parameter Estimation Comparison ---\n');
fprintf('alpha (MCMC): Mean = %.3f, 90%% CI = [%.3f, %.3f]\n', mean_alpha_mcmc, ci_alpha_mcmc(1), ci_alpha_mcmc(2));
fprintf('alpha (Grid): Mean = %.3f, 90%% CI = [%.3f, %.3f]\n', mean_a_grid, ci_a_grid(1), ci_a_grid(2));
fprintf('b = log(beta) (MCMC): Mean = %.3f, 90%% CI = [%.3f, %.3f]\n', mean_b_mcmc, ci_b_mcmc(1), ci_b_mcmc(2));
fprintf('b = log(beta) (Grid): Mean = %.3f, 90%% CI = [%.3f, %.3f]\n', mean_b_grid, ci_b_grid(1), ci_b_grid(2));
fprintf('Sample Mean of Data: %.3f\n', mean(x));
fprintf('Sample Median of Data: %.3f\n', median(x));

%% Plot 2: Posterior Contour with MCMC Samples
figure;
contour(A, B, pdf_grid, 10, 'LineWidth', 1.2); hold on;
scatter(alpha_mcmc(1:5:end), b_mcmc(1:5:end), 5, 'red', 'filled', 'MarkerFaceAlpha', 0.15);
xlabel('\alpha');
ylabel('b = \log(\beta)');
title('Posterior Samples (s = 0.5) over Exact Grid Contours');
legend('Grid Exact Posterior', 'MCMC Samples');
grid on;

%% Helper Function for Integrated Autocorrelation Time (IAT)
function tau = compute_iat(x)
    x = x - mean(x);
    N = length(x);
    [autocorr, lags] = xcorr(x, 'unbiased');
    autocorr = autocorr(lags >= 0) / autocorr(lags == 0);
    
    tau = 0.5;
    for M = 1:min(1000, length(autocorr)-1)
        tau = tau + autocorr(M+1);
        if M > 5 * tau
            break;
        end
    end
    tau = 1 + 2 * sum(autocorr(2:M+1));
end