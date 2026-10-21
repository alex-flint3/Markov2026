clear;
clc;
close all;


rng(42);

alpha_true = 1.5;
beta_true = 0.8;
n_vals = [10, 100, 1000];
num_sims = 2000;

iqr_mean = zeros(length(n_vals), 1);
iqr_median = zeros(length(n_vals), 1);

fprintf('--- Estimator Scatter Analysis ---\n');
for i = 1:length(n_vals)
    n = n_vals(i);

    % Generate Cauchy samples: alpha + beta * tan(Uniform(-pi/2, pi/2))
    theta = pi * (rand(num_sims, n) - 0.5);
    samples = alpha_true + beta_true * tan(theta);

    % Compute sample means and medians across each generated dataset
    means = mean(samples, 2);
    medians = median(samples, 2);

    % Measure scatter via IQR / 1.349
    iqr_mean(i) = (quantile(means, 0.75) - quantile(means, 0.25)) / 1.349;
    iqr_median(i) = (quantile(medians, 0.75) - quantile(medians, 0.25)) / 1.349;

    fprintf('n = %4d | Sample Mean Scatter: %8.3f | Sample Median Scatter: %6.3f\n', ...
        n, iqr_mean(i), iqr_median(i));
end