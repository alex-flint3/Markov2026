clear; clc; close all;
rng(42);

%% Parameters
lambda = 4;
t_end = 1;
N_sim = 200000;
mu = lambda * t_end; % Mean = 4

%% Helper Function for Poisson Random Variates without Toolbox
% Inverse Transform Sampling for Poisson(mu)
custom_poissrnd = @(m, N) arrayfun(@(~) find(rand() <= cumsum(exp(-m) * (m.^(0:25)) ./ factorial(0:25)), 1) - 1, zeros(N,1));

%% Construction 1: Exponential Gaps
N1_exp = zeros(N_sim, 1);
T3_exp = zeros(N_sim, 1);
T1_given_3_exp = [];
N_half_given_3_exp = [];
N_first_half = zeros(N_sim, 1);
N_second_half = zeros(N_sim, 1);

for i = 1:N_sim
    % Generate exponential gaps until sum exceeds 1 hour
    t_curr = 0;
    count = 0;
    arrivals = [];
    while true
        gap = -log(rand()) / lambda;
        t_curr = t_curr + gap;
        if t_curr <= 1
            count = count + 1;
            arrivals(end+1) = t_curr;
        else
            break;
        end
    end
    N1_exp(i) = count;
    
    % Store arrival details
    N_first_half(i) = sum(arrivals <= 0.5);
    N_second_half(i) = sum(arrivals > 0.5);
    
    if count == 3
        T1_given_3_exp(end+1) = arrivals(1);
        N_half_given_3_exp(end+1) = sum(arrivals <= 0.5);
    end
    
    % Generate T3 (sum of 3 Exp(lambda) gaps)
    T3_exp(i) = sum(-log(rand(3,1)) / lambda);
end

%% Construction 2: Poisson Count + Uniform Points (No Toolbox)
N1_poi = custom_poissrnd(mu, N_sim);
T1_given_3_poi = zeros(sum(N1_poi == 3), 1);
N_half_given_3_poi = zeros(sum(N1_poi == 3), 1);

idx3 = 1;
for i = 1:N_sim
    k = N1_poi(i);
    pts = sort(rand(k, 1) * t_end);
    if k == 3
        T1_given_3_poi(idx3) = pts(1);
        N_half_given_3_poi(idx3) = sum(pts <= 0.5);
        idx3 = idx3 + 1;
    end
end

%% Compute Correlation using corrcoef (Base MATLAB)
C = corrcoef(N_first_half, N_second_half);
r_corr = C(1, 2);

%% Numerical Results Output
fprintf('--- Simulation Results (%d runs) ---\n', N_sim);
fprintf('Exp Gaps Construction    : Mean N(1) = %.4f, Var N(1) = %.4f\n', mean(N1_exp), var(N1_exp));
fprintf('Uniform Points Construction: Mean N(1) = %.4f, Var N(1) = %.4f\n\n', mean(N1_poi), var(N1_poi));

fprintf('--- Among Hours with N(1) = 3 ---\n');
fprintf('Estimated P(N(1/2) = 0 | N(1)=3) : %.4f (Theoretical = 0.1250)\n', mean(N_half_given_3_exp == 0));
fprintf('Estimated E[T1 | N(1)=3]        : %.4f (Theoretical = 0.2500)\n', mean(T1_given_3_exp));
fprintf('Correlation Corr(N(1/2), N(1)-N(1/2)): %.4f (Theoretical = 0.0000)\n\n', r_corr);

%% Figure 1: N(1) Histograms vs Poisson(4) PMF
figure(1);
max_k = max([N1_exp; N1_poi]);
bin_edges = -0.5:1:(max_k + 0.5);

subplot(1,2,1);
histogram(N1_exp, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceAlpha', 0.6);
hold on;
k_vals = 0:max_k;
poi_pmf = exp(-mu) * (mu.^k_vals) ./ factorial(k_vals);
plot(k_vals, poi_pmf, 'r--o', 'LineWidth', 1.5);
title('N(1) via Exponential Gaps');
xlabel('N(1)'); ylabel('Probability'); grid on;

subplot(1,2,2);
histogram(N1_poi, 'BinEdges', bin_edges, 'Normalization', 'probability', 'FaceAlpha', 0.6);
hold on;
plot(k_vals, poi_pmf, 'r--o', 'LineWidth', 1.5);
title('N(1) via Uniform Points');
xlabel('N(1)'); ylabel('Probability'); grid on;

%% Figure 2: Ten Realizations Raster Plot
figure(2);
subplot(2,1,1); hold on;
for i = 1:10
    t_c = 0; pts = [];
    while t_c <= 1
        t_c = t_c + (-log(rand())/lambda);
        if t_c <= 1, pts(end+1) = t_c; end
    end
    plot(pts, i * ones(size(pts)), 'k|', 'MarkerSize', 12, 'LineWidth', 2);
end
ylim([0, 11]); xlim([0, 1]); title('10 Realizations: Exponential Gaps');
xlabel('Time (hours)'); ylabel('Realization Index'); grid on;

subplot(2,1,2); hold on;
for i = 1:10
    k = custom_poissrnd(mu, 1);
    pts = sort(rand(k, 1));
    plot(pts, i * ones(size(pts)), 'b|', 'MarkerSize', 12, 'LineWidth', 2);
end
ylim([0, 11]); xlim([0, 1]); title('10 Realizations: Uniform Points');
xlabel('Time (hours)'); ylabel('Realization Index'); grid on;

%% Figure 3: Histogram of T3 vs Gamma(3, 4) Density
figure(3);
histogram(T3_exp, 80, 'Normalization', 'pdf', 'FaceAlpha', 0.6); hold on;
t_grid = linspace(0, max(T3_exp), 300);
f_gamma = (lambda^3 .* t_grid.^2 .* exp(-lambda .* t_grid)) / factorial(2);
plot(t_grid, f_gamma, 'r-', 'LineWidth', 2);
title('Histogram of T_3 vs. Gamma(3, 4) Density');
xlabel('T_3 (hours)'); ylabel('Density'); legend('Simulation', 'Theoretical Density'); grid on;

%% Total Variation Distance vs Delta t
dt_vals = 1 ./ (10:10:2000);
tv_dist = zeros(length(dt_vals), 1);

for i = 1:length(dt_vals)
    dt = dt_vals(i);
    n_binom = round(1 / dt);
    p_binom = lambda * dt;
    
    k = 0:15; % Support for TV distance calculation
    
    % Native Binomial PMF calculation
    p_bin = zeros(size(k));
    for j = 1:length(k)
        kj = k(j);
        p_bin(j) = nchoosek(n_binom, kj) * (p_binom^kj) * ((1 - p_binom)^(n_binom - kj));
    end
    
    % Native Poisson PMF
    p_poi = exp(-lambda) * (lambda.^k) ./ factorial(k);
    
    tv_dist(i) = 0.5 * sum(abs(p_bin - p_poi));
end

figure(4);
loglog(dt_vals, tv_dist, 'b-', 'LineWidth', 1.5); hold on;
% Fit slope on log-log axes
p_fit = polyfit(log(dt_vals), log(tv_dist'), 1);
loglog(dt_vals, exp(p_fit(2)) * dt_vals.^p_fit(1), 'r--', 'LineWidth', 1.5);
title(sprintf('Total Variation Distance vs \\Delta t (Fitted Slope = %.4f)', p_fit(1)));
xlabel('\Delta t'); ylabel('Total Variation Distance');
legend('TV Distance', sprintf('Log-Log Fit (Slope \\approx %.2f)', p_fit(1))); grid on;