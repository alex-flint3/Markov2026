% MATLAB Script for Homework 8, Question 3(c) (Base MATLAB - No Toolboxes Required)
clear; clc; rng(100);

%% Simulation Setup
T_total = 720; % 30 days * 24 hours
lambda_max = 20;

% True intensity function
lambda_fun = @(t) 12 + 8 * cos(pi * (t - 16) / 12);
Lambda_true = @(s, t) 12*(t - s) + (96/pi)*(sin(pi*(t - 16)/12) - sin(pi*(s - 16)/12));

%% 1. Generate Arrivals via Thinning
% Step 1: Homogeneous Poisson process with rate lambda_max (Custom Base Poisson Generator)
N_proposals = my_poissrnd(lambda_max * T_total);
proposal_times = sort(rand(N_proposals, 1) * T_total);

% Step 2: Accept with probability lambda(t) / lambda_max
accept_probs = lambda_fun(proposal_times) / lambda_max;
keep_mask = (rand(N_proposals, 1) < accept_probs);

event_times = proposal_times(keep_mask);
n_events = length(event_times);

fprintf('==================================================\n');
fprintf('         QUESTION 3(c) SIMULATION RESULTS         \n');
fprintf('==================================================\n\n');

fprintf('1. PROPOSAL EFFICIENCY:\n');
fprintf('   Proposed Points: %d\n', N_proposals);
fprintf('   Kept Points:     %d\n', n_events);
fprintf('   Simulated Kept Fraction: %.4f (Theoretical: 0.6000)\n\n', n_events / N_proposals);

%% 2. Arrivals from 3 AM to 4 AM over 30 Days
% Hours 3 to 4 on day d correspond to [24*(d-1) + 3, 24*(d-1) + 4]
counts_3to4 = zeros(30, 1);
for d = 1:30
    t_start = 24*(d-1) + 3;
    t_end   = 24*(d-1) + 4;
    counts_3to4(d) = sum(event_times >= t_start & event_times < t_end);
end

lambda_3to4_true = Lambda_true(3, 4);
fprintf('2. 3 AM TO 4 AM STATISTICS (Over 30 Days):\n');
fprintf('   Simulated Mean:     %.4f | Theoretical Poisson(4.091): %.4f\n', mean(counts_3to4), lambda_3to4_true);
fprintf('   Simulated Variance: %.4f | Theoretical Poisson(4.091): %.4f\n\n', var(counts_3to4), lambda_3to4_true);

%% 3. Plot Mean Arrivals in Each Clock Hour
hourly_counts = zeros(24, 1);
for h = 0:23
    % Sum across all 30 days for clock hour h
    for d = 1:30
        t_s = 24*(d-1) + h;
        t_e = 24*(d-1) + h + 1;
        hourly_counts(h+1) = hourly_counts(h+1) + sum(event_times >= t_s & event_times < t_e);
    end
end
mean_hourly_sim = hourly_counts / 30;

% Plotting
figure('Name', 'ER Hourly Arrivals', 'Color', 'w');
t_grid = linspace(0, 24, 200);
bar(0.5:1:23.5, mean_hourly_sim, 1, 'FaceColor', [0.7 0.85 1], 'EdgeColor', 'b');
hold on;
plot(t_grid, lambda_fun(t_grid), 'r-', 'LineWidth', 2);
xlabel('Clock Hour of Day (t)');
ylabel('Average Arrivals / Hour');
title('Mean Hourly ER Arrivals over 30 Days');
legend('Simulated Hourly Mean', 'True \lambda(t)', 'Location', 'NorthWest');
grid on;

%% 4. Maximum Likelihood Estimation (MLE)
% Target: lambda(t; a, b, phi) = a + b * cos(pi * (t - phi) / 12)
% Log-Likelihood: sum_i log(lambda(t_i)) - Lambda(0, T_total)
% Integrated Lambda(0, T_total) for 30 full 24-hr cycles is simply a * T_total

neg_log_likelihood = @(params) - ( ...
    sum(log(max(params(1) + params(2) * cos(pi * (mod(event_times, 24) - params(3)) / 12), 1e-6))) ...
    - params(1) * T_total );

initial_guess = [10, 5, 14];
% Replaced fminunc & optimoptions with base MATLAB's fminsearch
opts = optimset('Display', 'off');
params_opt = fminsearch(neg_log_likelihood, initial_guess, opts);

LL_opt = -neg_log_likelihood(params_opt);

% Log-likelihood under Constant Rate model (lambda = n / T_total)
lambda_const = n_events / T_total;
LL_const = sum(log(repmat(lambda_const, n_events, 1))) - lambda_const * T_total;

fprintf('3. MAXIMUM LIKELIHOOD FITTING:\n');
fprintf('   Initial Guess (a, b, phi):  (10.0, 5.0, 14.0)\n');
fprintf('   MLE Estimates (a, b, phi):  (%.3f, %.3f, %.3f)  [True: (12, 8, 16)]\n', ...
    params_opt(1), params_opt(2), params_opt(3));
fprintf('   Maximized Log-Likelihood (Cyclic Model):   %.2f\n', LL_opt);
fprintf('   Log-Likelihood (Constant Rate Model):      %.2f\n', LL_const);
fprintf('   Log-Likelihood Ratio Gain:                  %.2f\n', LL_opt - LL_const);

%% Custom Helper Functions (Base MATLAB replacements)

function k = my_poissrnd(lambda)
    % Base MATLAB Poisson generator for large mean using Gaussian approximation
    % or exact inter-arrival accumulation for small values.
    if lambda > 100
        % Central Limit Theorem approximation for large values (lambda = 14400 here)
        k = round(lambda + sqrt(lambda) * randn());
    else
        L = exp(-lambda);
        k = 0;
        p = 1;
        while p > L
            k = k + 1;
            p = p * rand();
        end
        k = k - 1;
    end
end