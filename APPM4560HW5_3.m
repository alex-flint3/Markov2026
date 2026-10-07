clear;
clc;
close all;


%% Part (a)

% Define the 8x8 transition matrix P without random jumps
% Rows: current state, Columns: next state
P = [  0, 1/2, 1/2,   0,   0,   0,   0,   0;   % Node 1 -> 2, 3
       0,   0, 1/2, 1/2,   0,   0,   0,   0;   % Node 2 -> 3, 4
     1/2,   0,   0,   0, 1/2,   0,   0,   0;   % Node 3 -> 1, 5
     1/3,   0, 1/3,   0, 1/3,   0,   0,   0;   % Node 4 -> 1, 3, 5
       0, 1/3,   0,   0,   0, 1/3, 1/3,   0;   % Node 5 -> 2, 6, 7
       0,   0,   0,   0,   0,   1,   0,   0;   % Node 6 -> 6
       0,   0,   0,   0,   0,   0,   0,   1;   % Node 7 -> 8
       0,   0,   0,   0,   0,   0,   1,   0 ]; % Node 8 -> 7

% Initial uniform distribution q0
q0 = (1/8) * ones(1, 8);

% Power iteration for 101 steps
q = q0;
q_history = zeros(102, 8);
q_history(1, :) = q;

for n = 1:101
    q = q * P;
    q_history(n+1, :) = q;
end

q100 = q_history(101, :);
q101 = q_history(102, :);


%% Part (b)

d = 0.85;
G = d * P + ((1 - d) / 8) * ones(8, 8);

% Power iteration starting from uniform distribution
q_n = q0;
iter_count = 0;
tol = 1e-10;

while true
    q_next = q_n * G;
    iter_count = iter_count + 1;
    if norm(q_next - q_n, 1) < tol
        q_n = q_next;
        break;
    end
    q_n = q_next;
end

pi_power = q_n;

% Verification with linear solve: pi * (I - G) = 0 and sum(pi) = 1
A = [(eye(8) - G)'; ones(1, 8)];
b = [zeros(8, 1); 1];
pi_solve = (A \ b)';

[~, ranking] = sort(pi_power, 'descend');



%% Part (c)

R = 100;         % Number of surfers
T = 10^5;        % Total time steps
start_node = 1;  % All surfers start at node 1

% Precompute CDF matrix of G for vectorized inverse transform sampling
% Cumsum along rows (dimension 2)
G_cdf = cumsum(G, 2);

% Tracking visit counts across all surfers:
% state_counts(r, i) stores total visits to node i by surfer r
state_counts = zeros(R, 8);

% Array of current states for all R surfers (starts at node 1)
current_states = repmat(start_node, R, 1);

% Compute maximum absolute error for node 6 at sampled log-spaced time steps
sample_steps = round(logspace(2, 5, 50));
rms_error = zeros(length(sample_steps), 1);
sample_idx = 1;

% Pre-generate surfer state counts for RMS plot evaluation
% Vectorized time-stepping across all R surfers
surfer_paths = zeros(R, T + 1);
surfer_paths(:, 1) = start_node;

for t = 1:T
    % Draw uniform random numbers for all R surfers
    u = rand(R, 1);
    
    % Vectorized inverse transform sampling across rows of G_cdf
    % Find first column index where CDF >= u for each surfer's current state
    cdf_rows = G_cdf(current_states, :);
    next_states = sum(u > cdf_rows, 2) + 1;
    
    current_states = next_states;
    surfer_paths(:, t + 1) = current_states;
end

% Compute RMS error across time steps for log-log plot
for k = 1:length(sample_steps)
    t_curr = sample_steps(k);
    % Submatrix of visits up to step t_curr
    paths_t = surfer_paths(:, 1:t_curr+1);
    
    % Compute empirical pi_hat for each surfer at step t_curr
    pi_hat_t = zeros(R, 8);
    for r = 1:R
        pi_hat_t(r, :) = histcounts(paths_t(r, :), 1:9) / (t_curr + 1);
    end
    
    % Max error across all nodes per surfer, then RMS over surfers
    max_err_per_surfer = max(abs(pi_hat_t - pi_power), [], 2);
    rms_error(k) = sqrt(mean(max_err_per_surfer.^2));
end

% Final empirical stationary distribution estimate (averaged over all surfers at T = 10^5)
pi_hat_surfer1 = histcounts(surfer_paths(1, :), 1:9) / (T + 1);
pi_hat_avg = histcounts(surfer_paths(:), 1:9) / (R * (T + 1));



% 1. Bar Chart: One surfer's estimate vs True Stationary Distribution
figure('Name', 'Part (c) - Surfer Estimate vs True Pi', 'NumberTitle', 'off');
bar_data = [pi_hat_surfer1', pi_power'];
bar(1:8, bar_data);
xlabel('Page / Node ID');
ylabel('Probability Mass');
title('PageRank: Single Surfer Estimate (\hat{\pi}) vs True \pi (T = 10^5)');
legend({'Surfer 1 Estimate \hat{\pi}', 'True Stationary \pi'}, 'Location', 'NorthWest');
grid on;

% 2. Log-Log Plot: RMS Error vs T with fitted line (10^2 <= T <= 10^5)
figure('Name', 'Part (c) - RMS Error Convergence', 'NumberTitle', 'off');
loglog(sample_steps, rms_error, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;

% Linear fit in log-log space over 10^2 <= T <= 10^5
log_T = log10(sample_steps)';
log_err = log10(rms_error);
p_fit = polyfit(log_T, log_err, 1);
fitted_slope = p_fit(1);

fitted_y = 10.^(polyval(p_fit, log_T));
loglog(sample_steps, fitted_y, 'r--', 'LineWidth', 2);

xlabel('Time Steps T');
ylabel('RMS over surfers of max_i |\hat{\pi}_i - \pi_i|');
title(sprintf('Convergence Rate: Fitted Slope = %.3f (Expected \\approx -0.5)', fitted_slope));
legend({'Simulation RMS Error', sprintf('Fitted Line (Slope = %.3f)', fitted_slope)}, 'Location', 'SouthWest');
grid on;

