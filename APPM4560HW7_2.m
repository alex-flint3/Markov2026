clear;
clc;
close all;
rng(42);

%% 1. Dynamic Programming over 2^6 states (Group absorbing Markov chain)
lambda = [3.0, 2.0, 1.5, 1.2, 1.0, 0.75];
n = length(lambda);

% Map subset mask (1 to 2^n - 1) to expected remaining time
E_rem = zeros(1, 2^n - 1);

% Evaluate subsets by size from 1 to n
subsets = 1:(2^n - 1);
[~, idx] = sort(arrayfun(@(x) sum(dec2bin(x)=='1'), subsets));
subsets = subsets(idx);

for S = subsets
    members = find(bitget(S, 1:n));
    rate_sum = sum(lambda(members));
    
    if isscalar(members)
        E_rem(S) = 1 / rate_sum;
    else
        val = 1 / rate_sum;
        for i = members
            S_next = bitset(S, i, 0);
            val = val + (lambda(i) / rate_sum) * E_rem(S_next);
        end
        E_rem(S) = val;
    end
end
EM_exact_dp = E_rem(2^n - 1);

% Compute Probability each student is last via dynamic programming
P_last_state = zeros(n, 2^n - 1);
for S = subsets
    members = find(bitget(S, 1:n));
    rate_sum = sum(lambda(members));
    
    if length(members) == 1
        P_last_state(members(1), S) = 1.0;
    else
        for i = members
            S_next = bitset(S, i, 0);
            P_last_state(:, S) = P_last_state(:, S) + (lambda(i) / rate_sum) * P_last_state(:, S_next);
        end
    end
end
p_last_dp = P_last_state(:, 2^n - 1);

%% 2. Exact E[M] via Alternating Inclusion-Exclusion Formula
EM_exact_alt = 0;
for k = 1:n
    combos = nchoosek(1:n, k);
    for r = 1:size(combos, 1)
        group_rate = sum(lambda(combos(r, :)));
        EM_exact_alt = EM_exact_alt + ((-1)^(k+1)) / group_rate;
    end
end

%% 3. Monte Carlo Simulation (2 x 10^5 exams)
N_sim = 200000;
X_sim = -log(rand(N_sim, n)) ./ lambda; % Generate Exp(lambda_i) matrix
M_sim = max(X_sim, [], 2);
[~, last_idx_sim] = max(X_sim, [], 2);
[T1_sim, first_idx_sim] = min(X_sim, [], 2);

EM_sim = mean(M_sim);
p_last_sim = histcounts(last_idx_sim, 1:(n+1))' / N_sim;

%% Display Results Comparison
fprintf('--- Problem 2(d) Numerical Results ---\n');
fprintf('Exact E[M] (Dynamic Programming): %.5f hours\n', EM_exact_dp);
fprintf('Exact E[M] (Alternating Formula) : %.5f hours\n', EM_exact_alt);
fprintf('Simulated E[M] (%d runs)  : %.5f hours\n\n', N_sim, EM_sim);
fprintf('Student | Rate (lambda) | Exact Prob Last | Simulated Prob Last\n');
fprintf('--------------------------------------------------------------\n');
for i = 1:n
    fprintf('   %d    |     %5.2f     |     %7.4f     |      %7.4f\n', ...
        i, lambda(i), p_last_dp(i), p_last_sim(i));
end

%% Check Independence of T_1 and First Student Index
fprintf('\n--- Check First Student Finish Time Independence ---\n');
for i = 1:n
    mean_T1_given_i = mean(T1_sim(first_idx_sim == i));
    fprintf('Mean T_1 given Student %d finished first: %.4f (Theoretical Exp(Lambda) mean = %.4f)\n', ...
        i, mean_T1_given_i, 1/sum(lambda));
end

%% Plot 1: Histogram of M vs Exact Density
figure(1); % Explicit figure handle
% Fixed 'Bins' -> 'NumBins' or standard positional argument 80
histogram(M_sim, 80, 'Normalization', 'pdf', 'EdgeColor', 'none', 'FaceAlpha', 0.6);
hold on;

t_grid = linspace(0, max(M_sim), 400);

% Density f(t) = d/dt prod_i (1 - exp(-lambda_i * t))
f_density = zeros(size(t_grid));
for i = 1:n
    term = lambda(i) * exp(-lambda(i) * t_grid);
    for j = 1:n
        if j ~= i
            term = term .* (1 - exp(-lambda(j) * t_grid));
        end
    end
    f_density = f_density + term;
end

plot(t_grid, f_density, 'r-', 'LineWidth', 2);
xlabel('M (hours)');
ylabel('Probability Density');
title('Histogram of Last Student Finish Time M vs. Exact Density');
legend('Simulation Histogram', 'Exact Density');
grid on;

%% Plot 2: E[M] for lambda_k = k for k = 1 to 12
k_vals = 1:12;
EM_k = zeros(length(k_vals), 1);
for k = 1:12
    rates = 1:k;
    em_val = 0;
    for m = 1:k
        combos = nchoosek(rates, m);
        for r = 1:size(combos, 1)
            em_val = em_val + ((-1)^(m+1)) / sum(combos(r, :));
        end
    end
    EM_k(k) = em_val;
end

figure(2); % Explicit figure handle
plot(k_vals, EM_k, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b'); hold on;
yline(2 * log(2), 'r--', 'LineWidth', 1.5, 'Label', '2 ln 2 Upper Bound \approx 1.3863');
xlabel('Number of Students (k)');
ylabel('E[M]');
title('E[M] vs. k for Rates \lambda_i = i');
grid on;