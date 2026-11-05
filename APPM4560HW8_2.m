% MATLAB Script for Homework 8, Question 2(c) (Base MATLAB - No Toolbox Required)
clear; clc; rng(42); % Set seed for reproducibility

N_sim = 2e5; % Number of 5-second window simulations
T_window = 5.0;

%% 1. Simulation Method 1: Emit all photons, then thin (q = 0.4)
% Poisson generator helper function using inverse transform sampling
star_raw_counts = my_poissrnd(3 * T_window, N_sim);
sky_raw_counts  = my_poissrnd(2 * T_window, N_sim);

% Thinning step (Binomial sampling with q = 0.4 without binornd)
star_det_1 = my_binornd(star_raw_counts, 0.4);
sky_det_1  = my_binornd(sky_raw_counts, 0.4);
tot_det_1  = star_det_1 + sky_det_1;

%% 2. Simulation Method 2: Generate detected processes directly
% Rates: lambda_star_det = 1.2, lambda_sky_det = 0.8
star_det_2 = my_poissrnd(1.2 * T_window, N_sim);
sky_det_2  = my_poissrnd(0.8 * T_window, N_sim);
tot_det_2  = star_det_2 + sky_det_2;

%% 3. First Detection Times & Types Simulation
% Exponential generation using Inverse Transform: T = -mean * log(rand)
T_star_first = -(1 / 1.2) * log(rand(N_sim, 1));
T_sky_first  = -(1 / 0.8) * log(rand(N_sim, 1));

% Compute first detection time T1, max time M, and type indicator
T1 = min(T_star_first, T_sky_first);
M  = max(T_star_first, T_sky_first);
is_star_first = (T_star_first < T_sky_first); % 1 if star first, 0 if sky

%% Output & Comparisons
fprintf('==================================================\n');
fprintf('         QUESTION 2(c) SIMULATION RESULTS         \n');
fprintf('==================================================\n\n');

fprintf('1. COUNT STATISTICS (5s Window):\n');
fprintf('   Method 1 (Thinned):   Star Mean = %.4f (Var = %.4f), Sky Mean = %.4f (Var = %.4f)\n', ...
    mean(star_det_1), var(star_det_1), mean(sky_det_1), var(sky_det_1));
fprintf('   Method 2 (Direct):    Star Mean = %.4f (Var = %.4f), Sky Mean = %.4f (Var = %.4f)\n', ...
    mean(star_det_2), var(star_det_2), mean(sky_det_2), var(sky_det_2));
fprintf('   Theoretical:          Star Mean = %.4f (Var = %.4f), Sky Mean = %.4f (Var = %.4f)\n\n', ...
    1.2*5, 1.2*5, 0.8*5, 0.8*5);

fprintf('2. CORRELATION BETWEEN DETECTED STAR & SKY COUNTS:\n');
fprintf('   Method 1 Corr: %.5f\n', my_corr(star_det_1, sky_det_1));
fprintf('   Method 2 Corr: %.5f\n', my_corr(star_det_2, sky_det_2));
fprintf('   Theoretical:   0.00000 (Independent Poisson Processes)\n\n');

fprintf('3. TOTAL DETECTIONS vs POISSON(10):\n');
fprintf('   Total Detections Mean = %.4f, Var = %.4f\n', mean(tot_det_2), var(tot_det_2));
fprintf('   Theoretical Poisson(10): Mean = 10.0000, Var = 10.0000\n\n');

fprintf('4. CONDITIONAL PROBABILITY P(6 Star | Total = 10):\n');
idx_10 = (tot_det_2 == 10);
p_6star_given_10_sim = mean(star_det_2(idx_10) == 6);
% Replacement for binopdf(6, 10, 0.6)
p_6star_given_10_theory = nchoosek(10, 6) * (0.6^6) * (0.4^4);
fprintf('   Simulated:   %.4f\n', p_6star_given_10_sim);
fprintf('   Theoretical: %.4f\n\n', p_6star_given_10_theory);

fprintf('5. FIRST DETECTION TIMES & TYPES:\n');
p_star_first_sim = mean(is_star_first);
p_star_first_and_1s_sim = mean(is_star_first & (T1 <= 1.0));
mean_M_sim = mean(M);
corr_T1_type = my_corr(T1, double(is_star_first));

fprintf('   P(Star First):                    Sim = %.4f | Theory = %.4f\n', p_star_first_sim, 0.6);
fprintf('   P(Star First & T1 <= 1s):         Sim = %.4f | Theory = %.4f\n', p_star_first_and_1s_sim, (1 - exp(-2)) * 0.6);
fprintf('   E[M] (Time to detect both):       Sim = %.4f s | Theory = %.4f s\n', mean_M_sim, 19/12);
fprintf('   Corr(First Arrival Time T1, Type): Sim = %.5f | Theory = 0.00000\n', corr_T1_type);

%% Custom Helper Functions (Base MATLAB replacements)

function X = my_poissrnd(lambda, N)
    % Generates N Poisson random numbers with mean lambda
    % using Knuth's algorithm (fast for small-to-moderate lambda)
    L = exp(-lambda);
    X = zeros(N, 1);
    for i = 1:N
        k = 0;
        p = 1;
        while p > L
            k = k + 1;
            p = p * rand();
        end
        X(i) = k - 1;
    end
end

function K = my_binornd(N_trials, p)
    % Vectorized Binomial generator for varying N per sample
    K = zeros(size(N_trials));
    max_N = max(N_trials);
    if max_N > 0
        % Generate logical matrix of successes
        R = rand(length(N_trials), max_N) < p;
        % Mask out trials beyond each row's specified count
        col_indices = 1:max_N;
        mask = col_indices <= N_trials;
        K = sum(R .* mask, 2);
    end
end

function r = my_corr(x, y)
    % Base MATLAB correlation coefficient implementation
    C = cov(x, y);
    r = C(1, 2) / (std(x) * std(y));
end