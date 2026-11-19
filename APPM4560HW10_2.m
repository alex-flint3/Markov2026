%% Problem 2(c): Gillespie Simulation of M/M/infinity Bike Share (Base MATLAB)
rng(42); % Set seed for reproducibility

lambda = 12;          % Arrival rate (per hour)
mu = 3;               % Service rate (per hour, mean 20 min)
rho = lambda / mu;    % Stationary mean = 4
T = 500;              % Simulation horizon (hours)

% Initialize state and tracking arrays
t = 0;
n = 0;
max_states = 30;
time_in_state = zeros(1, max_states); % Index k+1 stores time in state k

% Gillespie algorithm loop
while t < T
    total_rate = lambda + n * mu;
    
    % Inverse transform sampling for Exponential(total_rate) without exprnd
    dt = -log(rand()) / total_rate;
    
    % Truncate final holding time at T
    if t + dt > T
        dt = T - t;
    end
    
    % Accumulate holding time in state n
    if n + 1 > length(time_in_state)
        time_in_state(n + 1) = 0;
    end
    time_in_state(n + 1) = time_in_state(n + 1) + dt;
    t = t + dt;
    
    % State transition if still before T
    if t < T
        if rand() < lambda / total_rate
            n = n + 1;
        else
            n = n - 1;
        end
    end
end

% Empirical distribution and time average
states = 0:(length(time_in_state) - 1);
emp_prob = time_in_state / T;

% Theoretical Poisson(rho) PMF without poisspdf (using gammaln for stability)
theo_prob = exp(-rho + states .* log(rho) - gammaln(states + 1));

max_diff = max(abs(emp_prob - theo_prob));
time_avg = sum(states .* emp_prob);

% Smallest fleet size c such that P(X > c) <= 0.01 without poissinv
cdf_vals = cumsum(theo_prob);
c_min = states(find(cdf_vals >= 0.99, 1, 'first'));
tail_prob = 1 - cdf_vals(c_min + 1);

% Display results
fprintf('Largest probability difference: %.4f\n', max_diff);
fprintf('Time average of X_t: %.4f (Theoretical rho = %.2f)\n', time_avg, rho);
fprintf('Smallest fleet size c: %d (P(X > %d) = %.4f)\n', c_min, c_min, tail_prob);

% Bar plot comparison (states 0 to 15)
plot_idx = 1:16;
figure;
bar(states(plot_idx), [emp_prob(plot_idx)', theo_prob(plot_idx)']);
xlabel('Number of Bikes Out (n)');
ylabel('Long-Run Fraction / Probability');
title('Gillespie Simulation (T = 500 hr) vs. Poisson(4)');
legend('Simulated Fraction of Time', 'Poisson(4) PMF', 'Location', 'northeast');
grid on;