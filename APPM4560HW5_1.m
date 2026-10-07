clear;
clc;
close all;

%% Part (b)

a1 = 1;
b1 = 1;

% Construct the 5x5 transition matrix P for states k = 0, 1, 2, 3, 4
P1 = zeros(5, 5);
for k = 0:4
    % Transition rate to k+1 (activation)
    if k < 4
        P1(k+1, k+2) = a1 * (4 - k) / 4;
    end
    % Transition rate to k-1 (deactivation)
    if k > 0
        P1(k+1, k) = b1 * k / 4;
    end
    % Self-loop probability
    P1(k+1, k+1) = 1 - (a1 * (4 - k) / 4) - (b1 * k / 4);
end

% Initial state: q0 = delta_0 (100% probability at state k = 0)
% State indices in MATLAB: state k corresponds to column/index k+1
q0 = [1, 0, 0, 0, 0];

% Track distributions up to n = 60
N_steps = 60;
q_history1 = zeros(N_steps + 1, 5);
q_history1(1, :) = q0;

q_curr = q0;
for n = 1:N_steps
    q_curr = q_curr * P1;
    q_history1(n+1, :) = q_curr;
end

% Extract q_50 and q_51 (MATLAB 1-indexed: row 51 and row 52)
q50 = q_history1(51, :);
q51 = q_history1(52, :);


% Compute running average for state k = 2 (index 3 in MATLAB)
% Running average S_n = (1 / (n + 1)) * sum_{k=0}^n q_k(2)
q_2_values = q_history1(:, 3); % q_n(2) for n = 0..60
running_avg_q2 = zeros(N_steps + 1, 1);
for n = 0:N_steps
    running_avg_q2(n+1) = mean(q_2_values(1:n+1));
end

% Plotting Part (b)
figure('Name', 'Question 1(b) - Trajectories and Running Average', 'NumberTitle', 'off');

n_axis = 0:N_steps;
plot(n_axis, q_history1(:, 3), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
plot(n_axis, q_history1(:, 5), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 4);
plot(n_axis, running_avg_q2, 'm--', 'LineWidth', 2);

% Analytical stationary value for comparison: pi(2) = binom(4,2) * (0.5^4) = 6/16 = 0.375
yline(6/16, 'k:', 'LineWidth', 1.5, 'DisplayName', '\pi(2) Stationary');

xlabel('Time Step (n)');
ylabel('Probability Mass');
title('Part (b): q_n(2), q_n(4), and Running Average of q_n(2) for a=1, b=1');
legend({'q_n(2)', 'q_n(4)', 'Running Avg of q_n(2)', '\pi(2) = 0.375'}, 'Location', 'Best');
grid on;


%% Part (c)

a2 = 0.3;
b2 = 0.1;

% Construct transition matrix P2
P2 = zeros(5, 5);
for k = 0:4
    if k < 4
        P2(k+1, k+2) = a2 * (4 - k) / 4;
    end
    if k > 0
        P2(k+1, k) = b2 * k / 4;
    end
    P2(k+1, k+1) = 1 - (a2 * (4 - k) / 4) - (b2 * k / 4);
end

% 1. Linear solve for stationary distribution pi: pi * (I - P2) = 0, sum(pi) = 1
A_eq = [(eye(5) - P2)'; ones(1, 5)];
b_eq = [zeros(5, 1); 1];
pi_solve = (A_eq \ b_eq)';

% Theoretical Binomial(4, theta) where theta = a / (a + b) = 0.3 / 0.4 = 0.75
theta = a2 / (a2 + b2);
pi_binom = zeros(1, 5);
for k = 0:4
    pi_binom(k+1) = nchoosek(4, k) * (theta^k) * ((1 - theta)^(4 - k));
end


% 2. Iteration to find smallest n where max_k |q_n(k) - pi(k)| < 1e-6
q_curr = q0;
n_conv = 0;
q_history2 = q0;

while true
    q_next = q_curr * P2;
    n_conv = n_conv + 1;
    q_history2(n_conv + 1, :) = q_next;
    
    if max(abs(q_next - pi_solve)) < 1e-6
        break;
    end
    q_curr = q_next;
end

% 3. Second largest eigenvalue modulus of P2
eig_vals = eig(P2);
abs_eigs = sort(abs(eig_vals), 'descend');
second_largest_eig_modulus = abs_eigs(2);



% Plotting Part (c) Panel
figure('Name', 'Question 1(c) - Convergence to Stationary Distribution', 'NumberTitle', 'off');

n_axis2 = 0:n_conv;
colors = lines(5);
hold on;
for k = 0:4
    plot(n_axis2, q_history2(:, k+1), 'LineWidth', 1.5, 'Color', colors(k+1, :), ...
        'DisplayName', sprintf('q_n(%d)', k));
    yline(pi_solve(k+1), '--', 'Color', colors(k+1, :), 'HandleVisibility', 'off');
end

xlabel('Time Step (n)');
ylabel('Probability Mass');
title(sprintf('Part (c): Convergence of q_n from \\delta_0 (a=0.3, b=0.1, Target n=%d)', n_conv));
legend('Location', 'East');
grid on;