% Problem 3(c): Day Trader Master Equation Simulation & Convergence
Q = [-2  2  0  0;
    1 -3  2  0;
    0  2 -4  2;
    0  0  3 -3];

% Stationary distribution pi
pi_stat = [3/19; 6/19; 6/19; 4/19];

% Eigenvalues
evals = sort(eig(Q), 'descend');
lambda2 = evals(2); % Second largest eigenvalue

% Time vector
t = linspace(0, 5, 500);
x_t = zeros(4, length(t));
TV = zeros(1, length(t));

x0 = [1; 0; 0; 0]; % Starting empty (state 0)

for k = 1:length(t)
    % x(t) = x(0) * expm(t*Q) -> vector form: x'(t) = Q' * x(t)
    x_t(:, k) = expm(Q' * t(k)) * x0;
    TV(k) = 0.5 * sum(abs(x_t(:, k) - pi_stat));
end

% Find time when TV = 0.01
idx = find(TV <= 0.01, 1, 'first');
t_target = t(idx);
t_theoretical = log(100) / abs(lambda2);

fprintf('Time when TV = 0.01: %.4f hours\n', t_target);
fprintf('Theoretical approximation ln(100)/|lambda2|: %.4f hours\n', t_theoretical);

% Plotting
figure;
subplot(2,1,1);
plot(t, x_t, 'LineWidth', 1.5);
xlabel('Time t (hours)'); ylabel('Probability x_i(t)');
legend('State 0', 'State 1', 'State 2', 'State 3');
title('State Probabilities x_i(t) over Time');
grid on;

subplot(2,1,2);
semilogy(t, TV, 'b-', 'LineWidth', 2); hold on;
% Reference line with slope lambda2
C = TV(1); 
semilogy(t, C * exp(lambda2 * t), 'r--', 'LineWidth', 1.5);
xlabel('Time t (hours)'); ylabel('TV Distance (log scale)');
title('Total Variation Distance TV(t)');
legend('TV(t)', sprintf('Slope \\lambda_2 = %.4f', lambda2));
grid on;