clear;
clc;
close all;


N = 10^5;

a = 0.9;
lambda_f = 1000;   % s^-1
lambda_s = 10;     % s^-1

% Bernoulli draw
B = rand(N,1) < a;


T = zeros(N,1);

T(B)  = -log(rand(sum(B),1)) / lambda_f;
T(~B) = -log(rand(sum(~B),1)) / lambda_s;


%% Empirical mean

empirical_mean = mean(T);

% Theoretical mean
theoretical_mean = a/lambda_f + (1-a)/lambda_s;


%% P(T > 50 ms)

threshold = 0.050;  

empirical_tail = mean(T > threshold);

% Theoretical probability
theoretical_tail = a*exp(-lambda_f*threshold) + ...
    (1-a)*exp(-lambda_s*threshold);


%% Histogram
edges = linspace(0, 1.2, 150);

figure;

histogram(T, edges, 'Normalization', 'pdf');
hold on;

t = linspace(1e-5, 1.2, 2000);

f = a*lambda_f*exp(-lambda_f*t) + ...
    (1-a)*lambda_s*exp(-lambda_s*t);

plot(t, f, 'r-', 'LineWidth', 2);

set(gca, 'YScale', 'log');

xlabel('Dwell time t (s)');
ylabel('Density');
title('Mixture of fast and slow exponential dwell times');

legend('Normalized histogram', 'f(t)', 'Location', 'northeast');
grid on;
