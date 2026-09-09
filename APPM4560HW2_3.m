clear;
clc;
close all;

N = 10^4;


%% Lambda = 0.5
lambda = 0.5;
c = 4/exp(1);

samples = zeros(N,1);
accepted = 0;
proposals = 0;

tic;

while accepted < N
    U1 = rand;
    X = -log(U1)/lambda;

    U2 = rand;
    proposals = proposals + 1;

    if U2 < exp(1)*X*exp(-X/2)/(2)
        accepted = accepted + 1;
        samples(accepted) = X;
    end
end

elapsed = toc;

acceptance_fraction = N/proposals;
time_per_sample = elapsed/N;


%% Plot for lambda = 0.5
x = linspace(0, max(samples), 500);
f = x .* exp(-x);

figure;
histogram(samples, 50, 'Normalization', 'pdf');
hold on;
plot(x, f, 'r-', 'LineWidth', 2);
xlabel('x');
ylabel('Density');
title('\lambda = 0.5');
legend('Normalized histogram', 'f(x) = xe^{-x}');
grid on;


%% Lambda = 0.2
lambda = 0.2;
c = 1/(0.16*exp(1));

samples2 = zeros(N,1);
accepted = 0;
proposals = 0;

tic;

while accepted < N
    U1 = rand;
    X = -log(U1)/lambda;

    U2 = rand;
    proposals = proposals + 1;

    accept_prob = X * exp(-(1-lambda)*X) / (c*lambda);

    if U2 < accept_prob
        accepted = accepted + 1;
        samples2(accepted) = X;
    end
end

elapsed = toc;

acceptance_fraction2 = N/proposals;
time_per_sample2 = elapsed/N;


%% Plot for lambda = 0.2
x2 = linspace(0, max(samples2), 500);
f2 = x2 .* exp(-x2);

figure;
histogram(samples2, 50, 'Normalization', 'pdf');
hold on;
plot(x2, f2, 'r-', 'LineWidth', 2);
xlabel('x');
ylabel('Density');
title('\lambda = 0.2');
legend('Normalized histogram', 'f(x) = xe^{-x}');
grid on;

