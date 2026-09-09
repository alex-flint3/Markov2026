clear;
clc;
close all;


%% Part (b)

N = 10^5;

U = rand(N,1);
z_samples = 1 - sqrt(1-U);

mean_b = mean(z_samples);


%% Part (c

mu = 1;
Nfinal = 1e4;
R = 1e3;

C = ones(R,1);

for n = 3:(Nfinal-1)

    % Probability of increasing C:
    p = mu * C / n;

    % One Bernoulli draw for each realization
    B = rand(R,1) < p;

    % Update all R realizations simultaneously
    C = C + B;
end

z = C / Nfinal;


%% Empirical statistics

empirical_mean = mean(z);
empirical_std = std(z);
empirical_ratio = empirical_std / empirical_mean;

minimum_core = min(C);
maximum_core = max(C);



%% Plot normalized histogram with h(z) overlaid

figure;

histogram(z, 30, 'Normalization', 'pdf');
hold on;

zgrid = linspace(0,1,500);
h = 2*(1-zgrid);

plot(zgrid, h, 'r-', 'LineWidth', 2);

xlabel('z = C/N');
ylabel('Density');
title('Distribution of Core Fraction z');
legend('Simulation', 'h(z) = 2(1-z)', 'Location', 'northeast');
grid on;
xlim([0 1]);
