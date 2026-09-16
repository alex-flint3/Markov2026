clear;
clc;
close all;

%% Parameters
R  = 2e4;       % number of independent runs
T  = 1e4;       % number of time steps
d0 = 10;         % initial lion-lamb separation

fit_lo = 1e2;   % lower end of fit window
fit_hi = 1e4;   % upper end of fit window

%% B

% Lamb starts at 0; lion starts at d0.
lamb = zeros(R,1);
lion = d0*ones(R,1);

alive1 = true(R,1);
S1 = zeros(T+1,1);
S1(1) = 1;       % S1(0) = 1

for t = 1:T

    % Only move runs in which the lamb is still alive.
    idx = find(alive1);

    % Independent random +/- 1 hops
    lamb_step = 2*(rand(length(idx),1) < 0.5) - 1;
    lion_step = 2*(rand(length(idx),1) < 0.5) - 1;

    lamb(idx) = lamb(idx) + lamb_step;
    lion(idx) = lion(idx) + lion_step;

    % Capture occurs when they occupy the same site.
    alive1(idx(lamb(idx) == lion(idx))) = false;

    % Survival probability
    S1(t+1) = mean(alive1);
end


%% Fit beta_1 over 10^2 <= t <= 10^4
t = (0:T)';

fit_idx = (t >= fit_lo) & (t <= fit_hi) & (S1 > 0);

p1 = polyfit(log(t(fit_idx)), log(S1(fit_idx)), 1);

slope1 = p1(1);
beta1 = -slope1;

% Continuum prediction
S1_cont = erf(d0 ./ (2*sqrt(t(2:end))));


%% C

% Reset positions.
lamb = zeros(R,1);
lion1 = d0*ones(R,1);
lion2 = d0*ones(R,1);

alive2 = true(R,1);
S2 = zeros(T+1,1);
S2(1) = 1;

for tstep = 1:T

    idx = find(alive2);

    n_alive = length(idx);

    % Independent +/- 1 hops for all three animals
    lamb_step  = 2*(rand(n_alive,1) < 0.5) - 1;
    lion1_step = 2*(rand(n_alive,1) < 0.5) - 1;
    lion2_step = 2*(rand(n_alive,1) < 0.5) - 1;

    lamb(idx)  = lamb(idx)  + lamb_step;
    lion1(idx) = lion1(idx) + lion1_step;
    lion2(idx) = lion2(idx) + lion2_step;

    % Capture if either lion occupies lamb's site.
    caught = (lamb(idx) == lion1(idx)) | ...
             (lamb(idx) == lion2(idx));

    alive2(idx(caught)) = false;

    S2(tstep+1) = mean(alive2);
end


%% Fit beta_2 over 10^2 <= t <= 10^4
fit_idx = (t >= fit_lo) & (t <= fit_hi) & (S2 > 0);

p2 = polyfit(log(t(fit_idx)), log(S2(fit_idx)), 1);

slope2 = p2(1);
beta2 = -slope2;


%% Tabulated results

times = [1e2; 1e3; 1e4];

S2_table  = S2(times + 1);
S1sq_table = S1(times + 1).^2;

results = table(times, S2_table, S1sq_table, ...
    'VariableNames', {'t','S2','S1_squared'});

disp(' ');
disp('Part (c): Survival probabilities');
disp(results);


%% Figure

fig = figure();
fig.Theme = 'light';
hold on;

% N = 1 simulation
loglog(t(2:end), S1(2:end), 'b-', ...
    'LineWidth', 1.5);

% N = 2 simulation
loglog(t(2:end), S2(2:end), 'r-', ...
    'LineWidth', 1.5);

% Continuum prediction for N = 1
loglog(t(2:end), S1_cont, 'k--', ...
    'LineWidth', 2);

% S1(t)^2
loglog(t(2:end), S1(2:end).^2, 'm:', ...
    'LineWidth', 2);

% Mark fitting region
xline(1e2, 'Color',[0.5 0.5 0.5], 'LineStyle','--');
xline(1e4, 'Color',[0.5 0.5 0.5], 'LineStyle','--');

% Force logarithmic axes
set(gca, 'XScale', 'log', 'YScale', 'log');

xlabel('Time, t');
ylabel('Survival probability, S_N(t)');

title(sprintf(['Capture of the Lamb: \\beta_1 = %.3f, \\beta_2 = %.3f\n' ...
               'Fit window: 10^2 \\leq t \\leq 10^4'], ...
               beta1, beta2));

legend({'S_1(t) simulation', ...
        'S_2(t) simulation', ...
        'erf[d_0/(2\sqrt{t})]', ...
        'S_1(t)^2'}, ...
        'Location','southwest');

grid on;
box on;

xlim([1 1e4]);
ylim([1/R 1]);

set(gca,'FontSize',12);

