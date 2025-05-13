%%
figure(2); clf;
set(gcf, 'position', [200 800 468*1.5 330])
ax = easy_gridOfEqualFigures([0.17 0.09], [0.32 0.03]);
landmarkNavigation(ax)
% saveFigurePdf(gcf, '~/Desktop/LandmarkNavigation')
%%
function landmarkNavigation(ax)

global AZred AZblue AZsand
ax.TickLabelInterpreter = 'latex';
fontsize = 14;

true_target = 2.5;
est_target = 2.2;
gray = [1 1 1]*0.9;

true_heading  = [0:1:2];
est_heading   = 0;
est_heading_s = 0.09;
feedback = 0.9;
feedback_s = 0.05;
ss = 1 / (1 / feedback_s^2 + 1 / est_heading_s^2);
mu = (feedback / feedback_s^2 + est_heading / est_heading_s^2) * ss;

% est_target = mu;

feedback_off = true_target/2;
est_heading2   = [mu 1.35 2.2 ];
est_heading2_s = [0.07  0.11  0.15 ];
norm_factor = normpdf(0, 0, 0.055);
x_vals = [-1:0.01:3.5];
norm_factor_fb = normpdf(0, 0, 0.04);

axes(ax); hold on; set(ax, 'ydir', 'reverse')
l = put_points( true_target, 1, 's', 10, AZred,  AZblue)
l = put_points(  est_target, 2, 's', 10, AZblue, AZblue)
l = put_points(true_heading, 3, 'V', 10, AZred,   AZred)
[l, l2] = put_pointsWithDist(    est_heading, 4, est_heading_s,     norm_factor, x_vals, '*', 10, AZblue, AZblue)

[l, l2] = put_pointsWithDist(    feedback, 5, feedback_s,     norm_factor_fb, x_vals, '*', 10, AZsand, AZsand)
% l = plot([0 feedback_off], [3 3], '-', 'color', AZsand,'LineWidth',8)

[l, l2] = put_pointsWithDist(est_heading2, 7, est_heading2_s, norm_factor, x_vals, '^', 10, AZblue,   AZblue)

plot([1 1]*est_target, [2 7], '--', 'color', AZblue)
plot([1 1]*true_target, [0.5 1], '--', 'color', 'k')
plot([1 1]*true_heading(end), [0.5 3], '--', 'color', 'k')
plot([est_heading est_heading feedback feedback], [4 5.5 5.5 5], 'k-')
plot(est_heading2(1)*[1 1], [5.5 7], 'k-')
% 
ylim([0 8])
xlim([-1.5 3])

a = put_arrow(ax, [0 est_heading2(1)], [6 6])
a = put_textbox(ax, [true_heading(end) true_target], [0.5 -0.5], 'measured error', fontsize, 'none', 'bottom', 'center')
a = put_doublearrow(ax, [true_heading(end) true_target], [0.5 0.5])
a = put_textbox(ax, [true_target 3], [6.5 4], 'respond when $\hat{m}_t = X$', fontsize, 'none', 'middle', 'center')
a = put_arrow(ax, [2.5 est_heading2(end)], [5 5])
a = put_textbox(ax, [-1.45 -0.05 ], [7.5 5.5], 'combine initial estimate and feedback', fontsize, 'none', 'middle', 'center')

set(gca, 'ytick', [1:5 7], ...
    'yticklabel', ...
    {'true target, $\alpha$' 
    'target estimate, $X$'
    'true heading, $\theta_t$'
    'initial estimate, $s_{t_f}$'
    'feedback, $f$'
    'combined estimate, $\hat{m}_t$'}, ...
    'fontsize', fontsize, ...
    'tickdir', 'out')

grid on
xlabel('heading distance after feedback is turned off', 'interpreter', 'latex')


end


