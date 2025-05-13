function [l, f] = plot_compareFitWithData_nofeedback_v2_TF( ax, sub, Xfit )
% function [l, f] = plot_compareFitWithData_nofeedback_v2_TF(sub, Xfit )

global AZblue AZred
% get data
% compare fit with nofeedback data ------------------------------------------


ind = sub.FB == 0;
target = sub.target(ind);
headAngle_t = sub.respond(ind);
actual_error = headAngle_t - target;

% [mean_error, var_error] = compute_noFeedback_v1( Xfit, target, headAngle_t );
[mean_error, var_error] = compute_noFeedback_v1( Xfit, target );


axes(ax); 
hold on;

[X, ind] = sort(target);

f = shadedErrorBars(X', mean_error(ind)', sqrt(var_error(ind))');
set(f, 'facecolor', AZblue*0.25 +0.75)


% 
% e = errorbar(target, mean_error, sqrt(var_error), ...
%     'linestyle', 'none', ...
%     'marker','.', 'color', AZblue*0.5+0.5)
l(2) = plot(target, actual_error, 'o', ...
    'color', AZred, 'markersize', 5, 'markerfacecolor', 'none', ...
    'linewidth', 1);
l(1) = plot(X, mean_error(ind), '-', 'color', AZblue, 'linewidth', 3)
plot([1 9], [0 0], 'k--')

% set(gca, 'ylim', [-4 4])
% set(gca, 'xlim',  [1 9])
%     'xtick', [0 180 360])

% xl = xlabel('target, \phi');
% yl = ylabel('response error, \theta_{t} - \phi');

% set([xl yl], 'interpreter', 'latex')
% ax(i).TickLabelInterpreter = 'latex';
% set(gca,'tickdir', 'out')