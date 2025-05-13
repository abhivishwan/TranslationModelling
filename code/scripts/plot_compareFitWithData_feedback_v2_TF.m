function l = plot_compareFitWithData_feedback_v2_TF( ax, sub, Xfit, model_flag, varargin )

if nargin < 5
    sf = 10;
else
    sf = varargin{1};
end

global AZblue AZred

% compare fit with feedback data ------------------------------------------
ind = sub.FB == 1;
target = sub.target(ind);
headAngle_t = sub.respond(ind);
headAngle_feedback = sub.fb_true(ind);
feedback = sub.fb(ind);
actual_error = headAngle_t - target;

% [mean_error, var_error, p_true] = compute_sampling_v1( Xfit, target,  headAngle_t, feedback, headAngle_feedback );
[mean_error, var_error] = compute_pureKalmanFilter_v1_TF(  Xfit, target, feedback, headAngle_feedback );
% [mean_error, var_error] = compute_pureLandmark_v1_TF(  Xfit, target, feedback, headAngle_feedback );

axes(ax);
hold on;

x = feedback - headAngle_feedback;
% ind = ~isnan(p_true) & (p_true ~= 1) & (p_true ~= 0);
% l(1) = scatter(x(ind), mean_error(ind,1), (1-p_true(ind))*sf, AZblue,'filled');
% l(2) = scatter(x(ind), mean_error(ind,2),p_true(ind)*sf, AZred, 'filled');
% l(1) = plot(x, mean_error(:,1),'.');
% l(2) = plot(feedback - headAngle_feedback, mean_error(:,2),'.');
l(1) = plot(feedback - headAngle_feedback, actual_error,'o','MarkerSize',5,'Color',[0.6 0.6 0.6],'MarkerFaceColor','w');
l(2) = scatter(x , mean_error, 10, AZred, 'filled');
set(gca, 'ylim', [-4.25 4.25])
set(gca, 'xlim', [-1.9 1.9])
% xl = get(gca, 'xlim');