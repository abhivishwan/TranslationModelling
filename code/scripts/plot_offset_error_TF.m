function l = plot_offset_error_TF( ax, sub, varargin )

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
trials = sub.trial(ind);

axes(ax);
hold on;

l(1) = plot(feedback - headAngle_feedback, actual_error,'o','MarkerSize',5,'Color',[0.6 0.6 0.6],'MarkerFaceColor','w');

% l(1) = scatter(feedback - headAngle_feedback, actual_error,8, trials,'filled');
% l(1) = scatter(trials, actual_error,8, feedback - headAngle_feedback,'filled');
% colormap("jet")

