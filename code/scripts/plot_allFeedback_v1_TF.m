function s = plot_allFeedback_v1_TF( ax, model_flag, X, target,  headAngle_t, feedback, headAngle_feedback )

global AZred AZblue

axes(ax); hold on;

switch model_flag
    
    case 1 % No Feedback model
        [mean_error, var_error] = compute_noFeedback_v1( X, target );
    
    case 2 % Pure Kalman Filter model
        [mean_error, var_error] = compute_pureKalmanFilter_v1_TF( X, target, feedback, headAngle_feedback );

    case 3 % Pure landmark navigation
        [mean_error, var_error] = compute_pureLandmark_v1_TF( X, target, feedback, headAngle_feedback );

    case 4 % Averaging model
        [mean_error, var_error] = compute_averaging_v1_TF( X, target,  headAngle_t, feedback, headAngle_feedback );
        
    case 5 % Sampling model
        scale = 50; bias = 0;
        [mean_error, var_error, p_true] = compute_sampling_v1_TF( X, target,  headAngle_t, feedback, headAngle_feedback );
        s(1) = scatter(feedback-headAngle_feedback, mean_error(:,1), (1-p_true)*scale+bias, AZblue, 'filled');
        s(2) = scatter(feedback-headAngle_feedback, mean_error(:,2), p_true*scale+bias, AZred, 'filled');
        return
        
    case 6 % Sampling model
        scale = 50; bias = 0;
        [mean_error, var_error, p_true] = compute_kalrand_v1_TF( X, target,  headAngle_t, feedback, headAngle_feedback );
        s(1) = scatter(feedback-headAngle_feedback, mean_error(:,1), (1-p_true)*scale+bias, AZblue, 'filled');
        s(2) = scatter(feedback-headAngle_feedback, mean_error(:,2), p_true*scale+bias, AZred, 'filled');
        return
end

s = plot(feedback-headAngle_feedback, mean_error, 'color', AZred);
