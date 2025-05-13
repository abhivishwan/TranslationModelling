% Code for Combination and competition between path integration and landmark navigation in the estimation of heading direction
% Authors: Robert C. Wilson & Sevan K. Harootonian
% 2021
%% directories
clear
% fundir = '/Users/bob/Work/Projects/arne/headDirection/ARNE_HeadDirectionCode/';
% fundir = '/Users/bob/Work/GitHub/ARNE_HeadDirectionCode/';
fundir = 'C:/Users/avishwanath/Documents/GitHub/RotationandTranslationModelling/';
% fundir = 'C:/Users/abhi2/Documents/GitHub/RotationandTranslationModelling/';

addpath([fundir, 'data'])
addpath(genpath([fundir, 'scripts']))
datadir = [fundir 'data/'];
% figdir= '~/Desktop/';
cd(fundir)
% figdir= 'C:/Users/avishwanath/Documents/GitHub/RotationandTranslationModelling/graphs/Translation_Abhi/';
%% default plot parameters
defaultPlotParameters
%% load data
% sub = load_data_v2(datadir,'dataconf.csv');
sub = load_data_v2(datadir,'data_Translation_30_M_OutlierCorr_FBtrue_halfdist_FB_halfplusoffset.csv'); % changed the load function to only get the good trials
%% ========================================================================
%% characterize data set, participants and trials %%
%% ========================================================================

%% FIGURE XXX - how many trials for each subject
figure; clf
set(gcf, 'position',[440   504   468   200])
b = plot_trialsPerSubject_v1(gca, sub);
saveFigurePdf(gcf, 'KF_trials')

%% Figure 2
% Plot response vs target plots for no feedback trails only for all
% subjects

sf = 1.3;
figure; clf;
set(gcf, 'position',[440   475   373   329])
hg = [0.2 0.1];
wg = [0.15 0.1];
[ax,hb,wb,ax2] = easy_gridOfEqualFigures(hg, wg);

for sn=1:length(sub)

    [l] = plot_response_target_TF( ax(1), sub(sn) )

    % plot([2 8],[2 8],'k--')

end
plot([2 8],[2 8],'k--')
set(ax2(1:end-1,:), 'xticklabel', [])
set(ax2(:,2:end), 'yticklabel', [])
set(ax, 'ylim', [0 11])
set(ax, 'xlim', [1.8 8.1])
a = add_oneExternalYLabel_v1(hg, wg, hb, wb, 90, 0, {'response distance (meters), \theta_{\itt}'});
set(a, 'fontsize', ceil(12))
% 
a = add_oneExternalXLabel_v1(hg, wg, hb, wb, -0.05, {'target distance (meters), \alpha'},'bottom');
set(a, 'fontsize', ceil(12))
set(ax,'tickdir','out','fontsize',ceil(8*sf))

%% Figure 6
% Plot error vs jitter plots for 3 good examples
% figure; clf
% t = tiledlayout('flow');
figure; clf;
set(gcf, 'position',[485 162 1074 364])
hg = [0.2 0.1];
wg = [0.15 0.05 0.05 0.1];
[ax,hb,wb,ax2]  = easy_gridOfEqualFigures(hg, wg);

for sn = 1:length(sub)
    actual_err_y = []; jitter_x = []; ind = [];
    ind = sub(sn).FB == 1;
    actual_err_y = sub(sn).respond - sub(sn).target;
    jitter_x = sub(sn).fb - sub(sn).fb_true;
    out = polyfit(jitter_x(ind),actual_err_y(ind),1);
    slope(sn) = out(1);
    yincept(sn) = out(2);
end
[~,indall] = sort(slope);
ind2 = [indall(1) indall(12) indall(end)];  
sub_lbs = {'participant 25' 'participant 12' 'participant 7'};
for sn=1:3
    axes(ax(sn)); hold on;
    [l] = plot_offset_error_TF( ax(sn), sub(ind2(sn)) )
    % [l] = plot_compareFitWithData_feedback_v2_TF( gca, sub(ind2(sn)), Xfit(ind2(sn),:,model_flag), model_flag, 20 );
    plot([-1.5 1.5], [-1.5 1.5],'k--')
    fancy_xlabel_v1({'feedback offset, ' ' $f - \theta_{t_f}$'},2,12)
    fancy_ylabel_v1({'response error, ' '$\theta_{t} - \alpha$'}, 2, 12)
    title(sub_lbs{sn},'FontSize',12)

end

addABCs(ax(1), [-0.035 0.1], 15*sf)
addABCs(ax(2), [-0.035 0.1], 15*sf, 'B')
addABCs(ax(3), [-0.035 0.1], 15*sf, 'C')

set(ax,'tickdir','out')
set(ax, 'ylim', [-4.15 4.15])
set(ax, 'xlim', [-1.9 1.9])

%% ========================================================================
%% no feedback condition %% no feedback condition %%
%% no feedback condition %% no feedback condition %%
%% no feedback condition %% no feedback condition %%
%% ========================================================================
%% fit path integration
data_flag = 1;
model_flag = 1;
n_fit = 100;
for sn = 1:length(sub)  
    [Xfit_nofb_purepath(sn,:)] = fit_any_v1_TF( sub(sn), data_flag, model_flag, n_fit );
end

%% FIGURE 5 - data vs model - feedback error vs response error
% Plotted after running path integration model with nfit = 100
model_flag = 1;

sf = 1.3;

figure(1); clf;
set(gcf, 'position',[440   204   1.3*468*sf   1.3*300*sf])
hg = ones(6,1)*0.05;
wg = ones(7,1)*0.02;
wg(1) = 0.1;
hg(1) = 0.11;
[ax,hb,wb,ax2] = easy_gridOfEqualFigures(hg, wg);
% t = tiledlayout('flow');

% sort by slope (g - gamma)
slope = Xfit_nofb_purepath(:,1) - Xfit_nofb_purepath(:,3);
[~,ind] = sort(slope);

for sn = 1:length(sub)
    [l, f] = plot_compareFitWithData_nofeedback_v2_TF(ax(sn),sub(ind(sn)), Xfit_nofb_purepath(ind(sn),:) )
end
set(ax2(1:end-1,:), 'xticklabel', [])
set(ax2(:,2:end), 'yticklabel', [])
set(ax, 'ylim', [-4 4])
set(ax, 'xlim',  [1 9])
a = add_oneExternalYLabel_v1(hg, wg, hb, wb, 90, 0, {'response error (meters), \theta_{\itt} - \alpha'});
set(a, 'fontsize', ceil(12*sf))
% 
a = add_oneExternalXLabel_v1(hg, wg, hb, wb, -0.035, {'target distance, \alpha'}, 'bottom');
set(a, 'fontsize', ceil(12*sf))
set(ax, 'TickDir', 'out', 'fontsize', ceil(8*sf))

%% Supplemntary FIGURE 1 - histograms of fit parameter values
% g           = X(1); % targetGain - gain on memory of target
% b           = X(2); % targetBias - bias in memory of target
% gamma       = X(3); % velGain - gain on velocity cue
% sigma_mem   = X(4); % actual memory noise
% sigma_vel   = X(5); % actual velocity noise
% s_0         = X(6); % subject estimate of initial uncertainty
% s_vel       = X(7); % subject estimate of velocity noise
% s_f         = X(8); % subject estimate of feedback noise
% omega       = X(9); % subject pTrue


var_name = {'target gain'
    'target bias'
    'velocity gain'
    'target noise'
    'velocity noise'
    {'participant' 'initial uncertainty'}
    {'participant' 'feedback noise'}
    {'participant prior' 'on true feedback'}};
var_symbol = {'\gamma_X'
    '\beta_X'
    '\gamma_d'
    '\sigma_X'
    '\sigma_d'
    's_{t_f}'
    's_f'
    'r'};

ind = [1 2 4 5];
n = var_name(ind);
s = var_symbol(ind);
XX = Xfit_nofb_purepath(:,ind);

figure; clf;
set(gcf, 'position',[440   365   541   439])
hg = [0.13 0.2 0.08];
wg = [0.12 0.12 0.02];
ax = easy_gridOfEqualFigures(hg, wg);

clear ttl xl yl
for i = 1:length(ax)
    axes(ax(i)); hold on;
    mns(i) = mean(XX(:,i));
    [h,x] = hist(XX(:,i));
    b = bar(x, h, 'facecolor', AZred*0.5+0.5, 'barwidth',1 );
    ttl(i) = title(n(i));
    xl(i) = xlabel(['$' s{i} '/' var_symbol{3} '$'], 'interpreter', 'latex');
    yl(i) = ylabel('count');
end

set(ax, 'fontsize', 12, 'tickdir', 'out');
set([xl], 'fontsize', 14)
set([yl], 'fontsize', 13)
set(ttl, 'fontsize', 14, 'fontweight', 'normal')
addABCs(ax, [-0.09 0.06], 24)
% saveFigurePdf(gcf, [figdir 'KF_noFeedbackParameters'])


%% ========================================================================
%% feedback condition %% feedback condition %% feedback condition %%
%% feedback condition %% feedback condition %% feedback condition %%
%% feedback condition %% feedback condition %% feedback condition %%
%% ========================================================================


%% FIGURE 3 - Illustration of qualitative properties of models for
%% feedback condition
model_name = {'Path Integration' 'Kalman Filter' 'Pure Landmark' 'Cue Combination' 'Hybrid' 'Random-Hybrid'};

feedback = [-1.5:0.05:1.5]';
headAngle_t = nan(size(feedback));
target = ones(size(feedback))*7.9;
headAngle_feedback = zeros(size(feedback));
scale = 50;
bias = 0;

X = [1 0 1 2.5 0.1 0.19 0.19 0.8];

clear ttl xl yl


figure; clf;
set(gcf, 'position',[440   468   858   336])
hg = [0.2 0.1];
wg = [0.15 0.05 0.05 0.1];
[ax,hb,wb,ax2]  = easy_gridOfEqualFigures(hg, wg);

set(ax, 'fontsize', 11, 'ylim', [-1.9 1.9])

l = [];
for i = 1:3

    if i == 5
        X = [1 0 1 2.5 0.1 0.19 0.19 0.5];
    end
    ii = i - 2;
    axes(ax(i)); hold on
    l = [l plot_allFeedback_v1_TF( ax(i), i, X, target,  headAngle_t, feedback, headAngle_feedback )];
    
    %xl(i) = xlabel('feedback offset, $f - \theta_{t_f}$', 'interpreter', 'latex')
    fancy_xlabel_v1({'feedback offset, ' ' $f - \theta_{t_f}$'},2,12)
    if i < 2
        fancy_ylabel_v1({'response error, ' '$\theta_{t} - \alpha$'}, 2, 12)
    end
    plot([-1.5 1.5],[0 0],'k--')
    plot([-1.5 1.5],[-1.5 1.5],'k--')
    title(model_name{i},'FontSize',14)
    % ttl(i) = title(model_name{i}, 'fontweight', 'normal')
end

set(l(1:3), 'linewidth', 3)
% set(ax2(1:end-1,:), 'xticklabel', [])
% set(ax2(:,2:end), 'yticklabel', [])

set(ax,  'tickdir', 'out')
set(ax, 'xlim',[-1.9 1.9])

% set(ttl, 'fontsize', 14)

% set([xl yl], 'fontsize', 14)
addABCs(ax, [-0.02 0.1], 20)
% saveFigurePdf(gcf, [figdir 'KF_modelPredictions'])
% s = plot_sampling_v1( ax(4), scale, bias, X, target,  headAngle_t, feedback, headAngle_feedback );

% a = add_oneExternalYLabel_v1(hg, wg, hb, wb, 90, -0.01, {'response error, \theta_{\itt} - \alpha'});
% set(a, 'fontsize', ceil(12*sf))
% 
% a = add_oneExternalXLabel_v1(hg, wg, hb, wb, -0.07, {'feedback offset (meters)'}, 'bottom');
% set(a, 'fontsize', ceil(12*sf))
% set(ax(6), 'visible', 'off')

%% 2022 - R21 NIMH - why is this lagging
model_name = {
    'Path Integration'
    'Landmark Navigation'
    {'Cue Combination' '(Kalman Filter)'} 
    {'Cue Combination' '(Variable Gain)'} 
    'Cue Competition'
    'Hybrid'};

feedback = [-180:180]';
headAngle_t = nan(size(feedback));
target = ones(size(feedback))*180;
headAngle_feedback = zeros(size(feedback));
scale = 50;
bias = 0;
clear X
X{1} = [1 0 1 10 1 20 0.1 20 0.8];
X{2} = [1 0 1 10 1 20 0.1 0 0.8];
X{3} = [1 0 1 10 1 20 0.1 20 0.8];
X{4} = [1 0 1 10 1 20 0.1 5 0.8];
X{5} = [1 0 1 10 1 20 0.1 0 0.8];
X{6} = [1 0 1 10 1 20 0.1 20 0.8];

mod_num = [1 2 2 3 4 4];

clear ttl xl yl

figure(1); clf;
set(gcf, 'position',[440   504   468   550])
ax = easy_gridOfEqualFigures([0.09 0.15 0.19 0.05], [0.1 0.13 0.02]);

set(ax, 'fontsize', 14, 'ylim', [-100 100])

l = [];
for i = 1:6
    l = [l plot_allFeedback_v1( ax(i), mod_num(i), X{i}, target,  headAngle_t, feedback, headAngle_feedback )];
    ttl(i) = title(model_name{i}, 'fontweight', 'normal', 'interpreter', 'latex')
    xl(i) = xlabel({'feedback offset, $f - \theta_{f}$'}, 'interpreter', 'latex')
    yl(i) = ylabel({'error, $\theta - \alpha$'}, 'interpreter', 'latex')
    
    %fancy_xlabel_v1({'feedback offset, ' ' $f - \theta_{t_f}$'},2,12)
    %fancy_ylabel_v1({'response error, ' '$\theta_{t} - \alpha$'}, 2, 12)
    plot([-180 180], [-180 180], 'k--')
    plot([-180 180], [0 0], 'k--')
    ax(i).TickLabelInterpreter = 'latex';
    
end

set(l(1:4), 'linewidth', 3)
set(l([1:4]), 'color', 'r')
set(l([5 7]), 'markerfacecolor', 'b')
set(l([6 8]), 'markerfacecolor', 'r')
set(ax, 'ylim', [-120 120], 'xlim', [-120 120],  'tickdir', 'out', ...
    'xtick', [-180:90:180], 'ytick', [-180:90:180])
set(ttl, 'fontsize', 18)

% set([xl yl], 'fontsize', 14)
% addABCs(ax, [-0.09 0.07], 20)
saveFigurePdf(gcf, [figdir 'KF_modelPredictions'])
% s = plot_sampling_v1( ax(4), scale, bias, X, target,  headAngle_t, feedback, headAngle_feedback );



%% fit all models
%% **take about 10min**
clear Xfit
data_flag   = 3;
n_fit       = 100;
tic
for model_flag  = 1:6
    for sn = 1:length(sub)
        disp([model_flag sn])
        % use fit_any_v1 for non-parallel evaluation of starting points
        %[Xfit(sn,:,model_flag), nLL(sn,model_flag), BIC(sn,model_flag), AIC(sn,model_flag)] ...
        %    = fit_any_v1( sub(sn), data_flag, model_flag, n_fit );
        [Xfit(sn,:,model_flag), nLL(sn,model_flag), BIC(sn,model_flag), AIC(sn,model_flag)] ...
            = fit_any_parallel_v1_TF( sub(sn), data_flag, model_flag, n_fit );
    end
end
toc

% save('MainTF_Xfit_All6_FBtrue_andFB_halfdist_gammadest','Xfit')
% save('MainTF_BIC_All6_FBtrue_andFB_halfdist_gammadest','BIC')

%% fit pure Landmark Navigation model
% clear Xfit
% data_flag   = 3;
% n_fit       = 100;
% tic
% for model_flag  = 5
%     for sn = 1:length(sub)
%         disp([model_flag sn])
%         % use fit_any_v1 for non-parallel evaluation of starting points
%         %[Xfit(sn,:,model_flag), nLL(sn,model_flag), BIC(sn,model_flag), AIC(sn,model_flag)] ...
%         %    = fit_any_v1( sub(sn), data_flag, model_flag, n_fit );
%         [Xfit(sn,:,model_flag), nLL(sn,model_flag), BIC(sn,model_flag), AIC(sn,model_flag)] ...
%             = fit_any_parallel_v1( sub(sn), data_flag, model_flag, n_fit );
%     end
% end
% toc
% save FK_modelFits_010521c_parallel

%% FIGURE 11 - comparison between data and model
% kalman fit
% Three models path, kalman, pure landmark
NewBIC = [];
NewBIC = [BIC(:,1:2) BIC(:,3)];

[~,ind] = min(NewBIC');
[BIC_sorted,idx_sorted] = sort(ind);
ixx = BIC_sorted == 2;
model_idx = idx_sorted(ixx);

model_flag = 2;
figure; clf;
set(gcf, 'position',[319.2000  125.0000  862.8000  678.3200])
hg = ones(6,1)*0.05;
wg = ones(7,1)*0.02;
wg(1) = 0.1;
hg(1) = 0.11;
[ax,hb,wb,ax2] = easy_gridOfEqualFigures(hg, wg);

XX = Xfit(:,:,model_flag);
KG = nan(1,length(sub));

for sn = 1:length(sub)
    ind                 = sub(sn).FB == 1;
    target              = sub(sn).target(ind);
    headAngle_t         = sub(sn).respond(ind);
    headAngle_feedback  = sub(sn).fb_true(ind);
    feedback            = sub(sn).fb(ind);

    [~, ~, rho_tf] = compute_pureKalmanFilter_v1_TF( XX(sn,:), target, feedback, headAngle_feedback );
    KG(1,sn) = rho_tf;
end
[KG_sort,idx] = sort(KG);

set(ax, 'fontsize', 12)
for sn = 1:length(sub)
    axes(ax(sn)); hold on;
    % sn = three_eg(i);
    [l] = plot_compareFitWithData_feedback_v2_TF( gca, sub(idx(sn)), Xfit(idx(sn),:,model_flag), model_flag, 20 );
    plot([-1.5 1.5], [-1.5 1.5], 'k--')
    title(['participant ' num2str(idx(sn))], 'fontsize', 14, 'fontweight', 'normal')

end

set(ax2(1:end-1,:), 'xticklabel', [])
set(ax2(:,2:end), 'yticklabel', [])

a = add_oneExternalYLabel_v1(hg, wg, hb, wb, 90, 0.005, {'response error (meters)'});
set(a, 'fontsize', ceil(14))
% 
a = add_oneExternalXLabel_v1(hg, wg, hb, wb, -0.03, {'feedback offset (meters)'},'bottom');
set(a, 'fontsize', ceil(14))
set(ax, 'tickdir', 'out')

[~,ind] = min(BIC');
ind = ind(idx);
i3 = ind == 1;
i1 = ind == 3;
set(ax([i3]), 'color', AZsand*0.25+0.75)
set(ax([i1]), 'color', AZcactus*0.25+0.75)
% addABCs(ax, [-0.11 0.06], 20)
% addABCs(ax, [-0.09 0.07], 20)
% saveFigurePdf(gcf, [figdir 'KF_modelVsData_examples'])

%% FIGURE 9 - histogram of fit parameters

% parameter order - to fit with order in which parameters are discussed in
% paper
par_order = [3 5 1 2 4 6 7 8];


model_flag = 2;

XX = Xfit(:, :, model_flag);
figure; clf;
set(gcf, 'position',[440   381   588   523])

ax = easy_gridOfEqualFigures([0.09 0.18 0.18 0.05], [0.08 0.1 0.1 0.01]);

for i = 1:6
    par_num = par_order(i)
    axes(ax(i)); hold on;
    [h,x] = hist(XX(:,par_num));
    rangmean(i,1) = min(XX(:,par_num));
    rangmean(i,2) = max(XX(:,par_num));
    rangmean(i,3) = mean(XX(:,par_num));
    b = bar(x, h, 'facecolor', AZred*0.5+0.5, 'barwidth',1 );
    ttl(i) = title(var_name{par_num});
    xl(i) = xlabel(['$' var_symbol{par_num} '$'], 'interpreter', 'latex');
    yl(i) = ylabel('count');
end

set(ax, 'fontsize', 11, 'tickdir', 'out')
% set(ax([1 3]), 'xlim', [0.4 1.6])
% set(ax([3]), 'xtick', [0.6 1 1.4])
set([xl], 'fontsize', 14)
set([yl], 'fontsize', 12)
set(ttl, 'fontweight', 'normal', 'fontsize', 12)


if model_flag == 1
    set(ax([1 6:9]), 'visible', 'off')
end

if model_flag == 2
    set(ax([7:9]), 'visible', 'off')
end
if model_flag == 4
    set(ax(9), 'visible', 'off')
end
addABCs(ax, [-0.06 0.05], 20)
% boxplot(Y);
% bar(M - M(4))
% saveFigurePdf(gcf, [figdir 'KF_feedbackParameters_' num2str(model_flag)])

%% mean parameter values
M = mean(XX(:,par_order))
var_symbol(par_order)

for i = 1:length(par_order)
    name = var_name{par_order(i)};
    if iscell(name)
        name = [name{1} ' ' name{2}];
    end
    disp(sprintf('%s, %s = %.2f', name, var_symbol{par_order(i)}, M(i)))
end



%% FIGURE 10 - implied Kalman gain with fit parameters
% Ordered by best fit for Path, kalman and pure landmark
NewBIC = [];
NewBIC = [BIC(:,1:2) BIC(:,3)];

[~,ind] = min(NewBIC');
[BIC_sorted,idx_sorted] = sort(ind);
ixx = BIC_sorted == 2;
KFmodel_idx = idx_sorted(ixx);
ixx = BIC_sorted == 1;
PImodel_idx = idx_sorted(ixx);
ixx = BIC_sorted == 3;
PLmodel_idx = idx_sorted(ixx);

model_flag = 2;
XX = Xfit(:,:,model_flag);
KG = nan(1,length(sub));

for sn = 1:length(sub)
    ind                 = sub(sn).FB == 1;
    target              = sub(sn).target(ind);
    headAngle_t         = sub(sn).respond(ind);
    headAngle_feedback  = sub(sn).fb_true(ind);
    feedback            = sub(sn).fb(ind);

    [~, ~, rho_tf] = compute_pureKalmanFilter_v1_TF( XX(sn,:), target, feedback, headAngle_feedback );
    KG(1:length(rho_tf),sn) = rho_tf;
end
[~,ind] = sort(KG(1,:));
% M = nanmean(KG,1)
[~,indPI] = sort(KG(1,PImodel_idx));
PImodel_idx = PImodel_idx(indPI);
[KF_KG,indKF] = sort(KG(1,KFmodel_idx));
KFmodel_idx = KFmodel_idx(indKF);
[~,indPL] = sort(KG(1,PLmodel_idx));
PLmodel_idx = PLmodel_idx(indPL);

ind = [PImodel_idx KFmodel_idx PLmodel_idx];
% boxplot(KG(:,ind))
figure; clf;
set(gcf, 'position',[585.2000  637.0000  706.8000  299.3200])
ax = easy_gridOfEqualFigures([0.23 0.03], [0.1 0.03]);
% ax(2) = easy_gridOfEqualFigures([0.23 0.03], [0.8 0.03]);
axes(ax(1)); hold on;
set(ax, 'fontsize', 14)

scatter(1:length(ind),KG(ind),"MarkerFaceColor",AZred,"MarkerEdgeColor",AZred)
hold on
plot([2.5 2.5], [0 1],'k--')
plot([26.5 26.5], [0 1],'k--')

fancy_ylabel_v1({'Kalman gain, ' '$K_{t_f}$'}, 2, 14)
xlim([0.5 30.5])
xlabel('Subjects (ordered by best fit model and Kalman gain)')
set(ax, 'tickdir', 'out')
% saveFigurePdf(gcf, [figdir 'KF_kalmanGain'])


%% FIGURE S5 - correlations between parameters 
par_order = [3 5 1 2 4 6 8];
% par_order = [3 5 1 2 4];

model_flag = 2;
XX = Xfit(:,:,model_flag);
% XX = Xfit_KalmanFilter(:,:,model_flag);
figure; clf;
set(gcf, 'position',[440   504   468   250])
% ax = easy_gridOfEqualFigures([0.48 0.03], [0.17 0.12])
% ax(2:3) = easy_gridOfEqualFigures([0.1 0.65], [0.1 0.12 0.05]);
% ax = easy_gridOfEqualFigures([0.1 0.1], [0.1 0.1 0.1 0.03]);
ax = easy_gridOfEqualFigures([0.12 0.03], [0.11 0.32]);
ax(2:3) = easy_gridOfEqualFigures([0.15 0.18 0.06], [0.76 0.03]);

plot_parameterCorrelations_v1(ax, Xfit(:,par_order,:), model_flag, var_symbol(par_order))
% saveFigurePdf(gcf, '~/Desktop/KF_parameterCorrelations')
ix = [3 3 ];%3 ]%3 3];
iy = [1 5 ];%8 ]%5 8];

a = 0.05;
b = 0.86;
c = 0.14;
tx = [a a a a a];
ty = [b b c b c];

% set(ax(2:3), 'xlim', [0.5 2.5]);%, 'xtick', [0.6 1 1.4])
clear xl yl
for i = 1:length(ix)

    axes(ax(i+1)); hold on;
    plot(XX(:,ix(i)), XX(:,iy(i)), 'o', 'markersize', 4, 'color', AZblue*0.5+0.5)
    l = lsline;
    set(l, 'color', AZred, 'linewidth', 3)
    [r,p] = corr(XX(:,ix(i)), XX(:,iy(i)), 'type', 'spearman');
    text(tx(i), ty(i), sprintf('$\\rho$ = %.2f', r), ...
        'units', 'normalized', 'interpreter', 'latex', ...
        'backgroundcolor', AZred*0.25+0.75, 'margin', 1)
    xl(i) = xlabel(['$' var_symbol{ix(i)} '$'], 'interpreter', 'latex', 'fontsize', 14);
    yl(i) = ylabel(['$' var_symbol{iy(i)} '$'], 'interpreter', 'latex', 'fontsize', 14);
end


xxx = [0.5 2.5];
yyy = 2*xxx - 1;
axes(ax(2)); hold on;
plot(xxx, yyy, 'k--')

xxx = [0.5 2.5];
yyy = xxx;
% axes(ax(3)); hold on;
% plot(xxx, yyy, 'k--')
set(ax, 'tickdir', 'out')
set([xl yl], 'fontsize', 14)
% set(ax(end), 'visible', 'off')
addABCs(ax(1), [-0.09 0.01], 20, 'A')
addABCs(ax(2:3), [-0.08 0.03], 20, 'BCDEF')
polyfit(XX(:,3), XX(:,5),1)
saveFigurePdf(gcf, [figdir 'KF_parameterCorrelations'])



%% ========================================================================
%% parameter recovery %% parameter recovery %% parameter recovery %%
%% parameter recovery %% parameter recovery %% parameter recovery %%
%% parameter recovery %% parameter recovery %% parameter recovery %%
%% ========================================================================


%% simulate fake data for parameter recovery
clear sim
Nsim = 30;

for model_flag = 1:6
    rng(10); % set random seed for reproducibility
    Xsim = generate_simulationParameters(3,  Nsim, Xfit(:,:,model_flag));
    expt = generate_simulationTrials(sub, Nsim);
    for count = 1:length(expt)
        sim(count, model_flag) = sim_any_v2_TF(model_flag, Xsim(count,:), expt(count));
    end
end

%% fit fake data for BOTH conditions (data_flag = 3)
data_flag = 3;
n_fit = 100;
tic
for model_flag = 1:6
    for sn = 1:length(sim)
        disp(['model_flag = ' num2str(model_flag) '; subject = ' num2str(sn)])
        % uncomment/comment to switch back from parallel
        %[sim(sn).Xfit] = fit_any_v1( sim(sn), data_flag, model_flag, n_fit );
        [sim(sn,model_flag).Xfit] = fit_any_parallel_v1_TF( sim(sn, model_flag), data_flag, model_flag, n_fit );
    end
end
toc
% save('MainTF_Parm_recovery_Sim_All6models_HalfDist_gammadest','sim')
%% Supplementary FIGURE 3 plot parameter recovery
var_name{6,:} = [];
var_name{6} = 'initial uncertainity';

for model_flag = 2
    [parm_r{model_flag}, parm_p{model_flag}]=plot_parameterRecovery_v1_TF( sim(:, model_flag), model_flag, var_name, var_symbol, par_order(1:6) );
end

%% plot parameter recovery correlations
figure; clf;
set(gcf, 'position',[440   504   468   300])
hg = [0.35 0.18];
wg = [0.2 0.2 0.2 0.03];

[ax, hb, wb] = easy_gridOfEqualFigures(hg, wg);
ord = [4 6 5];
for model_flag = 1:3

[t] = plot_confusionMatrix_v1(ax(model_flag), parm_r{model_flag}', var_symbol(par_order(1:ord(model_flag))), hg, wg, hb, wb);
set(t(parm_p{model_flag}'<=0.05), 'color', 'w')
set(t, 'fontsize', 14)
col = make_coolColorMap_v1(AZblue, [1 1 1], AZblue);
set(ax(model_flag), 'colormap', col)
end
%% FIGURE S6 does it introduce corrleations? No!!!
var_name{6,:} = [];
var_name{6} = 'initial uncertainity';
model_flag = 2;
figure; clf;
set(gcf, 'position',[440   504   468   250])
ax = easy_gridOfEqualFigures([0.2 0.04], [0.12 0.15 0.07]);
i = 1; j = 3;
plot_parameterRecoveryCorrelations_v1(ax(1), sim, model_flag, var_name, var_symbol, i, j)
addABCs(ax, [-0.07 0.0], 24,'AB')
i = 3; j = 5;
plot_parameterRecoveryCorrelations_v1(ax(2), sim, model_flag, var_name, var_symbol, i, j)

figure; clf;
set(gcf, 'position',[440   504   468   250])
ax = easy_gridOfEqualFigures([0.2 0.04], [0.12 0.15 0.07]);
i = 1; j = 5;
plot_parameterRecoveryCorrelations_v1(ax(1), sim, model_flag, var_name, var_symbol, i, j)
addABCs(ax, [-0.07 0.0], 24,'CD')
i = 3; j = 2;
plot_parameterRecoveryCorrelations_v1(ax(2), sim, model_flag, var_name, var_symbol, i, j)

%% parameter recovery for No Feedback model on no feedback trials
clear sim_NoFB
Nsim = 30;

model_flag = 1;
rng(10); % set random seed for reproducibility
Xsim = generate_simulationParameters(3,  Nsim, Xfit(:,:,model_flag));
expt = generate_simulationTrials(sub, Nsim);
for count = 1:length(expt)
    sim_NoFB(count, model_flag) = sim_any_v2_TF(model_flag, Xsim(count,:), expt(count));
end


%% fit fake data for no feedback condition only (data_flag = 1)
data_flag = 1;
n_fit = 100;
model_flag = 1;
for sn = 1:length(sim)
    disp(['model_flag = ' num2str(model_flag) '; subject = ' num2str(sn)])
    % uncomment/comment to switch back from parallel
    %[sim(sn).Xfit] = fit_any_v1( sim(sn), data_flag, model_flag, n_fit );
    [sim_NoFB(sn,model_flag).Xfit] = fit_any_parallel_v1_TF( sim_NoFB(sn, model_flag), data_flag, model_flag, n_fit );
end
save('MainTF_Parm_recovery_Sim_NoFB_HalfDist_Path','sim')
%% Supplementary Figure 2 plot parameter recovery
ind = [3 1 2 4 5 6 7 8];

model_flag = 1;
[r_NoFB,p_NoFB]=plot_parameterRecovery_v1_TF( sim_NoFB(:, model_flag), model_flag, var_name, var_symbol, ind)
% saveFigurePdf(gcf, ['~/Desktop/KF_parameterRecovery_noFeedback_' num2str(model_flag)]);


%% ========================================================================
%% confusion matrix %% confusion matrix %% confusion matrix %%
%% confusion matrix %% confusion matrix %% confusion matrix %%
%% confusion matrix %% confusion matrix %% confusion matrix %%
%% ========================================================================

%% simulate fake data for confusion matrix
Nsim = 30;

clear sim
for model_flag = 1:6
    Xsim = generate_simulationParameters(4,  Nsim, Xfit(:,:,model_flag));
    expt = generate_simulationTrials(sub, Nsim);
    for count = 1:length(expt)
        sim(count, model_flag) = sim_any_v2_TF(model_flag, Xsim(count,:), expt(count));
    end
end


%% compute confusion matrix
data_flag = 3;

n_fit  = 100;
for model_flag_sim  = 1:6
    for model_flag_fit  = 1:6

        for i = 1:length(sim(:, model_flag_sim))
            disp(sprintf('sim = %d, fit = %d, sub = %d', model_flag_sim, model_flag_fit, i));
            [~, ~, BIC_matrix(i, model_flag_fit, model_flag_sim), AIC_matrix(i, model_flag_fit, model_flag_sim)] ...
               = fit_any_parallel_v1_TF( sim(i, model_flag_sim), data_flag, model_flag_fit, n_fit );
        end
    end
end

% save('MainTF_BIC_All6_FBtrueFB_HalfDist_confusion_matrix','BIC')


%% FIGURE 7 - confusion matrix and Best fit 
% Plotting Best fit before confusion matrix
model_name = {'Path Integration' 'Kalman Filter' 'Pure Landmark' };

figure; clf;
sf = 1.5;
set(gcf, 'position',[440   504   468*sf   130*sf]*1)
hg = [0.35  0.18 ]*0.98;
wg = [0.2 0.1  0.2 0.03]*0.98;
[ax , hb, wb] = easy_gridOfEqualFigures(hg, wg);

[b, yl, nBestFit, best_model] = plot_modelComparison(ax(2), [BIC(:,1:2) BIC(:,3)], model_name);


set(ax(2), 'xticklabel', [])
axes(ax(1)); hold on;
NewBIC = [];
NewBIC = [BIC(:,1:2) BIC(:,3)];
Y = NewBIC(:,1:3) - repmat(NewBIC(:,2), [1 3]);
% Y = BIC - repmat(BIC(:,4), [1 4]);
r = (rand(size(Y,1),1)-0.5)*0.2;
plot([0.5 3.5], [0 0], 'k')
for i = 1:3
    plot(r+i, Y(:,i)','o', 'markersize', 8, 'color', AZred);
end

plot([0.5 3.5], [10 10], 'k--')
set(gca, 'view', [90 90], 'xtick', 1:5, 'xticklabel', model_name, ...
    'xlim', [0.5 3.5], 'ylim', [-200 400])
set(ax, ...
    'tickdir',      'out', ...
    'fontsize',     12);

ylabel('BIC(Model) - BIC(Kalman)')
% ylabel('BIC(Model) - BIC(Hybrid)')
% xlabel('model')

addABCs(ax(1), [-0.15 0.2], 15*sf)
addABCs(ax(2), [-0.04 0.2], 15*sf, 'B')

CM = zeros(3);
% go through each subject and each simulated model and find the best fit
for i = 1:size(sim,1)
    for model_flag_sim = 1:3
        [minBIC] = min(BIC_matrix(i,1:3,model_flag_sim));
        % get update to row of confusion matrix (allow for ties)
        dum = BIC_matrix(i,1:3,model_flag_sim) == minBIC;
        dum = dum / sum(dum);

        % update confusion matrix
        CM(model_flag_sim,:) = CM(model_flag_sim,:) + dum;
    end
end

% normalize confusion matrix to get p(fit | sim)
p_fitGivenSim = CM ./ repmat(sum(CM,2), [1 size(CM,2)]);

% normalize to get p(sim | fit)
p_simGivenFit = CM ./ repmat(sum(CM,1), [ size(CM,1) 1]);

[t] = plot_confusionMatrix_v1(ax(3), p_fitGivenSim, model_name, hg, wg, hb, wb);
set(t(p_fitGivenSim>=0.5), 'color', 'w')
set(t, 'fontsize', 12)

set(ax(3), 'fontsize', 12)

% xl3 = add_externalXlabelsTop(sum(wg(3:4))+wb(3), hg, wb, hb, 0.03, {'confusion matrix p(fit | sim)'});

set(ax(3), 'xticklabelrotation', 30, 'tickdir', 'out')
col = make_coolColorMap_v1(AZblue, [1 1 1], AZred);
% colormap(col)
colormap(ax(3), col)

addABCs(ax(3), [-0.15 0.2], 15*sf, 'C')

%% Supplementary Figure 4 Confusion matrix for all models
model_name = {'Path Integration' 'Kalman Filter' 'Pure Landmark' 'Cue Combination' 'Hybrid' 'Random Sampling'  };

% model_name{2} = 'Kalman Filter';
CM = zeros(6);
% go through each subject and each simulated model and find the best fit
for i = 1:size(sim,1)
    for model_flag_sim = 1:6
        [minBIC] = min(BIC_matrix(i,:,model_flag_sim));
        % get update to row of confusion matrix (allow for ties)
        dum = BIC_matrix(i,:,model_flag_sim) == minBIC;
        dum = dum / sum(dum);

        % update confusion matrix
        CM(model_flag_sim,:) = CM(model_flag_sim,:) + dum;
    end
end

% normalize confusion matrix to get p(fit | sim)
p_fitGivenSim = CM ./ repmat(sum(CM,2), [1 size(CM,2)]);

% normalize to get p(sim | fit)
p_simGivenFit = CM ./ repmat(sum(CM,1), [ size(CM,1) 1]);


figure; clf;
set(gcf, 'position',[440   504   468   300])
hg = [0.35 0.18];
wg = [0.26 0.08 0.03];

[ax, hb, wb] = easy_gridOfEqualFigures(hg, wg);
[t] = plot_confusionMatrix_v1(ax(1), p_fitGivenSim, model_name, hg, wg, hb, wb);
set(t(p_fitGivenSim'>=0.5), 'color', 'w')
set(t, 'fontsize', 14)
% [t] = plot_confusionMatrix_v1(ax(1), p_simGivenFit, model_name, hg, wg, hb, wb);
[t] = plot_confusionMatrix_v1(ax(2), p_simGivenFit, model_name, hg, wg, hb, wb);
set(t(p_simGivenFit'>0.5), 'color', 'w')
set(t, 'fontsize', 14)
set(ax(2), 'yticklabel', [])
set(ax, 'fontsize', 14)
% yl = add_externalYlabelsRotation_v1(wg, hg, wb, hb, 90, 0, {'simulated model'});
% xl = add_externalXlabelsBottom(wg, hg, wb, hb, -0.25, {'fit model'});
% xl2 = add_externalXlabelsBottom(sum(wg(1:2))+wb(1), hg, wb(1), hb, -0.25, {'fit model'});

xl3 = add_externalXlabelsTop(wg, hg, wb, hb, 0.03, {'confusion matrix   p(fit | sim)'});
xl4 = add_externalXlabelsTop(sum(wg(1:2))+wb(1), hg, wb, hb, 0.03, {'inversion matrix   p(sim | fit)'});
% set([xl yl xl2 xl3 xl4], 'fontsize', 16)
% set([xl3 xl4], 'fontsize', 16)


set(ax, 'xticklabelrotation', 40, 'tickdir', 'out')
% [t, xl, yl] = plot_confusionMatrix_v1(ax(2), p_simGivenFit, model_names, ...
%     hg, wb(1)+wg(1)+wg(2:end), hb, wb(2:end)); % positioning here is a hack
% set(ax(2), 'yticklabel', []);
% set(yl, 'visible', 'off')

% axes(ax(1));

col = make_coolColorMap_v1(AZblue, [1 1 1], AZred);
% colormap(col)
colormap(ax(1), col)
col = make_coolColorMap_v1(AZblue, [1 1 1], AZblue);
set(ax(2), 'colormap', col)

addABCs(ax, [-0.02 0.18], 22);
% saveFigurePdf(gcf, [figdir 'KF_confusionMatrix'])