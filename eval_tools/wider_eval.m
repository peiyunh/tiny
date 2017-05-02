% WIDER FACE Evaluation
% Conduct the evaluation on the WIDER FACE validation set. 
%
% Shuo Yang Dec 2015
%
clear;
close all;
addpath(genpath('./plot'));

%Please specify your prediction directory.
%pred_dir = '../result/half-withdown-normal-double';
%pred_name = 'half-withdown-normal-double';

%% what we are trying to reproduce 
% res101 tri res limit 30% hardmine train with neg borders (test as well, A+B)
% pred_dir = '../result/widerface-resnet-101-simple-dropout-fcn8s-sample256-posfrac0.5-N25-bboxreg-logistic-cluster-scaled-clusterNoResize-scaled-trires-limit30-hardmine-negborder-epoch47-val-probthresh0.03-nmsthresh0.3-testsizeNaN-multires-evalWithReg-nb-AB';
% pred_name = 'res101-scaled-trires-limit30-hardmine-negborders-nb-AB';

gt_dir = './ground_truth/wider_face_val.mat';

pred_dir = '../result/widerface-resnet-101-simple-sample256-posfrac0.5-N25-bboxreg-cluster-scaled-epoch25-val-probthresh0.03-nmsthresh0.3-multires-evalWithReg';
pred_name = 'hr_res101';

pred_list = read_pred(pred_dir, gt_dir);
norm_pred_list = norm_score(pred_list);

%evaluate on different settings
setting_name_list = {'easy_val';'medium_val';'hard_val'};
setting_class = 'setting_int';

%Please specify your algorithm name.
legend_name = pred_name; 
for i = 1:size(setting_name_list,1)
    fprintf('Current evaluation setting %s\n',setting_name_list{i});
    setting_name = setting_name_list{i};
    gt_dir = sprintf('./ground_truth/wider_%s.mat',setting_name);
    evaluation(norm_pred_list,gt_dir,setting_name,setting_class,legend_name);
end

fprintf('Plot pr curve under overall setting.\n');
dataset_class = 'Val';

% scenario-Int:
seting_class = 'int';
dir_int = sprintf('./plot/baselines/%s/setting_%s',dataset_class, seting_class);
wider_plot(setting_name_list,dir_int,seting_class,dataset_class);
