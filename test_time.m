clear all;

addpath matconvnet;
addpath matconvnet/matlab;
vl_setupnn;

addpath toolbox/nms;
addpath toolbox/export_fig;

gpu_id = 3;
%model_path = 'models/hr_res101.mat';
model_path = 'models/imagenet-resnet-101-dag.mat';

% loadng pretrained model (and some final touches) 
net = load(model_path);
%net = dagnn.DagNN.loadobj(net.net);
net = dagnn.DagNN.loadobj(net);
if gpu_id > 0 % for matconvnet it starts with 1 
    gpuDevice(gpu_id);
    net.move('gpu');
end
%net.layers(net.getLayerIndex('score4')).block.crop = [1,2,1,2];
%net.addLayer('cropx',dagnn.Crop('crop',[0 0]),...
%             {'score_res3', 'score4'}, 'score_res3c'); 
%net.setLayerInputs('fusex', {'score_res3c', 'score4'});
%net.addLayer('prob_cls', dagnn.Sigmoid(), 'score_cls', 'prob_cls');

%
iter_num = 100;

scale = 2; 

%
input_size = [1080*scale, 1920*scale, 3, 1];
img = gpuArray(rand(input_size, 'single'));
t1 = tic; 
for i = 1:iter_num
    net.eval({'data', img});
    wait(gpuDevice) ;
end
t2 = toc(t1);
fprintf('input size: [%d %d %d], time: %.6f\n', size(img), t2/iter_num);

%
input_size = [720*scale, 1280*scale, 3, 1];
img = gpuArray(rand(input_size, 'single'));
t1 = tic; 
for i = 1:iter_num
    net.eval({'data', img});
    wait(gpuDevice) ;
end
t2 = toc(t1);
fprintf('input size: [%d %d %d], time: %.6f\n', size(img), t2/iter_num);

%
input_size = [480*scale, 640*scale, 3, 1];
img = gpuArray(rand(input_size, 'single'));
t1 = tic; 
for i = 1:iter_num
    net.eval({'data', img});
    wait(gpuDevice) ;
end
t2 = toc(t1);
fprintf('input size: [%d %d %d], time: %.6f\n', size(img), t2/iter_num);