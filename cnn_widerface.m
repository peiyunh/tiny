%  FILE:   cnn_widerface.m
%
%    This function serves as the main function for training a model.
%  
%  INPUT:  configuration     (see code for details) 
%
%  OUTPUT: net               (trained network)
%          info              (training stats)

function [net, info] = cnn_widerface(varargin)
startup;


opts.keepDilatedZeros = false;
opts.inputSize = [500, 500];
opts.learningRate = 1e-4;

%% use customized training function ie. adam
opts.trainFn = '@cnn_train_dag';
opts.batchGetterFn = '@cnn_get_batch_logistic_zoom';
opts.freezeResNet = false;
opts.tag = '';
opts.clusterNum = 25;
opts.clusterName = '';
opts.bboxReg = true;
opts.skipLRMult = [1, 0.001, 0.0001, 0.00001];
opts.sampleSize = 256;
opts.posFraction = 0.5;
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.border = [0, 0];
opts.pretrainModelPath = 'matconvnet/pascal-fcn8s-tvg-dag.mat';
opts.dataDir = fullfile('data','widerface') ;
opts.modelType = 'pascal-fcn8s-tvg-dag' ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.minClusterSize = [2, 2]; 
opts.maxClusterSize = opts.inputSize;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.freezeResNet, sfx = ['freezed-' sfx] ; end
sfx = [sfx '-' 'sample' num2str(opts.sampleSize)] ; 
sfx = [sfx '-' 'posfrac' num2str(opts.posFraction)] ; 
sfx = [sfx '-' 'N' num2str(opts.clusterNum)];
if opts.bboxReg, sfx = [sfx '-' 'bboxreg']; end

if any(opts.inputSize~=500), 
    sz = opts.inputSize;
    sfx = [sfx '-input' num2str(sz(1)) 'x' num2str(sz(2))]; 
end
if ~isempty(opts.clusterName)
    sfx = [sfx '-' 'cluster-' opts.clusterName]; 
end

opts.expDir = fullfile('models', ['widerface-' sfx]) ;
if ~isempty(opts.tag)
    opts.expDir = [opts.expDir '-' opts.tag];
end
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.expDir

opts.batchSize = 10;
opts.numSubBatches = 1;
opts.numEpochs = 50;
opts.gpus = [1];
opts.numFetchThreads = 8;
opts.lite = false ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts = vl_argparse(opts, varargin) ;

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;

opts.train.gpus = opts.gpus;
opts.train.batchSize = opts.batchSize;
opts.train.numSubBatches = opts.numSubBatches;
opts.train.numEpochs = opts.numEpochs;
opts.train.learningRate = opts.learningRate;

opts.train.keepDilatedZeros = opts.keepDilatedZeros;

%% initialize model structure
fprintf('Trying to initialize the structure of %s\n', opts.modelType);
net = cnn_init('model', opts.modelType, ...
               'batchNormalization', opts.batchNormalization, ...
               'weightInitMethod', opts.weightInitMethod, ...
               'networkType', opts.networkType) ;

%% load pretrained weights
if ~isempty(opts.pretrainModelPath)
  fprintf('Loading pretrained weights from %s\n', opts.pretrainModelPath);
  if ~exist(opts.pretrainModelPath)
    [~,model_name,~] = fileparts(opts.pretrainModelPath);
    download_path = sprintf('./trained_models/%s.mat', model_name);
    download_url = sprintf('http://www.vlfeat.org/matconvnet/models/%s.mat', model_name); 
    fprintf('Downloading model %s from %s to %s ... \n', model_name, download_url, download_path);
    cmd = sprintf('wget -O %s %s', download_path, download_url); 
    system(cmd);
  end
  net = cnn_load_pretrain(net, opts.pretrainModelPath);
end

net.meta.inputSize = opts.inputSize;
net.meta.normalization.inputSize = opts.inputSize;
net.meta.normalization.border = opts.border;
net.meta.augmentation.transformation = 'none'; 
net.meta.augmentation.rgbVariance = [];

if ~exist(opts.expDir), mkdir(opts.expDir); end;

%% prepare data
if exist(opts.imdbPath)
    imdb = load(opts.imdbPath) ;
    fprintf('Loaded imdb from %s\n', opts.imdbPath);
else
    imdb = cnn_setup_imdb('dataDir', opts.dataDir);
    save(opts.imdbPath, '-struct', 'imdb') ;
    fprintf('Saved imdb to %s\n', opts.imdbPath);
end

%% save model options
optpath = fullfile(opts.expDir, 'opts.mat');
if ~exist(optpath), save(optpath, 'opts'); end


%% define batch getter function
%batchGetter = @cnn_get_batch_logistic_zoom;
if ~isempty(opts.batchGetterFn)
    batchGetter = str2func(opts.batchGetterFn);
end

%% clustering
minh = opts.minClusterSize(1); minw = opts.minClusterSize(2);
maxh = opts.maxClusterSize(1); maxw = opts.maxClusterSize(2);
if isempty(opts.clusterName)
    tmp_str = ['%s_N%d_' sprintf('min%dx%d_max%dx%d.mat',minh,minw,maxh,maxw)];
else
    tmp_str = ['%s_N%d_' opts.clusterName '.mat'];
end
clusterName = @(name,nd)(sprintf(tmp_str, name, nd));

%clusterName = @(name,nd,minh,minw,maxh,maxw)(sprintf( ...
%    '%s_N%d_min%dx%d_max%dx%d.mat',name,nd,minh,minw,maxh,maxw));
%clusterPath = fullfile(opts.dataDir, clusterName(...
%    'RefBox',opts.clusterNum,minh,minw,maxh,maxw));
boxName = 'RefBox'; 
clusterPath = fullfile(opts.dataDir, clusterName(boxName,opts.clusterNum));

fprintf('cluster path: %s\n', clusterPath);

if ~exist(clusterPath)
  clusters = cluster_rects(imdb, opts.clusterNum, [minh minw], [maxh maxw]);
  save(clusterPath, 'clusters');
else
  load(clusterPath);
end
net.meta.clusters = clusters;

%% compute image stats
imageStatsPath = fullfile(opts.dataDir, 'imageStats.mat') ;
if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
    [averageImage, rgbMean, rgbCovariance] = getImageStats(batchGetter, ...
                                                      opts, net.meta, imdb) ;
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
net.meta.augmentation.transformation = 'f5';
net.meta.normalization.averageImage = rgbMean ;
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
clear v d ;

%% add predictors/losses
switch opts.modelType
  case 'resnet-101-simple'
    net = cnn_add_loss_fcn8s_resnet101_simple(opts, net);
  otherwise
    error(sprintf('Not Implemented: model type %s', opts.modelType));
end

%% compute receptive fields and canonical variable sizes 
var2idx = containers.Map;
for i = 1:numel(net.vars)
    var2idx(net.vars(i).name) = i;
end
net.meta.var2idx = var2idx;

sz_ = opts.inputSize;
net.meta.varsizes = net.getVarSizes({'data',[sz_,3,1]});
net.meta.recfields = net.getVarReceptiveFields('data');

%% configure sampling hyperparameters
net.meta.sampleSize = opts.sampleSize;
net.meta.posFraction = opts.posFraction;
net.meta.posThresh = opts.posThresh;
net.meta.negThresh = opts.negThresh;

%% assign training function
trainFn = str2func(opts.trainFn); 

%% print out training options
opts
opts.train

%% start training (no validation)
derOutputs = {'loss_cls', 1, 'loss_reg', 1}; 
[net, info] = trainFn(net, imdb, getBatchFn(batchGetter, opts, net.meta), ...
                      'expDir', opts.expDir,...
                      'derOutputs', derOutputs, ...
                      'val', nan,...
                      opts.train) ;


%% wrapper for batch getter function
function fn = getBatchFn(batchGetter, opts, meta)
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.inputSize = meta.normalization.inputSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

if isfield(meta, 'sampleSize'), bopts.sampleSize = meta.sampleSize; end;
if isfield(meta, 'posFraction'), bopts.posFraction = meta.posFraction; end;
if isfield(meta, 'posThresh'), bopts.posThresh = meta.posThresh; end;
if isfield(meta, 'negThresh'), bopts.negThresh = meta.negThresh; end;

if isfield(meta, 'clusters'), bopts.clusters = meta.clusters; end;
if isfield(meta, 'var2idx'), bopts.var2idx = meta.var2idx; end;
if isfield(meta, 'varsizes'), bopts.varsizes = meta.varsizes; end;
if isfield(meta, 'recfields'), bopts.recfields = meta.recfields; end;
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(batchGetter, bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(batchGetter, bopts,useGpu,x,y) ;
end

%% interface to batch getter function
function inputs = getDagNNBatch(batchGetter, opts, useGpu, imdb, batch)
imagePaths = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
imageSizes = imdb.images.size(batch, :);
%isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;
isVal = 0;
if isfield(opts, 'clusters') && ~isVal 
    labelRects = imdb.labels.rects(batch);
else
    labelRects = [];
end

[images, clsmaps, regmaps] = batchGetter(imagePaths, imageSizes, labelRects, ...
                                       opts, 'prefetch', nargout == 0) ;

if nargout > 0
    % if we are training
    if useGpu && ~isempty(clsmaps) && ~isempty(regmaps)
        images = gpuArray(images) ;
    end
    inputs = {'data', images};
    
    if ~isempty(clsmaps)
        inputs(end+1:end+2) = {'label_cls', clsmaps}; 
    end
    if ~isempty(regmaps)
        inputs(end+1:end+2) = {'label_reg', regmaps};
    end
end

%% function to collect image statistics 
function [averageImage, rgbMean, rgbCovariance] = ...
    getImageStats(batchGetter, opts, meta, imdb)
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'dagnn' ;
fn = getBatchFn(batchGetter, opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    temp = gather(temp{2});
    z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
    n = size(z,2) ;
    avg{end+1} = mean(temp, 4) ;
    rgbm1{end+1} = sum(z,2)/n ;
    rgbm2{end+1} = z*z'/n ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;


