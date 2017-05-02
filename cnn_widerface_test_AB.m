%  FILE:   cnn_widerface.m
%
%    This function serves as the main function for testing a model.
%  
%  INPUT:  configuration (see code for details)
%
%  OUTPUT: none          (detections will be written to files)

function cnn_widerface_test_AB(varargin)

startup;

opts.allscale_templateIds = [];
opts.onescale_templateIds = [];

opts.overWrite = false;
opts.noOrgres = false;
opts.noDownsampling = false;
opts.noUpsampling = false;
opts.testMultires = false;
opts.testTag = '';
opts.inputSize = [500, 500]; 

opts.noContext = false; 

opts.evalWithReg = true;
opts.tag = '';
opts.bboxReg = true;
opts.skipLRMult = [1, 0.001, 0.0001, 0.00001];
opts.vis = true;
opts.useDropout = true;
opts.testEpoch = 0;
opts.testIter = 0;
opts.testSet = 'val';
opts.resDir = 'result/';
opts.clusterNum = 25;
opts.clusterName = ''; 
opts.sampleSize = 256;
opts.posFraction = 0.5;
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.border = [0, 0];
opts.pretrainModelPath = 'matconvnet/pascal-fcn8s-tvg-dag.mat';
opts.dataDir = fullfile('data','widerface') ;
opts.modelType = 'pascal-fcn8s-tvg-dag' ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.probThresh = 0.05;
opts.nmsThresh = 0.3;

sfx = opts.modelType ;
if opts.useDropout, sfx = [sfx '-dropout']; end
sfx = [sfx '-' 'sample' num2str(opts.sampleSize)] ; 
sfx = [sfx '-' 'posfrac' num2str(opts.posFraction)] ; 
sfx = [sfx '-' 'N' num2str(opts.clusterNum)];

if opts.bboxReg, sfx = [sfx '-' 'bboxreg']; end
if opts.noContext, sfx = [sfx '-' 'nocontext']; end;

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
opts.numEpochs = 20;
opts.gpus = [];
opts.learningRate = 0.0001;
opts.numFetchThreads = 4 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts = vl_argparse(opts, varargin) ;

opts

% load imdb
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_setup_imdb('dataDir', opts.dataDir);
  if ~exist(opts.expDir), mkdir(opts.expDir) ; end
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% load model for testing 
opts.networkType = 'dagnn' ;

% test latest model if no other options
if opts.testEpoch == 0
  testEpoch = findLastCheckpoint(opts.expDir);
  if testEpoch==0
    error('No available model for evaluation.');
  end
else
  testEpoch = opts.testEpoch; 
end
testSet = opts.testSet;
testModelName = sprintf('net-epoch-%d.mat',testEpoch);
testModelPath = fullfile(opts.expDir, testModelName);
fprintf('Evaluating model from %s\n', testModelPath);

% generate tag based on test settings 
testName = strrep(opts.expDir, 'models/', '');
testName = sprintf('%s-epoch%d-%s', testName, testEpoch, testSet);
testName = [testName '-probthresh' num2str(opts.probThresh)]; 
testName = [testName '-nmsthresh' num2str(opts.nmsThresh)]; 
if opts.testMultires, testName = [testName '-multires']; end
if opts.noOrgres, testName = [testName '-noorgres']; end
if opts.noUpsampling, testName = [testName '-noupsampling']; end
if opts.noDownsampling, testName = [testName '-nodownsampling']; end
if opts.evalWithReg, testName = [testName '-evalWithReg']; end

% append additional tags 
if ~isempty(opts.testTag),
  testName = [testName '-' opts.testTag];
end

% generate result directory 
resDir = fullfile(opts.resDir, testName);
if numel(resDir) > 255, resDir = strrep(resDir, '-nmsthresh0.3', ''); end
if numel(resDir) > 255, resDir = strrep(resDir, 'widerface-resnet-50-',''); end
if numel(resDir) > 255, resDir = strrep(resDir, 'bboxreg-logistic-cluster-',''); end
if ~exist(resDir) mkdir(resDir); end
for i = 1:numel(imdb.events.name)
  eventDir = fullfile(resDir, imdb.events.name{i});
  if ~exist(eventDir), mkdir(eventDir); end
end

% load core net 
net_ = load(testModelPath);
if isfield(net_, 'net')
  net = dagnn.DagNN.loadobj(net_.net);
else
  net = dagnn.DagNN.loadobj(net_);
end
clear net_;

% make sure matconvnet does not clear my precious variables 
%net.vars(net.getVarIndex('score_res4')).precious = 1; 
%net.vars(net.getVarIndex('score_res3')).precious = 1; 
%net.vars(net.getVarIndex('score_final')).precious = 1; 

% add cropping so that our network adjusts to different resolution at test time
% (note: during training we always look at cropped regions with fixed size)
net.layers(net.getLayerIndex('score4')).block.crop = [1,2,1,2];
net.addLayer('cropx',dagnn.Crop('crop',[0 0]), {'score_res3', 'score4'}, 'score_res3c'); % post crop
net.setLayerInputs('fusex', {'score_res3c', 'score4'});

% add a probability output layer
net.addLayer('prob_cls', dagnn.Sigmoid(), 'score_cls', 'prob_cls');

% build a dictionary from variable name to index
var2idx = containers.Map;
for i = 1:numel(net.vars)
  var2idx(net.vars(i).name) = i;
end
net.meta.var2idx = var2idx;
net.meta.recfields = net.getVarReceptiveFields('data');

% move network onto gpu
net.mode = 'test' ;
if ~isempty(opts.gpus),
  assert(numel(opts.gpus) == 1);
  gpuDevice(opts.gpus);
  net.move('gpu'); 
end

% 
vis = opts.vis;

% setup test set
switch opts.testSet
  case 'val'
    test = find(imdb.images.set==2);
  case 'test'
    test = find(imdb.images.set==3);
end

% canonical shapes 
clusters = net.meta.clusters;
clusters_h = clusters(:,4) - clusters(:,2) + 1;
clusters_w = clusters(:,3) - clusters(:,1) + 1;
clusters_scale = clusters(:,5); 
max_cluster_h = max(clusters_h); 
max_cluster_w = max(clusters_w);

% average image
averageImage = reshape(net.meta.normalization.averageImage,1,1,3);


% Type A and Type B template Ids 
if isempty(opts.allscale_templateIds) && isempty(opts.onescale_templateIds)
  error('at least specify allscale templates or onescale template');
else
  allscale_templateIds = opts.allscale_templateIds;
  onescale_templateIds = opts.onescale_templateIds;
  ignoredTemplateIds = setdiff(1:size(clusters,1), [allscale_templateIds onescale_templateIds]);
end

% Run evaluation image-by-image
probThresh = opts.probThresh;
nmsThresh = opts.nmsThresh;

t1 = tic;
for i = 1:numel(test)
  % input file
  imagePath = fullfile(imdb.imageDir, imdb.images.name{test(i)});
  
  % output file
  [~,imname,imext] = fileparts(imagePath);
  eventId = imdb.labels.eventid(test(i));
  event = imdb.events.name{eventId};
  ofile = fullfile(resDir, event, [imname '.txt']);

  if ~opts.overWrite && ~vis && exist(ofile),
    continue;
  end

  imgsize = imdb.images.size(test(i),:);
  minside = min(imdb.images.size(test(i),:));
  if opts.testMultires
    scales = [-2, -1, 0, 1];
  else
    scales = [0] ;
  end

  if opts.noUpsampling
    scales(scales > 0) = []; 
  end

  if opts.noDownsampling
    scales(scales < 0) = []; 
  end

  if opts.noOrgres
    scales(scales == 0) = [];
  end

  if isempty(scales),
    error('Error: no scale to test (scales is empty)');
  end

  drects = [];
  for s = 2 .^ scales
    testSize = minside * s; 

    % scale-specific template selection
    if s < 1 % if we down-sample, no med/small templates
      invalid_onescale_idx = find(clusters_scale(onescale_templateIds) >= 1);
    elseif s == 1 % if no re-sample, only med templates
      invalid_onescale_idx = find(clusters_scale(onescale_templateIds) ~= 1);
    elseif s > 1 % if we up-sample, no big/med templates
      invalid_onescale_idx = find(clusters_scale(onescale_templateIds) <= 1);
    end
    invalid_tid = [ignoredTemplateIds onescale_templateIds(invalid_onescale_idx)];

    % read input (only resize when we have to) 
    if testSize ~= min(imdb.images.size(test(i),:))
      ims = vl_imreadjpeg({imagePath}, 'resize', testSize);
    else
      ims = vl_imreadjpeg({imagePath});
    end
    im = uint8(ims{1});

    % give up when input resolution is too huge 
    if s > 1 && (size(im,1) > 5000 || size(im,2) > 5000)
      continue;
    end

    % subtract average image and feed into the network
    im_ = bsxfun(@minus, single(im), averageImage);
    if strcmp(net.device, 'gpu')
      im_ = gpuArray(im_);
    end
    inputs = {'data', im_};
    net.eval(inputs(1:2));
    
    % compute classification and regression scores
    score_cls = gather(net.vars(net.getVarIndex('score_cls')).value);
    score_reg = gather(net.vars(net.getVarIndex('score_reg')).value);
    prob_cls = gather(net.vars(net.getVarIndex('prob_cls')).value);

    % decompose regression predictions
    tx = score_reg(:,:,1:opts.clusterNum); 
    ty = score_reg(:,:,opts.clusterNum+1:opts.clusterNum*2); 
    tw = score_reg(:,:,opts.clusterNum*2+1:opts.clusterNum*3); 
    th = score_reg(:,:,opts.clusterNum*3+1:opts.clusterNum*4); 
    
    % 
    net.meta.inputSize = size(im_);
    net.meta.normalization.inputSize = size(im_);
    net.meta.varsizes = net.getVarSizes({'data', [size(im_),1]});

    % NOTE: no need to do this since we allow negative appearing on the border during training
    %viomask = violation(net.meta,clusters,size(im_,1),size(im_,2));
    %prob_cls(viomask) = 0;

    % zero out prediction from templates that are invalid on this scale
    prob_cls(:,:,invalid_tid) = 0 ;
    
    % gather initial detections
    idx = find(prob_cls > probThresh);
    scores = score_cls(idx);

    % translate feature-level coordinates to pixel-level coordinates
    [fy,fx,fc] = ind2sub(size(prob_cls), idx);
    [cy,cx] = backtrack(net.meta, fy, fx);
    cw = clusters(fc,3)-clusters(fc,1)+1;
    ch = clusters(fc,4)-clusters(fc,2)+1;

    % apply regression refinement if necessary
    if opts.evalWithReg
      dcx = cw .* tx(idx); 
      dcy = ch .* ty(idx);
      rcx = cx + dcx;
      rcy = cy + dcy;
      rcw = cw .* exp(tw(idx));
      rch = ch .* exp(th(idx));
      tmp_drects = [rcx-rcw/2,rcy-rch/2,rcx+rcw/2,rcy+rch/2];
    else
      tmp_drects = [cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2];
    end
    
    % convert detections to the original scale 
    original_size = imdb.images.size(test(i),:);
    factor = min(original_size ./ testSize); 
    tmp_drects = bsxfun(@times, tmp_drects, factor);
    tmp_drects = horzcat(tmp_drects, fc, scores);
    drects = vertcat(drects, tmp_drects) ;
  end

  % append score and do nms
  ridx = nms(drects(:,[1:4 end]), nmsThresh);
  srects = drects(ridx, :);
  
  % round to pixel coordinates
  if ~isempty(srects)
    srects(:,1:4) = round(srects(:,1:4));
  end
  
  % output to file
  fout = fopen(ofile, 'w');
  fprintf(fout, '%s/%s%s\n', event, imname, imext);
  fprintf(fout, '%d\n', size(srects,1));
  for j = 1:size(srects, 1)
    fprintf(fout, '%d %d %d %d %d %f\n', ...
            srects(j,1), srects(j,2), ...
            srects(j,3)-srects(j,1)+1, ...
            srects(j,4)-srects(j,2)+1, ...
            srects(j,5), srects(j,6));
  end
  fclose(fout);
  
  if vis
    figure(1);
    clf;

    im = imread(imagePath);
    imshow(im);
    hold on ;
    if test(i) <= numel(imdb.labels.rects)
      grects = imdb.labels.rects{test(i)};
    else
      grects = [];
    end
    
    if ~isempty(grects),
      % get rid of ground truth that are too small 
      grects(find(grects(:,4)-grects(:,2)+1<10),:) = [];
      % for each ground truth, compute the best overlap with any detection
      ovlp = 1 - pdist2(srects(:,1:4), grects, @rect_dist, 'smallest', 1);
      colors = [0 0 0];
      if all(ovlp==0)                % no detection associated with this gt at all
        colors = [1 0 0];            
      else
        colors(ovlp >= 0.5, 2) = 1;  % we got this one 
        colors(ovlp < 0.5, 1) = 1;   % we failed this one 
      end
      plotBoxes(grects(:,1), grects(:,2), ...
                grects(:,3)-grects(:,1)+1, grects(:,4)-grects(:,2)+1,...
                colors, 3*ones(size(grects,1),1));
      rec = sum(ovlp>=0.5);          % we can compute a per-image recall rate
      rec_rate = rec / size(grects, 1);
    else
      rec = nan;
      rec_rate = nan;
    end

    % plot detections
    srects = double(srects);
    % plotBoxes(srects(:,1), srects(:,2), ...
    %           srects(:,3)-srects(:,1)+1, ...
    %           srects(:,4)-srects(:,2)+1,...
    %           [0 0 1], 2*ones(size(srects,1),1));
    visualize_detection([], srects, 0.3);
    hold off;
    
    msg = sprintf('recall %d%%(%d/%d), at thresh %.2f\n', ...
                  round(rec_rate*100), rec, size(grects,1), ...
                  opts.probThresh);
    title(msg);

    drawnow;
    keyboard;
  end
  t2 = toc(t1);
  fprintf('Testing epoch %d: processed %d/%d, %.2f img/sec\n', ...
          testEpoch, i, numel(test), i/t2);
end

function viomask = violation(opts,clusters,h,w)
rf = opts.recfields(opts.var2idx('prob_cls'));
ofx = rf.offset(2); 
ofy = rf.offset(1); 
stx = rf.stride(2); 
sty = rf.stride(1); 

%
varsize = opts.varsizes{opts.var2idx('score_cls')};
vsx = varsize(2); 
vsy = varsize(1); 

% 
[xx,yy] = meshgrid(ofx+[0:vsx-1]*stx, ofy+[0:vsy-1]*sty);
nc = size(clusters,1);
dx1 = reshape(clusters(:,1),1,1,nc); 
dy1 = reshape(clusters(:,2),1,1,nc); 
dx2 = reshape(clusters(:,3),1,1,nc); 
dy2 = reshape(clusters(:,4),1,1,nc); 
areas  = reshape((dx2-dx1+1).*(dy2-dy1+1), 1,1,nc);

xx1 = bsxfun(@plus, xx, dx1);
yy1 = bsxfun(@plus, yy, dy1);
xx2 = bsxfun(@plus, xx, dx2); 
yy2 = bsxfun(@plus, yy, dy2);

% compute four corners of the rect 
viox1 = xx1 <= 0;
vioy1 = yy1 <= 0;
viox2 = xx2 > w; 
vioy2 = yy2 > h;
viomask = viox1 | vioy1 | viox2 | vioy2;

function [cy,cx] = backtrack(opts,fy,fx)
rf = opts.recfields(opts.var2idx('score_cls'));
ofx = rf.offset(2); 
ofy = rf.offset(1); 
stx = rf.stride(2); 
sty = rf.stride(1); 
cx = (fx-1)*stx + ofx; 
cy = (fy-1)*sty + ofy;

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;

tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
% find latest epoch
for i = 1:numel(tokens)
  token = tokens{i}{1}; 
  ep = str2num(token{1});
  if ep >= epoch 
    epoch = ep; 
  end
end

