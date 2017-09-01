%  FILE:   cnn_widerface.m
%
%    This function serves as the main function for evaluating a model. It loads
%    detection and ground truth and evaluates at all three difficulty levels at
%    once. 
% 
%    Note that this evaluation script consistently produces slightly lower
%    numbers (mAPs) comparing to the official evaluation script. I implement my
%    own version because it runs faster and it is more customizable, in terms of
%    adding visualization etc.
%  
%  INPUT:  configuration (see code for details)
%
%  OUTPUT: none         

function cnn_widerface_eval(varargin)

%
opts.noUpsampling = false;
opts.noOrgres = false;
opts.noDownsampling = false;
opts.testMultires = false;
opts.testTag = '';
opts.inputSize = [500, 500];
[opts, varargin] = vl_argparse(opts, varargin);

%
opts.evalWithReg = true;
opts.tag = '';
opts.bboxReg = true;
opts.skipLRMult = [1, 0.001, 0.0001, 0.00001];
opts.penalizeDuplicate = true;
opts.nmsThresh = 0.3;
opts.probThresh = 0.8;
opts.draw = false;
opts.minOverlap = [0.3:0.2:0.7];
opts.testEpoch = 0; 
opts.testIter = 0; 
opts.testSet = 'val';
opts.resDir = 'results/';
opts.testName = '';
opts.clusterNum = 25;
opts.clusterName = '';
opts.sampleSize = 256;
opts.posFraction = 0.5; 
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.pretrainModelPath = 'matconvnet/pascal-fcn8s-tvg-dag.mat';
opts.dataDir = fullfile('data','widerface') ;
opts.modelType = 'pascal-fcn8s-tvg-dag' ;
opts.networkType = 'dagnn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

%
opts.minClusterSize = [2, 2]; 
opts.maxClusterSize = opts.inputSize;
[opts, varargin] = vl_argparse(opts, varargin) ;

%
sfx = opts.modelType ;
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

%
opts.expDir = fullfile('models', ['widerface-' sfx]);
if ~isempty(opts.tag)
    opts.expDir = [opts.expDir '-' opts.tag];
end
[opts, varargin] = vl_argparse(opts, varargin) ;

% imdb
opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

%

if exist(opts.imdbPath) 
    imdb = load(opts.imdbPath) ;
else
    imdb = cnn_setup_imdb('dataDir', opts.dataDir);
    if ~exist(opts.expDir), mkdir(opts.expDir) ; end
    save(opts.imdbPath, '-struct', 'imdb') ;
end

%
if opts.testEpoch == 0
    testEpoch = findLastCheckpoint(opts.expDir);
    if testEpoch==0
        error('No available model for evaluation.');
    end
else
    testEpoch = opts.testEpoch; 
end
testSet = opts.testSet;

if isempty(opts.testName)
    testName = strrep(opts.expDir, 'models/', '');
    testName = sprintf('%s-epoch%d-%s', testName, testEpoch, testSet);
    testName = [testName '-probthresh' num2str(opts.probThresh)]; 
    testName = [testName '-nmsthresh' num2str(opts.nmsThresh)]; 
    if opts.testMultires, testName = [testName '-multires']; end
    if opts.noOrgres, testName = [testName '-noorgres']; end
    if opts.noUpsampling, testName = [testName '-noupsampling']; end
    if opts.noDownsampling, testName = [testName '-nodownsampling']; end
    if opts.evalWithReg, testName = [testName '-evalWithReg']; end
else
    testName = opts.testName; 
end

if ~isempty(opts.testTag),
    testName = [testName '-' opts.testTag];
end

if strcmp(opts.resDir, 'results/')
    resDir = fullfile(opts.resDir, testName);
    if numel(resDir) > 255, resDir = strrep(resDir, '-nmsthresh0.3', ''); end
    if numel(resDir) > 255, resDir = strrep(resDir, 'widerface-resnet-50-',''); end
    if numel(resDir) > 255, resDir = strrep(resDir, 'bboxreg-logistic-cluster-',''); end
else
    resDir = opts.resDir;
end

% load ground truth info 
%fprintf('Loading ground truth annotation for %s.\n', opts.testSet);
switch opts.testSet 
  case 'val'
    test = find(imdb.images.set == 2);
  case 'test'
    error('Evaluting on test set - you must be kidding me...\n')
end

hard_info = load('eval_tools/ground_truth/wider_hard_val.mat');
med_info = load('eval_tools/ground_truth/wider_medium_val.mat');
easy_info = load('eval_tools/ground_truth/wider_easy_val.mat');

% thresholds 
minOverlap = opts.minOverlap;
nt = numel(opts.minOverlap);

%
img_idx_map = containers.Map();
for i = 1:numel(hard_info.file_list)
    for j = 1:numel(hard_info.file_list{i})
        img_idx_map(hard_info.file_list{i}{j}) = j;
    end
end

% number of files
easy_npos = 0 ;
med_npos = 0 ;
hard_npos = 0 ;

gt(numel(test)) = struct('BB', [], 'det', []);
easy_gt(numel(test)) = struct('BB', [], 'det', []);
med_gt(numel(test)) = struct('BB', [], 'det', []);
hard_gt(numel(test)) = struct('BB', [], 'det', []);
for i = 1:numel(test)
    eventId = imdb.labels.eventid(test(i));
    
    gtrects = imdb.labels.rects{test(i)};
    gt(i).BB = gtrects';
    easy_gt(i).det = false(size(gtrects,1), nt);
    med_gt(i).det = false(size(gtrects,1), nt);
    hard_gt(i).det = false(size(gtrects,1), nt);
    
    gt_h = gtrects(:,4)-gtrects(:,2)+1;
    gt_w = gtrects(:,3)-gtrects(:,1)+1;

    % NOTE: using released evaluation tools
    img_size = imdb.images.size(test(i),:);
    [~, img_name, ~] = fileparts(imdb.images.name{test(i)});
    img_idx = img_idx_map(img_name);

    easy_gt(i).ign = ones(size(gtrects, 1), 1);
    med_gt(i).ign = ones(size(gtrects, 1), 1);
    hard_gt(i).ign = ones(size(gtrects, 1), 1);

    % easy_gt(i).ign(easy_info.ignore_list{eventId}{img_idx}) = 0;
    % med_gt(i).ign(med_info.ignore_list{eventId}{img_idx}) = 0;
    % hard_gt(i).ign(hard_info.ignore_list{eventId}{img_idx}) = 0;
    easy_gt(i).ign(easy_info.gt_list{eventId}{img_idx}) = 0;
    med_gt(i).ign(med_info.gt_list{eventId}{img_idx}) = 0;
    hard_gt(i).ign(hard_info.gt_list{eventId}{img_idx}) = 0;
    
    easy_npos = easy_npos + sum(easy_gt(i).ign == 0);
    med_npos = med_npos + sum(med_gt(i).ign == 0);
    hard_npos = hard_npos + sum(hard_gt(i).ign == 0);
end

% collect all predictions
detrespath = fullfile(resDir, sprintf('detections.txt'));
if ~exist(detrespath) || 1
    fout = fopen(detrespath, 'w');
    for i = 1:numel(test)
        eventId = imdb.labels.eventid(test(i)); 

        event = imdb.events.name{eventId}; 
        [~,imname,~] = fileparts(imdb.images.name{test(i)});
        resfile = fullfile(resDir, event, [imname '.txt']); 
        if ~exist(resfile), continue; end
        o = fopen(resfile);
        C = textscan(o,'%d %d %d %d %f','HeaderLines', 2);
        % if this is an older result
        if any(cellfun(@(x)(any(isnan(x))), C))
            fclose(o);
            o = fopen(resfile);
            C = textscan(o,'%d %d %d %d %d %f','HeaderLines', 2);
        end
        x1 = C{1}; y1 = C{2}; w = C{3}; h = C{4}; 
        x2 = x1 + (w-1); y2 = y1 + (h-1);
        confidence = C{end};

        fclose(o);
        for j = 1:numel(confidence)
            fprintf(fout, '%d %f %d %d %d %d\n',i,confidence(j), ...
                    x1(j),y1(j),x2(j),y2(j));
        end
    end
    fclose(fout);
end

%
[ids,confidence,b1,b2,b3,b4] = textread(detrespath, '%d %f %d %d %d %d');
BB = [b1 b2 b3 b4]';

% 
bh = b4 - b2 + 1;
bw = b3 - b1 + 1;

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
BB=BB(:,si);
%tids = tids(si);

% assign detections to ground truth objects
nd=length(confidence);

easy_tp=zeros(nd,nt);
easy_fp=zeros(nd,nt);
med_tp=zeros(nd,nt);
med_fp=zeros(nd,nt);
hard_tp=zeros(nd,nt);
hard_fp=zeros(nd,nt);

ovmaxs=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        %fprintf('face: pr: compute: %d/%d\n',d,nd);
        drawnow;
        tic; 
   end
    
    % find ground truth image
    i=ids(d);

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    jmax = 0 ;

    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    
    % assign detection as true positive/don't care/false positive
    for k=1:nt
        if ovmax>=opts.minOverlap(k)
            %if ~easy_gt(i).ign(j)
            if ~easy_gt(i).ign(jmax)
                if ~easy_gt(i).det(jmax,k) || ~opts.penalizeDuplicate
                    easy_tp(d,k)=1;            % true positive
                    easy_gt(i).det(jmax,k)=true;
                else
                    easy_fp(d,k)=1;            % false positive (multiple detection)
                end
            end
            %if ~med_gt(i).ign(j)
            if ~med_gt(i).ign(jmax)
                if ~med_gt(i).det(jmax,k) || ~opts.penalizeDuplicate
                    med_tp(d,k)=1;            % true positive
                    med_gt(i).det(jmax,k)=true;
                else
                    med_fp(d,k)=1;            % false positive (multiple detection)
                end
            end
            %if ~hard_gt(i).ign(j)
            if ~hard_gt(i).ign(jmax)
                if ~hard_gt(i).det(jmax,k) || ~opts.penalizeDuplicate
                    hard_tp(d,k)=1;            % true positive
                    hard_gt(i).det(jmax,k)=true;
                else
                    hard_fp(d,k)=1;            % false positive (multiple detection)
                end
            end
        else
            easy_fp(d,k)=1;                    % false positive
            med_fp(d,k)=1;                    % false positive
            hard_fp(d,k)=1;                    % false positive
        end
    end
    ovmaxs(d)=ovmax;
end

% compute precision/recall
easy_fp_ = easy_fp; 
easy_tp_ = easy_tp; 

easy_fp=cumsum(easy_fp,1);
easy_tp=cumsum(easy_tp,1);
easy_rec=easy_tp/easy_npos;
easy_prec=easy_tp./(easy_fp+easy_tp);

easy_aps = zeros(1, nt);
for k = 1:nt
    easy_aps(k)=VOCap(easy_rec(:,k),easy_prec(:,k));
    fprintf('class: face, epoch: %d, subset: %s (easy), minOverlap=%.2f, AP = %.3f\n', ...
            testEpoch, testSet, opts.minOverlap(k), easy_aps(k))
end

med_fp_ = med_fp; 
med_tp_ = med_tp; 

med_fp=cumsum(med_fp,1);
med_tp=cumsum(med_tp,1);
med_rec=med_tp/med_npos;
med_prec=med_tp./(med_fp+med_tp);

med_aps = zeros(1, nt);
for k = 1:nt
    med_aps(k)=VOCap(med_rec(:,k),med_prec(:,k));
    fprintf('class: face, epoch: %d, subset: %s (medium), minOverlap=%.2f, AP = %.3f\n', ...
            testEpoch, testSet, opts.minOverlap(k), med_aps(k))
end

hard_fp_ = hard_fp; 
hard_tp_ = hard_tp; 

hard_fp=cumsum(hard_fp,1);
hard_tp=cumsum(hard_tp,1);
hard_rec=hard_tp/hard_npos;
hard_prec=hard_tp./(hard_fp+hard_tp);

hard_aps = zeros(1, nt);
for k = 1:nt
    hard_aps(k)=VOCap(hard_rec(:,k),hard_prec(:,k));
    fprintf('class: face, epoch: %d, subset: %s (hard), minOverlap=%.2f, AP = %.3f\n', ...
            testEpoch, testSet, opts.minOverlap(k), hard_aps(k))
end

if opts.draw
    close all; 
    for i = 1:3
        switch i 
          case 1
            rec = easy_rec;
            prec = easy_prec;
            aps = easy_aps;
            diffLevel = 'easy';
          case 2
            rec = med_rec;
            prec = med_prec;
            aps = med_aps;
            diffLevel = 'medium';
          case 3
            rec = hard_rec;
            prec = hard_prec;
            aps = hard_aps;
            diffLevel = 'hard';
        end
            
        figure(i);
        %% plot our precision/recall
        hold on;
        for k = 1:nt
            plot(rec(:,k),prec(:,k),'-','linewidth',3);
            [~,ii] = max(rec(:,k).*prec(:,k)); 
        end

        hold off;
        grid;
        xlabel 'recall'
        ylabel 'precision'
        %title(sprintf('class: face, subset: %s, AP = %.3f',testSet,ap));
        title(sprintf('class face, epoch %d, subset %s(%s), AP(0.5) %.3f', ...
                      testEpoch, testSet, diffLevel, aps(opts.minOverlap==0.5)));
        xlim([0,1]); 
        ylim([0,1]);
        print('-dpng', fullfile(resDir, ['wider_' diffLevel '.png']));
    end
end

% -------------------------------------------------------------------------
function [epoch, iter] = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;

epoch = 0;

tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
% find latest epoch
for i = 1:numel(tokens)
  token = tokens{i}{1}; 
  ep = str2num(token{1});
  if ep >= epoch 
    epoch = ep; 
  end
end
