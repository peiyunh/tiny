% 

clear all;

addpath matconvnet;
addpath matconvnet/matlab;
vl_setupnn;

addpath toolbox/nms;
addpath toolbox/export_fig;

% specify pretrained model (download if needed)
model_path = 'models/hr_res101.mat';
if ~exist(model_path)
    url = 'https://www.cs.cmu.edu/~peiyunh/tiny/hr_res101.mat';
    cmd = ['wget -O ' model_path ' ' url];
    system(cmd);
end

% use gpu or not (make sure you have matconvnet compiled with gpu)
use_gpu = true;

% setup testing threshold
thresh = 0.5;
nmsthresh = 0.1;

% loadng pretrained model (and some final touches) 
net = load(model_path);
net = dagnn.DagNN.loadobj(net.net);
if use_gpu
    net.move('gpu');
end
net.layers(net.getLayerIndex('score4')).block.crop = [1,2,1,2];
net.addLayer('cropx',dagnn.Crop('crop',[0 0]),...
             {'score_res3', 'score4'}, 'score_res3c'); 
net.setLayerInputs('fusex', {'score_res3c', 'score4'});
net.addLayer('prob_cls', dagnn.Sigmoid(), 'score_cls', 'prob_cls');
averageImage = reshape(net.meta.normalization.averageImage,1,1,3);

% reference boxes of templates
clusters = net.meta.clusters; 

% by default, we look at three resolutions (.5X, 1X, 2X)
scales = [-1 0 1];

% run through all images under demo/data
files = dir('demo/data/*');
for f = dir('demo/data/*')'
    if strcmp(f.name, '.') || strcmp(f.name, '..'),
        continue;
    end
    % load input
    img = imread(fullfile('demo/data', f.name));
    [~,name,~] = fileparts(f.name);
    img = single(img);

    % initialize variables that store bounding boxes
    reg_bbox = [];
    raw_bbox = [];
    for s = 2.^scales
        img_ = imresize(img, s, 'bilinear');
        img_ = bsxfun(@minus, img_, averageImage);

        if strcmp(net.device, 'gpu')
            img_ = gpuArray(img_);
        end

        % in case it goes beyond memory limit (12GB)
        if size(img_, 1) > 10000 || size(img_, 2) > 10000
            continue;
        end

        % we don't run every template on every scale
        % ids of templates to ignore 
        tids = [];
        if s <= 1, tids = 5:12;
        else, tids = [5:12 19:25];
        end
        ignoredTids = setdiff(1:size(clusters,1), tids);

        % run through the net
        [img_h, img_w, ~] = size(img_);
        inputs = {'data', img_};
        net.eval(inputs);

        % collect scores 
        score_cls = gather(net.vars(net.getVarIndex('score_cls')).value);
        score_reg = gather(net.vars(net.getVarIndex('score_reg')).value);
        prob_cls = gather(net.vars(net.getVarIndex('prob_cls')).value);
        prob_cls(:,:,ignoredTids) = 0;

        % threshold for detection
        idx = find(prob_cls > thresh);
        [fy,fx,fc] = ind2sub(size(prob_cls), idx);

        % interpret heatmap into bounding boxes 
        cy = (fy-1)*8 - 1; cx = (fx-1)*8 - 1;
        ch = clusters(fc,4) - clusters(fc,2) + 1;
        cw = clusters(fc,3) - clusters(fc,1) + 1;

        % filter out bounding boxes cross boundary 
        x1 = cx - cw/2; y1 = cy - ch/2;
        x2 = cx + cw/2; y2 = cy + ch/2;
        x1 = max(1, min(x1, img_w));
        y1 = max(1, min(y1, img_h));
        x2 = max(1, min(x2, img_w));
        y2 = max(1, min(y2, img_h));

        % extract bounding box refinement
        Nt = size(clusters, 1); 
        tx = score_reg(:,:,1:Nt); 
        ty = score_reg(:,:,Nt+1:2*Nt); 
        tw = score_reg(:,:,2*Nt+1:3*Nt); 
        th = score_reg(:,:,3*Nt+1:4*Nt); 

        % refine bounding boxes
        dcx = cw .* tx(idx); 
        dcy = ch .* ty(idx);
        rcx = cx + dcx;
        rcy = cy + dcy;
        rcw = cw .* exp(tw(idx));
        rch = ch .* exp(th(idx));

        %
        scores = score_cls(idx);
        tmp_raw_bbox = [cx-cw/2, cy-ch/2, cx+cw/2, cy+ch/2];
        tmp_reg_bbox = [rcx-rcw/2, rcy-rch/2, rcx+rcw/2, rcy+rch/2];

        tmp_raw_bbox = horzcat(tmp_raw_bbox ./ s, fc, scores);
        tmp_reg_bbox = horzcat(tmp_reg_bbox ./ s, fc, scores);

        raw_bbox = vertcat(raw_bbox, tmp_raw_bbox);
        reg_bbox = vertcat(reg_bbox, tmp_reg_bbox);
    end

    % nms 
    ridx = nms(reg_bbox(:,[1:4 end]), nmsthresh); 
    reg_bbox = reg_bbox(ridx,:);
    raw_bbox = raw_bbox(ridx,:);

    % visualize results (faces shorter than 10px are not shown)
    visualize_detection(uint8(img), reg_bbox, thresh);

    % (optional) export figure 
    export_fig('-dpng', '-native', '-opengl', '-transparent', fullfile('demo/visual', [name '.png']));
end
