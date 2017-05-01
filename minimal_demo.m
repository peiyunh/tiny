% 
clear all;

addpath matconvnet;
addpath matconvnet/matlab;
vl_setupnn;

addpath toolbox/nms;
addpath toolbox/export_fig;

%
MAX_INPUT_DIM = 9000;
MAX_DISP_DIM = 3000;

% specify pretrained model (download if needed)
model_path = 'models/hr_res101.mat';
if ~exist(model_path)
    url = 'https://www.cs.cmu.edu/~peiyunh/tiny/hr_res101.mat';
    cmd = ['wget -O ' model_path ' ' url];
    system(cmd);
end

% use gpu or not (make sure you have matconvnet compiled with gpu)
gpu_id = 4;

% setup testing threshold
thresh = 0.9;
nmsthresh = 0.1;

% loadng pretrained model (and some final touches) 
net = load(model_path);
net = dagnn.DagNN.loadobj(net.net);
if gpu_id > 0 % for matconvnet it starts with 1 
    gpuDevice(gpu_id);
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
clusters_h = clusters(:,4) - clusters(:,2) + 1;
clusters_w = clusters(:,3) - clusters(:,1) + 1;
normal_idx = find(clusters(:,5) == 1);

% by default, we look at three resolutions (.5X, 1X, 2X)
%scales = [-1 0 1]; % update: adapt to image resolution (see below)

% run through all images under demo/data
files = dir('demo/data/*.jpg');
files = files(4);
for f = files'
    if strcmp(f.name, '.') || strcmp(f.name, '..'),
        continue;
    end
    % load input
    [~,name,ext] = fileparts(f.name);
    if ~strcmp(lower(ext), '.jpg') && ~strcmp(lower(ext), '.png')
        continue;
    end
    try
        raw_img = imread(fullfile('demo/data', f.name));
    catch
        continue;
    end
    raw_img = single(raw_img);

    % 
    [raw_h, raw_w, ~] = size(raw_img) ;
    min_scale = min(floor(log2(max(clusters_w(normal_idx)/raw_w))),...
                    floor(log2(max(clusters_h(normal_idx)/raw_h))));
    % <=1: avoid too much artifacts due to interpolation
    % 5000: in case run out of memory 
    % max_scale = min(1, -log2(max(raw_h, raw_w)/MAX_INPUT_DIM));
    max_scale = 0;
    %scales = min_scale : 0.5 : max_scale;
    scales = [min_scale:0, 0.5:0.5:max_scale];
    %2.^scales .* raw_h
    %2.^scales .* raw_w

    % initialize variables that store bounding boxes
    reg_bbox = [];
    for s = 2.^scales
        img = imresize(raw_img, s, 'bilinear');
        fprintf('processing an image with %d x %d size.\n', round(size(img,1)), round(size(img,2)));
        img = bsxfun(@minus, img, averageImage);

        if strcmp(net.device, 'gpu')
            img = gpuArray(img);
        end

        % we don't run every template on every scale
        % ids of templates to ignore 
        tids = [];
        if s <= 1, tids = 5:12;
        else, tids = [5:12 19:25];
        end
        ignoredTids = setdiff(1:size(clusters,1), tids);

        % run through the net
        [img_h, img_w, ~] = size(img);
        inputs = {'data', img};
        net.eval(inputs);

        % collect scores 
        score_cls = gather(net.vars(net.getVarIndex('score_cls')).value);
        score_reg = gather(net.vars(net.getVarIndex('score_reg')).value);
        prob_cls = gather(net.vars(net.getVarIndex('prob_cls')).value);
        prob_cls(:,:,ignoredTids) = 0;
        %max(prob_cls(:))

        % threshold for detection
        idx = find(prob_cls > thresh);
        [fy,fx,fc] = ind2sub(size(prob_cls), idx);

        % interpret heatmap into bounding boxes 
        cy = (fy-1)*8 - 1; cx = (fx-1)*8 - 1;
        ch = clusters(fc,4) - clusters(fc,2) + 1;
        cw = clusters(fc,3) - clusters(fc,1) + 1;

        % filter out bounding boxes cross boundary 
        %x1 = cx - cw/2; y1 = cy - ch/2;
        %x2 = cx + cw/2; y2 = cy + ch/2;
        %x1 = max(1, min(x1, img_w));
        %y1 = max(1, min(y1, img_h));
        %x2 = max(1, min(x2, img_w));
        %y2 = max(1, min(y2, img_h));
        
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
        tmp_reg_bbox = [rcx-rcw/2, rcy-rch/2, rcx+rcw/2, rcy+rch/2];

        tmp_reg_bbox = horzcat(tmp_reg_bbox ./ s, fc, scores);

        reg_bbox = vertcat(reg_bbox, tmp_reg_bbox);
    end

    % nms 
    ridx = nms(reg_bbox(:,[1:4 end]), nmsthresh); 
    reg_bbox = reg_bbox(ridx,:);

    %
    reg_bbox(:,[2 4]) = max(1, min(raw_h, reg_bbox(:,[2 4])));
    reg_bbox(:,[1 3]) = max(1, min(raw_w, reg_bbox(:,[1 3])));

    %
    vis_img = raw_img;
    vis_bbox = reg_bbox;
    if max(raw_h, raw_w) > MAX_DISP_DIM
        vis_scale = MAX_DISP_DIM/max(raw_h, raw_w);
        vis_img = imresize(raw_img, vis_scale);
        vis_bbox(:,1:4) = vis_bbox(:,1:4) * vis_scale;
    end
    visualize_detection(uint8(vis_img), vis_bbox, thresh);

    %
    drawnow;

    keyboard;
    % (optional) export figure 
    export_fig('-dpng', '-native', '-opengl', '-transparent', fullfile('demo/visual', [name '.png']), '-r300');

    fprintf('finish %s\n', f.name);

    keyboard;
end

gpuDevice([]); 