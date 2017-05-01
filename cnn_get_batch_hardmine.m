% Hardmine: basically don't sample negatives here, instead handle
% sampling in the loss layer based on the loss 
function [images, clsmaps, regmaps] = cnn_get_batch_hardmine( ...
    imagePaths, imageSizes, labelRects, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation

opts.inputSize = [227, 227] ;
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
% opts.numAugments = 1; % no need
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;

opts.rfs = [];
opts.lossType = [];
opts.clusterType = [];
opts.clusters = [];
opts.posThresh = 0.7;
opts.negThresh = 0.3;
opts.var2idx = [];
opts.varsizes = []; 
opts.recfields = [];
opts.sampleSize = 64;
opts.posFraction = 0.5;
opts = vl_argparse(opts, varargin);

%% 
inputSize = opts.inputSize;

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(imagePaths) >= 1 && ischar(imagePaths{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

% TODO use Resize for vl_imreadjpeg
if prefetch 
    vl_imreadjpeg(imagePaths, 'numThreads', opts.numThreads, 'prefetch') ;
    images = [];
    clsmaps = []; 
    regmaps = []; 
    return ;
end

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
    opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
    opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

for i = 1:numel(labelRects)
    if ~isempty(labelRects{i})
        labelRects{i} = labelRects{i}(:,1:4);
    end
end

% most time spent here
if fetch
    %% NOTE: previously we were resizing while reading 
    %minLen = min(inputSize(1:2));
    %imageCells = vl_imreadjpeg(imagePaths,'numThreads', opts.numThreads, ...
    %                           'resize', minLen) ;
    %% NOTE: now we crop after reading
    imageCells = vl_imreadjpeg(imagePaths, 'numThreads', opts.numThreads);
    pasteBoxes = zeros(numel(imagePaths), 4);
    for i = 1:numel(imageCells)
        % create a buffer with all zeros to fill in
        imageSize = imageSizes(i,:);
        labelRect = labelRects{i};

        if ~isempty(opts.averageImage)
            imageBuffer = repmat(opts.averageImage, inputSize);
        else
            imageBuffer = zeros([inputSize 3]);
        end
        
        rnd = rand(1);
        if rnd < 1/3
            imageSize = ceil(imageSize / 2);
            labelRect = labelRect / 2;
            img = imresize(imageCells{i}, 0.5, 'bilinear');
        elseif rnd > 2/3
            imageSize = ceil(imageSize * 2);
            labelRect = labelRect * 2;
            img = imresize(imageCells{i}, 2, 'bilinear');
        else
            img = imageCells{i};
        end
        if size(img,3) == 1
            img = cat(3, img, img, img) ;
        end

        % crop image
        crop_y1 = randi([1 max(1, imageSize(1)+1-inputSize(1))]);
        crop_x1 = randi([1 max(1, imageSize(2)+1-inputSize(2))]);
        crop_y2 = min(imageSize(1), crop_y1+inputSize(1)-1);
        crop_x2 = min(imageSize(2), crop_x1+inputSize(2)-1);
        crop_h = crop_y2-crop_y1+1;
        crop_w = crop_x2-crop_x1+1;
        
        paste_y1 = randi([1, inputSize(1)-crop_h+1]);
        paste_x1 = randi([1, inputSize(2)-crop_w+1]);
        paste_y2 = paste_y1 + crop_h - 1;
        paste_x2 = paste_x1 + crop_w - 1;
        pasteBoxes(i,:) = [paste_x1, paste_y1, paste_x2, paste_y2];

        imageBuffer(paste_y1:paste_y2,paste_x1:paste_x2,:) = ...
            img(crop_y1:crop_y2, crop_x1:crop_x2,:);

        if 0
            subplot(121);
            imshow(uint8(img));
            hold on;
            rectangle('position', [crop_x1, crop_y1, crop_w, crop_h], ...
                      'EdgeColor', 'r');
            for j = 1:size(labelRect, 1)
                rectangle('position', [labelRect(j,1), labelRect(j,2) ...
                                    labelRect(j,3)-labelRect(j,1)+1, ...
                                    labelRect(j,4)-labelRect(j,2)+1], ...
                          'EdgeColor', 'b');
            end
            hold off; 
            drawnow; 
        end

        if ~isempty(labelRect)
            % if label rects are severly truncated due to cropping
            tLabelRect = labelRect;
            tLabelRect(:,1) = max(tLabelRect(:,1), crop_x1); 
            tLabelRect(:,2) = max(tLabelRect(:,2), crop_y1); 
            tLabelRect(:,3) = min(tLabelRect(:,3), crop_x2); 
            tLabelRect(:,4) = min(tLabelRect(:,4), crop_y2);
            tovlp = 1 - rect_dist(tLabelRect, labelRect);
            
            % change bounding boxes
            labelRect = bsxfun(@minus, labelRect, [crop_x1,crop_y1,crop_x1,crop_y1])+1;
            labelRect = bsxfun(@plus, labelRect, [paste_x1,paste_y1,paste_x1,paste_y1])-1;
            
            labelRect(:,1) = min(inputSize(2), max(1, labelRect(:,1)));
            labelRect(:,2) = min(inputSize(1), max(1, labelRect(:,2)));
            labelRect(:,3) = min(inputSize(2), max(1, labelRect(:,3)));
            labelRect(:,4) = min(inputSize(1), max(1, labelRect(:,4)));
        
            invalid_idx = find(labelRect(:,3)<=labelRect(:,1) | ...
                               labelRect(:,4)<=labelRect(:,2) | ...
                               tovlp < opts.negThresh);
            labelRect(invalid_idx,:) = [];
        end
        
        if 0
            subplot(122);
            imshow(uint8(imageBuffer));
            hold on;
            for j = 1:size(labelRect, 1)
                rectangle('position', [labelRect(j,1), labelRect(j,2) ...
                                    labelRect(j,3)-labelRect(j,1)+1, ...
                                    labelRect(j,4)-labelRect(j,2)+1], ...
                          'EdgeColor', 'g');
            end
            hold off;
            drawnow;
            keyboard;
        end

        % update both image and annotation
        imageSizes(i,:) = inputSize; % pretend we read crops 
        imageCells{i} = imageBuffer;
        labelRects{i} = labelRect;
    end
else
    imageCells = imagePaths ;
end

tfs = [] ;
switch opts.transformation
  case 'none'
    tfs = [
        .5 ;
        .5 ;
        0 ] ;
  case 'f5'
    tfs = [...
        .5 0 0 1 1 .5 0 0 1 1 ;
        .5 0 1 0 1 .5 0 1 0 1 ;
        0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'stretch'
  otherwise
    error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(imagePaths)), 1) ;

%% decide to use clusters or subclusters 
centers = opts.clusters;

% define grids 
if ~isempty(opts.recfields) && ~isempty(opts.varsizes)
    rf = opts.recfields(opts.var2idx('score_cls'));
    ofx = rf.offset(2); 
    ofy = rf.offset(1);
    stx = rf.stride(2);
    sty = rf.stride(1);

    %
    varsize = opts.varsizes{opts.var2idx('score_cls')};
    vsx = varsize(2); 
    vsy = varsize(1);

    % 
    [coarse_xx,coarse_yy] = meshgrid(ofx+(0:vsx-1)*stx, ofy+(0:vsy-1)*sty);
    %% NOTE: go for pixel-level overlap
    %[xx,yy] = meshgrid(ofx + (0:vsx*stx-1), ofy + (0:vsy*sty-1));
    %% NOTE: compensate the shifting factor
    %% now it stays the same after we max pool
    %[fine_xx,fine_yy] = meshgrid(ofx-(stx-1)/2 + (0:vsx*stx-1), ...
    %                             ofy-(sty-1)/2 + (0:vsy*sty-1));
    %xx = gpuArray(xx);
    %yy = gpuArray(yy);
    
    nt = size(centers,1);
    dx1 = reshape(centers(:,1),1,1,nt); 
    dy1 = reshape(centers(:,2),1,1,nt); 
    dx2 = reshape(centers(:,3),1,1,nt); 
    dy2 = reshape(centers(:,4),1,1,nt); 
    areas  = reshape((dx2-dx1+1).*(dy2-dy1+1), 1,1,nt);
    
    coarse_xx1 = bsxfun(@plus, coarse_xx, dx1);
    coarse_yy1 = bsxfun(@plus, coarse_yy, dy1);
    coarse_xx2 = bsxfun(@plus, coarse_xx, dx2);
    coarse_yy2 = bsxfun(@plus, coarse_yy, dy2);
    
    %fine_xx1 = bsxfun(@plus, fine_xx, dx1);
    %fine_yy1 = bsxfun(@plus, fine_yy, dy1);
    %fine_xx2 = bsxfun(@plus, fine_xx, dx2);
    %fine_yy2 = bsxfun(@plus, fine_yy, dy2);

    % NOTE now since we are doing random pasting, viomask will be
    % different depending on where we paste
    
    % compute four corners of the rect 
%     viox1 = coarse_xx1 <= 0;
%     vioy1 = coarse_yy1 <= 0;
%     viox2 = coarse_xx2 > inputSize(2); 
%     vioy2 = coarse_yy2 > inputSize(1);
%     coarse_viomask = viox1 | vioy1 | viox2 | vioy2;

    % paste-related viomask
    pad_viomasks = cell(1, numel(imagePaths));
    for i = 1:numel(imagePaths)
        padx1 = coarse_xx1 < pasteBoxes(i,1); 
        pady1 = coarse_yy1 < pasteBoxes(i,2); 
        padx2 = coarse_xx2 > pasteBoxes(i,3); 
        pady2 = coarse_yy2 > pasteBoxes(i,4);
        pad_viomasks{i} = padx1 | pady1 | padx2 | pady2; 
    end
end

% init inputs
images = zeros(inputSize(1), inputSize(2), 3, ...
               numel(imagePaths), 'single') ;

% init targets
if isempty(labelRects) || isempty(opts.varsizes)
    clsmaps = []; 
    regmaps = []; 
else
    clsmaps = -ones(vsy, vsx, nt, numel(imagePaths), 'single');
    regmaps = zeros(vsy, vsx, nt*4, numel(imagePaths), 'single');
end

%% for logging some stats
%logfile = fopen('pos_size_stats.txt', 'a+');
%overlap_gt_num = zeros(numel(imagePaths), size(centers,1));
%overlap_pos_num = zeros(numel(imagePaths), size(centers,1));
%sample_pos_num = zeros(numel(imagePaths), size(centers,1));
%sample_neg_num = zeros(numel(imagePaths), size(centers,1));

% enumerate
for i=1:numel(imagePaths)
    % acquire image
    if isempty(imageCells{i}) % i.e. png (use imread)
        imt = imread(imagePaths{i}) ;
        imt = single(imt) ; % faster than im2single (and multiplies by 255)
    else
        imt = imageCells{i} ;
    end
    if size(imt,3) == 1
        imt = cat(3, imt, imt, imt) ;
    end
    
    % acquire labelRects 
    labelRect = [];
    if ~isempty(labelRects)
        labelRect = labelRects{i};
    end
    
    % resize
    %w = size(imt,2) ;
    %h = size(imt,1) ;
    % NOTE imt should be resized already, 
    %      read image size from imdb info
    h = imageSizes(i, 1);
    w = imageSizes(i, 2);
    factor = [(inputSize(1)+opts.border(1))/h ...
              (inputSize(2)+opts.border(2))/w];
    if opts.keepAspect
        factor = max(factor) ;
    end
    
    if any(abs(factor - 1) > 0.0001)
        % NOTE resize has been done in vl_imreadjpeg
        %imt = imresize(imt, ...
        %               'scale', factor, ...
        %               'method', opts.interpolation) ;
        if ~isempty(labelRect),
            labelRect=labelRect.*factor;
        end
    end

    % crop & flip 
    % NOTE read resized height and width
    w = size(imt,2) ;
    h = size(imt,1) ;
    switch opts.transformation
      case 'stretch'
        error('Did not expected stretch transformation');
        sz = round(min(inputSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [h;w])) ;
        dx = randi(w - sz(2) + 1, 1) ;
        dy = randi(h - sz(1) + 1, 1) ;
        flip = rand > 0.5 ;
      otherwise
        transid = transformations(randi(size(transformations,1)), i);
        tf = tfs(:, transid) ;
        sz = inputSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = tf(3) ;
    end
    sx = round(linspace(dx, sz(2)+dx-1, inputSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, inputSize(1))) ;
    if ~isempty(labelRect)
        labelRect(:,[1 3]) = bsxfun(@minus, labelRect(:,[1 3]), dx-1);
        labelRect(:,[2 4]) = bsxfun(@minus, labelRect(:,[2 4]), dy-1);
        labelRect(:,1) = min(inputSize(2), max(1, labelRect(:,1)));
        labelRect(:,2) = min(inputSize(1), max(1, labelRect(:,2)));
        labelRect(:,3) = min(inputSize(2), max(1, labelRect(:,3)));
        labelRect(:,4) = min(inputSize(1), max(1, labelRect(:,4)));
    end

    if flip,
        sx = fliplr(sx) ;
        if ~isempty(labelRect)
            % TODO use a wrapper called labelFlipFunc
            lx1 = labelRect(:,1); lx2 = labelRect(:,3);
            % NOTE bug fixed 
            %labelRect(:,1) = opts.inputSize(2) - lx1;
            %labelRect(:,3) = opts.inputSize(2) - lx2;
            labelRect(:,3) = inputSize(2) - lx1 + 1;
            labelRect(:,1) = inputSize(2) - lx2 + 1;
        end
        % NOTE: should flip pad_viomask as well due to random
        % pasting (credit JW)
        pad_viomasks{i} = fliplr(pad_viomasks{i});
    end

    if ~isempty(opts.averageImage)
        offset = opts.averageImage ;
        if ~isempty(opts.rgbVariance)
            offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3));
        end
        images(:,:,:,i) = bsxfun(@minus, imt(sy,sx,:), offset) ;
    else
        images(:,:,:,i) = imt(sy,sx,:) ;
    end

    % NOTE we should never continue in this for loop,
    % because then we will overwhelm the batch with all potential negatives
    %if isempty(labelRect), continue; end;
    if ~isempty(labelRect) 
        invalid_idx = find(labelRect(:,3)<=labelRect(:,1) | ...
                           labelRect(:,4)<=labelRect(:,2));
        labelRect(invalid_idx,:) = [];
    end    
    % NOTE we cannot just let go, because it leaves target all -1 
    %if isempty(labelRect),
    %    continue;
    %end

    %% NOTE if this holds, it means we are still extracting stats 
    if isempty(opts.recfields) || isempty(opts.varsizes)
        continue;
    end
    
    % NOTE we use an option now 
    % save memory when computing iou 
    %labelRect = int16(labelRect);
    
    % 1% images in train split have more than 100 faces
    saveMemory = size(labelRect,1) > 100;

    % initialize IOU matrix 
    iou = [];
    
    % NOTE I've seen 1749 faces in one image. The code below is written in
    % a way that avoids memory issue as far as I can, while keeping
    % it vectorized, instead of using for-loop. 
    ng = size(labelRect, 1);
    if ng > 0
        gx1 = labelRect(:,1);
        gy1 = labelRect(:,2);
        gx2 = labelRect(:,3);
        gy2 = labelRect(:,4);

        % compute max overlap across interpolated coordinates
        iou = compute_dense_overlap(ofx,ofy,stx,sty,vsx,vsy,...
                                    dx1,dy1,dx2,dy2,...
                                    gx1,gy1,gx2,gy2,...
                                    1,1);
        
        fxx1 = reshape(labelRect(:,1),1,1,1,ng);
        fyy1 = reshape(labelRect(:,2),1,1,1,ng); 
        fxx2 = reshape(labelRect(:,3),1,1,1,ng); 
        fyy2 = reshape(labelRect(:,4),1,1,1,ng);

%        fareas = reshape((fyy2-fyy1+1).*(fxx2-fxx1+1),1,1,1,ng);
%        fareas = single(fareas);
%        
%        ixx1 = bsxfun(@max, fxx1, fine_xx1);
%        ixx2 = bsxfun(@min, fxx2, fine_xx2);
%        iw = ixx2-ixx1+1; 
%        %iw = gpuArray(ixx2)-gpuArray(ixx1)+1;
%        if saveMemory, clear ixx1 ixx2; end
%        
%        iyy1 = bsxfun(@max, fyy1, fine_yy1); 
%        iyy2 = bsxfun(@min, fyy2, fine_yy2);
%        ih = iyy2-iyy1+1;
%        %ih = gpuArray(iyy2)-gpuArray(iyy1)+1;
%        if saveMemory, clear iyy1 iyy2; end
%
%        iareas = single(iw.*ih);
%        iareas(iw<0) = 0;
%        if saveMemory, clear iw; end
%        iareas(ih<0) = 0 ;
%        if saveMemory, clear ih; end
%        %iareas(ixx2<=ixx1 | iyy2<=iyy1) = 0;
%
%        sumareas = bsxfun(@plus,areas,fareas);
%        uareas = bsxfun(@minus,sumareas,iareas);
%        fine_iou = iareas ./ uareas;
%        if saveMemory, clear iareas; end
%
%        %% NOTE: 
%        % coarse_xx is exactly the same as the result of pooling
%        % over fine_xx; same goes to yy 
%        %xx = vl_nnpool(xx, [sty stx], 'stride', [sty stx], 'Method', 'avg');
%        %yy = vl_nnpool(yy, [sty stx], 'stride', [sty stx], 'Method', 'avg');
%        iou = vl_nnpool(fine_iou, [sty stx], 'stride', [sty stx], 'Method', 'max');
%        iou = gather(iou);

        % compute reg targets
        dhh = dy2-dy1+1;
        dww = dx2-dx1+1;
        fcx = (fxx1 + fxx2)/2; 
        fcy = (fyy1 + fyy2)/2;
        tx = bsxfun(@rdivide, bsxfun(@minus,fcx,coarse_xx), dww);
        ty = bsxfun(@rdivide, bsxfun(@minus,fcy,coarse_yy), dhh);
        fhh = fyy2-fyy1+1;
        fww = fxx2-fxx1+1;
        tw = log(bsxfun(@rdivide, fww, dww)); 
        th = log(bsxfun(@rdivide, fhh, dhh));
    end

    % add this line for perturbation so that we will not always choose
    % left corner, when candidate window is smaller than ground truth
    if ~isempty(iou)
        iou = iou + 1e-6*rand(size(iou));
    end
    % NOTE this may be a bug (making best_face_per_loc really
    % messy)

    % NOTE this is a change where we always want to pick the central
    % canditate as best shot if there are multiple candidates with the
    % same overlap (for example, this happens when candidate boxes are
    % smaller than ground truth so that any position the candidate
    % is totally within the ground truth, it is the same overlap)
%    label_cys = (labelRect(:,2) + labelRect(:,4))/2;
%    dist_ys = abs(bsxfun(@minus, label_cys, coarse_yy(:,1)'));
%    [~, min_fys] = min(dist_ys, [], 2);
%    label_cxs = (labelRect(:,1) + labelRect(:,3))/2;
%    dist_xs = abs(bsxfun(@minus, label_cxs, coarse_xx(1,:)));
%    [~, min_fxs] = min(dist_xs, [], 2);
%    for j = 1:ng
%        cfx = min_fxs(j); cfy = min_fys(j);
%        iou(cfy,cfx,:,j) = iou(cfy,cfx,:,j) + 1e-6;
%    end

    % temp targets
    clsmap = -ones(vsy, vsx, nt, 'single');
    regmap = zeros(vsy, vsx, 4*nt, 'single');

%    tmp_h = labelRect(:,4) - labelRect(:,2) + 1;
%    tmp_w = labelRect(:,3) - labelRect(:,1) + 1;
%    tmp_label = [-(tmp_w-1)/2, -(tmp_h-1)/2, (tmp_w-1)/2, (tmp_h-1)/2];
%    perfect_ovlp = zeros(size(tmp_label,1), size(centers,1));
%    for j = 1:size(centers,1)
%        perfect_ovlp(:,j) = 1 - rect_dist(centers(j,:), tmp_label);
%    end
%    overlap_gt_num(i,:) = sum(perfect_ovlp >= 0.7, 1); 

    if ng > 0
        [best_iou,best_face_per_loc] = max(iou, [], 4);
        regidx = sub2ind([vsy*vsx*nt, ng], (1:vsy*vsx*nt)', ...
                         best_face_per_loc(:));
        tx = reshape(tx(regidx), vsy, vsx, nt);
        ty = reshape(ty(regidx), vsy, vsx, nt); 
        tw = repmat(tw, vsy, vsx, 1, 1);
        tw = reshape(tw(regidx), vsy, vsx, nt);
        th = repmat(th, vsy, vsx, 1, 1);
        th = reshape(th(regidx), vsy, vsx, nt);
        % reg target (regress to best overlap face)
        regmap = cat(3, tx, ty, tw, th);
        
        % for each face, the best overlapped
        [iou_,fbest_idx] = max(reshape(iou,[],ng),[],1);
        
        % NOTE stick with original strategy 
        %clsmap(fbest_idx) = 1;

        % NOTE the original strategy does not make sense any more
        %      when we have only one template 
        clsmap(fbest_idx(iou_>opts.negThresh)) = 1;

        % +: all anchors with 
        %best_iou = max(iou,[],4);
        clsmap = max(clsmap, (best_iou>=opts.posThresh)*2-1);

        % 0: all non-positive anchors between [lo,hi)
        gray = -ones(size(clsmap));
        gray(opts.negThresh <= best_iou & best_iou < opts.posThresh) = 0;
        clsmap = max(clsmap, gray);
    end

    % 0: boundary (including the introduced boundary due to cropping) crossing 
    clsmap(pad_viomasks{i}) = 0;
    regmap(pad_viomasks{i}) = 0;

    % TODO limit the number of samples 
    %pos_maxnum = opts.sampleSize*opts.posFraction;
    %pos_idx = find(clsmap(:)==1);
    %if numel(pos_idx) > pos_maxnum
    %    didx = Shuffle(numel(pos_idx), 'index', numel(pos_idx)-pos_maxnum);
    %    clsmap(pos_idx(didx)) = 0;
    %end
    %
    %neg_maxnum = pos_maxnum*(1-opts.posFraction)/opts.posFraction;
    %neg_idx = find(clsmap(:)==-1);
    %if numel(neg_idx) > neg_maxnum
    %    ridx = Shuffle(numel(neg_idx), 'index', gather(neg_maxnum));
    %    didx = [1:numel(neg_idx)];
    %    didx(ridx) = [];
    %    clsmap(neg_idx(didx)) = 0;
    %end

    clsmaps(:,:,:,i) = clsmap;
    regmaps(:,:,:,i) = regmap;
   
    if 0
        subplot(131);
        imagesc(images(:,:,:,i)./255);
        axis image;
        
        tx = regmap(:,:,1:nt); 
        ty = regmap(:,:,nt+1:nt*2); 
        tw = regmap(:,:,nt*2+1:nt*3); 
        th = regmap(:,:,nt*3+1:nt*4); 
        
        idx = find(clsmap > 0); 

        [fy,fx,fc] = ind2sub(size(clsmap), idx);
        [cy,cx] = backtrack(opts, fy, fx);
        cw = opts.clusters(fc,3)-opts.clusters(fc,1)+1;
        ch = opts.clusters(fc,4)-opts.clusters(fc,2)+1;

        feat_idx = sub2ind(size(clsmap), fy, fx, fc);
        box_ovlp = best_iou(feat_idx);

        dcx = cw .* tx(idx); 
        dcy = ch .* ty(idx);
        rx = cx + dcx;
        ry = cy + dcy; 
        % apply regression to width and height 
        rw = cw .* exp(tw(idx));
        rh = ch .* exp(th(idx));
        
        cr = [cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2];
        dr = [rx-rw/2,ry-rh/2,rx+rw/2,ry+rh/2];
        
        plotBoxes(cr(:,1),cr(:,2),cr(:,3)-cr(:,1)+1,cr(:,4)-cr(:,2)+1, ...
                  [1 0 0], 1); 
        plotBoxes(dr(:,1),dr(:,2),dr(:,3)-dr(:,1)+1,dr(:,4)-dr(:,2)+1, ...
                  [0 0 1], 1);
        if ~isempty(labelRect)
            plotBoxes(labelRect(:,1),labelRect(:,2),labelRect(:,3)- ...
                      labelRect(:,1)+1,labelRect(:,4)-labelRect(:,2)+1, ...
                      [ 0 1 0] , 1);
        end
        hold off;
        
        subplot(132);
        imagesc(tile3DHeat(clsmap));
        
        subplot(133);
        imagesc(tile3DHeat(regmap));
        caxis([-50,50]);
        
        keyboard;
    end
end

%% clean up gpu memory
%clear fine_xx1 fine_xx2 fine_yy1 fine_yy2;

%overlap_gt_num = sum(overlap_gt_num, 1);
%overlap_pos_num = sum(overlap_pos_num, 1); 
%sample_pos_num = sum(sample_pos_num, 1); 
%sample_neg_num = sum(sample_neg_num, 1);
%dlmwrite('overlap_gt_stats_maxpixlv.csv', overlap_gt_num, '-append');
%dlmwrite('overlap_pos_stats_maxpixlv.csv', overlap_pos_num, '-append');
%dlmwrite('sample_pos_stats_maxpixlv.csv', sample_pos_num, '-append');
%dlmwrite('sample_neg_stats_maxpixlv.csv', sample_neg_num, '-append');

function [cx,cy] = backtrack(opts, fx, fy)
rf = opts.recfields(opts.var2idx('score_cls'));
ofx = rf.offset(2); 
ofy = rf.offset(1); 
stx = rf.stride(2); 
sty = rf.stride(1); 
cx = (fx-1)*stx + ofx; 
cy = (fy-1)*sty + ofy;
