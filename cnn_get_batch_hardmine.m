%  FILE:   cnn_get_batch_hardmine.m
% 
%    This function takes a batch of images (including paths and annotations) and
%    generate input and ground truth that will be fed into the detection
%    network.
% 
%  INPUT:  imagePaths (image paths of a batch of images)
%          imageSizes (image sizes of the same batch of images)
%          labelRects (ground truth bounding boxes)
% 
%  OUTPUT: images (500x500 random cropped regions)
%          clsmaps (ground truth classification heat map)
%          regmaps (ground truth regression heat map)

function [images, clsmaps, regmaps] = cnn_get_batch_hardmine(imagePaths, imageSizes, labelRects, varargin)

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
  % we crop after reading images
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

    % resize images with a random scaling factor
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
  h = imageSizes(i, 1);
  w = imageSizes(i, 2);
  factor = [(inputSize(1)+opts.border(1))/h ...
            (inputSize(2)+opts.border(2))/w];
  if opts.keepAspect
    factor = max(factor) ;
  end
  
  if any(abs(factor - 1) > 0.0001)
    if ~isempty(labelRect),
      labelRect=labelRect.*factor;
    end
  end

  % crop & flip 
  w = size(imt,2) ;
  h = size(imt,1) ;
  switch opts.transformation
    case 'stretch'
      error('Did not expected stretch transformation');
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
      lx1 = labelRect(:,1); lx2 = labelRect(:,3);
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

  %% NOTE if this holds, it means we are getting average images 
  if isempty(opts.recfields) || isempty(opts.varsizes)
    continue;
  end
  
  % % 1% images in train split have more than 100 faces
  % saveMemory = size(labelRect,1) > 100;

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
  % the same candidate if all candidates have the same overlap
  % when candidate window is smaller than ground truth
  if ~isempty(iou)
    iou = iou + 1e-6*rand(size(iou));
  end

  % temp targets
  clsmap = -ones(vsy, vsx, nt, 'single');
  regmap = zeros(vsy, vsx, 4*nt, 'single');

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
    
    % original strategy from Faster RCNN
    %clsmap(fbest_idx) = 1;
    % NOTE: we avoid defining best-shots too aggressively 
    clsmap(fbest_idx(iou_>opts.negThresh)) = 1;

    % +: all anchors with 
    %best_iou = max(iou,[],4);
    clsmap = max(clsmap, (best_iou>=opts.posThresh)*2-1);

    % 0: all non-positive anchors between [lo,hi)
    gray = -ones(size(clsmap));
    gray(opts.negThresh <= best_iou & best_iou < opts.posThresh) = 0;
    clsmap = max(clsmap, gray);
  end

  % 0: boundary
  % (note: including the introduced boundary due to cropping/padding) 
  nonneg_border = (pad_viomasks{i} & clsmap~=-1);

  clsmap(nonneg_border) = 0;
  regmap(nonneg_border) = 0;

  clsmaps(:,:,:,i) = clsmap;
  regmaps(:,:,:,i) = regmap;

  % NOTE: balancing sampling is handled in cnn_train_dag_hardmine.m
  
  if 0
    subplot(131);
    imagesc(images(:,:,:,i)./255);
    axis image;

    % (sanity-check) read positive sliding windows off the generated ground truth
    idx = find(clsmap > 0); 
    [fy,fx,fc] = ind2sub(size(clsmap), idx);
    [cy,cx] = backtrack(opts, fy, fx);
    cw = opts.clusters(fc,3)-opts.clusters(fc,1)+1;
    ch = opts.clusters(fc,4)-opts.clusters(fc,2)+1;
    feat_idx = sub2ind(size(clsmap), fy, fx, fc);
    box_ovlp = best_iou(feat_idx);

    % refine bounding box
    tx = regmap(:,:,1:nt); 
    ty = regmap(:,:,nt+1:nt*2); 
    tw = regmap(:,:,nt*2+1:nt*3); 
    th = regmap(:,:,nt*3+1:nt*4);
    dcx = cw .* tx(idx); 
    dcy = ch .* ty(idx);
    rx = cx + dcx;
    ry = cy + dcy;
    rw = cw .* exp(tw(idx));
    rh = ch .* exp(th(idx));

    % assemble bounding boxes [x1,y1,x2,y2]
    cr = [cx-cw/2,cy-ch/2,cx+cw/2,cy+ch/2];
    dr = [rx-rw/2,ry-rh/2,rx+rw/2,ry+rh/2];

    % visualize positive anhor boxes 
    plotBoxes(cr(:,1),cr(:,2),cr(:,3)-cr(:,1)+1,cr(:,4)-cr(:,2)+1, ...
              [1 0 0], 1);
    % visualize positive anchor boxes after bounding box refinement 
    plotBoxes(dr(:,1),dr(:,2),dr(:,3)-dr(:,1)+1,dr(:,4)-dr(:,2)+1, ...
              [0 0 1], 1);
    % visualize ground truth bounding box 
    if ~isempty(labelRect)
      plotBoxes(labelRect(:,1),labelRect(:,2),labelRect(:,3)- ...
                labelRect(:,1)+1,labelRect(:,4)-labelRect(:,2)+1, ...
                [ 0 1 0] , 1);
    end
    hold off;
    
    % visualize classification ground truth heat map
    subplot(132);
    imagesc(tile3DHeat(clsmap));
    axis square;

    % visualize regression ground truth heat map  
    subplot(133);
    imagesc(tile3DHeat(regmap));
    caxis([-50,50]);
    axis square;
    
    keyboard;
  end
end

function [cx,cy] = backtrack(opts, fx, fy)
rf = opts.recfields(opts.var2idx('score_cls'));
ofx = rf.offset(2); 
ofy = rf.offset(1); 
stx = rf.stride(2); 
sty = rf.stride(1); 
cx = (fx-1)*stx + ofx; 
cy = (fy-1)*sty + ofy;
