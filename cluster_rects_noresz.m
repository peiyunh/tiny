function C  = cluster_rects_noresz(imdb, N, minsz, maxsz, vis)
if nargin < 5
    vis = 0;
end

subplot = @(m,n,k) subtightplot(m,n,k,[0.1,0.1]);

%% use all trainval images
idx = find(imdb.images.set <= 2);

%% NOTE ignore translation and upside-down flag 
rects = imdb.labels.rects(idx);

%% resize rects to how we view them during training
%sizes = imdb.images.size(idx,:);
%for i = 1:size(rects,1)
%    rects{i} = rects{i} ./ min(sizes(i,:)./maxsz);
%end
rects = vertcat(rects{:});

%% centralize 
hs = rects(:,4) - rects(:,2) + 1;
ws = rects(:,3) - rects(:,1) + 1;

%% quantize 
rects = [-(ws-1)/2, -(hs-1)/2, (ws-1)/2, (hs-1)/2];
%% add multi-scale versions 
%rects = vertcat(rects/2, rects, rects*2);
%rects = vertcat(rects, rects*2);
%rects = vertcat(rects/2, rects);

%% ignore faces with size out of the range
idx = find(hs<=maxsz(1)&ws<=maxsz(2)&hs>=minsz(1)&ws>=minsz(2));
rects = rects(idx,:);
fprintf('Ignored faces smaller than %dx%d, %d bboxes left.\n', ...
        minsz(1),minsz(2),numel(idx));

%% subsample for faster clustering
rects = rects(randsample(size(rects,1), min(size(rects,1), 5e4)), :);
fprintf('Clustering on %d/%d face bounding boxes.\n', ...
        size(rects,1), numel(idx));

%% build kmedoids
[Cidx,C,sumd,D,midx] = kmedoids(rects, N, 'Options', ...
                                statset('UseParallel', true), ...
                                'Distance', @rect_dist);

% sort C based on areas
[~,I] = sort(C(:,3).*C(:,4),'descend');
C = C(I,:);

if ~vis
    return;
end

clf; 
[SI,SJ] = factorize(N);
for i = 1:N
    subplot(SI,SJ,i);
    plotBoxes(C(i,1),C(i,2),C(i,3)-C(i,1)+1,C(i,4)-C(i,2)+1,rand(1,3),0.5);
    title(num2str(i));
    axis([-250,250,-250,250]);
end
