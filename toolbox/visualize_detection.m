function visualize_detection(img, bbox, thr)

if ~isempty(img), imshow(img); end

colors = parula(100);
scores = vl_nnsigmoid(bbox(:,end));

hold on;
for i = 1:size(bbox, 1)
    if scores(i) < thr, continue; end
    color = colors(ceil(100*scores(i)), :);
    bw = bbox(i,3) - bbox(i,1) + 1;
    bh = bbox(i,4) - bbox(i,2) + 1;
    if min([bw bh]) <= 20
        lw = 0.5;
    else
        lw = max(2, min(3, min([bh/20, bw/20])));
    end
    rectangle('position', [bbox(i,1:2) bbox(i,3:4)-bbox(i,1:2)+1], ...
              'EdgeColor', color, 'LineWidth', lw);
end
hold off;
%colormap parula;
colormap(colors);
colorbar;
axis off;
drawnow;