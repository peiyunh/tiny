function plotBoxes(x, y, w, h, c, lw)
nx = numel(x);
ny = numel(y);
nw = numel(w);
nh = numel(h);
nc = size(c,1); 
nl = size(lw,1);
if any(nx ~= [ny nw nh])
    error('Please input (x,y,w,h[,c,w]) with same dim');
end
hold on; 
for i = 1:nx
    clr = c(min(nc,i),:);
    lwd = lw(min(nl,i));
    if lwd == 0, continue; end;
    rectangle('position', [x(i), y(i), w(i), h(i)], ...
              'EdgeColor', clr, 'LineWidth', lwd);
end
hold off;