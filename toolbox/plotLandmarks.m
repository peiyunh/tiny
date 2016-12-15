function plotLandmarks(lands)
[H, W, P, D] = size(pose4d); 
p = randsample(P,1);

hold on ;
cmap = hsv(H*W);
for i = 1:H
    for j = 1:W
        id = (i-1)*W + j;
        c = cmap(id,:);
        x = pose4d(i,j,p,1:21); 
        y = pose4d(i,j,p,22:42); 
        scatter(x,y,30,'filled','MarkerFaceColor',c);
        
        x1 = min(x); 
        y1 = min(y); 
        x2 = max(x); 
        y2 = max(y);
        rectangle('position', [x1 y1 x2-x1+1 y2-y1+1], ...
                  'EdgeColor', c, 'LineStyle', '--');
    end
end
hold off; 
