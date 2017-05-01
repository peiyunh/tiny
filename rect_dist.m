function D = rect_dist(XI,XJ)

%XI = round(XI); 
%XJ = round(XJ);

aI = (XI(:,3)-XI(:,1)+1) .* (XI(:,4)-XI(:,2)+1);
aJ = (XJ(:,3)-XJ(:,1)+1) .* (XJ(:,4)-XJ(:,2)+1);

x1 = max(XI(:,1), XJ(:,1)); 
y1 = max(XI(:,2), XJ(:,2)); 
x2 = min(XI(:,3), XJ(:,3)); 
y2 = min(XI(:,4), XJ(:,4));

aIJ = (x2-x1+1) .* (y2-y1+1) .* (x2>x1 & y2>y1);

iou = aIJ ./ (aI+aJ-aIJ);
D = max(0, min(1, 1 - iou));
