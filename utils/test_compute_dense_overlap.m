load('test_data.mat')
iou_ = compute_dense_overlap(ofx,ofy,stx,sty,vsx,vsy,dx1,dy1,dx2,dy2,gx1,gy1,gx2,gy2,1,1);
err = abs(iou_ - iou);
if all(err(:)) < 1e-12
    fprintf('Test for compute_dense_overlap [passed]\n');
else
    fprintf('Test for compute_dense_overlap [failed]\n');
end
    
    
