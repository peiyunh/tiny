#include "mex.h"
#include "math.h"
#include <algorithm>
#include <string.h>
using namespace std; 

/*
 *                              0     1      2      3     4     5
 * function o = compute_overlap(bbox, fdimy, fdimx, dimy, dimx, scale, 
 *                              6     7     8
 *                              padx, pady, imsize)
 * bbox   bounding box image coordinates [x1 y1 x2 y2]
 * fdimy  number of rows in filter
 * fdimx  number of cols in filter
 * dimy   number of rows in feature map
 * dimx   number of cols in feature map
 * scale  image scale the feature map was computed at
 * padx   x padding added to feature map
 * pady   y padding added to feature map
 * imsize size of the image [h w]
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) { 
  /* inputs */
  const double ofx = mxGetScalar(prhs[0]);
  const double ofy = mxGetScalar(prhs[1]);
  const double stx = mxGetScalar(prhs[2]);
  const double sty = mxGetScalar(prhs[3]);
  const int vsx = mxGetScalar(prhs[4]);
  const int vsy = mxGetScalar(prhs[5]);

  const int nt = mxGetNumberOfElements(prhs[6]);
  const double *dx1 = mxGetPr(prhs[6]);
  const double *dy1 = mxGetPr(prhs[7]);
  const double *dx2 = mxGetPr(prhs[8]);
  const double *dy2 = mxGetPr(prhs[9]);

  const int ng = mxGetNumberOfElements(prhs[10]);
  const double *gx1 = mxGetPr(prhs[10]);
  const double *gy1 = mxGetPr(prhs[11]);
  const double *gx2 = mxGetPr(prhs[12]);
  const double *gy2 = mxGetPr(prhs[13]);

  const double zmx = mxGetScalar(prhs[14]);
  const double zmy = mxGetScalar(prhs[15]);

  /* outputs */
  const int dims[] = {vsy, vsx, nt, ng};
  /* printf("dims: %d %d %d %d\n", vsy, vsx, nt, ng); */

  mxArray *mx_overlap = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
  double *overlap = (double *)mxGetPr(mx_overlap);
  plhs[0] = mx_overlap;

  /* temporary buffer */
  const int tmp_dims[] = {(vsy-1)*zmy+1, (vsx-1)*zmx+1, nt};
  mxArray *mx_tmp_overlap = mxCreateNumericArray(3, tmp_dims, mxDOUBLE_CLASS, mxREAL);
  double *tmp_overlap = (double *)mxGetPr(mx_tmp_overlap);
  
  /* setup pooling input & output */
  mxArray *inputs[6];
  
  /* run max pooling over the temporary buffer */
  inputs[0] = mx_tmp_overlap;
  mxArray *mx_pool = mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS,mxREAL);
  double *pool = mxGetPr(mx_pool);
  pool[0] = zmy;
  pool[1] = zmx;
  inputs[1] = mx_pool;

  /* set stride */
  inputs[2] = mxCreateString("Stride");
  mxArray *mx_stride = mxCreateNumericMatrix(1,2,mxDOUBLE_CLASS,mxREAL);
  double *stride = mxGetPr(mx_stride);
  stride[0] = zmy;
  stride[1] = zmx;
  inputs[3] = mx_stride;

  /* set padding */
  inputs[4] = mxCreateString("Pad");
  mxArray *mx_pad = mxCreateNumericMatrix(1,4,mxDOUBLE_CLASS,mxREAL);
  double *pad = mxGetPr(mx_pad);
  pad[0] = floor(zmy/2);
  pad[1] = ceil(zmy/2);
  pad[2] = floor(zmx/2);
  pad[3] = ceil(zmx/2);
  inputs[5] = mx_pad;

  /* declare outputs */
  mxArray *outputs[1];

  int i, j, x, y; 
  /* compute overlap for each placement of the filter */
  for (int i = 0; i < ng; i ++) { 
    double bbox_x1 = gx1[i];
    double bbox_y1 = gy1[i];
    double bbox_x2 = gx2[i];
    double bbox_y2 = gy2[i];

    double bbox_h = bbox_y2 - bbox_y1 + 1;
    double bbox_w = bbox_x2 - bbox_x1 + 1; 
    double bbox_area = bbox_h * bbox_w;

    int gidx = vsy * vsx * nt * i;
    /* NOTE: we do not have to reset everytime, once we make sure we
     * overwrite every element down below */
    /* memset(tmp_overlap, 0.0, tmp_dims[0]*tmp_dims[1]*tmp_dims[2]*sizeof(double)); */

    for (j = 0; j < nt; j ++) {
      /*int tid = omp_get_thread_num();
	printf("Thread id: %d\n", tid);*/
      
      double delta_x1 = dx1[j]; 
      double delta_y1 = dy1[j]; 
      double delta_x2 = dx2[j]; 
      double delta_y2 = dy2[j];

      double filter_h = delta_y2 - delta_y1 + 1;
      double filter_w = delta_x2 - delta_x1 + 1;
      double filter_area = filter_h * filter_w;

      int tidx = tmp_dims[0] * tmp_dims[1] * j;

      int xmax = (vsx-1)*zmx;
      int ymax = (vsy-1)*zmy;

      /* enumerate spatial locations */
      for (x = 0; x <= xmax; x ++){
	for (y = 0; y <= ymax; y ++){
	  double cx = ofx + x*(stx/zmx);
	  int xidx = tmp_dims[0] * x;
	  double cy = ofy + y*(sty/zmy);
	    
	  double x1 = delta_x1 + cx;
	  double y1 = delta_y1 + cy;
	  double x2 = delta_x2 + cx;
	  double y2 = delta_y2 + cy;

	  double xx1 = max(x1, bbox_x1); 
	  double yy1 = max(y1, bbox_y1); 
	  double xx2 = min(x2, bbox_x2); 
	  double yy2 = min(y2, bbox_y2);

	  double int_w = xx2 - xx1 + 1;
	  double int_h = yy2 - yy1 + 1;
	  if (int_w > 0 && int_h > 0){
	    double int_area = int_w * int_h; 
	    double union_area = filter_area + bbox_area - int_area;
	    *(tmp_overlap+tidx+xidx+y) = int_area / union_area;
	  }else{
	    *(tmp_overlap+tidx+xidx+y) = 0;
	  }
	}
      }
    }

    /* call max pooling if needed */
    double *pooled_overlap;
    if (zmx != 1 || zmy != 1){
      mexCallMATLAB(1, outputs, 6, inputs, "vl_nnpool");
      pooled_overlap = mxGetPr(outputs[0]);
    }else{
      pooled_overlap = tmp_overlap;
    }

    /* copy results from buffer to final output */ 
    memcpy(overlap+gidx, pooled_overlap, vsy*vsx*nt*sizeof(double));
    
    /*for (int k = 0; k < 6; k ++ )
      mxDestroyArray(inputs[k]);
      mxDestroyArray(outputs[0]);*/
  }
}
