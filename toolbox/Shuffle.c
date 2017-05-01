// Shuffle.c
// Random permutation of array elements
// SHUFFLE works in Inplace, Index and Derange mode:
//
// 1. INPLACE MODE: Y = Shuffle(X, Dim)   --------------------------------------
// The elements of the input array are randomly re-ordered along the specified
// dimension. In opposite to X(RANDPERM(LENGTH(X)) no temporary memory is used,
// such that SHUFFLE is much more efficient for large arrays.
// INPUT:
//   X: Array of any size. Types: DOUBLE, SINGLE, CHAR, LOGICAL,
//      (U)INT64/32/16/8. The processed dimension must be shorter than 2^32
//      elements. To shuffle larger or complex arrays, use the Index mode.
//   Dim: Dimension to operate on.
//      Optional, default: [], operate on first non-singleton dimension.
// OUTPUT:
//   Y: Array of same size and type as X, but with shuffled elements.
//
// 2. INDEX MODE: Index = Shuffle(N, 'index', NOut)   --------------------------
// A vector of shuffled indices is created as by RANDPERM, but faster and using
// the smallest possible integer type to save memory. The number of output
// elements can be limited. This method works for cells, structs or complex
// arrays also.
// INPUT:
//   N:      Numeric scalar >= 0.
//   String: 'index'.
//   NOut:   Optional, number of outputs, NOut <= N. Default: N.
// OUTPUT:
//   Index:  [1 x nOut] vector containing the shuffled elements of the vector
//           [1:N] vector. To limit the memory usage the smallest possible type
//           is used: UINT8/16/32 or INT64.
//
// 3. DERANGEMENT MODE: Index = Shuffle(N, 'derange', NOut)  -------------------
// Equivalent to Index mode, but all elements Index[i] ~= i.
//
// CONTROL COMMANDS:  ----------------------------------------------------------
// Shuffle(S, 'seed'): S is either empty, scalar or a [1 x 4] DOUBLE vector to
//      set the random number  generator to a certain state. 4 integers between
//      0 and 2^32-1 are recommended for maximum entropy. If S is empty the
//      default seed is used.
// Shuffle([], 'lock'): Lock Mex in the memory such that "clear all" does not
//      reset the random  number generator. If the Mex was compiled with
//      -DAUTOLOCK it is locked automatically.
// Shuffle([], 'unlock'): Unlock the Mex file to allow a recompilation.
// Shuffle(): Display compiler settings and status: Fast/exact integer, accept
//      cell/struct, current lock status, auto-lock mode.
//
// EXAMPLES:
//   Shuffle(1234567890, 'seed');
//   R = Shuffle(1:8)             %  [5, 8, 7, 4, 2, 6, 3, 1]
//   R = Shuffle('abcdefg')       %  'gdfbcea'
//   R = Shuffle([1:4; 5:8], 2)   %  [3, 4, 2, 1;  8, 6, 7, 5]
//   I = Shuffle(8, 'index');     %  UINT8([3, 6, 5, 2, 4, 7, 8, 1])
// Choose 10 different rows from a 1000 x 100 matrix:
//   X = rand(1000, 100);       Y = X(Shuffle(1000, 'index', 10), :);
// Operate on cells or complex arrays:
//   C = {9, 's', 1:5};         SC = C(Shuffle(numel(C), 'index'));
//   M = rand(3) + i * rand(3); SM = M(:, Shuffle(size(C, 2), 'index'))
//
// NOTES: Shuffle(X) is about 50% to 85% faster than: Y = X(randperm(numel(X)).
//   It uses the Knuth-shuffle algorithm, also known as Fisher-Yates-shuffle.
//
//   The random numbers are created by the cute KISS algorithm of George
//   Marsaglia, which has a period of about 2^124 (~10^37). If somebody needs
//   "more" randomness, I can create a version using Marsaglias CMWC4096 with a
//   period of 10^33000, which remarkably tops e.g. MT19937. But then 4096
//   seeds with 32 bit and high entropy are needed...
//
//   To my surprise Shuffle can process CELLs and STRUCT arrays directly
//   treating them as arrays with elements of size mwSize. But this is not
//   documented by MathWorks and may change in the future (tested with Matlab
//   5.3, 6.5, 7.7, 7.8). After compiling with "-DSLOPPY_INPUT" this works:
//     r = Shuffle({'string', [], 9})   %  ==> {[], 'string', 9}
//   But it is faster and safer to permute the indices (shared data copies!).
//
//   Two methods are implemented to create a random integer between 0 and i:
//   1. EXACT: KISS_n32(i): A bias is avoided by rejecting all RNG values,
//      which are greater than the greatest multiple of i and replying
//      mod(KISS, i). This is the default.
//   2. FAST: (mwSize) (i * KISS() / 2^32)  with KISS is a UINT32
//            or for i >= 2^32:
//            (mwSize) (i * KISS_d53())     with 0.0 <= KISS_d < 1.0
//      This is e.g. 40% faster for generating a 1e6 index vector, but there is
//      a tiny bias: some values appear more often then others: Example:
//        i = 2^32-1, j = (uint32) (i*KISS()/2^32); ==>
//        Probability for i==j is 2^-31, but for i!=j it is 2^-32.
//      For "small" i, the bias is tiny and really hard to detect and the
//      shuffling reduces the influence in addition. If you need speed and
//      accept a small bias, compile with -DFASTRAND.
//
// COMPILATION:
//  -Unlock the mex-file for recompilations on demand: Shuffle([], 'unlock')
//  -Standard (no cell input, exact rand integers):
//     mex -O Shuffle.c
//  -Faster random integers with tiny bias:
//     mex -O -DFAST_RAND Shuffle.c
//  -Operate on CELLs and STRUCT arrays directly (experimental!):
//     mex -O -DSLOPPY_INPUT Shuffle.c
//  -Enable the automatic locking of the Mex file:
//     mex -O -AUTOLOCK Shuffle.c
//  -Fastest executable with MSVC 2008: Insert these optimization flags in
//   FULLFILE(prefdir, 'mexopts.bat'): OPTIMFLAGS = ... /arch:SSE2 /fp:fast ...
//   => 30% faster index method!
//  -Linux:
//     mex -O CFLAGS="\$CFLAGS -std=c99" Shuffle.c
//  -Precompiled Mex:
//     http://www.n-simon.de/mex
//  -Run the unit-test uTest_Shuffle after compiling or to test speed!
//  -The compiler directives can be combined.
//
// Tested: Matlab 6.5, 7.7, 7.8, WinXP, 32bit
//         Compiler: LCC2.4/3.8, BCC5.5, OWC1.8, MSVC2008
// Assumed Compatibility: higher Matlab versions, Mac, Linux, 64bit
// Author: Jan Simon, Heidelberg, (C) 2010-2011 j@n-simon.de
//
// See also RAND, RANDPERM.
// FEX:

/*
% $JRev: R-K V:037 Sum:ojzpgXwpSyUJ Date:07-Mar-2011 00:48:45 $
% $License: BSD (use/copy/change/redistribute on own risk, mention the author) $
% $UnitTest: uTest_Shuffle $
% $File: Tools\Mex\Source\Shuffle.c $
% History:
% 001: 24-Mar-2010 10:51, Stable version.
% 004: 31-Mar-2010 12:06, Dimension to operate on can be secified.
%      Use uint32_T instead of unsigned long to satisfy 64 bit compilers. This
%      was needed at least for GCC 4.4 on Unbuntu.
% 009: 05-Apr-2010 10:53, Index mode with limited output length.
% 016: 22-Apr-2010 22:15, Workaround for LCC 2.4 (shipped with Matlab).
%      James Tursa has created a workaround for the LCC 2.4. Based on this
%      KISS() and KISS_d() was modified to avoid mixed 32/64 bit integer
%      arithmetics.
% 020: 12-Jul-2010 21:03, BUGFIX: Seed with 4 numbers failed.
%      Mex file is locked in memory now. 'unlock' command. Empty matrix to use
%      the default seed without warming up the RNG.
% 031: 25-Jan-2011 15:44, 10% faster index vectors, thanks: Derek O'Connor.
%      See Knuth, Section 3.4.2, TAOCP, Vol 2, 3rd Ed.
%      Now the automatic locking of the function can be controlled by a
%      compiler directive. As default the function must be locked manually.
%      BUGFIX: Bad replies for 64 bit Index fixed. There was a very tiny chance
%      to get repeated indices for the Index method with > 2^32 elements.
% 037: 06-Mar-2011 14:02, Derangement mode.
%      After a long discussion in CSSM I've implemented a simple rejection
%      method: Call the Fisher-Yates-shuffle of the Index method until all
%      Y[i] != i. Including the check inside the Mex has the advantage, that the
%      memory is allocated once only. Therefore it is 4.4 times faster than
%      randpermfull(10000), and 30% faster than GRDmex(10000).
*/

#include "mex.h"
#include "tmwtypes.h"
#include <math.h>

// Fast or exact random integers:
#if defined(FASTRAND)
#define FAST_MODE 1
#else
#define FAST_MODE 0
#endif

// Lock the Mex file, such that CLEAR('all') does not reset the RNG:
#if defined(AUTOLOCK)
#define AUTOLOCK_MODE 1
#else
#define AUTOLOCK_MODE 0
#endif

// Error messages do not contain the function name in Matlab 6.5! This is not
// necessary in Matlab 7, but it does not bother:
#define ERR_ID   "JSimon:Shuffle:"
#define ERR_HEAD "*** Shuffle[mex]: "

// LCC 2.4 (shipped with Matlab) cannot compile unsigned long long. For
// signed long long, calculations with mixed 32 and 64 bit integers reply
// unexpected results. Solution: Run KISS with signed 64 bit values and cast
// intermediate values to 64 bit explicitely.
// LCC 3.8 (from the net) works fine with signed long long.
#if defined(__LCC__)
   typedef long long ULONG64;
   typedef long long int64_T;
#  define SPECIAL_CAST (long long)
#  define VALUE_698769069 698769069LL

#else  // Tested with BCC 5.5, MSVC 2008, OWC 1.8:
   typedef uint64_T ULONG64;
#  define SPECIAL_CAST          // Empty!
#  if defined(__BORLANDC__)     // No "ULL" for BCC 5.5
#    define VALUE_698769069 698769069UL
#  else
#    define VALUE_698769069 698769069ULL
#  endif
#endif

// 32 bit addressing for Matlab 6.5: -------------------------------------------
// See MEX option "compatibleArrayDims" for MEX in Matlab >= 7.7.
#ifndef MWSIZE_MAX
#define mwSize       int32_T           // Defined in tmwtypes.h
#define mwIndex      int32_T
#define MWSIZE_MAX   2147483647UL
#define MWINDEX_MAX  2147483647UL
#define MWSINDEX_MAX 2147483647L
#endif

// Parameters for the KISS random number generator: ----------------------------
static uint32_T kx = 123456789, ky = 362436000, kz = 521288629,
                kc = 7654321;
uint32_T KISS(void);
double KISS_d(void);
double KISS_d53(void);
uint32_T KISS_n32(const uint32_T n);
void SeedKISS(const mxArray *Seed);

// Prototypes: -----------------------------------------------------------------
void Identify(void);
mwSize GetStep(const mwSize *XDim, const mwSize Dim);
mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim);
void GetNandNOut(int nrhs, const mxArray *prhs[], const char *Mode,
                 double *N_d, double *NOut_d);

// "elements use <x> Bytes, Dimension <I>":
void Shuffle_8B_D1(double  *X, mwSize nV, mwSize nT);
void Shuffle_4B_D1(int32_T *X, mwSize nV, mwSize nT);
void Shuffle_2B_D1(int16_T *X, mwSize nV, mwSize nT);
void Shuffle_1B_D1(int8_T  *X, mwSize nV, mwSize nT);

void Shuffle_8B_DN(double  *X, mwSize Step, mwSize nV, mwSize nT);
void Shuffle_4B_DN(int32_T *X, mwSize Step, mwSize nV, mwSize nT);
void Shuffle_2B_DN(int16_T *X, mwSize Step, mwSize nV, mwSize nT);
void Shuffle_1B_DN(int8_T  *X, mwSize Step, mwSize nV, mwSize nT);

mxArray *Shuffle_Index(double N_d, double NOut_d);
void Index_8B_Full(double   *X, mwSize N);
void Index_8B_Part(double   *X, mwSize N, mwSize NOut);
void Index_4B_Full(uint32_T *X, mwSize N);
void Index_4B_Part(uint32_T *X, mwSize N, mwSize NOut);
void Index_2B_Full(uint16_T *X, mwSize N);
void Index_2B_Part(uint16_T *X, mwSize N, mwSize NOut);
void Index_1B_Full(uint8_T  *X, mwSize N);
void Index_1B_Part(uint8_T  *X, mwSize N, mwSize NOut);

mxArray *Derange_Index(double N_d, double NOut_d);
void Derange_8B(double   *X, mwSize N, mwSize NOut, int Partial);
void Derange_4B(uint32_T *X, mwSize N, mwSize NOut, int Partial);
void Derange_2B(uint16_T *X, mwSize N, mwSize NOut, int Partial);
void Derange_1B(uint8_T  *X, mwSize N, mwSize NOut, int Partial);

// Main function ===============================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  mwSize Dim, Step, XnDim, nX, nV, nT;
  const  mwSize *XDim;
  mxChar *C;
  double N_d, NOut_d, Dim_d;  // Indices as DOUBLE for secure check of limits
  void   *Data;
  
  // Lock Mex in the memory to prevent the RNG status from CLEAR ALL: ----------
#if AUTOLOCK_MODE
  if (!mexIsLocked()) {
     mexLock();
  }
#endif

  // Check number of inputs and outputs: ---------------------------------------
  if (nrhs == 0 || nrhs > 3) {
     if (nrhs == 0 && nlhs == 0) {
        Identify();
        return;
     } else {
        mexErrMsgIdAndTxt(ERR_ID   "BadNInput",
                          ERR_HEAD "1 or 3 inputs allowed.");
     }
  }
  if (nlhs > 1) {
     mexErrMsgIdAndTxt(ERR_ID   "BadNOutput",
                       ERR_HEAD "1 output allowed.");
  }
  
  // Get extent of 1st input: --------------------------------------------------
  // [X], or [N] for index mode, or [Seed] for seeding:
  XnDim = mxGetNumberOfDimensions(prhs[0]);
  XDim  = mxGetDimensions(prhs[0]);
  nX    = mxGetNumberOfElements(prhs[0]);
  
  if (nrhs == 1) {                 // 1 input: ---------------------------------
     // Operate on first non-singelton dimension - a check for scalar or empty
     // input is done later:
     Dim = FirstNonSingeltonDim(XnDim, XDim);

  } else {                         // 2 or 3 inputs:
     if (mxIsNumeric(prhs[1])) {   // 2nd input is [Dim]: ----------------------
        if (nrhs > 2) {
           mexErrMsgIdAndTxt(ERR_ID "BadNInput_Dim",
                   ERR_HEAD "Only 2 inputs are allowed if [Dim] is specified.");
        }
        
        switch (mxGetNumberOfElements(prhs[1])) {
           case 0:                 // Dim is empty - use the default:
              Dim = FirstNonSingeltonDim(XnDim, XDim);
              break;
              
           case 1:                 // Dim is a numeric scalar:
              Dim_d = mxGetScalar(prhs[1]) - 1;  // Zero based index
              if (Dim_d != floor(Dim_d) || !mxIsFinite(Dim_d)) {
                 mexErrMsgIdAndTxt(ERR_ID   "BadInput2",
                                   ERR_HEAD "[Dim] must be integer.");
              } else if (Dim_d < 0.0 || Dim_d >= (double) XnDim) {
                 mexErrMsgIdAndTxt(ERR_ID   "BadInput2",
                                   ERR_HEAD "[Dim] exceeds array dimensions.");
              }
              Dim = (mwSize) Dim_d;
              break;
           
           default:
              mexErrMsgIdAndTxt(ERR_ID   "BadInput2",
                                ERR_HEAD "[Dim] must be empty or scalar.");
        }
        
     } else if (mxIsChar(prhs[1])) {            // Special actions: ------------
        // The string need at least one character:
        if (mxGetNumberOfElements(prhs[1]) == 0) {
           mexErrMsgIdAndTxt(ERR_ID   "BadInput2",
                             ERR_HEAD "2nd input needs at least 1 character.");
        }
        
        // Check 1st character to branch to "seed", "index", "unlock" call:
        C = (mxChar *) mxGetData(prhs[1]);
        if (*C == L'I' || *C == L'i') {         // Shuffle index: --------------
           GetNandNOut(nrhs, prhs, "Index", &N_d, &NOut_d);
           plhs[0] = Shuffle_Index(N_d, NOut_d);
           
        } else if (*C == L'D' || *C == L'd') {  // Derangement index: ----------
           GetNandNOut(nrhs, prhs, "Derange", &N_d, &NOut_d);
           plhs[0] = Derange_Index(N_d, NOut_d);
           
        } else if (*C == L'S' || *C == L's') {  // Seed the RNG: ---------------
           SeedKISS(prhs[0]);                   // Input checked in subfunction
           
        } else if (*C == L'L' || *C == L'l') {  // Lock mex: -------------------
           if (!mexIsLocked()) {
              mexLock();
              mexPrintf("::: Shuffle[mex]: locked\n");
           }
        
        } else if (*C == L'U' || *C == L'u') {  // Unlock mex: -----------------
           if (mexIsLocked()) {
              mexUnlock();
              mexPrintf("::: Shuffle[mex]: unlocked\n");
           }
           
        } else {
           mexErrMsgIdAndTxt(ERR_ID "BadInput2",
                        ERR_HEAD "Unknown string command. "
                        "Known: 'seed', 'index', 'derange', 'lock', 'unlock'.");
        }
        
        return;
       
     } else {  // 2nd input is neither CHAR nor numeric: -----------------------
        mexErrMsgIdAndTxt(ERR_ID "BadInput2",
                     ERR_HEAD "2nd input must be the dimension to operate on.");
     }
  }
  
  // SHUFFLE(X, Dim) mode: -----------------------------------------------------
  // Refuse complex input:
  if (mxIsComplex(prhs[0])) {
     mexErrMsgIdAndTxt(ERR_ID   "ComplexInput",
                       ERR_HEAD "Use index mode for complex input!");
  }

#if !defined(SLOPPY_INPUT)
  // The method works for STRUCT arrays and CELLs also, but this is not
  // documented by The MathWorks! This stricter test rejects unexpected inputs:
  if (!mxIsNumeric(prhs[0]) && !mxIsChar(prhs[0])) {
     mexErrMsgIdAndTxt(ERR_ID   "BadInputType",
                       ERR_HEAD "The input must be numeric or a string.");
  }
#endif
  
  // Get step width between elements of the subvectors in dimension [Dim]:
  Step = GetStep(XDim, Dim);
  
  // Limit input size on 64-bit systems, because KISS replies just 32 bits.
  nV = XDim[Dim];
  if ((nV >> 31) > 2) {  // nV > 2^32, compatible with 32 bit mwSize
     mexErrMsgIdAndTxt(ERR_ID   "ToLargeInput",
                       ERR_HEAD "Specified dimension exceeds 2^32 elements.");
  }
    
  // Create the output with the same size, type and data as the input:
  plhs[0] = mxDuplicateArray(prhs[0]);
  
  // No shuffling for a scalar or empty input:
  if (nV <= 1) {
     return;
  }
  
  // Call different functions depending on the number of bytes per element:
  Data = mxGetData(plhs[0]);
  if (Step == 1) {  // Operate on first non-singleton dimension: ---------------
     nT = nX / nV;  // Product of final dimensions
     switch (mxGetElementSize(plhs[0])) {
       case 8:  Shuffle_8B_D1((double  *) Data, nV, nT);  break;
       case 4:  Shuffle_4B_D1((int32_T *) Data, nV, nT);  break;
       case 2:  Shuffle_2B_D1((int16_T *) Data, nV, nT);  break;
       case 1:  Shuffle_1B_D1((int8_T  *) Data, nV, nT);  break;
       default:
          mxDestroyArray(plhs[0]);
          mexErrMsgIdAndTxt(ERR_ID   "BadInputType",
                            ERR_HEAD "Unknown input type.");
     }
     
   } else {   // Operate on any dimension: -------------------------------------
     // Extent of subvector (from 1st to last element):
     nT = nX / (nV * Step);
     
     switch (mxGetElementSize(plhs[0])) {
       case 8:  Shuffle_8B_DN((double  *) Data, Step, nV, nT);  break;
       case 4:  Shuffle_4B_DN((int32_T *) Data, Step, nV, nT);  break;
       case 2:  Shuffle_2B_DN((int16_T *) Data, Step, nV, nT);  break;
       case 1:  Shuffle_1B_DN((int8_T  *) Data, Step, nV, nT);  break;
       default:
          mxDestroyArray(plhs[0]);
          mexErrMsgIdAndTxt(ERR_ID   "BadInputType",
                            ERR_HEAD "Unknown input type.");
     }
  }
  
  return;
}

// =============================================================================
mwSize FirstNonSingeltonDim(const mwSize Xndim, const mwSize *Xdim)
{
  // Get first non-singelton dimension - zero based.
  mwSize N;
  
  for (N = 0; N < Xndim; N++) {
     if (Xdim[N] != 1) {
        return (N);
     }
  }
  
  return (0);  // Use the first dimension if all dims are 1
}

// =============================================================================
mwSize GetStep(const mwSize *Xdim, const mwSize N)
{
  // Get step size between elements of a subvector in the N'th dimension.
  // This is the product of the leading dimensions.
  const mwSize *XdimEnd, *XdimP;
  mwSize       Step;
  
  Step    = 1;
  XdimEnd = Xdim + N;
  for (XdimP = Xdim; XdimP < XdimEnd; Step *= *XdimP++) ; // empty loop
  
  return (Step);
}

// =============================================================================
void GetNandNOut(int nrhs, const mxArray *prhs[], const char *Mode,
                 double *Np, double *NOutp)
{
  // Get number of indices and of outputs for Index mode.
  double N_d, NOut_d;
  
  if (mxGetNumberOfElements(prhs[0]) != 1 || !mxIsNumeric(prhs[0])) {
     mexErrMsgIdAndTxt(ERR_ID "BadInput1",
                       ERR_HEAD "%s mode: "
                       "1st input must be a numeric scalar.", Mode);
  }
  
  N_d = mxGetScalar(prhs[0]);        // Length of index vector as DOUBLE
  if (nrhs == 2) {
     NOut_d = N_d;                   // Same as number of indices
  } else {                           // 3 input is number of outputs:
     if (mxGetNumberOfElements(prhs[2]) != 1 || !mxIsNumeric(prhs[2])) {
        mexErrMsgIdAndTxt(ERR_ID "BadInput3",
                          ERR_HEAD "%s mode: "
                          "3rd input must be a numeric scalar.", Mode);
     }
     NOut_d = mxGetScalar(prhs[2]);  // As DOUBLE
  }
  
  // Work with doubles for N and NOut at first to avoid errors for checking
  // the limits of UINT32 values with 32 bit compilers:
  if (N_d >= (double) MWINDEX_MAX) {
     mexErrMsgIdAndTxt(ERR_ID   "BadIndex",
                       ERR_HEAD "%s mode: Index is too large.", Mode);
  } else if (N_d != floor(N_d) || NOut_d != floor(NOut_d)) {
     mexErrMsgIdAndTxt(ERR_ID   "BadIndex",
                       ERR_HEAD "%s mode: Indices must be integer.", Mode);
  } else if (N_d < 0.0 || NOut_d < 0.0 || NOut_d > N_d ||
             !mxIsFinite(N_d) || !mxIsFinite(NOut_d)) {
     mexErrMsgIdAndTxt(ERR_ID   "BadIndex",
                       ERR_HEAD "%s mode: 0 <= [NOut] <= [N].", Mode);
  }
  
  // Now a conversion to an integer type is safe, but DOUBLEs are replied to
  // allow a safe check of the 32 bit limits on 32 bit machines:
  *Np    = N_d;
  *NOutp = NOut_d;
  
  return;
}


// =============================================================================
// === Inplace shuffling or the input array                                  ===
// =============================================================================
void Shuffle_8B_D1(double *X, mwSize nV, mwSize nT)
{
  // Knuth or Fisher-Yates shuffle for DOUBLE and (U)INT64.
  // INPUT:
  //   X:  Pointer to output (operating inplace!).
  //   nV: Length of dimension to operate on.
  //   nT: Product of trailing dimensions.
  // Operate along first dimension.
  // There are 4 ways to shuffle an array: Start at first or last element, and
  // shuffle elements from 0 to i or from i to end. I cannot find a difference
  // for the sorting quality, while starting at the last element is minimally
  // faster for some compilers.
  
  double t;
  mwSize w, i;
  
  while (nT-- != 0) {  // Loop over trailing dimensions
    i = nV;
    while (i != 0) {   // Loop over subvector
       // Get swap index: 0 <= w <= i-1
#if FAST_MODE
       w    = (mwSize) (i * KISS_d());   // Tiny bias, but faster
#else
       w    = (mwSize) KISS_n32(i);
#endif
       t    = X[w];
       X[w] = X[--i];
       X[i] = t;
    }
    
    X += nV;
  }

  return;
}

// =============================================================================
void Shuffle_4B_D1(int32_T *X, mwSize nV, mwSize nT)
{
  // Knuth or Fisher-Yates shuffle for SINGLE and (U)INT32.
  // To my surprise this works for CELL and STRUCT arrays also.
  // Operate along first dimension.
  // Inputs and comments see: Shuffle_8B_D1()
  
  int32_T t;
  mwSize w, i;
  
  while (nT-- != 0) {
    i = nV;
    while (i != 0) {
#if FAST_MODE
       w    = (mwSize) (i * KISS_d());   // 0 <= w <= i-1
#else
       w    = KISS_n32(i);
#endif
       t    = X[w];
       X[w] = X[--i];
       X[i] = t;
    }

    X += nV;
  }

  return;
}

// =============================================================================
void Shuffle_2B_D1(int16_T *X, mwSize nV, mwSize nT)
{
  // Knuth or Fisher-Yates shuffle for CHAR and (U)INT16.
  // Operate along first dimension.
  // Inputs and comments see: Shuffle_8B_D1()
  
  int16_T t;
  mwSize w, i;
  
  while (nT-- != 0) {
    i = nV;
    while (i != 0) {
#if FAST_MODE
       w    = (mwSize) (i * KISS_d());   // 0 <= w <= i-1
#else
       w    = KISS_n32(i);
#endif
       t    = X[w];
       X[w] = X[--i];
       X[i] = t;
    }
    
    X += nV;
 }

  return;
}

// =============================================================================
void Shuffle_1B_D1(int8_T *X, mwSize nV, mwSize nT)
{
  // Knuth or Fisher-Yates shuffle for (U)INT8 and LOGICAL.
  // Operate along first dimension.
  // Inputs and comments see: Shuffle_8B_D1()
  
  int8_T t;
  mwSize w, i;
  
  while (nT-- != 0) {
    i = nV;
    while (i != 0) {
#if FAST_MODE
       w    = (mwSize) (i * KISS_d());   // 0 <= w <= i-1
#else
       w    = KISS_n32(i);
#endif
       t    = X[w];
       X[w] = X[--i];
       X[i] = t;
    }
    
    X += nV;
 }

  return;
}

// =============================================================================
void Shuffle_8B_DN(double *X, mwSize Step, mwSize nV, mwSize nT)
{
  // Shuffle array with 8 byte element size along a specified dimension. The
  // actual type of the data does not matter and this function works for
  // DOUBLE and (U)INT64.
  // X:  Pointer to output array (inplace shuffling).
  // Step: Step length between elements of the processed dimension. This is
  //     the product of the leading dimensions.
  // nV: Number of elements in the processed dimension.
  // nT: Product of trailing dimensions.
  // Operate along any dimension.
   
  double tmp, *Xp;                    // Valid for DOUBLE, U/INT64
  mwSize i, j, k, w, iS;
#if FAST_MODE
  double  i1d;
#else
  mwSize i1;
#endif
  
  for (k = 0; k < nT; k++) {          // Loop over trailing dimensions
     iS = 0;
     for (i = 1; i < nV; i++) {       // Process i.th element of subvectors
        Xp  = X;
        iS += Step;                   // Offset to i.th element
#if FAST_MODE                      // Faster but tiny bias
        i1d = (double) (i + 1) / 4294967296.0;  // Multiplication from KISS_d!
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * (mwSize) (i1d * KISS());  // (0 <= w <= i) * Step
#else
        i1  = i + 1;
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * KISS_n32(i1);
#endif
           tmp    = Xp[w];
           Xp[w]  = Xp[iS];
           Xp[iS] = tmp;
           Xp++;                      // Next row
        }
     }
     
     X += Step * nV;                  // Proceed to next chunk
  }

  return;
}

// =============================================================================
void Shuffle_4B_DN(int32_T *X, mwSize Step, mwSize nV, mwSize nT)
{
  // Shuffle array with 4 byte element size along a specified dimension.
  // Inputs and comments see: Shuffle_8B_DN()
  // Operate along any dimension.
  
  int32_T tmp, *Xp;                   // Valid for SINGLE, U/INT32
  mwSize  i, j, k, w, iS;
#if FAST_MODE
  double  i1d;
#else
  mwSize i1;
#endif
  
  for (k = 0; k < nT; k++) {          // Loop over trailing dimensions
     iS = 0;
     for (i = 1; i < nV; i++) {       // Process i.th element of subvectors
        Xp  = X;
        iS += Step;                   // Offset to i.th element
#if FAST_MODE
        i1d = (double) (i + 1) / 4294967296.0;  // Multiplication from KISS_d!
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * (mwSize) (i1d * KISS());  // (0 <= w <= i) * Step
#else
        i1  = i + 1;
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * KISS_n32(i1);
#endif
           tmp    = Xp[w];
           Xp[w]  = Xp[iS];
           Xp[iS] = tmp;
           Xp++;                      // Next row
        }
     }
     
     X += Step * nV;                  // Proceed to next chunk
  }
  
  return;
}

// =============================================================================
void Shuffle_2B_DN(int16_T *X, mwSize Step, mwSize nV, mwSize nT)
{
  // Shuffle array with 2 byte element size along a specified dimension.
  // Inputs and comments see: Shuffle_8B_DN()
  // Operate along any dimension.
  
  int16_T tmp, *Xp;                   // Valid for CHAR, U/INT16
  mwSize  i, j, k, w, iS;
#if FAST_MODE
  double  i1d;
#else
  mwSize i1;
#endif
  
  for (k = 0; k < nT; k++) {          // Loop over trailing dimensions
     iS = 0;
     for (i = 1; i < nV; i++) {       // Process i.th element of subvectors
        Xp  = X;
        iS += Step;                   // Offset to i.th element
#if FAST_MODE
        i1d = (double) (i + 1) / 4294967296.0;  // Multiplication from KISS_d!
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * (mwSize) (i1d * KISS());  // (0 <= w <= i) * Step
#else
        i1  = i + 1;
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * KISS_n32(i1);
#endif
           tmp    = Xp[w];
           Xp[w]  = Xp[iS];
           Xp[iS] = tmp;
           Xp++;                      // Next row
        }
     }
     
     X += Step * nV;                  // Proceed to next chunk
  }
  
  return;
}

// =============================================================================
void Shuffle_1B_DN(int8_T *X, mwSize Step, mwSize nV, mwSize nT)
{
  // Shuffle array with 1 byte element size along a specified dimension.
  // Inputs and comments see: Shuffle_8B_DN()
  // Operate along any dimension.
  
  int8_T tmp, *Xp;  // The actual type does not matter
  mwSize i, j, k, w, iS;
#if FAST_MODE
  double  i1d;
#else
  mwSize i1;
#endif
  
  for (k = 0; k < nT; k++) {          // Loop over trailing dimensions
     iS = 0;
     for (i = 1; i < nV; i++) {       // Process i.th element of subvectors
        Xp  = X;
        iS += Step;                   // Offset to i.th element
#if FAST_MODE
        i1d = (double) (i + 1) / 4294967296.0;  // Multiplication from KISS_d!
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * (mwSize) (i1d * KISS());  // (0 <= w <= i) * Step
#else
        i1  = i + 1;
        for (j = 0; j < Step; j++) {            // Loop over leading dimensions
           w = Step * KISS_n32(i1);
#endif
           tmp    = Xp[w];
           Xp[w]  = Xp[iS];
           Xp[iS] = tmp;
           Xp++;                      // Next row
        }
     }
     
     X += Step * nV;                  // Proceed to next chunk
  }
  
  return;
}


// =============================================================================
// === Shuffled index vector                                                 ===
// =============================================================================
mxArray *Shuffle_Index(double N_d, double NOut_d)
{
  // Create a vector [1:N] with the smallest possible unsigned intger type and
  // shuffle it afterwards. Reply the 1st nOut elements only.
  
  mxArray *Out;
  mwSize  N, NOut;
  
  // Number of indices and outputs as integers - otherwise a signed 32 bit
  // mwSize will invalidate the "N <= 2147483647" comparison for N=2^31.
  // The values have been checked in GetNanNOut() before:
  N    = (mwSize) N_d;
  NOut = (mwSize) NOut_d;
  
  // Early return on empty output:
  if (N == 0 || NOut == 0) {
     return (mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL));
  }
            
  // Create the index vector [1:N] with the smallest possible INT* type and
  // shuffle it:
  if (N <= 255) {
     Out = mxCreateNumericMatrix(1, N, mxUINT8_CLASS, mxREAL);
     if (NOut < (N * 0.9)) {
        Index_1B_Part((uint8_T *) mxGetData(Out), N, NOut);
     } else {
        Index_1B_Full((uint8_T *) mxGetData(Out), N);
     }
     
  } else if (N <= 65535) {
     Out = mxCreateNumericMatrix(1, N, mxUINT16_CLASS, mxREAL);
     if (NOut < (N * 0.9)) {
        Index_2B_Part((uint16_T *) mxGetData(Out), N, NOut);
     } else {
        Index_2B_Full((uint16_T *) mxGetData(Out), N);
     }
     
  } else if (N_d <= 2147483647.0) {   // Double N and sign for 32 bit systems
     Out = mxCreateNumericMatrix(1, N, mxUINT32_CLASS, mxREAL);
     if (NOut < (N * 0.75)) {
        Index_4B_Part((uint32_T *) mxGetData(Out), N, NOut);
     } else {
        Index_4B_Full((uint32_T *) mxGetData(Out), N);
     }
     
  } else {
     // For 32 bit systems, N >= MWINDEX_MAX has excluded 64 bit arrays already.
#if defined(__LCC__) || defined(__BORLANDC__)
     mexErrMsgIdAndTxt(ERR_ID "No64Bit",
                  ERR_HEAD "No 64 bit addressing if compiled with LCC or BCC.");
#endif
     Out = mxCreateNumericMatrix(1, N, mxDOUBLE_CLASS, mxREAL);
     if (NOut < (N * 0.5)) {
        Index_8B_Part(mxGetPr(Out), N, NOut);
     } else {
        Index_8B_Full(mxGetPr(Out), N);
     }
  }
  
  // Cut output vector if wanted - set new dimensions and realloc:
  if (NOut != N) {
     mxSetN(Out, NOut);
     mxSetPr(Out, mxRealloc(mxGetData(Out), NOut * mxGetElementSize(Out)));
  }
  
  return (Out);
}

// =============================================================================
void Index_8B_Full(double *X, mwSize N)
{
  // Knuth or Fisher-Yates shuffle for [1:N] for N < MWINDEX_MAX
  // Create the values dynamically, then no swapping is needed.
  
  int64_T i;
  double  *Xj;
  
  *X = 1;
  i  = 1;
  while (i < N) {
     Xj   = X + (mwSize) ((i + 1) * KISS_d53());  // X <= Xj < X+(i+1)
     X[i] = *Xj;
     *Xj  = (double) ++i;
  }

  return;
}

// =============================================================================
void Index_8B_Part(double *X, mwSize N, mwSize NOut)
{
  // Knuth or Fisher-Yates shuffle for INT64(1:N). Works on a vector only, but
  // allows limitation of shuffled elements.
  // NOT TESTED!
  // INPUT:
  //   X:    Pointer to output (operating inplace!).
  //   N:    Length of index vector.
  //   NOut: Number of elements to reply. The later elements are shuffled
  //         partially only and cleared by the caller.
  
  // METHOD 1:
  // Using array indices seems to be 1% slower with LCC, BCC, OWC and MSVC, but
  // this is below the measurement precision, so it is a question of taste!
  //
  // int64_T w, i = 0;
  // double t, *Y;
  //
  // t = 1;                        // Create [1:N] vector at first
  // Y = X;
  // while ((*Y++ = t++) != N) ;   // empty loop
  //
  // while (i < NOut) {
  //    w      = i + (mwSize) (N-- * KISS_d53());   // i <= w <= n
  //    t      = X[w];
  //    X[w]   = X[i];
  //    X[i++] = t;
  // }
  
  // METHOD 2:
  // Access the vector elements by two pointers:
  // double t, *Y, *Xf = X + NOut;
  //
  // t = 1;                       // Create [1:N] vector at first
  // Y = X;
  // while ((*Y++ = t++) != N) ;  // empty loop
  //
  // while (X < Xf) {
  //    Y    = X + (mwSize) (N-- * KISS_d53());   // i <= w <= n
  //    t    = *Y;
  //    *Y   = *X;
  //    *X++ = t;
  // }
   
  // METHOD 3:
  // Create vector elements and shuffle at the same time - this is much faster
  // if NOut is much smaller than N.
  
  double *Xj, t;
  mwSize w, i = 0;
  
  t  = 1;                          // Create [1:NOut] vector at first
  Xj = X;
  while ((*Xj++ = t++) != NOut) ;  // empty loop
  
  while (i < NOut) {
     w      = i + (mwSize) (N-- * KISS_d53());   // i <= w < n
     t      = X[w];                      // BCC 5.5: warning "loss of precision"
     X[w]   = X[i];
     X[i++] = t ? t : (double) (w + 1);  // Create element if not done before
  }
  
  return;
}

// =============================================================================
void Index_4B_Full(uint32_T *X, mwSize N)
{
  // Knuth or Fisher-Yates shuffle for [1:N] for N < MAX_uint32_T
  // Create the values dynamically, then no swapping is needed.
  
  mwSize   i;
  uint32_T *Xj;
  
  *X = 1;
  i  = 1;
  while (i < N) {
#if FAST_MODE
     Xj   = X + (mwSize) ((i + 1) * KISS_d());   // X <= Xj < X+(i+1)
#else
     Xj   = X + KISS_n32(i + 1);
#endif
     X[i] = *Xj;
     *Xj  = ++i;
  }

  return;
}

// =============================================================================
void Index_4B_Part(uint32_T *X, mwSize N, mwSize NOut)
{
  // Knuth or Fisher-Yates shuffle for [1:N] for N < MAX_uint32_T
  // Create the reduced index vector [1:NOut] at first and higher elements
  // on demand only. This is faster if NOut is about 10% smaller than N.
  
  uint32_T t, *Xj, NOut_u32 = (uint32_T) NOut;
  mwSize i, w;

  t  = 1;                              // Create [1:NOut] vector at first
  Xj = X;
  while ((*Xj++ = t++) != NOut_u32) ;  // empty loop
  
  i = 0;
  while (i < NOut) {
#if FAST_MODE
     w      = i + (mwSize) (N-- * KISS_d());  // i <= w < n
#else
     w      = i + KISS_n32(N--);
#endif
     t      = X[w];
     X[w]   = X[i];
     X[i++] = t ? t : (w + 1);         // Create element if not done before
  }

  return;
}

// =============================================================================
void Index_2B_Full(uint16_T *X, mwSize N)
{
  // Knuth or Fisher-Yates shuffle [1:N] for N < MAX_uint16_T.
  // Create the values dynamically, then no swapping is needed.
  
  mwSize   i;
  uint16_T *Xj;
  
  *X = 1;
  i  = 1;
  while (i < N) {
#if FAST_MODE
     Xj   = X + (mwSize) ((i + 1) * KISS_d());   // X <= Xj < X+(i+1)
#else
     Xj   = X + KISS_n32(i + 1);
#endif
     X[i] = *Xj;
     *Xj  = (uint16_T) (++i);
  }

  return;
}

// =============================================================================
void Index_2B_Part(uint16_T *X, mwSize N, mwSize NOut)
{
  // Knuth or Fisher-Yates shuffle [1:N] for N <= MAX_uint16T.
  // Create the complete index vector and shuffle afterwards.
  // This is faster if NOut is much smaller than N.
  
  uint16_T t, *Xj, NOut_u16 = (uint16_T) NOut;
  mwSize i, w;

  t  = 1;                              // Create [1:NOut] vector at first
  Xj = X;
  while ((*Xj++ = t++) != NOut_u16) ;  // empty loop
  
  i = 0;
  while (i < NOut) {
#if FAST_MODE
     w      = i + (mwSize) (N-- * KISS_d());  // i <= w < n
#else
     w      = i + KISS_n32(N--);
#endif
     t      = X[w];
     X[w]   = X[i];
     X[i++] = t ? t : (w + 1);         // Create element if not done before
  }

  return;
}

// =============================================================================
void Index_1B_Full(uint8_T *X, mwSize N)
{
  // Knuth or Fisher-Yates shuffle [1:N] for N <= MAX_uint8T.
  // Create the values dynamically, then no swapping is needed.
  
  uint8_T i, *Xj, N_u8 = (uint8_T) N;
  
  *X = 1;
  i  = 1;
  while (i < N_u8) {
#if FAST_MODE
     Xj   = X + (mwSize) ((i + 1) * KISS_d());   // X <= Xj < X+(i+1)
#else
     Xj   = X + KISS_n32(i + 1);
#endif
     X[i] = *Xj;
     *Xj  = ++i;
  }

  return;
}

// =============================================================================
void Index_1B_Part(uint8_T *X, mwSize N, mwSize NOut)
{
  // Knuth or Fisher-Yates shuffle [1:N] for N <= MAX_uint8T.
  // Create the complete index vector and shuffle afterwards.
  // This is faster if NOut is smaller than N.
  
  uint8_T t, *Xj, *Xf = X + NOut, N_u8 = (uint8_T) N;
  
  t  = 1;                          // Create [1:N] vector at first
  Xj = X;
  while ((*Xj++ = t++) != N_u8) ;  // empty loop, not NOut!
  
  while (X < Xf) {                 // Shuffle just the output elements
#if FAST_MODE
     Xj   = X + (mwSize) (N-- * KISS_d());  // X <= Xj < X+n
#else
     Xj   = X + KISS_n32(N--);
#endif
     t    = *Xj;
     *Xj  = *X;
     *X++ = t;
  }
  
  return;
}


// =============================================================================
// === Derangement index vector                                              ===
// =============================================================================
mxArray *Derange_Index(double N_d, double NOut_d)
{
  // Create an index vector with NOut elements of the range [1:N].
  // Reject this vector until all Y[i] != i.
  
  mxArray *Out;
  mwSize  N, NOut;
  
  // Number of indices and outputs as integers - otherwise a signed 32 bit
  // mwSize will invalidate the "N <= 2147483647" comparison for N=2^31.
  // The values have been checked in GetNanNOut() before:
  N    = (mwSize) N_d;
  NOut = (mwSize) NOut_d;
    
  // There is no derangement of length 1:
  if (N < 2) {
     mexErrMsgIdAndTxt(ERR_ID   "NoScalarDerange",
                       ERR_HEAD "A derangement needs at least 2 elements.");
  }
  
  // Early return on empty output:
  if (NOut == 0) {
     return (mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL));
  }
  
  // Create the index vector [1:N] with the smallest possible INT* type and
  // shuffle it:
  if (N <= 255) {
     Out = mxCreateNumericMatrix(1, N, mxUINT8_CLASS, mxREAL);
     Derange_1B((uint8_T *) mxGetData(Out), N, NOut, NOut < (N * 0.9));
     
  } else if (N <= 65535) {
     Out = mxCreateNumericMatrix(1, N, mxUINT16_CLASS, mxREAL);
     Derange_2B((uint16_T *) mxGetData(Out), N, NOut, NOut < (N * 0.9));
     
  } else if (N_d <= 2147483647.0) {   // Double N and sign for 32 bit systems
     Out = mxCreateNumericMatrix(1, N, mxUINT32_CLASS, mxREAL);
     Derange_4B((uint32_T *) mxGetData(Out), N, NOut, NOut < (N * 0.75));
     
  } else {
     // For 32 bit systems, N >= MWINDEX_MAX has excluded 64 bit arrays already.
#if defined(__LCC__) || defined(__BORLANDC__)
     mexErrMsgIdAndTxt(ERR_ID "No64Bit",
                  ERR_HEAD "No 64 bit addressing if compiled with LCC or BCC.");
#endif
     Out = mxCreateNumericMatrix(1, N, mxDOUBLE_CLASS, mxREAL);
     Derange_8B(mxGetPr(Out), N, NOut, NOut < (N * 0.5));
  }
  
  // Cut output vector if wanted - set new dimensions and realloc:
  if (NOut != N) {
     mxSetN(Out, NOut);
     mxSetPr(Out, mxRealloc(mxGetData(Out), NOut * mxGetElementSize(Out)));
  }
  
  return (Out);
}

// =============================================================================
void Derange_8B(double *X, mwSize N, mwSize NOut, int Partial)
{
  // Call the Index method to create the vector until all X[i] != i.
  
  int64_T i, NOut_64 = NOut;
  
  while (1) {
     if (Partial) {                   // Get a shuffle index vector:
        Index_8B_Part(X, N, NOut);
     } else {
        Index_8B_Full(X, N);
     }
     
     for (i = 0; i < NOut_64; i++) {  // Check for X[i] != i
        if (X[i] == (double) (i + 1)) {
           break;
        }
     }

     if (i == NOut_64) {              // Accept or restart:
        return;
     }
     if (Partial) {                   // Clear values:
        memset(X + NOut, 0, (N - NOut) * sizeof(double));
     }
  }

  return;
}

// =============================================================================
void Derange_4B(uint32_T *X, mwSize N, mwSize NOut, int Partial)
{
  // Call the Index method to create the vector until all X[i] != i.
  
  uint32_T i, NOut_32 = NOut;

  while (1) {
     if (Partial) {                   // Get a shuffle index vector:
        Index_4B_Part(X, N, NOut);
     } else {
        Index_4B_Full(X, N);
     }
     
     for (i = 0; i < NOut_32; i++) {  // Check for X[i] != i
        if (X[i] == (i + 1)) {
           break;
        }
     }

     if (i == NOut_32) {              // Accept or restart:
        return;
     }
     if (Partial) {                   // Clear values:
        memset(X + NOut, 0, (N - NOut) * sizeof(uint32_T));
     }
  }

  return;
}

// =============================================================================
void Derange_2B(uint16_T *X, mwSize N, mwSize NOut, int Partial)
{
  // Call the Index method to create the vector until all X[i] != i.
  
  uint16_T i, NOut_16 = NOut;

  while (1) {
     if (Partial) {                   // Get a shuffle index vector:
        Index_2B_Part(X, N, NOut);
     } else {
        Index_2B_Full(X, N);
     }
     
     for (i = 0; i < NOut_16; i++) {  // Check for X[i] != i
        if (X[i] == (i + 1)) {
           break;
        }
     }

     if (i == NOut_16) {              // Accept or restart:
        return;
     }
     if (Partial) {                   // Clear values:
        memset(X + NOut, 0, (N - NOut) * sizeof(uint16_T));
     }
  }

  return;
}

// =============================================================================
void Derange_1B(uint8_T *X, mwSize N, mwSize NOut, int Partial)
{
  // Call the Index method to create the vector until all X[i] != i.
  
  uint8_T i = 0, NOut_8 = NOut;
  
  while (i != NOut_8) {
     if (Partial) {                   // Get a shuffle index vector:
        Index_1B_Part(X, N, NOut);
     } else {
        Index_1B_Full(X, N);
     }
     
     for (i = 0; i < NOut_8; i++) {   // Check for X[i] != i
        if (X[i] == (i + 1)) {
           break;
        }
     }
  }

  return;
}

// =============================================================================
// === Random number generator:                                              ===
// =============================================================================
uint32_T KISS(void) {
  // Random integer, 0 < x < 2^32-1
  // George Marsaglia: Keep It Simple Stupid
  // Features: 32 bit numbers, solves DIEHARD test, period > 2^124 (== 2.1e37)
  // The status is implemented as global statics for easier seeding.
  
  ULONG64 t, a = VALUE_698769069;
  
  kx  = 69069 * kx + 12345;
  ky ^= ky << 13;
  ky ^= ky >> 17;
  ky ^= ky << 5;
  t   = a * SPECIAL_CAST kz + SPECIAL_CAST kc;
  kc  = (uint32_T) (t >> 32);
  
  return (kx + ky + (kz = (uint32_T) t));
}

// =============================================================================
double KISS_d(void) {
  // Random double, 0 <= x < 1, 32 bit precision.
  
  ULONG64 t, a = VALUE_698769069;
  
  kx  = 69069 * kx + 12345;
  ky ^= ky << 13;
  ky ^= ky >> 17;
  ky ^= ky << 5;
  t   = a * SPECIAL_CAST kz + SPECIAL_CAST kc;  // SPECIAL for ULONG64 and LCC2
  kc  = (uint32_T) (t >> 32);
  
  // Direct approach using UINT->DOUBLE:
  // return ((kx + ky + (kz = (uint32_T) t)) / 4294967296.0);
  
  // 17% faster: INT->DOUBLE is implemented as processor command, but
  // UINT->DOUBLE is not! The results differ by +-0.5 !
  return ((int32_T) (kx + ky + (kz = (uint32_T) t)) *
          (1.0 / 4294967296.0) + 0.5);
}

// =============================================================================
double KISS_d53(void) {
  // Random double, 0 <= x < 1, 53 bit precision.
  // Uses 64 bit of 2 KISS UINT32 numbers.
  
  // Direct approach using UINT->DOUBLE:
  // return ((2097152.0 * KISS() + (KISS() >> 11)) / 9007199254740992.0);

  // 14% faster - INT->DOUBLE is implemented as processor command, but
  // UINT->DOUBLE is not! The results differ by 0.5!
  int32_T a = (int32_T) (KISS() >> 5);   // Signed, but result is positive!
  int32_T b = (int32_T) (KISS() >> 6);   // Signed, but result is positive!
  
  return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
  
  // 0<=x<=1: return (a * 67108864.0 + b) * (1.0 / 9007199254740991.0);
}

// =============================================================================
uint32_T KISS_n32(const uint32_T n)
{
  // Random integer number, 0 <= x < n.
  // The 32 bit random number created by KISS is *not* simply truncated by MOD
  // or DIV to avoid the bias, if n is no divider of MAX_uint32_T.
  
  uint32_T value, max_u = (uint32_T) 0xFFFFFFFFU;  // MAX_uint32_T
  
  // Only 0 is <= 1:
  if (n < 2) {
     return (uint32_T) 0;
  }
   
  // Draw numbers until one is not greater than the biggest multiple of n,
  // which is smaller than MAX_uint32_T:
  max_u -= max_u % n;
  while ((value = KISS()) >= max_u) ;  // Empty loop
  
  return (value % n);
}

// =============================================================================
void SeedKISS(const mxArray *Seed)
{
  // Seed the status of the KISS random number generator.
  
  uint32_T S, n;
  double   *Sd;
  mwSize   nSeed;
  
  // Actually UINT32 values are the natural type, but it is not worth to
  // program 2 different input methods.
  if (!mxIsDouble(Seed)) {
     mexErrMsgIdAndTxt(ERR_ID "BadSeed",
             ERR_HEAD "Input [Seed] must be a DOUBLE with 0, 1 or 4 elements.");
  }
  
  // Initial status as in Marsaglia's publication:
  kx = 123456789UL;
  ky = 362436000UL;
  kz = 521288629UL;
  kc = 7654321UL;

  nSeed = mxGetNumberOfElements(Seed);
  if (nSeed != 0) {
     if (nSeed == 1) {         // Cheap seeding with a single scalar:
        S   = (uint32_T) mxGetScalar(Seed);
        kx ^= S;
        ky ^= KISS() ^ S;
        kz ^= KISS() ^ S;
        kc ^= KISS() ^ S;
        kc %= 698769069UL;     // Limit kc
        
     } else if (nSeed == 4) {  // 4 elements for more entropy:
        Sd  = mxGetPr(Seed);
        kx ^= (uint32_T) Sd[0];
        ky ^= (uint32_T) Sd[1];
        kz ^= (uint32_T) Sd[2];
        kc ^= (uint32_T) Sd[3];
        kc %= 698769069UL;     // Limit kc
        
     } else if (nSeed != 0) {
        mexErrMsgIdAndTxt(ERR_ID "BadInput2",
             ERR_HEAD "Input [Seed] must be a DOUBLE with 0, 1 or 4 elements.");
     }
     
     // Warm up - not necessarily needed also:
     n = 67 + KISS() % 63;
     while (n--) {
        KISS();
     }
  }

  return;
}

// =============================================================================
void Identify(void)
{
  // Display date and time of compilation:
  mexPrintf(__FILE__ "\nCompiled:          " __DATE__ " " __TIME__ "\n");
  
#if FAST_MODE
    mexPrintf("Integer mode:      fast\n");
#else
    mexPrintf("Integer mode:      exact\n");
#endif

#if defined(LOOSE_INPUT_TYPE)
    mexPrintf("Cell/struct input: accepted\n");
#else
    mexPrintf("Cell/struct input: rejected\n");
#endif

  if (mexIsLocked()) {
     mexPrintf("Locked mex:        on\n");
  } else {
     mexPrintf("Locked mex:        off\n");
  }
  
#if AUTOLOCK_RNG
    mexPrintf("Auto lock:         on\n");
#else
    mexPrintf("Auto lock:         off\n");
#endif

  return;
}
