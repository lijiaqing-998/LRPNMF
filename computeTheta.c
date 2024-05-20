#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"


#define random(x) (rand()%x)



void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[]) {
    
    /*this fucntion uses the randomized median finding algorithm to compute theta
     input:  vector b with all positive real elements 				 
     	     scalar z which is equal to alpha/L
        
     output: a scalar, the nonnegative theta
             
    */
    double chosen_val, z, S=0.0, deltaS, theta, swapt;
    int allsumless = 1, N=0, deltaN, rb, cb, lenb, tempsize;
    
    int i, randIdx;
    const mxArray * bvec; 
    
    double * bvecptr;
    double * tempvec, * currStart, * currEnd, * lowp, * highp;
    
    if (nrhs !=2)
		mexErrMsgTxt("Must have 2 input arguments");
	if (nlhs !=1)
		mexErrMsgTxt("Must have 1 output arguments");
    
    bvec = prhs[0];
    z = mxGetScalar(prhs[1]);
    bvecptr = mxGetPr(bvec);
    
    
    rb = mxGetM(bvec);
    cb = mxGetN(bvec);
    
    if (rb == 1)
        lenb = cb;
    else if (cb == 1)
        lenb = rb;
    else
        mexErrMsgTxt("Error, the first argument must be a vector!");
    
    tempvec = (double *)mxCalloc(lenb,sizeof(double));
    
    for (i = 0; i < lenb; i++)
        tempvec[i] = bvecptr[i];
    
    currStart = tempvec;
    currEnd = tempvec + lenb - 1;
    while (currEnd >= currStart)
    {
        tempsize = currEnd - currStart +1;
        randIdx = random(tempsize);
        chosen_val = *(currStart+randIdx);
        if (currStart+randIdx != currEnd)
            *(currStart+randIdx) = *currEnd;
        lowp = currStart;
        highp = currEnd-1;
        deltaS = chosen_val;
        deltaN = 1;
        while (lowp < highp)
        {
            while (*highp < chosen_val && highp > lowp) highp--;
            while (*lowp >= chosen_val && highp > lowp) 
            {
                deltaS += *lowp;
                deltaN++;
                lowp++;
            }
            if (highp != lowp)
            {
                deltaS += *highp;
                deltaN++;
                swapt = *lowp;
                *lowp = *highp;
                *highp = swapt;
                lowp++;
                if (lowp != highp)
                    highp--;
            }
        }
        
        if (lowp == highp && *lowp >= chosen_val)
        {
            
            deltaS += *lowp;
            
            deltaN++;
        } else lowp--;
        
        
        if ( ((S + deltaS) - (N + deltaN)*chosen_val) < z )
        {
            S += deltaS;
            N += deltaN;
            currStart = lowp+1;
            currEnd--;
        } else {
            allsumless = 0;
            currEnd = lowp;
        }
    }
    if (allsumless == 1 && S < z) 
        theta = 0;
    else
        theta = (S-z)/N;
    
    mxFree(tempvec);
    plhs[0] = mxCreateDoubleScalar(theta);

}