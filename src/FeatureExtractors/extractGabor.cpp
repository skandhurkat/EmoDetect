#include <FeatureExtractors/extractGabor.h>
#include <Infrastructure/exceptions.h>
#define  _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include "cxcore.h"
#include "cv.h"
#include "highgui.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib>
#include <string>
#include <iostream>
#include <getopt.h>
#include <climits>
#include <vector>
#include <cstdio>

#define IMAGE_FILE_JPG "jpg"
#define IMAGE_DIR "images"
#define M_PI 3.14159265359

using namespace std;
using namespace cv;

int save_mat_image(CvMat* mat, char* name)
{
  double maxVal,minVal;
  CvPoint minLoc, maxLoc;
  cvMinMaxLoc( mat, &minVal, &maxVal,&minLoc, &maxLoc, 0 );
  CvSize imsize= {mat->rows,mat->cols};
  IplImage* tt=cvCreateImage(imsize,IPL_DEPTH_32F,0);
  tt = cvGetImage(mat,  tt);
  IplImage* ipl=cvCreateImage(imsize ,8,1);
  cvConvertScale(tt,ipl,255.0/(maxVal-minVal),-minVal*255.0/(maxVal-minVal));
  cvSaveImage(name,ipl);
  cvReleaseImage(&ipl);
  cvReleaseImage(&tt);
  return 1;
}

double MeanVector(double* v, int vSize)
{
  int i;
  double mean=0.0;
  double* ptr = v;

  for(i=0; i<vSize; i++)
    {

      mean+= *ptr;
      ptr++;
    }
  mean /=(double)vSize;
  return mean;
}

void ZeroMeanUnitLength( double* v, int vSize)
{

  double sqsum = 0.0;
  double mean = MeanVector(v,vSize);
  double* ptr = v;
  int i;

  for(i=0; i<vSize; i++)
    {

      (*ptr)  -= mean;
      sqsum += (*ptr)*(*ptr);
      ptr++;
    }
  double a = 1.0f/(double)(sqrt(sqsum));
  ptr = v;
  for(i=0; i<vSize; i++)
    {

      (*ptr) *= a;
      ptr++;
    }
}

CvMat*  GetMat2( FILE *fh, bool isVector )
{
  FILE *Stream;				//Stream for output.
  int cIndex=0;				//Counter for elements in the matrix.
  double Dim[2];				//The dimensions of the loaded matrix.
  double* Elem;				//Pointer to the elements the matrix.

  assert( fh!=NULL );
  Stream = fh;

  if(fread((void*)Dim,sizeof(double),2,Stream) <2 )//Retrive the dimensions.
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  int rows,cols;
  if(!isVector)
    {
      rows = (int)Dim[0];//Save the dimensions of the matrix.
      cols = (int)Dim[1];
    }
  else
    {
      cols = (int)Dim[0];//Save the dimensions of the matrix.
      rows = (int)Dim[1];
    }

  Elem=(double*)malloc(rows*cols*2*sizeof(double));	//Allocate memory to retrive the data.
  if(Elem==NULL)
    {
      printf("Out of memory.");
    }

  if(fread((void*)Elem,sizeof(double),rows*cols*2,Stream) < (unsigned)(rows*cols*2) ) //Retrive the data.
    {
      printf("Problem reading from file.");
    }
  CvMat* m  = cvCreateMatHeader(rows,cols,CV_64FC2);
  cvCreateData( m);
  int cI,cJ;
  for(cI=0; cI<rows; cI++)			//Put the elements into the matrix.
    {
      for(cJ=0; cJ<cols; cJ++)
        {
	  ((double*)(m->data.ptr + m->step*cI))[cJ*2]=Elem[cIndex];
	  cIndex++;
	  ((double*)(m->data.ptr + m->step*cI))[cJ*2+1]=Elem[cIndex];
	  cIndex++;
        }
    }
  free(Elem);

  return m;//Free allocated memory.
}

CvMat*  GetMat( FILE *fh, bool isVector )
{

  FILE *Stream;				//Stream for output.
  int cIndex=0;				//Counter for elements in the matrix.
  double Dim[2];				//The dimensions of the loaded matrix.
  double* Elem;				//Pointer to the elements the matrix.

  assert( fh!=NULL );
  Stream = fh;

  if(fread((void*)Dim,sizeof(double),2,Stream) <2 )//Retrive the dimensions.
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  int rows,cols;
  if(!isVector)
    {
      rows = (int)Dim[0];//Save the dimensions of the matrix.
      cols = (int)Dim[1];
    }
  else
    {
      cols = (int)Dim[0];//Save the dimensions of the matrix.
      rows = (int)Dim[1];
    }
  Elem=(double*)malloc(rows*cols*sizeof(double));	//Allocate memory to retrive the data.
  if(Elem==NULL)
    {
      printf("Out of memory.");
    }

  if(fread((void*)Elem,sizeof(double),rows*cols,Stream) < (unsigned)(rows*cols) ) //Retrive the data.
    {
      printf("Problem reading from file.");
    }
  CvMat* m  = cvCreateMatHeader(rows,cols,CV_64FC1);
  cvCreateData( m);
  int cI,cJ;
  for(cI=0; cI<rows; cI++)			//Put the elements into the matrix.
    {
      for(cJ=0; cJ<cols; cJ++)
        {
	  CV_MAT_ELEM( *m,double,cI,cJ)=Elem[cIndex];
	  cIndex++;
        }
    }
  free(Elem);
  return m;//Free allocated memory.
}

int WriteMat(CvMat* m, FILE *fh, bool isVector )
{
  FILE *Stream;					//Stream for output.
  int cIndex=2;					//Counter for elements in the matrix.
  int cSize=m->rows *m->cols+2;	//The number of elements to save.
  double*Elem;					//Pointer to all the element to save.

  assert( fh!=NULL );
  Stream = fh;

  Elem=(double*)malloc(cSize*sizeof(double));//Allocate memory to Elem.
  if(Elem==NULL)
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  if(!isVector)
    {
      Elem[0]=(double)m->rows;//Save the dimensions of the matrix.
      Elem[1]=(double)m->cols;
    }
  else
    {
      Elem[0]=1.0;//Save the dimensions of the matrix.
      Elem[1]=(double)m->rows;
    }
  int cI,cJ;
  for(cI=0; cI<m->rows ; cI++)	//Write the elements of the matrix to Elem.
    {
      for(cJ=0; cJ<m->cols; cJ++)
        {
	  Elem[cIndex]=CV_MAT_ELEM( *m,double,cI,cJ);
	  cIndex++;
        }
    }
  if(fwrite((void*)Elem,sizeof(double),cSize,Stream) < (unsigned)cSize) //Save the data.
    {
      printf("Error, can't save the matrix!");
      return(0);
    }
  free(Elem);	//Free memory.
  return(1);
}

int WriteMat2(CvMat* m, FILE *fh, bool isVector )
{
  FILE *Stream;					//Stream for output.
  int cIndex=2;					//Counter for elements in the matrix.
  int cSize=m->rows *m->cols*2+2;	//The number of elements to save.
  double*Elem;					//Pointer to all the element to save.

  assert( fh!=NULL );
  Stream = fh;

  Elem=(double*)malloc(cSize*sizeof(double));//Allocate memory to Elem.
  if(Elem==NULL)
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  if(!isVector)
    {
      Elem[0]=(double)m->rows;//Save the dimensions of the matrix.
      Elem[1]=(double)m->cols;
    }
  else
    {
      Elem[0]=1.0;//Save the dimensions of the matrix.
      Elem[1]=(double)m->rows;
    }
  int cI,cJ;
  for(cI=0; cI<m->rows ; cI++)	//Write the elements of the matrix to Elem.
    {
      for(cJ=0; cJ<m->cols; cJ++)
        {
	  Elem[cIndex]=((double*)(m->data.ptr + m->step*cI))[cJ*2];
	  cIndex++;
	  Elem[cIndex]=((double*)(m->data.ptr + m->step*cI))[cJ*2+1];
	  cIndex++;
        }
    }
  if(fwrite((void*)Elem,sizeof(double),cSize,Stream) < (unsigned)cSize) //Save the data.
    {
      printf("Error, can't save the matrix!");
      return(0);
    }
  free(Elem);	//Free memory.
  return(1);
}

CvMat** LoadGaborFFT(char* fldname)
{
  int i,j;
  char filename[512];
  //gabor_filter("gabor");
  CvMat** mGabor=(CvMat**)malloc(40*sizeof(CvMat));

  for(i=0; i<5; i++)
    for(j=0; j<8; j++)
      {
	sprintf(filename,"%s/FFT%s_%d_%d.data",fldname,"gabor",i,j);
	FILE *ifp = fopen(filename, "rb");
	mGabor[i*8+j]	= GetMat(ifp,false);
	fclose(ifp);
      }
  return(mGabor);

}

void UnloadGaborFFT(CvMat** mGabor)
{
  int i,j;
  for(i=0; i<5; i++)
    for(j=0; j<8; j++)
      cvReleaseMat(&mGabor[i*8+j]);

  cvReleaseMat(mGabor);

}

int Mulfft3( const CvArr* srcAarr, const CvArr* srcBarr, CvArr* dstarr )
{
  CvMat *srcA = (CvMat*)srcAarr;
  CvMat *srcB = (CvMat*)srcBarr;
  CvMat *dst = (CvMat*)dstarr;

  int i,j, rows, cols;
  rows = srcA->rows;
  cols = srcA->cols;
  double c_re,c_im;

  for( i=0; i<rows; i++ )
    {
      for( j = 0; j < cols; j ++ )
        {
//  cout<<(i*2)<< " " <<(j*2)<<endl;
	  c_re = ((double*)(srcA->data.ptr + srcA->step*i))[j*2]*((double*)(srcB->data.ptr + srcB->step*i))[j*2] -
	    ((double*)(srcA->data.ptr + srcA->step*i))[j*2+1]*((double*)(srcB->data.ptr + srcB->step*i))[j*2+1];
	  c_im = ((double*)(srcA->data.ptr + srcA->step*i))[j*2]*((double*)(srcB->data.ptr + srcB->step*i))[j*2+1] +
	    ((double*)(srcA->data.ptr + srcA->step*i))[j*2+1]*((double*)(srcB->data.ptr + srcB->step*i))[j*2];

	  ((double*)(dst->data.ptr + dst->step*i))[j*2]	= c_re;
	  ((double*)(dst->data.ptr + dst->step*i))[j*2+1]	= c_im;

        }
    }

  return 1;
}

int writeData(double* v, int length, FILE* fh)
{
  FILE *Stream;					//Stream for output.
  double*Elem;					//Pointer to all the element to save.

  assert( fh!=NULL );
  Stream = fh;

  Elem=(double*)malloc((length+2)*sizeof(double));//Allocate memory to Elem.
  if(Elem==NULL)
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  Elem[0]=(double)1;//Save the dimensions of the matrix.
  Elem[1]=(double)length;

  int i;
  for(i=0; i<length ; i++)
    {
      Elem[i+2]=v[i];
      //cout << v[i] << endl;
    }


  if(fwrite((void*)Elem,sizeof(double),(length+2),Stream) < (unsigned)(length+2)) //Save the data.
    {
      printf("Error, can't save the matrix!");
      return(0);
    }
  free(Elem);
  return(1);
}

double* getData(FILE* fh, int &length)
{
  FILE *Stream;					//Stream for output.
  double*Elem;					//Pointer to all the element to save.
  double Dim[2];				//The dimensions of the loaded matrix.

  assert( fh!=NULL );
  Stream = fh;

  if(fread((void*)Dim,sizeof(double),2,Stream) <2 )//Retrive the dimensions.
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }

  length=(int)Dim[1];

  Elem=(double*)malloc(length*sizeof(double));//Allocate memory to Elem.
  if(Elem==NULL)
    {
      printf("Can't allocate memory for saving the matrix!");
      return(0);
    }


  if(fread((void*)Elem,sizeof(double),length,Stream) < (unsigned)length) //Save the data.
    {
      printf("Error, can't save the matrix!");
      return(0);
    }
  return Elem;
}

int gabor_extraction(IplImage* img,double* object,CvMat** mGabor)
{
  int w,h;
  w=128;
  h=128;
  CvSize img_size = cvGetSize( img );
  IplImage* imtmp=cvCreateImage(img_size,IPL_DEPTH_64F,0);

  cvConvertScale(img,imtmp,1.0,0);

  int i,j,x,y,n;

  int dft_M = cvGetOptimalDFTSize( w+h- 1 );
  int dft_N = cvGetOptimalDFTSize( w+h- 1);
  
  CvMat* imdft=cvCreateMat(dft_M, dft_N, CV_64FC1 );
  cvZero(imdft);
  for(i=0; i<h; i++)
    for(j=0; j<w; j++)
      ((double*)(imdft->data.ptr + (imdft->step)*i))[j] = ((double*)(imtmp->imageData + imtmp->widthStep*i))[j];

  cvDFT( imdft, imdft, CV_DXT_FORWARD, w );
  n=w*h/64;


  for (i=0; i<5; i++)
    {
      for (j=0; j<8; j++)
        {
	  CvMat* gout=cvCreateMatHeader(dft_M, dft_N, CV_64FC1 );
	  cvCreateData( gout);

	  cvMulSpectrums( imdft,mGabor[i*8+j], gout, 0);
	  cvDFT( gout, gout, CV_DXT_INVERSE,w+h-1);

	  /*downsample sacle factor 64*/
	  for (x=4; x<w; x+=8)
	    for (y=4; y<h; y+=8)
	      {
		double sum=((double*)(gout->data.ptr + gout->step*(x+h/2)))[(y+w/2)*2]*
		  ((double*)(gout->data.ptr + gout->step*(x+h/2)))[(y+w/2)*2]+
		  ((double*)(gout->data.ptr + gout->step*(x+h/2)))[(y+w/2)*2+1]*
		  ((double*)(gout->data.ptr + gout->step*(x+h/2)))[(y+w/2)*2+1];

		object[(i*8+j)*n+x/8*h/8+y/8]=sqrt(sum);
	      }
	  cvReleaseMat(&gout);
	  ZeroMeanUnitLength(object,n);
        }
    }

  cvReleaseImage(&imtmp);
  cvReleaseMat(&imdft);
  
  return(1);
}

void extractGaborFeatures(const IplImage* img, Mat& gb)
{
  unsigned long long nsize = NUM_GABOR_FEATURES;
  CvSize size = cvSize(128,128);
  CvSize img_size = cvGetSize( img );
  IplImage*	ipl=	cvCreateImage(img_size,8,0);
  if(img->nChannels==3)
    {
      cvCvtColor(img,ipl,CV_BGR2GRAY);
    }
  else
    {
      cvCopy(img,ipl,0);
    }

  gb.release();
  gb = Mat::zeros(1, NUM_GABOR_FEATURES, CV_32FC1);
  if((size.width!=img_size.width)||(size.height!=img_size.height))
    {
      IplImage* tmpsize=cvCreateImage(size,8,0);
      cvResize(ipl,tmpsize,CV_INTER_LINEAR);
      cvReleaseImage( &ipl);
      ipl=cvCreateImage(size,8,0);
      cvCopy(tmpsize,ipl,0);
      cvReleaseImage( &tmpsize);
    }
    
  double* object=(double*)malloc(nsize*sizeof(double));
  IplImage* tmp=cvCreateImage(size,IPL_DEPTH_64F,0);

  cvConvertScale(ipl,tmp,1.0,0);
  CvMat** mGabor= LoadGaborFFT("gabor");
  /*Gabor wavelet*/
  gabor_extraction(tmp,object,mGabor);
  ZeroMeanUnitLength(object,nsize);
  cvReleaseImage( &tmp);
  cvReleaseImage( &ipl);

  UnloadGaborFFT(mGabor);
  int actualSize = 0;

  for(int i=0;i<NUM_GABOR_FEATURES;i++) {
    gb.at<float>(0, actualSize++) = static_cast<float>(object[i]);
  }
}

int gabor_kernel(int scale,int orientation,int mask_size,double kmax,double sigma,char* filename)
{

  FILE *cout, *sout;
  register int u,v,x,y;

  char coutfilename[512],soutfilename[512];

  double* gabor_cos = (double *)malloc (mask_size*mask_size*sizeof (double));
  double* gabor_sin = (double *)malloc (mask_size*mask_size*sizeof (double));
  int offset = mask_size / 2;

  double f=sqrt(2.0);
  double sig=sigma*sigma;
  for (v=0; v<scale; v++)
    for (u=0; u<orientation; u++)
      {

	double kv=kmax/pow(f,v);
	double phiu=u*M_PI/8.0;
	double kv_mag=kv*kv;

	for (x = 0; x < mask_size; x++)
	  for (y = 0; y< mask_size; y++)
	    {
	      int i=x-offset;
	      int j=y-offset;
	      double mag=(double)(i*i+j*j);
	      gabor_cos[x*mask_size+y] = kv_mag/sig*exp(-0.5*kv_mag*mag/sig)*
		(cos(kv*(i*cos(phiu)+j*sin(phiu)))-exp(-1.0*sig/2.0));
	      gabor_sin[x*mask_size+y] = kv_mag/sig*exp(-0.5*kv_mag*mag/sig)*
		(sin(kv*(i*cos(phiu)+j*sin(phiu))));//-exp(-1.0*sig/2.0)
	    }

	/* Let's deal with the cosine part first */

	sprintf(coutfilename,"gabor//C%s_%d_%d.data",filename,v,u);
	sprintf(soutfilename,"gabor//S%s_%d_%d.data",filename,v,u);

	if ((cout = fopen(coutfilename, "wb")) == NULL)
	  {
	    printf("Cannot open file %s!\n",coutfilename);
	    return(0);
	  }

	CvMat cmat = cvMat( mask_size, mask_size,CV_64FC1,gabor_cos );
	WriteMat(&cmat,cout,false);
	fclose(cout);
	sprintf(coutfilename,"gabor//C%s_%d_%d.bmp",filename,v,u);
	save_mat_image(&cmat,coutfilename);
	/* Now let's deal with the sine part */
	if ((sout = fopen(soutfilename, "wb")) == NULL)
	  {
	    printf("Cannot open file %s!\n",soutfilename);
	    return(0);
	  }

	CvMat smat = cvMat( mask_size, mask_size,CV_64FC1,gabor_sin );
	WriteMat(&smat,sout,false);
	sprintf(soutfilename,"gabor//S%s_%d_%d.bmp",filename,v,u);
	save_mat_image(&smat,soutfilename);
	fclose(sout);

      }
  free(gabor_cos);
  free(gabor_sin);
  return 1;
}

int gabor_filter(char* base_fld)
{
  int img_size=128;
  int i,j,x,y;
  char coutfilename[512];
  fprintf(stdout,"Building gabor filter bank data: ");
  gabor_kernel(5,8,img_size,M_PI/2.0,2*M_PI, "gabor");
  fprintf(stdout," OK!\n");
  fprintf(stdout,"Calculating Gabor filter bank FFT: ");

  int dft_M = cvGetOptimalDFTSize( img_size + img_size - 1 );
  int dft_N = cvGetOptimalDFTSize( img_size + img_size - 1 );
  for (i=0; i<5; i++)
    {
      for (j=0; j<8; j++)
        {
	  sprintf(coutfilename,"%s\\C%s_%d_%d.data",base_fld,"gabor",i,j);
	  FILE *ifp = fopen(coutfilename, "rb");
	  CvMat* mGaborReal	= GetMat(ifp,false);
	  fclose(ifp);
	  sprintf(coutfilename,"%s\\S%s_%d_%d.data",base_fld,"gabor",i,j);
	  FILE *ifps = fopen(coutfilename, "rb");
	  CvMat* mGaborImg	= GetMat(ifps,false);
	  fclose(ifps);

	  CvMat* mGabor=cvCreateMatHeader( dft_M, dft_N, CV_64FC1 );
	  cvCreateData( mGabor);

	  CvMat* mGaborFFT=cvCreateMatHeader( dft_M, dft_N, CV_64FC1);
	  cvCreateData( mGaborFFT);

	  cvZero(mGabor);
	  for(x=0; x<img_size; x++)
	    for(y=0; y<img_size; y++)
	      {
		((double*)(mGabor->data.ptr + mGabor->step*x))[y]=
		  CV_MAT_ELEM(*mGaborReal,double,x,y);
	      }

	  cvDFT(mGabor,mGaborFFT,CV_DXT_FORWARD,img_size);

	  sprintf(coutfilename,"%s\\FFT%s_%d_%d.data",base_fld,"gabor",i,j);
	  FILE *ofps = fopen(coutfilename, "wb");
	  WriteMat(mGaborFFT,ofps,false);
	  fclose(ofps);

	  cvReleaseMat(&mGaborFFT);

	  cvReleaseMat(&mGabor);
	  cvReleaseMat(&mGaborReal);
	  cvReleaseMat(&mGaborImg);
        }
    }
  fprintf(stdout," OK!\n");
  return 1;
}
