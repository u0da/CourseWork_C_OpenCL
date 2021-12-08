#include "clFFT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h> 
#include "2.12.2/include/clFFT.h"

/////////////////////////////////////
// OpenCL FFT 2D function ///////////
/////////////////////////////////////

int FFT_2D_OpenCL(float *tab[], clfftDirection direction, int sizex, int sizey, cl_device_id target_device)
{

 // Index
 int i;

 // OpenCL variables
 cl_int err;
 cl_context ctx = 0;
 cl_command_queue queue = 0;

 // Input and Output buffer
 cl_mem buffersIn[2]  = {0, 0};
 cl_mem buffersOut[2] = {0, 0};

 // Temporary buffer
 cl_mem tmpBuffer = 0;

 // Size of temp buffer
 size_t tmpBufferSize = 0;
 int status = 0;
 int ret = 0;

 // Total size of FFT
 size_t N = sizex*sizey;

 // FFT library realted declarations
 clfftPlanHandle planHandle;
 clfftDim dim = CLFFT_2D;
 size_t clLengths[2] = {sizex, sizey};
 
 // Create an OpenCL context
 ctx = clCreateContext(NULL, 1, &target_device, NULL, NULL, &err);

 // Create a command queue
 queue = clCreateCommandQueue(ctx, target_device, 0, &err);
//  queue = clCreateCommandQueueWithProperties(ctx, device, 0, &err);

 // Setup clFFT
 clfftSetupData fftSetup;
 err = clfftInitSetupData(&fftSetup);
 err = clfftSetup(&fftSetup);

 // Create a default plan for a complex FFT
 err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

 // Set plan parameters
 err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);
 err = clfftSetLayout(planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
 err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);

 // Bake the plan
 err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);

 // Real and Imaginary arrays
 cl_float* inReal  = (cl_float*) malloc (N * sizeof (cl_float));
 cl_float* inImag  = (cl_float*) malloc (N * sizeof (cl_float));
 cl_float* outReal = (cl_float*) malloc (N * sizeof (cl_float));
 cl_float* outImag = (cl_float*) malloc (N * sizeof (cl_float));

 // Initialization of inReal, inImag, outReal and outImag
 for(i = 0; i < N; i++)
 {
  inReal[i]  = tab[0][i];
  inImag[i]  = tab[1][i];
  outReal[i] = 0.0f;
  outImag[i] = 0.0f;
 }

 // Create temporary buffer
 status = clfftGetTmpBufSize(planHandle, &tmpBufferSize);

 if ((status == 0) && (tmpBufferSize > 0))
 {
  tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
  if (err != CL_SUCCESS)
   printf("Error with tmpBuffer clCreateBuffer\n");
 }

 // Prepare OpenCL memory objects : create buffer for input
 buffersIn[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), inReal, &err);
 if (err != CL_SUCCESS)
  printf("Error with buffersIn[0] clCreateBuffer\n");

 // Enqueue write tab array into buffersIn[0]
 err = clEnqueueWriteBuffer(queue, buffersIn[0], CL_TRUE, 0, N * sizeof(float), inReal, 0, NULL, NULL);
 if (err != CL_SUCCESS)
  printf("Error with buffersIn[0] clEnqueueWriteBuffer\n");

 // Prepare OpenCL memory objects : create buffer for input
 buffersIn[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), inImag, &err);
 if (err != CL_SUCCESS)
  printf("Error with buffersIn[1] clCreateBuffer\n");

 // Enqueue write tab array into buffersIn[1]
 err = clEnqueueWriteBuffer(queue, buffersIn[1], CL_TRUE, 0, N * sizeof(float), inImag, 0, NULL, NULL);
 if (err != CL_SUCCESS)
  printf("Error with buffersIn[1] clEnqueueWriteBuffer\n");

 // Prepare OpenCL memory objects : create buffer for output
 buffersOut[0] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), outReal, &err);
 if (err != CL_SUCCESS)
  printf("Error with buffersOut[0] clCreateBuffer\n");

 buffersOut[1] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(cl_float), outImag, &err);
 if (err != CL_SUCCESS)
  printf("Error with buffersOut[1] clCreateBuffer\n");

 err = clfftEnqueueTransform(planHandle, direction, 1, &queue, 0, NULL, NULL, buffersIn, buffersOut, tmpBuffer);

 // Wait for calculations to be finished
 err = clFinish(queue);

 // Fetch results of calculations : Real and Imaginary
 err = clEnqueueReadBuffer(queue, buffersOut[0], CL_TRUE, 0, N * sizeof(float), tab[0], 0, NULL, NULL);
 err = clEnqueueReadBuffer(queue, buffersOut[1], CL_TRUE, 0, N * sizeof(float), tab[1], 0, NULL, NULL);

 // Release OpenCL memory objects
 clReleaseMemObject(buffersIn[0]);
 clReleaseMemObject(buffersIn[1]);
 clReleaseMemObject(buffersOut[0]);
 clReleaseMemObject(buffersOut[1]);
 clReleaseMemObject(tmpBuffer);

 // Release the plan
 err = clfftDestroyPlan(&planHandle);

 // Release clFFT library
 clfftTeardown();

 // Release OpenCL working objects
 clReleaseCommandQueue(queue);
 clReleaseContext(ctx);

 return ret;
}

void FFT_Shift(float *tab[], int sizex, int sizey)
{
  int half_sizex =  sizex/2;
  int half_sizey =  sizey/2;

  for(int k = 0; k < 2; k++)
  {
    float *tab_k = tab[k];
    for (int i = 0; i < sizex; i++)
    {
      int row_start_index = i*sizey;
      for (int j = 0; j < half_sizey; j++)
      {
        float tmp = tab_k[row_start_index+j]; 
        tab_k[row_start_index+j] =  tab_k[row_start_index+half_sizey+j];
        tab_k[row_start_index+half_sizey+j] = tmp;
      }
    }


    for (int j = 0; j < sizey; j++)
    {
      // int col_start_index = j*sizex;
      for (int i = 0; i < half_sizex; i++)
      {
        float tmp = tab_k[i*sizey+j]; 
        tab_k[i*sizey+j] =  tab_k[(i+half_sizex)*sizey+j];
        tab_k[(i+half_sizex)*sizey+j] = tmp;
      }
    }
  }
}



int M (float x, float y)
{
    if((pow(x, 2) + pow(y, 2)) < pow(M_PI/2, 2))
        return (1);
    else
        return (0);
}

float p_s (float x, float y, float delta_z)
{
  // float d_1 = 57.4f * 10e-3;
  // float d_0 = 37.0f * 10e-3;
  // float r_0 = 4.5f * 10e-3;
  // float lamba = 0.55f * 10e-6;
  // float w = 2.34f * 10e-5;
  
    return (0.375 * delta_z * M_PI * (pow(x, 2) + pow(y, 2)));
}

float p (float x, float y)
{
    return (0.5 * M_PI * (pow(x, 2) + pow(y, 2)));
}

int main(void)
{
 cl_int err;
 cl_platform_id platform = 0;
 cl_uint number_of_devices = 0;

// Setup OpenCL environment
 err = clGetPlatformIDs(1, &platform, NULL);

 // получение "устройства", котором будет выполнятся вычисление
 err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU , 0, NULL, &number_of_devices); 
 
 printf("Кол-во устройств: %u\n", number_of_devices);
 printf("Список устройств:\n");
 
 cl_device_id *devices  = (cl_device_id*) malloc (number_of_devices * sizeof(cl_device_id));
 err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU , number_of_devices, devices, NULL);
 for (int i = 0; i < number_of_devices; i++)
 {
   char name[128] = {'\0'};
   err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
   printf("\t\t[%d]%s\n", i, name); 
 }

// выбираем нужный девайс

cl_device_id device = devices[1];

float delta_z =  0 * M_PI;
float x = 0, y = 0;
    
 // Signal array and FFT output array
float *Array[2];
float *h[2];
float *result[2];

 // Number of sampling points
int sizex = 1024;
int sizey = 1024;

 // Total size of FFT
int N = sizex*sizey;

 // Allocation of Array
Array[0] = (float*) calloc(N,sizeof(float));
Array[1] = (float*) calloc(N,sizeof(float));
result[0]= (float*) calloc(N,sizeof(float));
result[1]= (float*) calloc(N,sizeof(float));
h[0]     = (float*) calloc(512*512,sizeof(float));
h[1]     = (float*) calloc(512*512,sizeof(float));
//=================================================================

FILE *input_pic;
input_pic = fopen("image512-1024.txt", "r"); // уже получили матрицу в нужном формате, она расширина

// если сгенерированный файл 1024х1024 (те изначальное изображение 512х512)
printf("Считываю картинку из файла\n");
for(int i = 0; i < N; i++)
{
  fscanf(input_pic, "%f ", &Array[0][i]);
}
fclose(input_pic);
printf("Положил картинку в Array[0]\n");


// ФОРМИРУЕМ МАТРИЦУ h
for(int i = 0; i < sizex/2; i++)
{
   for(int j = 0; j < sizey/2; j++)
   {
       float complex tmp;
       x = (M_PI / 512) * (i - 255);
       y = (M_PI / 512) * (j - 255);
       tmp = M(x, y) * cexp((I * p_s(x, y, delta_z)) + (I * p(x, y)));
       h[0][i*sizey/2+j] = creal(tmp);
       h[1][i*sizey/2+j] = cimag(tmp);
   }
}

//  Perform Forward FFT для H
if (FFT_2D_OpenCL(h, CLFFT_FORWARD, 512, 512, device) == 0)
  printf("FFT for h func passed !\n");


FFT_Shift(h, 512, 512);

for(int i = 0; i < 512*512; i++)
{
    h[0][i] = pow(cabs(h[0][i] + I * h[1][i]), 2);
    h[1][i] = 0.0f;
}

float *h_rash[2];
h_rash[0]     = (float*) calloc(N,sizeof(float));
h_rash[1]     = (float*) calloc(N,sizeof(float));

for(int i = 0; i < 512; i++)
{
  memcpy(&(h_rash[0][i*sizey]), &(h[0][i*512]), 512*sizeof(float));
}

FILE *h_rash_out;
h_rash_out = fopen("h_rash_out.txt", "wb");
for (int i = 0; i < N; i++)
{
  if((i % 1024 == 0 ) && (i != 0) )
  {
    putc('\n', h_rash_out);
  }
  fprintf(h_rash_out, "%f ", h_rash[0][i]); 
}
fclose(h_rash_out);

if (FFT_2D_OpenCL(Array, CLFFT_FORWARD, sizex, sizey, device) == 0)
  printf("FFT for Image passed !\n");

FFT_Shift(Array, sizex, sizey);


if (FFT_2D_OpenCL(h_rash, CLFFT_FORWARD, sizex, sizey, device) == 0)
  printf("FFT for h func passed !\n");
FFT_Shift(h_rash, sizex, sizey);


for (int i = 0; i < N; i++)
{
    result[0][i] = Array[0][i] * h_rash[0][i] - Array[1][i] * h_rash[1][i];
    result[1][i] = Array[0][i] * h_rash[1][i] + Array[1][i] * h_rash[0][i];
}

if (FFT_2D_OpenCL(result, CLFFT_BACKWARD, sizex, sizey, device) == 0)
  printf("IFFT for result passed !\n");

for(int i = 0; i < 1024*1024; i++)
{
    result[0][i] = cabs(result[0][i] + I * result[1][i]);
    result[1][i] = 0.0f;
}

FILE *fp;
  fp = fopen("result.txt", "wb");
  for (int i = 0; i < N; i++)
  {
    if((i % sizey == 0 ) && (i != 0) )
    {
      putc('\n', fp);
    }
    fprintf(fp, "%f ", result[0][i]); 
  }

  fclose(fp);

free(h_rash[0]);
free(h_rash[1]);
free(Array[0]);
free(Array[1]);
free(result[0]);
free(result[1]);
free(h[0]);
free(h[1]);  

return 0;
}
