#include "clFFT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>

#define MAX_SOURCE_SIZE (0x100000)

/////////////////////////////////////
// OpenCL FFT 2D function ///////////
/////////////////////////////////////

struct Cl_Buffer_pair
{
    cl_mem buffers[2];
};

cl_int InitCl_Buffer_pair(cl_context ctx, cl_command_queue queue, cl_bitfield mode, int N , struct  Cl_Buffer_pair *pair)
{
    cl_int err = CL_SUCCESS;
    memset(pair, 0, sizeof(*pair)); // побайтовое обнуление всей структуры pair

    const float zero = 0;
    for (int i = 0; i < 2; i++)
    {
        pair->buffers[i] = clCreateBuffer(ctx, mode, N * sizeof(cl_float), NULL, &err);
        if (err != CL_SUCCESS) {
            printf("InitCl_Buffer_pair: Error with buffers[%d] clCreateBuffer\n", i);
            return err;
        }
        err = clEnqueueFillBuffer(queue, pair->buffers[i], &zero, sizeof(zero), 0, N * sizeof(float), 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            printf("InitCl_Buffer_pair: Error with buffers[%d] clEnqueueFillBuffer\n", i);
            return err;
        }

    }
    clFinish(queue);
    return  err;
}

void DeInItCl_Buffer_pair(struct Cl_Buffer_pair *pair)
{
    // Release OpenCL memory objects
    clReleaseMemObject(pair->buffers[0]);
    clReleaseMemObject(pair->buffers[1]);
    memset(pair, 0, sizeof(*pair)); // побайтовое обнуление всей структуры pair
}

struct FFT_OpenCL_data {
    int sizex;
    int sizey;

    struct Cl_Buffer_pair buffersOut;

    // Temporary buffer
    cl_mem tmpBuffer;

    // FFT library realted declarations
    clfftPlanHandle planHandle;
};

cl_int InitFFT_OpenCL_data(int sizex, int sizey, cl_context ctx, cl_command_queue queue, struct FFT_OpenCL_data *data) {
    cl_int err = CL_SUCCESS;
    memset(data, 0, sizeof(*data)); // побайтовое обнуление всей структуры data
    data->sizex = sizex;
    data->sizey = sizey;
    int N = sizex * sizey;
    size_t clLengths[2] = {sizex, sizey};
    // Size of temp buffer
    size_t tmpBufferSize = 0;

    // Create a default plan for a complex FFT
    err = clfftCreateDefaultPlan(&data->planHandle, ctx, CLFFT_2D, clLengths);
    if (err != CL_SUCCESS)
        return err;

    // Set plan parameters
    err = clfftSetPlanPrecision(data->planHandle, CLFFT_SINGLE);
    if (err != CL_SUCCESS)
        return err;
    err = clfftSetLayout(data->planHandle, CLFFT_COMPLEX_PLANAR, CLFFT_COMPLEX_PLANAR);
    if (err != CL_SUCCESS)
        return err;
    err = clfftSetResultLocation(data->planHandle, CLFFT_OUTOFPLACE);
    if (err != CL_SUCCESS)
        return err;

    // Bake the plan
    err = clfftBakePlan(data->planHandle, 1, &queue, NULL, NULL);
    if (err != CL_SUCCESS)
        return err;

    // Create temporary buffer
    err = clfftGetTmpBufSize(data->planHandle, &tmpBufferSize);
    if (err != CL_SUCCESS)
        return err;

    if (tmpBufferSize > 0)
    {
        data->tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, 0, &err);
        if (err != CL_SUCCESS) {
            printf("Error with tmpBuffer clCreateBuffer\n");
            return err;
        }
    }

    err = InitCl_Buffer_pair(ctx, queue, CL_MEM_WRITE_ONLY, N, &data->buffersOut);

    return err;
}

void DeInItFFT_OpenCL_data(struct FFT_OpenCL_data *data)
{
    // Release OpenCL memory objects
    DeInItCl_Buffer_pair(&data->buffersOut);
    clReleaseMemObject(data->tmpBuffer);
    clfftDestroyPlan(&data->planHandle);
    memset(data, 0, sizeof(*data)); // побайтовое обнуление всей структуры data
}

int FFT_2D_OpenCL(struct Cl_Buffer_pair input, clfftDirection direction, cl_command_queue queue, struct FFT_OpenCL_data *data) {
    cl_int err;
    size_t N = data->sizex * data->sizey;

    // заполнение буферов на GPU нулями
    const float zero = 0;
    for (int i = 0; i < 2; i++)
        err = clEnqueueFillBuffer(queue, data->buffersOut.buffers[i], &zero, sizeof(zero), 0, N * sizeof(float), 0, NULL, NULL);

    err = clfftEnqueueTransform(data->planHandle, direction, 1, &queue, 0, NULL, NULL,
                                input.buffers, data->buffersOut.buffers, data->tmpBuffer);

    // Wait for calculations to be finished
    err = clFinish(queue);

    return err;
}

int M(float x, float y) {
    if ((pow(x, 2) + pow(y, 2)) < pow(M_PI / 2, 2))
        return (1);
    else
        return (0);
}

float p_s(float x, float y, float delta_z) {
    // float d_1 = 57.4f * 10e-3;
    // float d_0 = 37.0f * 10e-3;
    // float r_0 = 4.5f * 10e-3;
    // float lamba = 0.55f * 10e-6;
    // float w = 2.34f * 10e-5;

    return (0.375 * delta_z * M_PI * (pow(x, 2) + pow(y, 2)));
}

float p(float x, float y) {
    return (0.5 * M_PI * (pow(x, 2) + pow(y, 2)));
}

int main(void) {

    cl_int err;
    cl_int ret;
    cl_platform_id platform = 0;
    cl_uint number_of_devices = 0;

    cl_context ctx = 0;
    cl_command_queue queue = 0;


// Setup OpenCL environment
    err = clGetPlatformIDs(1, &platform, NULL);

    // получение "устройства", котором будет выполнятся вычисление
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, 0, NULL, &number_of_devices);

    printf("Кол-во устройств: %u\n", number_of_devices);
    printf("Список устройств:\n");

    cl_device_id *devices = (cl_device_id *) malloc(number_of_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, number_of_devices, devices, NULL);
    for (int i = 0; i < number_of_devices; i++)
    {
        char name[128] = {'\0'};
        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        printf("\t\t[%d]%s\n", i, name);
    }

    printf("\n");
// выбираем нужный девайс
    cl_device_id device = devices[1];

// Create an OpenCL context
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

// Create a command queue
    queue = clCreateCommandQueue(ctx, device, 0, &err);
//  queue = clCreateCommandQueueWithProperties(ctx, device, 0, &err);

    // Setup clFFT
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    struct FFT_OpenCL_data fft_orig_size;
    struct FFT_OpenCL_data fft_rash_size;


    err = InitFFT_OpenCL_data(512, 512, ctx, queue, &fft_orig_size);
    err = InitFFT_OpenCL_data(1024, 1024, ctx, queue, &fft_rash_size);

    float delta_z = 0 * M_PI;
    float x = 0, y = 0;

    float *Array[2];
    float *h[2];
    float *result[2];

    // Number of sampling points
    int sizex = 1024;
    int sizey = 1024;
    int half_sizex = sizex / 2;
    int half_sizey = sizey / 2;

    // Total size of FFT
    int N = sizex * sizey;
    int half_N = half_sizex * half_sizey;

    // Allocation of arrays
    h[0]      = (float *) calloc(half_N, sizeof(float));
    h[1]      = (float *) calloc(half_N, sizeof(float));
    Array[0]  = (float *) calloc(N, sizeof(float));
    Array[1]  = (float *) calloc(N, sizeof(float));
    result[0] = (float *) calloc(N, sizeof(float));
    result[1] = (float *) calloc(N, sizeof(float));

//=================================================================
// КАРТИНКА
    FILE *input_pic;
    input_pic = fopen("image512-1024-3.txt", "r"); //уже получили матрицу в нужном формате, она расширина

// если сгенерированный файл 1024х1024 (те изначальное изображение 512х512)
    printf("Считываю картинку из файла\n");
    for (int i = 0; i < N; i++)
        fscanf(input_pic, "%f ", &Array[0][i]);

    fclose(input_pic);
    printf("Положил картинку в Array[0]\n");

    struct Cl_Buffer_pair Array_CL;
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N, &Array_CL);
    for (int i = 0; i < 2; i++)
    {
        // Кладем картинку (расшир размерности) в буфферы GPU
        err = clEnqueueWriteBuffer(queue, Array_CL.buffers[i], CL_TRUE, 0,
                                   fft_rash_size.sizex * fft_rash_size.sizey * sizeof(float), Array[i], 0, NULL, NULL);
        if (err != CL_SUCCESS)
            printf("Error with Array_CL.buffers[%d] clEnqueueWriteBuffer\n", i);
    }

    if (FFT_2D_OpenCL(Array_CL, CLFFT_FORWARD, queue, &fft_rash_size) == 0)
        printf("FFT for Image passed !\n");


    for (int i = 0; i < 2; i++)
    {
        err = clEnqueueCopyBuffer(queue, fft_rash_size.buffersOut.buffers[i], Array_CL.buffers[i],  0, 0, N * sizeof(cl_float), 0, NULL, NULL);
        if (err != CL_SUCCESS)
            printf("Error with clEnqueueCopyBuffer \n");
    }
// КАРТИНКА

// clEnqueueNDRangeKernel
// вызов kernel для модуля комплексного числа

    FILE *rash_kernel;
    char *source_str;
    size_t source_size;

    rash_kernel = fopen("rash_kernel.cl", "r");
    if (!rash_kernel)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, rash_kernel);
    fclose(rash_kernel);
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&source_str, NULL, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(ret != CL_SUCCESS)
    {
        printf("Problems w/ building program\n");
        size_t source_size_log;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0 , NULL, &source_size_log);
        char *log= malloc(sizeof(char) * source_size_log);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, source_size_log, log, &source_size_log);
        printf("%s\n", log);
        free(log);
        clReleaseProgram(program);
        clfftTeardown();
        exit(1);
    }
    free(source_str);
    source_str = NULL;

    // Execute the OpenCL kernel on the list
    size_t global_item_size = N; // Process the entire lists
//    size_t local_item_size = 64;


    struct Cl_Buffer_pair h_CL;
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, half_N, &h_CL);

    cl_kernel h_init_kernel = clCreateKernel(program, "h_init_kernel", &ret);
    ret = clSetKernelArg(h_init_kernel, 0, sizeof(delta_z), &delta_z);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for delta_z h_init_kernel\n");
    ret = clSetKernelArg(h_init_kernel, 1, sizeof(cl_mem), &h_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[0] h_init_kernel\n");
    ret = clSetKernelArg(h_init_kernel, 2, sizeof(cl_mem), &h_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[1] h_init_kernel\n");

    size_t global_group_size[] = {half_sizex, half_sizey};
    ret = clEnqueueNDRangeKernel(queue, h_init_kernel, 2, NULL, global_group_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clEnqueueNDRangeKernel h_init_kernel");
    ret = clFinish(queue);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clFinish");

    //  Perform Forward FFT для H
    if (FFT_2D_OpenCL(h_CL, CLFFT_FORWARD, queue, &fft_orig_size) == 0)
        printf("FFT for h func passed !\n");


    for (int i = 0; i < 2; i++)
    {
        err = clEnqueueCopyBuffer(queue, fft_orig_size.buffersOut.buffers[i], h_CL.buffers[i], 0, 0, half_N * sizeof(cl_float), 0, NULL, NULL);
        if (err != CL_SUCCESS)
            printf("Error with clEnqueueCopyBuffer \n");
    }

    cl_kernel h_squared_abs_kernel = clCreateKernel(program, "h_squared_abs_kernel", &ret);

    ret = clSetKernelArg(h_squared_abs_kernel, 0, sizeof(cl_mem), &h_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[0] h_squared_abs_kernel\n");
    ret = clSetKernelArg(h_squared_abs_kernel, 1, sizeof(cl_mem), &h_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[1] h_squared_abs_kernel\n");

    ret = clEnqueueNDRangeKernel(queue, h_squared_abs_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clEnqueueNDRangeKernel h_squared_abs_kernel");
    clFinish(queue);


    cl_kernel fft_shift_row_kernel = clCreateKernel(program, "fft_shift_row_kernel", &ret);
    cl_kernel fft_shift_col_kernel = clCreateKernel(program, "fft_shift_col_kernel", &ret);

    for (int i = 0; i < 2; i++)
    {
        ret = clSetKernelArg(fft_shift_row_kernel, 0, sizeof(cl_mem), &h_CL.buffers[i]);
        if(ret != CL_SUCCESS)
            printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);
        ret = clSetKernelArg(fft_shift_row_kernel, 1, sizeof(half_sizex), &half_sizex);
        if(ret != CL_SUCCESS)
            printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);

        size_t sizey_t = half_sizey;
        ret = clEnqueueNDRangeKernel(queue, fft_shift_row_kernel, 1, NULL, &sizey_t, NULL, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clEnqueueNDRangeKernel fft_shift_row_kernel");



        ret = clSetKernelArg(fft_shift_col_kernel, 0, sizeof(cl_mem), &h_CL.buffers[i]);
        if(ret != CL_SUCCESS)
            printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);
        ret = clSetKernelArg(fft_shift_col_kernel, 1, sizeof(half_sizey), &half_sizey);
        if(ret != CL_SUCCESS)
            printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);


        size_t sizex_t = half_sizex;
        ret = clEnqueueNDRangeKernel(queue, fft_shift_col_kernel, 1, NULL, &sizex_t, NULL, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clEnqueueNDRangeKernel fft_shift_col_kernel");
        clFinish(queue);
    }

    struct Cl_Buffer_pair h_rash_CL;
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N, &h_rash_CL);

    for (int i = 0; i < 2; i++)
        for(int j = 0; j < half_sizey; j++)
        {
            err = clEnqueueCopyBuffer(queue, h_CL.buffers[i], h_rash_CL.buffers[i], j * half_sizex * sizeof(cl_float), j * sizex * sizeof (cl_float), half_sizex * sizeof(cl_float), 0, NULL, NULL);
            if (err != CL_SUCCESS)
                printf("Error with clEnqueueCopyBuffer %d\n", j);

        }

    if (FFT_2D_OpenCL(h_rash_CL, CLFFT_FORWARD, queue, &fft_rash_size) == 0)
        printf("FFT for h_rash func passed !\n");


    // &fft_rash_size.buffersOut.buffers[0] = h_rash_CL after FFT
    cl_kernel multiply_kernel = clCreateKernel(program, "multiply_kernel", &ret);

    ret = clSetKernelArg(multiply_kernel, 0, sizeof(cl_mem), &Array_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for Array[0] multiply\n");
    ret = clSetKernelArg(multiply_kernel, 1, sizeof(cl_mem), &Array_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for Array[1] multiply\n");
    ret = clSetKernelArg(multiply_kernel, 2, sizeof(cl_mem), &fft_rash_size.buffersOut.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[0] multiply\n");
    ret = clSetKernelArg(multiply_kernel, 3, sizeof(cl_mem), &fft_rash_size.buffersOut.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[1] multiply\n");

    struct Cl_Buffer_pair result_CL;
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N, &result_CL);
    ret = clSetKernelArg(multiply_kernel, 4, sizeof(cl_mem), &result_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[0] multiply\n");
    ret = clSetKernelArg(multiply_kernel, 5, sizeof(cl_mem), &result_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[1] multiply\n");

    ret = clEnqueueNDRangeKernel(queue, multiply_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clEnqueueNDRangeKernel multiply");
    clFinish(queue);

    if (FFT_2D_OpenCL(result_CL, CLFFT_BACKWARD, queue, &fft_rash_size) == 0)
        printf("IFFT for result passed !\n");

    // Create the OpenCL kernel
    cl_kernel abs_rash_kernel = clCreateKernel(program, "abs_rash_kernel", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(abs_rash_kernel, 0, sizeof(cl_mem), &fft_rash_size.buffersOut.buffers[0]);
            if(ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for result[0] abs\n");
    ret = clSetKernelArg(abs_rash_kernel, 1, sizeof(cl_mem), &fft_rash_size.buffersOut.buffers[1]);
            if(ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for result[1] abs\n");

    ret = clEnqueueNDRangeKernel(queue, abs_rash_kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clEnqueueNDRangeKernel abs");
    clFinish(queue);


    ret = clEnqueueReadBuffer(queue, fft_rash_size.buffersOut.buffers[0], CL_TRUE, 0,
                              N * sizeof(float), result[0], 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clEnqueueReadBuffer");

    DeInItCl_Buffer_pair(&Array_CL);
    DeInItCl_Buffer_pair(&h_CL);
    DeInItCl_Buffer_pair(&h_rash_CL);
    DeInItCl_Buffer_pair(&result_CL);

    FILE *fp;
    fp = fopen("result.txt", "wb");
    for (int i = 0; i < N; i++)
    {
        if ((i % sizey == 0) && (i != 0))
            putc('\n', fp);
        fprintf(fp, "%f ", result[0][i]);
    }
    fclose(fp);

    DeInItFFT_OpenCL_data(&fft_orig_size);
    DeInItFFT_OpenCL_data(&fft_rash_size);
    clReleaseProgram(program);
// Release clFFT library
    clfftTeardown();
// Release OpenCL working objects
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(Array[0]);
    free(Array[1]);
    free(result[0]);
    free(result[1]);
    free(h[0]);
    free(h[1]);

    return 0;
}