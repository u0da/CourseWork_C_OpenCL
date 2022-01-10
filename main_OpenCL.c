#include "clFFT.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <png.h>
#include <inttypes.h>
#include <stdarg.h>

#define MAX_SOURCE_SIZE (0x100000)
FILE *log_file;
const float zero = 0;

void show_status_string(const char *format, ...)
{
    char str[256]={'\0'};

    va_list args;
    va_start(args, format);

    vsprintf(str, format, args);

    va_end(args);

    printf( "### ");
    puts(str);

    fputs(str, log_file);
    fputc('\n', log_file);
}


struct Cl_Buffer_pair
{
    cl_mem buffers[2];
};

cl_int  InitCl_Buffer_pair(cl_context ctx, cl_command_queue queue, cl_bitfield mode, size_t N, struct  Cl_Buffer_pair *pair)
{
    cl_int err = CL_SUCCESS;
    memset(pair, 0, sizeof(*pair)); // побайтовое обнуление всей структуры pair

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
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("InitCl_Buffer_pair: Error with clFinish\n");
        return err;
    }
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

    // Temporary buffer
    cl_mem tmpBuffer;

    // FFT library realted declarations
    clfftPlanHandle planHandle;
};

cl_int InitFFT_OpenCL_data(int sizex, int sizey, cl_context ctx, cl_command_queue queue, int amount_of_buffers_to_transform, clfftDirection direction_normalize, struct FFT_OpenCL_data *data) {
    cl_int err = CL_SUCCESS;
    memset(data, 0, sizeof(*data)); // побайтовое обнуление всей структуры data
    data->sizex = sizex;
    data->sizey = sizey;
    int N = sizex * sizey;
    size_t clLengths[2] = {sizex, sizey};
    size_t tmpBufferSize = 0; // Size of temp buffer

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

    err = clfftSetResultLocation(data->planHandle, CLFFT_INPLACE);
    if (err != CL_SUCCESS)
        return err;


    err = clfftSetPlanBatchSize(data->planHandle, amount_of_buffers_to_transform);
    if (err != CL_SUCCESS)
        return err;

    err = clfftSetPlanDistance(data->planHandle, N, N);
    if (err != CL_SUCCESS)
        return err;

    err = clfftSetPlanScale(data->planHandle, direction_normalize, 1.0f / sqrtf(N));
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
        data->tmpBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tmpBufferSize, NULL, &err);
        if (err != CL_SUCCESS) {
            printf("Error with tmpBuffer clCreateBuffer\n");
            return err;
        }
    }

    return err;
}

void DeInItFFT_OpenCL_data(struct FFT_OpenCL_data *data)
{
    // Release OpenCL memory objects
    clReleaseMemObject(data->tmpBuffer);
    clfftDestroyPlan(&data->planHandle);
    memset(data, 0, sizeof(*data)); // побайтовое обнуление всей структуры data
}

///////// OpenCL FFT 2D function ///////////
int FFT_2D_OpenCL(struct Cl_Buffer_pair *input_output, clfftDirection direction, cl_command_queue queue, cl_int finishFlag,
                  struct FFT_OpenCL_data *data){
    cl_int err;
    // заполнение буферов на GPU нулями
    err = clfftEnqueueTransform(data->planHandle, direction, 1, &queue, 0, NULL, NULL,
                                input_output->buffers, input_output->buffers, data->tmpBuffer);

    // Wait for calculations to be finished
    if (finishFlag == CL_TRUE)
        err = clFinish(queue);

    return err;
}

struct Image{
    int width;
    int height;

    png_bytep* row_pointers;
};

struct Image read_png_file(char* file_name)
{
    png_byte color_type;
    png_byte bit_depth;

    png_structp png_ptr;
    png_infop info_ptr;
    int number_of_passes;

    struct Image image;
    memset(&image, 0, sizeof(image));

    png_byte header[8];    // 8 is the maximum size that can be checked

    /* open file and test for it being a png */
    FILE *fp = fopen(file_name, "rb");
    if (!fp)
    {
        printf("[read_png_file] File %s could not be opened for reading\n", file_name);
        return image;
    }

    fread(header, 1, 8, fp);
    if (png_sig_cmp(header, 0, 8))
    {
        printf("[read_png_file] File %s is not recognized as a PNG file\n", file_name);
        return image;
    }


    /* initialize stuff */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
    {
        printf("[read_png_file] png_create_read_struct failed\n");
        return  image;
    }


    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        printf("[read_png_file] png_create_info_struct failed\n");
        return image;
    }


    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[read_png_file] Error during init_io\n");
        return image;
    }


    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, 8);

    png_read_info(png_ptr, info_ptr);

    image.width = png_get_image_width(png_ptr, info_ptr);
    image.height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    number_of_passes = png_set_interlace_handling(png_ptr);
//    png_read_update_info(png_ptr, info_ptr);


    /* read file */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[read_png_file] Error during read_image\n");
        return  image;
    }

    if ((color_type & PNG_COLOR_MASK_COLOR)!= PNG_COLOR_TYPE_GRAY )
        png_set_rgb_to_gray(png_ptr, 1, 0, 0);
    png_read_update_info(png_ptr, info_ptr);

    image.row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * image.height);
    for (int y = 0; y < image.height; y++)
        image.row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

    png_read_image(png_ptr, image.row_pointers);

    fclose(fp);
    return image;
}

void write_png_file(struct Image image, const char* file_name)
{
    png_byte color_type = PNG_COLOR_TYPE_GRAY;
    png_byte bit_depth = 8;

    png_structp png_ptr;
    png_infop info_ptr;


    /* create file */
    FILE *fp = fopen(file_name, "wb");
    if (!fp)
    {
        printf("[write_png_file] File %s could not be opened for writing", file_name);
        return;
    }

    /* initialize stuff */
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (!png_ptr)
    {
        printf("[write_png_file] png_create_write_struct failed");
        return;
    }


    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)
    {
        printf("[write_png_file] png_create_info_struct failed");
        return;
    }


    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[write_png_file] Error during init_io");
        return;
    }


    png_init_io(png_ptr, fp);


    /* write header */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[write_png_file] Error during writing header");
        return;
    }


    png_set_IHDR(png_ptr, info_ptr, image.width, image.height,
                 bit_depth, color_type, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);


    /* write bytes */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[write_png_file] Error during writing bytes");
        return;
    }


    png_write_image(png_ptr, image.row_pointers);


    /* end write */
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        printf("[write_png_file] Error during end of write");
        return;
    }

    png_write_end(png_ptr, NULL);

    fclose(fp);
}

struct Cl_Buffer_pair read_and_fft_pics(cl_context ctx, cl_command_queue queue, int amount_of_pics, int sizex) {
    cl_int err;
    struct Cl_Buffer_pair all_pics_buffer;
    struct FFT_OpenCL_data fft_rash_size;
    size_t N = sizex*sizex;
    float *Array;

    clock_t creation_of_helpers_time_start = clock();
    Array  = (float *) calloc(N, sizeof(float));
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N*amount_of_pics, &all_pics_buffer);
    InitFFT_OpenCL_data(sizex, sizex, ctx, queue, amount_of_pics, CLFFT_BACKWARD, &fft_rash_size);
    clock_t creation_of_helpers_time_end = clock();
    show_status_string("Time for initiating buffer(helpers) for pics: %f", (float)(creation_of_helpers_time_end-creation_of_helpers_time_start)/CLOCKS_PER_SEC);

    clock_t  sumtime = 0;

    const size_t pic_size_in_bytes = N * sizeof(cl_float);

    for (int i = 0; i < amount_of_pics; i++)
    {
        clock_t start_time_load_pic = clock();
        char filename[64] = {'\0'};
        sprintf(filename, "%dx%d/image%02d.png", fft_rash_size.sizex/2, fft_rash_size.sizey/2, i+1);
        printf("### filename: %s\n", filename);

        struct Image image;
        image = read_png_file(filename);

        if (image.row_pointers == NULL || image.width*2 != fft_rash_size.sizex || image.height*2 != fft_rash_size.sizey)
        {
            DeInItCl_Buffer_pair(&all_pics_buffer);
            if (image.row_pointers != NULL)
            {
                for(int l = 0; l < image.width; l++)
                    free(image.row_pointers[l]);
                free(image.row_pointers);
            }
            free(Array);
            return all_pics_buffer;
        }

        for (int l = 0; l < image.height; l++)
        {
            for (int p = 0; p < image.width; p++)
                Array[l*fft_rash_size.sizey+p] = image.row_pointers[l][p];
        }

        cl_event write_future = 0;
        err = clEnqueueWriteBuffer(queue, all_pics_buffer.buffers[0], CL_FALSE, pic_size_in_bytes*i,
                                   pic_size_in_bytes, Array, 0, NULL, &write_future);


        for(int l = 0; l < image.width; l++)
            free(image.row_pointers[l]);
        free(image.row_pointers);
        if (err != CL_SUCCESS)
        {
            printf("Error with pics[%d].buffers[0] clEnqueueWriteBuffer\n", i);
            clReleaseEvent(write_future);
            DeInItCl_Buffer_pair(&all_pics_buffer);
            free(Array);
            return all_pics_buffer;
        }
        err = clWaitForEvents(1, &write_future);
        cl_int err1 = clReleaseEvent(write_future);
        if (err != CL_SUCCESS || err1 != CL_SUCCESS)
        {
            printf("ERROR with events\n");
            DeInItCl_Buffer_pair(&all_pics_buffer);
            free(Array);
            return all_pics_buffer;
        }

        clock_t tmpTime = clock() - start_time_load_pic;
        sumtime += tmpTime;
        printf("### %d loaded pic: %f seconds", i+1, (float)tmpTime/CLOCKS_PER_SEC);
        printf("\n");
    }

    /// Прямое ПФ для КАРТИНК

    clock_t fft_start = clock();
    if (FFT_2D_OpenCL(&all_pics_buffer, CLFFT_FORWARD, queue, CL_TRUE , &fft_rash_size) == 0)
    {

        clock_t fft_end = clock();
        printf("### all pics fft: %f seconds\n", (float)(fft_end - fft_start)/CLOCKS_PER_SEC);
    }
    else
    {
        printf("Problems w/ FFT\n");
        DeInItCl_Buffer_pair(&all_pics_buffer);
    }

    printf("\n");
    DeInItFFT_OpenCL_data(&fft_rash_size);
    free(Array);
    return all_pics_buffer;
}

cl_program init_kernel_program(cl_context ctx, cl_device_id device)
{
    // Execute the OpenCL kernel on the list
    cl_int ret;
    FILE *rash_kernel;
    char *source_str;

    /// Открываем файл с kernel на чтение
    rash_kernel = fopen("rash_kernel.cl", "r");
    if (!rash_kernel)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        return 0;
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    fread( source_str, 1, MAX_SOURCE_SIZE, rash_kernel);
    fclose(rash_kernel);

    // Create program
    cl_program program = clCreateProgramWithSource(ctx, 1, (const char **)&source_str, NULL, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    /// в случае ошибок выводим log
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
        return 0;
    }
    free(source_str);
    source_str = NULL;

    return program;
}



//// СКОЛЬКО ПАМЯТИ ТРАТИТСЯ ////
// x^2 - размер одой картинки в пикселях ( оригинальный )
// тк мы работаем с раширенными матрицами => (2x)^2 - размер одной картинки в пикселях ( расширенный )
// переводим в байты => (2x)^2 * sizeof(float) - размер одной картинки в байтах ( расширенный )
// а тк у нас есть и мнимая часть => (2x)^2 * sizeof(float) * 2 - размер Фурье-образа одной картики в байтах ( расширенный )
// тк у нас h_rash_CL тоже имеет размер (2x)^2 * sizeof(float) * 2 => (2x)^2 * sizeof(float) * 2 * 2 - минимальный необходимый объем памяти на GPU
// также нужно место для части результата ( матрица и тут должна быть расширена ) => (2x)^2 * sizeof(float) * 2 (result_part_CL)
// нужно (2x)^2 * sizeof(float) - для итогового результата ( сумма получивших картинок ) (result_CL)

// тогда минимальный объем памяти для N картинок на GPU - (2x)^2 * sizeof(float) * 2 * (2*N + 1) + x^2* sizeof(float)*2  + (2x)^2 * sizeof(float) ( предпоследнее слагаемое - h оригинального размера, последнее - результат )
// x^2 * sizeof(float) * (16N + 14)

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

    log_file = fopen("log_file.txt", "wb");

    printf("### Amount of devices: %u\n", number_of_devices);
    fprintf(log_file, "Amount of devices: %u\n", number_of_devices);
    printf("### List of devices:\n");

    cl_device_id *devices = (cl_device_id *) malloc(number_of_devices * sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, number_of_devices, devices, NULL);
    for (int i = 0; i < number_of_devices; i++)
    {
        char name[128] = {'\0'};

        err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, NULL);
        show_status_string("\t\t[%d]%s", i, name);
    }
    printf("\n");

// выбираем нужный девайс
    int ptr = -1;

    while (ptr >= number_of_devices || ptr < 0)
    {
        printf("Choose device: ");
        scanf("%d", &ptr);
        printf("\n");
    }
    cl_device_id device = devices[ptr];

    fprintf(log_file, "You chose this device: [%d]\n", ptr);
    free(devices);

    char version[128] = {'\0'};
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION,  sizeof(version), version, NULL);
    fprintf(log_file, "Your OpenCL version: %s\n", version);

    int major_version = 0;
    int minor_version = 0;

    sscanf(version, "OpenCL %d.%d",&major_version, &minor_version);

    if (major_version <= 1 && minor_version < 2)
    {
        printf("Program requires OpenCL 1.2 and higher\n");
        exit(1);
    }

    ptr = 0;
    printf("Choose image size (like 512, 1024 etc): ");
    scanf("%d", &ptr);
    printf("\n");
    fprintf(log_file, "You chose image size: %dx%d\n", ptr, ptr);
    int amount_of_pics = 0;

    while (amount_of_pics < 1)
    {
        printf("Choose amount of pics: ");
        scanf("%d", &amount_of_pics);
        printf("\n");
    }
    fprintf(log_file, "You chose this amount of pics: %d\n", amount_of_pics);

    cl_ulong device_memsize_in_bytes = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(device_memsize_in_bytes), &device_memsize_in_bytes, NULL);

    show_status_string("GPU mem space: %"PRIu64" MB", device_memsize_in_bytes/((cl_ulong)1024*(cl_ulong)1024));

    cl_ulong min_memsize_in_bytes_required = (cl_ulong)(ptr*ptr) * sizeof(float) * (16 * amount_of_pics + 14);
    if (min_memsize_in_bytes_required >= device_memsize_in_bytes)
    {
        printf("### Not enough GPU memory\n");
        printf("### Min required GPU mem space: %"PRIu64" MB\n", min_memsize_in_bytes_required/((cl_ulong)1024*(cl_ulong)1024));
        return 1;
    }


// Create an OpenCL context
    ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

// Create a command queue
    queue = clCreateCommandQueue(ctx, device, 0, &err);

    show_status_string("Initializing FFT library...");
    // Setup clFFT
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);


    int sizex = ptr*2;
    int sizey = ptr*2;
    int half_sizex = sizex / 2;
    int half_sizey = sizey / 2;

    // Total size of FFT( по сути размер расширенных матриц )
    size_t N = sizex * sizey;
    // исходный размер картинки ( используется только для h )
    size_t half_N = half_sizex * half_sizey;

    cl_program program = init_kernel_program(ctx, device);
    if (program == 0)
    {
        clfftTeardown(); // Release clFFT library
        clReleaseCommandQueue(queue); // Release OpenCL working objects
        clReleaseContext(ctx);
        fclose(log_file);
        exit(1);
    }

///=================================================================

/// НАЧАЛО РАБОТЫ С КАРТИНКОЙ
    struct Cl_Buffer_pair all_pics_buffer;

    show_status_string("Reading and FFT-ing input pics...");
    clock_t start = clock();
    all_pics_buffer = read_and_fft_pics(ctx, queue, amount_of_pics, sizex);
    printf("### Reading and fft'ing pics ends in: %f seconds\n", (float)(clock()-start)/CLOCKS_PER_SEC);
    if (all_pics_buffer.buffers[0] == 0)
    {
       clfftTeardown(); // Release clFFT library
       clReleaseCommandQueue(queue); // Release OpenCL working objects
       clReleaseProgram(program);
       clReleaseContext(ctx);
       fclose(log_file);
       exit(1);
    }

/// КОНЕЦ РАБОТЫ С КАРТИНКОЙ


/// НАЧАЛО РАБОТЫ С h

    clock_t start_h_CL_time = clock();
    struct FFT_OpenCL_data fft_orig_size;
    err = InitFFT_OpenCL_data(half_sizex, half_sizey, ctx, queue, 1, CLFFT_FORWARD, &fft_orig_size);

    // кол-во картинок равно 3 => amount_of_pics = 3;
    int amount_of_h = amount_of_pics;

    /// Создаем пару буферов для h размером исходной картинки и h расширенной
    struct Cl_Buffer_pair h_rash_CL[amount_of_h];
    struct Cl_Buffer_pair h_CL_k;

    show_status_string("Init buffer for h_original_size and h_extended_size");

    for (int i = 0; i < amount_of_h; i++)
    {
        InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N, &h_rash_CL[i]);
    }
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, half_N, &h_CL_k);

    // Создаем kernel для инициализации h и передаем туда аргументы ( delta_z и два буфера для вещественной и мнимой части )
    cl_kernel h_init_kernel = clCreateKernel(program, "h_init_kernel", &ret);
    cl_kernel h_squared_abs_kernel = clCreateKernel(program, "h_squared_abs_kernel", &ret);
    cl_kernel fft_shift_row_kernel = clCreateKernel(program, "fft_shift_row_kernel", &ret);
    cl_kernel fft_shift_col_kernel = clCreateKernel(program, "fft_shift_col_kernel", &ret);


    ret = clSetKernelArg(h_init_kernel, 1, sizeof(cl_mem), &h_CL_k.buffers[0]);
    if (ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[0] h_init_kernel\n");
    ret = clSetKernelArg(h_init_kernel, 2, sizeof(cl_mem), &h_CL_k.buffers[1]);
    if (ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[1] h_init_kernel\n");


    ret = clSetKernelArg(h_squared_abs_kernel, 0, sizeof(cl_mem), &h_CL_k.buffers[0]);
    if (ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[0] h_squared_abs_kernel\n");
    ret = clSetKernelArg(h_squared_abs_kernel, 1, sizeof(cl_mem), &h_CL_k.buffers[1]);
    if (ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for h[1] h_squared_abs_kernel\n");


    for (int k = 0; k < amount_of_h; k++)
    {

        float delta_z = k * M_PI;

        ret = clSetKernelArg(h_init_kernel, 0, sizeof(delta_z), &delta_z);
        if (ret != CL_SUCCESS)
            printf("Problems w/ setting KernelArgs for delta_z h_init_kernel\n");


        size_t global_group_size[] = {half_sizex, half_sizey};
        /// Кладем в очередь команды для вызова kernel, который создает матрицу h размерами исходной картинки

        ret = clEnqueueNDRangeKernel(queue, h_init_kernel, 2, NULL, global_group_size, NULL, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clEnqueueNDRangeKernel h_init_kernel");
        ret = clFinish(queue);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clFinish");



        show_status_string("Making FFT for h_original_size");
        /// Прямое ПФ для h
        if (FFT_2D_OpenCL(&h_CL_k, CLFFT_FORWARD, queue, CL_TRUE ,&fft_orig_size) == 0)
            ;
        else
            printf("FFT for h func NOT passed !\n");

        /// FFTShift для h

        for (int i = 0; i < 2; i++)
        {
            ret = clSetKernelArg(fft_shift_row_kernel, 0, sizeof(cl_mem), &h_CL_k.buffers[i]);
            if (ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);
            ret = clSetKernelArg(fft_shift_row_kernel, 1, sizeof(half_sizex), &half_sizex);
            if (ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h[%d] fft_shift_row_kernel\n", i);

            size_t sizey_t = half_sizey;
            ret = clEnqueueNDRangeKernel(queue, fft_shift_row_kernel, 1, NULL, &sizey_t, NULL, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clEnqueueNDRangeKernel fft_shift_row_kernel");
            ret = clFinish(queue);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clFinish");

            ret = clSetKernelArg(fft_shift_col_kernel, 0, sizeof(cl_mem), &h_CL_k.buffers[i]);
            if (ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);
            ret = clSetKernelArg(fft_shift_col_kernel, 1, sizeof(half_sizey), &half_sizey);
            if (ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h[%d] fft_shift_col_kernel\n", i);


            size_t sizex_t = half_sizex;
            ret = clEnqueueNDRangeKernel(queue, fft_shift_col_kernel, 1, NULL, &sizex_t, NULL, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clEnqueueNDRangeKernel fft_shift_col_kernel");
            ret = clFinish(queue);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clFinish");

        }

        /// Модуль для h^2

        size_t half_N_local = half_N;
        ret = clEnqueueNDRangeKernel(queue, h_squared_abs_kernel, 1, NULL, &half_N_local, NULL, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clEnqueueNDRangeKernel h_squared_abs_kernel");
        ret = clFinish(queue);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clFinish");

        // Расширяем матрицу h ( теперь она становится h_rash )
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < half_sizey; j++)
            {
                err = clEnqueueCopyBuffer(queue, h_CL_k.buffers[i], h_rash_CL[k].buffers[i],
                                          j * half_sizex * sizeof(cl_float),  j * sizex * sizeof(cl_float),
                                          half_sizex * sizeof(cl_float), 0, NULL, NULL);
                if (err != CL_SUCCESS)
                {
                    printf("Error with clEnqueueCopyBuffer %d\n", j);
                    for (int l = 0; l < amount_of_h; l++)
                        DeInItCl_Buffer_pair(&h_rash_CL[l]);
                    DeInItCl_Buffer_pair(&all_pics_buffer);
                    fclose(log_file);
                    DeInItFFT_OpenCL_data(&fft_orig_size);
                    clReleaseProgram(program);
                    clfftTeardown(); // Release clFFT library
                    clReleaseCommandQueue(queue); // Release OpenCL working objects
                    clReleaseContext(ctx);
                    return err;
                }

            }

        ret = clFinish(queue);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clFinish after copy");
    }


    clReleaseKernel(h_init_kernel);
    clReleaseKernel(h_squared_abs_kernel);
    clReleaseKernel(fft_shift_col_kernel);
    clReleaseKernel(fft_shift_row_kernel);

    DeInItFFT_OpenCL_data(&fft_orig_size);
    DeInItCl_Buffer_pair(&h_CL_k);

    clock_t end_h_CL_time = clock();

    clock_t start_h_CL_fft_time = clock();
    struct FFT_OpenCL_data fft_rash_size;
    err = InitFFT_OpenCL_data(sizex, sizey, ctx, queue, 1, CLFFT_BACKWARD, &fft_rash_size);

    for (int k = 0; k < amount_of_h; k++)
    {
        // Прямое ПФ для расширенной матрицы h
        if (FFT_2D_OpenCL(&h_rash_CL[k], CLFFT_FORWARD, queue, CL_FALSE, &fft_rash_size) == 0)
            ;
        else
            printf("FFT for h_rash func NOT passed !\n");
    }
    ret = clFinish(queue);
    if (ret != CL_SUCCESS)
        printf("Problems w/ clFinish after 2nd FFT for h_rash");

    clock_t end_h_CL_fft_time = clock();


/// РАБОТА С h ЗАКОНЧЕНА
    float h_gen_time = (float)(end_h_CL_time - start_h_CL_time)/CLOCKS_PER_SEC;
    float h_fft_time = (float)(end_h_CL_fft_time - start_h_CL_fft_time)/CLOCKS_PER_SEC;

    show_status_string("");
    show_status_string("Time for generating h: %f",  h_gen_time);
    show_status_string("Time for h fft: %f",  h_fft_time);
    show_status_string("Total time for generating and fft'ing h: %f", h_gen_time + h_fft_time);
    show_status_string("");

/// Умножение картинки и элементов матрицы h_rash

    clock_t calculations_start = clock();

    cl_kernel multiply_kernel = clCreateKernel(program, "multiply_kernel", &ret);
    cl_kernel add_normalized_abs_part_kernel = clCreateKernel(program, "add_normalized_abs_part_kernel", &ret);

    /// Создаем два буфера, которые будут хранить результат произведения
    cl_mem result_CL;
    result_CL = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(cl_float), NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Init result_CL clCreateBuffer ERROR\n");
        return err;
    }

    struct Cl_Buffer_pair result_part_CL;
    InitCl_Buffer_pair(ctx, queue, CL_MEM_READ_WRITE, N, &result_part_CL);

    ret = clSetKernelArg(multiply_kernel, 0, sizeof(cl_mem), &all_pics_buffer.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for all_pics_buffer.buffers[0 ]multiply\n");
    ret = clSetKernelArg(multiply_kernel, 1, sizeof(cl_mem), &all_pics_buffer.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for all_pics_buffer.buffers[1] multiply\n");

    ret = clSetKernelArg(multiply_kernel, 5, sizeof(cl_mem), &result_part_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[0] multiply\n");
    ret = clSetKernelArg(multiply_kernel, 6, sizeof(cl_mem), &result_part_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[1] multiply\n");


    ret = clSetKernelArg(add_normalized_abs_part_kernel, 0, sizeof(cl_mem), &result_part_CL.buffers[0]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[0] abs\n");
    ret = clSetKernelArg(add_normalized_abs_part_kernel, 1, sizeof(cl_mem), &result_part_CL.buffers[1]);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[1] abs\n");

    float scaling = 1 / (powf(half_sizex, 3.0f)*amount_of_pics);
    ret = clSetKernelArg(add_normalized_abs_part_kernel, 2, sizeof(scaling), &scaling);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for scaling add_normalized_abs_part_kernel\n");
    ret = clSetKernelArg(add_normalized_abs_part_kernel, 3, sizeof(cl_mem), &result_CL);
    if(ret != CL_SUCCESS)
        printf("Problems w/ setting KernelArgs for result[1] abs\n");


    float *result;
    struct Image image_result;

    image_result.width = half_sizex;
    image_result.height = half_sizey;
    image_result.row_pointers = malloc(image_result.height* sizeof(image_result.row_pointers[0]));


    for(int i = 0; i < image_result.height; i++)
        image_result.row_pointers[i] = malloc(image_result.width* sizeof(image_result.row_pointers[0][0]));

    result = (float *) calloc(N, sizeof(float));

    float time_multiply_full = 0;
    for (int m = 0; m < amount_of_pics; m++)
    {
        err = clEnqueueFillBuffer(queue, result_CL, &zero, sizeof(zero), 0, N * sizeof(float), 0, NULL, NULL);
        if (err != CL_SUCCESS)
        {
            printf("Init result_CL clEnqueueFillBuffer ERROR\n");
            return err;
        }
        ret = clFinish(queue);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clFinish after copy");

        for (int n = 0; n < amount_of_pics; n++)
        {
            cl_ulong offset = N * n;
            ret = clSetKernelArg(multiply_kernel, 2, sizeof(offset), &offset);
            if(ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for offset multiply\n");

            int h_rash_CL_index = abs(n-m);
            ret = clSetKernelArg(multiply_kernel, 3, sizeof(cl_mem), &h_rash_CL[h_rash_CL_index].buffers[0]);
            if(ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h_rash_CL[0] multiply\n");
            ret = clSetKernelArg(multiply_kernel, 4, sizeof(cl_mem), &h_rash_CL[h_rash_CL_index].buffers[1]);
            if(ret != CL_SUCCESS)
                printf("Problems w/ setting KernelArgs for h_rash_CL[1] multiply\n");


            clock_t  multiply_start_time = clock();

            ret = clEnqueueNDRangeKernel(queue, multiply_kernel, 1, NULL, &N, NULL, 0, NULL, NULL);

            if (ret != CL_SUCCESS)
                printf("Problems w/ clEnqueueNDRangeKernel multiply: %d\n", ret);
            ret = clFinish(queue);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clFinish");

            clock_t  multiply_end_time = clock();

            show_status_string("Time for multiplying 1 layer: %f", (float)(multiply_end_time-multiply_start_time)/CLOCKS_PER_SEC);
            time_multiply_full += (float)(multiply_end_time-multiply_start_time)/CLOCKS_PER_SEC;
            printf("### index_result:%d index_input:%d\n", m, n);



            /// Обратное ПФ для результата
            if (FFT_2D_OpenCL(&result_part_CL, CLFFT_BACKWARD, queue, CL_TRUE, &fft_rash_size) == 0)
                ;
                //            printf("IFFT for result passed !\n");
            else
                printf("IFFT for result NOT passed !\n");

            ret = clEnqueueNDRangeKernel(queue, add_normalized_abs_part_kernel, 1, NULL, &N, NULL, 0, NULL, NULL);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clEnqueueNDRangeKernel abs");
            ret = clFinish(queue);
            if (ret != CL_SUCCESS)
                printf("Problems w/ clFinish");
        }

        show_status_string("Time for multiplying all layers: %f", time_multiply_full);

        clock_t start_writing_result = clock();
        ret = clEnqueueReadBuffer(queue, result_CL, CL_TRUE, 0,
                                  N * sizeof(float), result, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
            printf("Problems w/ clEnqueueReadBuffer");


//#pragma omp parallel for
        for (int k = 0; k < image_result.height; k++)
            for(int l = 0; l < image_result.width; l++)
            {
                float res = result[(k+image_result.height/2)*fft_rash_size.sizex+(l+image_result.width/2)];
                image_result.row_pointers[k][l] = (png_byte)res;
            }

        char filename_png[64] = {'\0'};
        sprintf(filename_png, "result/image%02d.png",  m+1);
        show_status_string("Writing data to file");

        write_png_file(image_result, filename_png);
        clock_t end_writing_result = clock();
        calculations_start += end_writing_result - start_writing_result;
    }

    clReleaseKernel(multiply_kernel);
    clReleaseKernel(add_normalized_abs_part_kernel);
    clReleaseMemObject(result_CL);
    DeInItCl_Buffer_pair(&result_part_CL);

    show_status_string("");
    show_status_string("Full time of calculations(multiply+add): %g seconds\n", (float)(clock() - calculations_start )/CLOCKS_PER_SEC);
    printf("### Cleaning...\n");

    for (int i = 0; i < image_result.height; i++)
        free(image_result.row_pointers[i]);
    free(image_result.row_pointers);
    free(result);

    /// Удаляем ненужные нам буфферы

    DeInItCl_Buffer_pair(&all_pics_buffer);
    for (int i = 0; i < amount_of_h; i++)
    {
        DeInItCl_Buffer_pair(&h_rash_CL[i]);
    }

    fclose(log_file);
    DeInItFFT_OpenCL_data(&fft_rash_size);
    clReleaseProgram(program);
    clfftTeardown(); // Release clFFT library
    clReleaseCommandQueue(queue); // Release OpenCL working objects
    clReleaseContext(ctx);

    return 0;
}