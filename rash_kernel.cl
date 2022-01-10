#define M_PI 3.1415927f

__kernel void add_normalized_abs_part_kernel(__global float *result_part_real, __global float *result_part_imag, 
                                             const float scaling, __global float *result)
{
    // Get the index of the current element to be processed
    int i = get_global_id(0);
 
    float res_real = result_part_real[i];
    float res_imag = result_part_imag[i];
    
    
    // Do the operation
    // result[i] += sqrt(res_real*res_real + res_imag*res_imag)*scaling;
    result[i] = min(sqrt(res_real*res_real + res_imag*res_imag)*scaling + result[i], 255.0f);
}

__kernel void multiply_kernel(__global const float *images_real, __global const float *images_imag, 
                                const ulong image_start_offset,
                              __global const float *h_real, __global const float *h_imag,
                              __global float *result_real, __global float *result_imag)
{
    int i = get_global_id(0);
    ulong pixel_offset = i + image_start_offset; 
    float im_real =  images_real[pixel_offset];
    float im_imag =  images_imag[pixel_offset];
    float h_r = h_real[i];
    float h_i = h_imag[i];

    result_real[i] = im_real * h_r - im_imag * h_i;
    result_imag[i] = im_real * h_i + im_imag * h_r;
}


int M(float x, float y) 
{
    if ((pown(x, 2) + pown(y, 2)) < pown(M_PI*0.5f, 2))
        return (1);
    else
        return (0);
}

float p_s(float x, float y, float delta_z) 
{
    // float d_1 = 57.4f * 10e-3;
    // float d_0 = 37.0f * 10e-3;
    // float r_0 = 4.5f * 10e-3;
    // float lamba = 0.55f * 10e-6;
    // float w = 2.34f * 10e-5;

    // return (-M_PI * lamba * (d_1*d_1)*delta_z * (pown(x, 2) + pown(y, 2)))/pown(d_0 + w, 2);
    return (0.375 * fabs(delta_z) * M_PI * (pown(x, 2) + pown(y, 2)));
}

float p(float x, float y) {
    return (M_PI*0.5f * (pown(x, 2) + pown(y, 2)));
}


__kernel void h_init_kernel(const float delta_z, __global float *h_real,  __global float *h_imag)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    int sizex = get_global_size(0);
    int sizey = get_global_size(1);
    
    // float d_1 = 57.4f * 1e-3;
    // float d_0 = 37.0f * 1e-3;
    // float r_0 = 4.5f * 1e-3;
    // float lamba = 0.55f * 1e-6;
    // float w = 2.34f * 1e-4;
    // float a = 4.0f * r_0/(lamba * d_1);

    int index = i * sizey + j;
    float x = 0, y = 0;

    x = (M_PI / sizex) * (i - sizex/2);
    y = (M_PI / sizey) * (j - sizey/2);
    // float m_result = M(x*lamba*d_1, y*lamba*d_1);
    float m_result = M(x, y);
    float p_result = p(x, y);
    float p_s_result = p_s(x, y, delta_z);
    
    float cos_result = 0.0f;
    float sin_result = sincos(p_result + p_s_result, &cos_result);
    
    h_real[index] = m_result * cos_result;
    h_imag[index] = m_result * sin_result;
}

__kernel void h_squared_abs_kernel(__global float *h_real, __global float *h_imag)
{
    int i = get_global_id(0);

    h_real[i] = h_real[i]*h_real[i] + h_imag[i]*h_imag[i];
    h_imag[i] = 0.0f; 
}

__kernel void fft_shift_row_kernel(__global float *array, const int num_col)
{
    int i = get_global_id(0);

    int half_num_col = num_col/2;
    int row_start_index = i * num_col;
    
    for (int j = 0; j < half_num_col; j++)
    {
        float tmp = array[row_start_index + j];
        array[row_start_index + j] = array[row_start_index + half_num_col + j];
        array[row_start_index + half_num_col + j] = tmp;
    }
    
}

__kernel void fft_shift_col_kernel(__global float *array, const int num_row)
{
    int j = get_global_id(0);
    int num_col = get_global_size(0);

    // sizex = num_col
    // sizey = num_row

    int half_num_row = num_row/2;

    for (int i = 0; i < half_num_row; i++)
    {
        float tmp = array[i * num_col + j];
        array[i * num_col + j] = array[(i + half_num_row) * num_col + j];
        array[(i + half_num_row) * num_col + j] = tmp;
    }
}

