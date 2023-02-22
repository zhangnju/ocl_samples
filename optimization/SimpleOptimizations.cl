
__kernel void
SimpleKernel( const __global float *input, __global float *output)
{
    size_t index = get_global_id(0);
    for(int i = 0; i < ITER_NUM; i++)
    {
        output[index] = sin(fabs(input[index]));
    }
}

__kernel /*__attribute__((vec_type_hint(float4))) */ void
SimpleKernel4( const __global float4 *input, __global float4 *output)
{
    size_t index = get_global_id(0);
    for(int i = 0; i < ITER_NUM; i++)
    {
        output[index] = sin(fabs(input[index]));
    }
}
