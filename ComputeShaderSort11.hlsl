//--------------------------------------------------------------------------------------
// File: ComputeShaderSort11.hlsl
//
// This file contains the compute shaders to perform GPU sorting using DirectX 11.
// 
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

#define BITONIC_BLOCK_SIZE 512

#define TRANSPOSE_BLOCK_SIZE 16

//--------------------------------------------------------------------------------------
// Constant Buffers
//--------------------------------------------------------------------------------------
cbuffer CB : register( b0 )
{
    unsigned int g_iLevel;
    unsigned int g_iLevelMask;
    unsigned int g_iWidth;
    unsigned int g_iHeight;
};

//--------------------------------------------------------------------------------------
// Structured Buffers
//--------------------------------------------------------------------------------------
Buffer<uint2> Input : register( t0 );
RWBuffer<uint> Data : register( u0 );

//--------------------------------------------------------------------------------------
// Bitonic Sort Compute Shader
//--------------------------------------------------------------------------------------
groupshared double shared_data[BITONIC_BLOCK_SIZE];

[numthreads(BITONIC_BLOCK_SIZE, 1, 1)]
void BitonicSort( uint3 Gid : SV_GroupID, 
                  uint3 DTid : SV_DispatchThreadID, 
                  uint3 GTid : SV_GroupThreadID, 
                  uint GI : SV_GroupIndex )
{
    // Load shared data
	uint2 t = uint2(Data[DTid.x * 2], Data[DTid.x * 2 + 1]);
    shared_data[GI] = asdouble(t.x, t.y);
    GroupMemoryBarrierWithGroupSync();
    
    // Sort the shared data
    for (unsigned int j = g_iLevel >> 1 ; j > 0 ; j >>= 1)
    {
        double result = ((shared_data[GI & ~j] <= shared_data[GI | j]) == (bool)(g_iLevelMask & DTid.x))? shared_data[GI ^ j] : shared_data[GI];
        GroupMemoryBarrierWithGroupSync();
        shared_data[GI] = result;
        GroupMemoryBarrierWithGroupSync();
    }
    
	asuint(shared_data[GI], t.x, t.y);
    // Store shared data
    Data[DTid.x * 2] = t.x;
    Data[DTid.x * 2 + 1] = t.y;
}

//--------------------------------------------------------------------------------------
// Matrix Transpose Compute Shader
//--------------------------------------------------------------------------------------
groupshared uint2 transpose_shared_data[TRANSPOSE_BLOCK_SIZE * TRANSPOSE_BLOCK_SIZE];

[numthreads(TRANSPOSE_BLOCK_SIZE, TRANSPOSE_BLOCK_SIZE, 1)]
void MatrixTranspose( uint3 Gid : SV_GroupID, 
                      uint3 DTid : SV_DispatchThreadID, 
                      uint3 GTid : SV_GroupThreadID, 
                      uint GI : SV_GroupIndex )
{
    transpose_shared_data[GI] = Input[DTid.y * g_iWidth + DTid.x];
    GroupMemoryBarrierWithGroupSync();
    uint2 XY = DTid.yx - GTid.yx + GTid.xy;
    uint2 result = transpose_shared_data[GTid.x * TRANSPOSE_BLOCK_SIZE + GTid.y];
	uint idx = XY.y * g_iHeight + XY.x;
	Data[idx * 2] = result.x;
	Data[idx * 2 + 1] = result.y;
}
