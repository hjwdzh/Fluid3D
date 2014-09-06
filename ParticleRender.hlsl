//--------------------------------------------------------------------------------------
// File: FluidRender.hlsl
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Particle Rendering
//--------------------------------------------------------------------------------------

struct Particle {
    float3 position;
    float3 velocity;
	float pressure;
};

struct ParticleDensity {
    float density;
};

StructuredBuffer<float3> ParticlesRO : register( t0 );
StructuredBuffer<ParticleDensity> ParticleDensityRO : register( t1 );

cbuffer cbMarchingCubesConstants : register( b1 )
{
	uint4 gridSize;
	uint4 gridSizeShift;
	uint4 gridSizeMask;
	float4 voxelSize;
};

cbuffer cbRenderConstants : register( b0 )
{
	matrix g_mViewProjection[2];
	matrix g_mView;
	float4 g_fVolSlicing;
	float4 g_fTessFactor;
	float4 g_fEyePos;
	float4 g_fRMAssist;
	float4 g_fRMAssist2;
	float g_fSmoothlen;
	float g_fParticleSize;
	float g_fParticleAspectRatio;
	float g_fInvGridSize;
};

uint3 calcGridPos(uint i)
{
	uint3 gridPos;
	gridPos.x = i & gridSizeMask.x;
	gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
	gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
	return gridPos;
}

struct VSParticleOut
{
    float3 position : POSITION;
    float4 color : COLOR;
};

struct GSParticleOut
{
    float4 position : SV_Position;
    float4 color : COLOR;
    float2 texcoord : TEXCOORD;
};


//--------------------------------------------------------------------------------------
// Visualization Helper
//--------------------------------------------------------------------------------------

static const float4 Rainbow[5] = {
    float4(1, 0, 0, 1), // red
    float4(1, 1, 0, 1), // orange
    float4(0, 1, 0, 1), // green
    float4(0, 1, 1, 1), // teal
    float4(0, 0, 1, 1), // blue
};

float4 VisualizeNumber(float n)
{
    return lerp( Rainbow[ floor(n * 4.0f) ], Rainbow[ ceil(n * 4.0f) ], frac(n * 4.0f) );
}

float4 VisualizeNumber(float n, float lower, float upper)
{
    return VisualizeNumber( saturate( (n - lower) / (upper - lower) ) );
}


//--------------------------------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------------------------------

VSParticleOut ParticleVS(uint ID : SV_VertexID)
{
    VSParticleOut Out = (VSParticleOut)0;
    Out.position = ParticlesRO[ID];
    Out.color = VisualizeNumber(ParticleDensityRO[ID].density, 1000.0f, 2000.0f);
    return Out;
}


//--------------------------------------------------------------------------------------
// Particle Geometry Shader
//--------------------------------------------------------------------------------------

static const float2 g_positions[4] = { float2(-1, 1), float2(1, 1), float2(-1, -1), float2(1, -1) };
static const float2 g_texcoords[4] = { float2(0, 1), float2(1, 1), float2(0, 0), float2(1, 0) };

[maxvertexcount(4)]
void ParticleGS(point VSParticleOut In[1], inout TriangleStream<GSParticleOut> SpriteStream)
{
    float4 position = mul(float4(In[0].position, 1.0f), g_mViewProjection[0]);
    GSParticleOut Out = (GSParticleOut)0;
    Out.color = In[0].color;

    [unroll]
    for (int i = 0; i < 4; i++)
    {
		Out.position = position + g_fParticleSize * float4(g_positions[i] * float2(1.0f, g_fParticleAspectRatio), 0, 0);
        Out.texcoord = g_texcoords[i];
        SpriteStream.Append(Out);
    }
    SpriteStream.RestartStrip();
}


//--------------------------------------------------------------------------------------
// Pixel Shader
//--------------------------------------------------------------------------------------

float4 ParticlePS(GSParticleOut In) : SV_Target
{
	float2 tx = In.texcoord * 2.0f - 1.0f;
	float dis = length(tx);
	float w = dis < 0.85? 1.0f : (exp(-80.0f * (dis - 0.85f) * (dis - 0.85f)));
    return float4(In.color.xyz, w);
}
