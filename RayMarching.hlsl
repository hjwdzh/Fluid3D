Texture3D<float> DensityFieldRO : register( t0 );
Texture2D<float4> texBBox : register( t1 );
SamplerState	g_pSampLinear : register( s0 );

cbuffer cbRenderConstants : register( b0 )
{
	matrix g_mViewProjection;
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

cbuffer cbGridDimConstants : register( b4 ) 
{
	int4 g_iGridMin;
	int4 g_iGridDim;
	int4 g_iGridDot;
};

const static float3 vertices[] =
{
    { 0.0f, 1.0f, 0.0f },
    { 1.0f, 1.0f, 0.0f  },
    { 1.0f, 1.0f, 1.0f   },
    { 0.0f, 1.0f, 1.0f  },
    { 0.0f, 0.0f, 0.0f},
    { 1.0f, 0.0f, 0.0f },
    { 1.0f, 0.0f, 1.0f  },
    { 0.0f, 0.0f, 1.0f },
};

const static uint indices[] =
{
    3,1,0,
    2,1,3,

    0,5,4,
    1,5,0,

    3,4,7,
    0,4,3,

    1,6,5,
    2,6,1,

    2,7,6,
    3,7,2,

    6,4,5,
    7,4,6,
};

struct VSTriangleOut
{
	float4 position : SV_POSITION;
	float3 wpos		: TEXCOORD0;
};

VSTriangleOut BoundingBoxVS(uint ID : SV_VertexID)
{
    VSTriangleOut Out = (VSTriangleOut)0;
    Out.wpos = (vertices[indices[ID]] * (float3) g_iGridDim.xyz + (float3) g_iGridMin.xyz) * g_fSmoothlen;
	Out.position = mul(float4(Out.wpos, 1.0f), g_mViewProjection);

    return Out;
}

float4 BoundingBoxPS(VSTriangleOut input) : SV_Target0
{
	return float4(input.wpos, 1.0f);
}

float distance_field(float3 pos) 
{
	float3 tex = pos * g_fRMAssist.xyz + g_fRMAssist2.xyz;
	return DensityFieldRO.Sample(g_pSampLinear, tex.xzy, 0);
}

float4 RayMarchingPS(VSTriangleOut input) : SV_Target0
{
	float3 spos = input.wpos;

	float4 epos = texBBox[(uint2) input.position.xy];

	clip(epos.w - 0.5f);

	float3 dirnn = epos.xyz - spos;
	float maxlen = length(dirnn);
	float3 dir = normalize(dirnn);

	float3 pos_cur = spos;

	float dv_old = 1.0f;
	float dv = distance_field(pos_cur);
	float dvacc = 0;

	int i = 0;

	[unroll]
	while(i < 32 && dv > g_fSmoothlen * 2.0f && dvacc < maxlen && dv <= dv_old) {
		pos_cur = pos_cur + min(dv, g_fSmoothlen * 0.5f) * dir;
		dv = distance_field(pos_cur);
		dvacc += min(dv, g_fSmoothlen * 0.5f);
		dv_old = dv;
		i++;
	}

	if(dv > dv_old)
		pos_cur = pos_cur + (dv_old - dv) * dir;

	clip(g_fSmoothlen * 2.0f - dv);

	return float4(pos_cur, 1.0f);
}