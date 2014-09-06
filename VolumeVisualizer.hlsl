Texture3D<float> DensityFieldRO : register( t0 );

SamplerState g_sampLinear : register( s0 );

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

const static float2 QuadBuf[] =
{
	{-1, 1},
	{1, 1},
	{1, -1},
	{-1, 1},
	{1, -1},
	{-1, -1}
};

struct VVSOutput {
	float4 Pos  : SV_Position;
	float2 tex	: TEXCOORD0;
};

VVSOutput VisualizeVS(uint ID : SV_VertexID)
{
	VVSOutput o;
	o.Pos = float4(QuadBuf[ID], 0.5f, 1.0f);
	o.tex = o.Pos.xy * float2(0.5f, -0.5f) + 0.5f;
	return o;
}

float4 VisualizePS(VVSOutput input) : SV_Target
{
	float2 tex = input.tex;
	float4 fVS = g_fVolSlicing;
	


	float2 utex = tex / fVS.zw;
	// Which Piece in Vol
	float2 wp = tex * fVS.xy;
	uint2 iwp = (uint2) wp;

	// i = piece in Vol
	uint i = iwp.y * (uint) fVS.x + iwp.x;

	// Which pos in piece
	float2 fp = frac(wp);
	float fpz = (float)i * g_fInvGridSize;

	clip(1.0f - fpz);

	return DensityFieldRO.Sample(g_sampLinear, float3(fp, fpz));
}