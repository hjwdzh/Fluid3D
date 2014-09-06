cbuffer cbMarchingCubesConstants : register( b1 )
{
	uint4 gridSize;
	uint4 gridSizeShift;
	uint4 gridSizeMask;
	float4 voxelSize;
};

cbuffer cbRenderConstants : register( b0 )
{
	matrix g_mViewProjection;
	matrix g_mView;
	float4 g_fVolSlicing;
	float4 g_fTessFactor;
	float4 g_fEyePos;
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