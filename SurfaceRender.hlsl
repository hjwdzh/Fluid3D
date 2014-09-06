
struct Vertex
{
	float3 vPosition;
	float3 vNormal;
};

struct Triangle
{
	Vertex v[3];
};

StructuredBuffer<Triangle> VerticesRO : register( t0 );
TextureCube<float4> EnvMap : register( t1 );
Texture2D<float4> PositionMap : register( t2 );
Texture2D<float4> NormalMap : register( t3 );
Texture2D<float4> ColorMap : register( t4 );
Texture2D<float4> LightMap : register( t20 );


SamplerState g_samLinear : register( s0 );

#define IOR 1.333333
#define R0Constant (((1.0- (1.0/IOR) )*(1.0- (1.0/IOR) ))/((1.0+ (1.0/IOR) )*(1.0+ (1.0/IOR) )))
#define R0Inv (1.0 - R0Constant)

float FresnelApprox( float3 incident, float3 normal )
{
     return R0Constant + R0Inv * pow( 1.0-dot(incident,normal),5.0 );
}

struct VS_OUTPUT {
	float4 vPosition  : SV_POSITION;
	float3 f3Position : TEXCOORD0;
	float3 f3Normal	  : TEXCOORD1;
};

#define CT_REFRA 1
#define CT_REFLE 2

struct VS_OUTPUT3 {
	float4 vPosition  : SV_POSITION;
	float4 vLightDir  : TEXCOORD0;
	float3 f3Normal	  : TEXCOORD1;
	int iType		  : TEXCOORD2;
};

struct VS_OUTPUT4 {
	float4 vPosition  : SV_POSITION;
	float2 tex			: TEXCOORD0;
};

struct VS_OUTPUT5 {
	float4 vPosition  : SV_POSITION;
	float3 wPos			: TEXCOORD0;
};

struct VS_OUTPUT2 {
	float3 f3PositionOri	  : TEXCOORD0;
	float3 f3PositionCurRefra : TEXCOORD1;
	float3 f3PositionCurRefle : TEXCOORD2;
	float3 f3Normal			  : TEXCOORD3;
	int iavail				  : TEXCOORD4;
};

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


struct HS_Input
{
	float3 f3Position   : POSITION;
	float3 f3Normal     : NORMAL;
};

struct HS_ConstantOutput
{
	// Tess factor for the FF HW block
	float fTessFactor[3]    : SV_TessFactor;
	float fInsideTessFactor : SV_InsideTessFactor;

	// Geometry cubic generated control points
	float3 f3B210    : POSITION3;
	float3 f3B120    : POSITION4;
	float3 f3B021    : POSITION5;
	float3 f3B012    : POSITION6;
	float3 f3B102    : POSITION7;
	float3 f3B201    : POSITION8;
	float3 f3B111    : CENTER;

/*	// Normal quadratic generated control points
	float3 f3N110    : NORMAL3;      
	float3 f3N011    : NORMAL4;
	float3 f3N101    : NORMAL5;*/
};

struct HS_ControlPointOutput
{
	float3 f3Position    : POSITION;
	float3 f3Normal      : NORMAL;
};

struct DS_Output
{
	float4 f4Position   : SV_Position;
	float3 f3Position	: TEXCOORD0;
	float3 f3Normal      : TEXCOORD1;
};


//--------------------------------------------------------------------------------------
// This hull shader passes the tessellation factors through to the HW tessellator, 
// and the 10 (geometry), 6 (normal) control points of the PN-triangular patch to the domain shader
//--------------------------------------------------------------------------------------
HS_ConstantOutput HS_PNTrianglesConstant( InputPatch<HS_Input, 3> I )
{
	HS_ConstantOutput O = (HS_ConstantOutput)0;
	bool bViewFrustumCull = false;
	bool bBackFaceCull = false;
	float fEdgeDot[3];

#ifdef USE_VIEW_FRUSTUM_CULLING

	// Perform view frustum culling test
	bViewFrustumCull = ViewFrustumCull( I[0].f3Position, I[1].f3Position, I[2].f3Position, g_f4ViewFrustumPlanes, g_f4GUIParams2.y );

#endif

#ifdef USE_BACK_FACE_CULLING

	// Perform back face culling test

	// Aquire patch edge dot product between patch edge normal and view vector 
	fEdgeDot[0] = GetEdgeDotProduct( I[2].f3Normal, I[0].f3Normal, g_f4ViewVector.xyz );
	fEdgeDot[1] = GetEdgeDotProduct( I[0].f3Normal, I[1].f3Normal, g_f4ViewVector.xyz );
	fEdgeDot[2] = GetEdgeDotProduct( I[1].f3Normal, I[2].f3Normal, g_f4ViewVector.xyz );

	// If all 3 fail the test then back face cull
	bBackFaceCull = BackFaceCull( fEdgeDot[0], fEdgeDot[1], fEdgeDot[2], g_f4GUIParams1.x );

#endif

	// Skip the rest of the function if culling
	if( !bViewFrustumCull && !bBackFaceCull )
	{
		// Use the tessellation factors as defined in constant space 
		O.fTessFactor[0] = O.fTessFactor[1] = O.fTessFactor[2] = g_fTessFactor.w;
		float fAdaptiveScaleFactor;

#if defined( USE_SCREEN_SPACE_ADAPTIVE_TESSELLATION )

		// Get the screen space position of each control point, so we can compute the 
		// desired tess factor based upon an ideal primitive size
		float2 f2EdgeScreenPosition0 = GetScreenSpacePosition( I[0].f3Position, g_f4x4ViewProjection,  g_f4ScreenParams.x,  g_f4ScreenParams.y );
		float2 f2EdgeScreenPosition1 = GetScreenSpacePosition( I[1].f3Position, g_f4x4ViewProjection,  g_f4ScreenParams.x,  g_f4ScreenParams.y );
		float2 f2EdgeScreenPosition2 = GetScreenSpacePosition( I[2].f3Position, g_f4x4ViewProjection,  g_f4ScreenParams.x,  g_f4ScreenParams.y );
		// Edge 0
		fAdaptiveScaleFactor = GetScreenSpaceAdaptiveScaleFactor( f2EdgeScreenPosition2, f2EdgeScreenPosition0, g_f4TessFactors.x, g_f4GUIParams1.w );
		O.fTessFactor[0] = lerp( 1.0f, O.fTessFactor[0], fAdaptiveScaleFactor ); 
		// Edge 1
		fAdaptiveScaleFactor = GetScreenSpaceAdaptiveScaleFactor( f2EdgeScreenPosition0, f2EdgeScreenPosition1, g_f4TessFactors.x, g_f4GUIParams1.w );
		O.fTessFactor[1] = lerp( 1.0f, O.fTessFactor[1], fAdaptiveScaleFactor ); 
		// Edge 2
		fAdaptiveScaleFactor = GetScreenSpaceAdaptiveScaleFactor( f2EdgeScreenPosition1, f2EdgeScreenPosition2, g_f4TessFactors.x, g_f4GUIParams1.w );
		O.fTessFactor[2] = lerp( 1.0f, O.fTessFactor[2], fAdaptiveScaleFactor ); 

#else

#if defined( USE_DISTANCE_ADAPTIVE_TESSELLATION )

		// Perform distance adaptive tessellation per edge
		// Edge 0
		fAdaptiveScaleFactor = GetDistanceAdaptiveScaleFactor(    g_f4Eye.xyz, I[2].f3Position, I[0].f3Position, g_f4TessFactors.z, g_f4TessFactors.w * g_f4GUIParams1.z );
		O.fTessFactor[0] = lerp( 1.0f, O.fTessFactor[0], fAdaptiveScaleFactor ); 
		// Edge 1
		fAdaptiveScaleFactor = GetDistanceAdaptiveScaleFactor(    g_f4Eye.xyz, I[0].f3Position, I[1].f3Position, g_f4TessFactors.z, g_f4TessFactors.w * g_f4GUIParams1.z );
		O.fTessFactor[1] = lerp( 1.0f, O.fTessFactor[1], fAdaptiveScaleFactor ); 
		// Edge 2
		fAdaptiveScaleFactor = GetDistanceAdaptiveScaleFactor(    g_f4Eye.xyz, I[1].f3Position, I[2].f3Position, g_f4TessFactors.z, g_f4TessFactors.w * g_f4GUIParams1.z );
		O.fTessFactor[2] = lerp( 1.0f, O.fTessFactor[2], fAdaptiveScaleFactor ); 

#endif

#if defined( USE_SCREEN_RESOLUTION_ADAPTIVE_TESSELLATION )

		// Use screen resolution as a global scaling factor
		// Edge 0
		fAdaptiveScaleFactor = GetScreenResolutionAdaptiveScaleFactor( g_f4ScreenParams.x, g_f4ScreenParams.y, g_fMaxScreenWidth * g_f4GUIParams2.x, g_fMaxScreenHeight * g_f4GUIParams2.x );
		O.fTessFactor[0] = lerp( 1.0f, O.fTessFactor[0], fAdaptiveScaleFactor ); 
		// Edge 1
		fAdaptiveScaleFactor = GetScreenResolutionAdaptiveScaleFactor( g_f4ScreenParams.x, g_f4ScreenParams.y, g_fMaxScreenWidth * g_f4GUIParams2.x, g_fMaxScreenHeight * g_f4GUIParams2.x );
		O.fTessFactor[1] = lerp( 1.0f, O.fTessFactor[1], fAdaptiveScaleFactor ); 
		// Edge 2
		fAdaptiveScaleFactor = GetScreenResolutionAdaptiveScaleFactor( g_f4ScreenParams.x, g_f4ScreenParams.y, g_fMaxScreenWidth * g_f4GUIParams2.x, g_fMaxScreenHeight * g_f4GUIParams2.x );
		O.fTessFactor[2] = lerp( 1.0f, O.fTessFactor[2], fAdaptiveScaleFactor ); 

#endif

#endif

#ifdef USE_ORIENTATION_ADAPTIVE_TESSELLATION

#ifndef USE_BACK_FACE_CULLING

		// If back face culling is not used, then aquire patch edge dot product
		// between patch edge normal and view vector 
		fEdgeDot[0] = GetEdgeDotProduct( I[2].f3Normal, I[0].f3Normal, g_f4ViewVector.xyz );
		fEdgeDot[1] = GetEdgeDotProduct( I[0].f3Normal, I[1].f3Normal, g_f4ViewVector.xyz );
		fEdgeDot[2] = GetEdgeDotProduct( I[1].f3Normal, I[2].f3Normal, g_f4ViewVector.xyz );    

#endif

		// Scale the tessellation factors based on patch orientation with respect to the viewing
		// vector
		// Edge 0
		fAdaptiveScaleFactor = GetOrientationAdaptiveScaleFactor( fEdgeDot[0], g_f4GUIParams1.y );
		float fTessFactor0 = lerp( 1.0f, g_f4TessFactors.x, fAdaptiveScaleFactor ); 
		// Edge 1
		fAdaptiveScaleFactor = GetOrientationAdaptiveScaleFactor( fEdgeDot[1], g_f4GUIParams1.y );
		float fTessFactor1 = lerp( 1.0f, g_f4TessFactors.x, fAdaptiveScaleFactor ); 
		// Edge 2
		fAdaptiveScaleFactor = GetOrientationAdaptiveScaleFactor( fEdgeDot[2], g_f4GUIParams1.y );
		float fTessFactor2 = lerp( 1.0f, g_f4TessFactors.x, fAdaptiveScaleFactor ); 

#if defined( USE_SCREEN_SPACE_ADAPTIVE_TESSELLATION ) || defined( USE_DISTANCE_ADAPTIVE_TESSELLATION )

		O.fTessFactor[0] = ( O.fTessFactor[0] + fTessFactor0 ) / 2.0f;    
		O.fTessFactor[1] = ( O.fTessFactor[1] + fTessFactor1 ) / 2.0f;    
		O.fTessFactor[2] = ( O.fTessFactor[2] + fTessFactor2 ) / 2.0f;    

#else

		O.fTessFactor[0] = fTessFactor0;    
		O.fTessFactor[1] = fTessFactor1;    
		O.fTessFactor[2] = fTessFactor2;    

#endif

#endif

		// Now setup the PNTriangle control points...

		// Assign Positions
		float3 f3B003 = I[0].f3Position;
		float3 f3B030 = I[1].f3Position;
		float3 f3B300 = I[2].f3Position;
		// And Normals
		float3 f3N002 = I[0].f3Normal;
		float3 f3N020 = I[1].f3Normal;
		float3 f3N200 = I[2].f3Normal;

		// Compute the cubic geometry control points
		// Edge control points
		O.f3B210 = ( ( 2.0f * f3B003 ) + f3B030 - ( dot( ( f3B030 - f3B003 ), f3N002 ) * f3N002 ) ) / 3.0f;
		O.f3B120 = ( ( 2.0f * f3B030 ) + f3B003 - ( dot( ( f3B003 - f3B030 ), f3N020 ) * f3N020 ) ) / 3.0f;
		O.f3B021 = ( ( 2.0f * f3B030 ) + f3B300 - ( dot( ( f3B300 - f3B030 ), f3N020 ) * f3N020 ) ) / 3.0f;
		O.f3B012 = ( ( 2.0f * f3B300 ) + f3B030 - ( dot( ( f3B030 - f3B300 ), f3N200 ) * f3N200 ) ) / 3.0f;
		O.f3B102 = ( ( 2.0f * f3B300 ) + f3B003 - ( dot( ( f3B003 - f3B300 ), f3N200 ) * f3N200 ) ) / 3.0f;
		O.f3B201 = ( ( 2.0f * f3B003 ) + f3B300 - ( dot( ( f3B300 - f3B003 ), f3N002 ) * f3N002 ) ) / 3.0f;
		// Center control point
		float3 f3E = ( O.f3B210 + O.f3B120 + O.f3B021 + O.f3B012 + O.f3B102 + O.f3B201 ) / 6.0f;
		float3 f3V = ( f3B003 + f3B030 + f3B300 ) / 3.0f;
		O.f3B111 = f3E + ( ( f3E - f3V ) / 2.0f );

		// Compute the quadratic normal control points, and rotate into world space
/*		float fV12 = 2.0f * dot( f3B030 - f3B003, f3N002 + f3N020 ) / dot( f3B030 - f3B003, f3B030 - f3B003 );
		O.f3N110 = normalize( f3N002 + f3N020 - fV12 * ( f3B030 - f3B003 ) );
		float fV23 = 2.0f * dot( f3B300 - f3B030, f3N020 + f3N200 ) / dot( f3B300 - f3B030, f3B300 - f3B030 );
		O.f3N011 = normalize( f3N020 + f3N200 - fV23 * ( f3B300 - f3B030 ) );
		float fV31 = 2.0f * dot( f3B003 - f3B300, f3N200 + f3N002 ) / dot( f3B003 - f3B300, f3B003 - f3B300 );
		O.f3N101 = normalize( f3N200 + f3N002 - fV31 * ( f3B003 - f3B300 ) );*/
	}
	else
	{
		// Cull the patch
		O.fTessFactor[0] = 0.0f;
		O.fTessFactor[1] = 0.0f;
		O.fTessFactor[2] = 0.0f;
	}

	// Inside tess factor is just the average of the edge factors
	O.fInsideTessFactor = ( O.fTessFactor[0] + O.fTessFactor[1] + O.fTessFactor[2] ) / 3.0f;

	return O;
}

[domain("tri")]
[partitioning("pow2")]
[outputtopology("triangle_ccw")]
[patchconstantfunc("HS_PNTrianglesConstant")]
[outputcontrolpoints(3)]
[maxtessfactor(9.0f)]
HS_ControlPointOutput SurfaceHS( InputPatch<HS_Input, 3> I, uint uCPID : SV_OutputControlPointID )
{
	HS_ControlPointOutput O = (HS_ControlPointOutput)0;

	// Just pass through inputs = fast pass through mode triggered
	O.f3Position = I[uCPID].f3Position;
	O.f3Normal = I[uCPID].f3Normal;

	return O;
}


//--------------------------------------------------------------------------------------
// This domain shader applies contol point weighting to the barycentric coords produced by the FF tessellator 
//--------------------------------------------------------------------------------------
[domain("tri")]
DS_Output SurfaceDS( HS_ConstantOutput HSConstantData, const OutputPatch<HS_ControlPointOutput, 3> I, float3 f3BarycentricCoords : SV_DomainLocation )
{
	DS_Output O = (DS_Output)0;

	// The barycentric coordinates
	float fU = f3BarycentricCoords.x;
	float fV = f3BarycentricCoords.y;
	float fW = f3BarycentricCoords.z;

	// Precompute squares and squares * 3 
	float fUU = fU * fU;
	float fVV = fV * fV;
	float fWW = fW * fW;
	float fUU3 = fUU * 3.0f;
	float fVV3 = fVV * 3.0f;
	float fWW3 = fWW * 3.0f;

	// Compute position from cubic control points and barycentric coords
	float3 f3Position = I[0].f3Position * fWW * fW +
		I[1].f3Position * fUU * fU +
		I[2].f3Position * fVV * fV +
		HSConstantData.f3B210 * fWW3 * fU +
		HSConstantData.f3B120 * fW * fUU3 +
		HSConstantData.f3B201 * fWW3 * fV +
		HSConstantData.f3B021 * fUU3 * fV +
		HSConstantData.f3B102 * fW * fVV3 +
		HSConstantData.f3B012 * fU * fVV3 +
		HSConstantData.f3B111 * 6.0f * fW * fU * fV;

	// Compute normal from quadratic control points and barycentric coords
	float3 f3Normal =   I[0].f3Normal * fWW +
		I[1].f3Normal * fUU +
		I[2].f3Normal * fVV;/* +
		HSConstantData.f3N110 * fW * fU +
		HSConstantData.f3N011 * fU * fV +
		HSConstantData.f3N101 * fW * fV;*/

	// Normalize the interpolated normal    
	f3Normal = normalize( f3Normal );

	// Calc diffuse color    
	O.f3Normal = f3Normal;
	O.f3Position = f3Position.xyz;

	// Transform model position with view-projection matrix
	O.f4Position = mul( float4( f3Position.xyz, 1.0 ), g_mViewProjection[0] );

	return O;
}

const static float3 ldir = {0.57735f, 0.4714f, -0.333333f};
const static float3 bodycolor = {1,1,1};

float3 intersectP(float3 v, float3 r) 
{
	const static float3 np = {0, 1, 0};
	float3 vrp = (r - dot(r, np) * np) * dot(v, np);
	return float3(vrp.x + v.x, 0, vrp.z + v.z);
}

VS_OUTPUT2 SimpleCausticVS(uint ID : SV_VertexID)
{
	uint pid = ID / 3;
	uint vid = ID % 3;
	Vertex vWorld = VerticesRO[pid].v[vid];
	float3 v = vWorld.vPosition;
	float3 n = vWorld.vNormal.xyz;
	float3 refra = normalize(refract(-ldir, n, 1.0f / 1.3333f));
	float3 refle = normalize(reflect(-ldir, n));
	bool dall = (dot(-ldir, n) < 0);
	int drefra = dall && (dot(refra, float3(0,1,0)) < 0);
	int drefle = dall && (dot(refle, float3(0,1,0)) < 0);


	VS_OUTPUT2 o;
	o.iavail = (drefle * CT_REFLE) | (drefra * CT_REFRA);
	o.f3PositionOri = v;
	o.f3PositionCurRefra = drefra ? intersectP(v, refra) : 0;
	o.f3PositionCurRefle = drefle ? intersectP(v, refle) : 0;
	o.f3Normal = vWorld.vNormal.xyz;
	return o;
}

float4 SimpleShadowVS(uint ID : SV_VertexID) : SV_POSITION
{
	uint pid = ID / 3;
	uint vid = ID % 3;
	Vertex vWorld = VerticesRO[pid].v[vid];
	float3 v = vWorld.vPosition;
	float3 r = -ldir;

    return mul(float4(intersectP(v, r), 1.0f), g_mViewProjection[1]);
}

const static float4 PlaneColor={0.3f, 0.5f, 1.0f, 1};

float4 SimpleShadowPS(float4 input : SV_POSITION) : SV_TARGET
{
	return PlaneColor * 0.2f;
}

HS_Input SurfaceVS(uint ID : SV_VertexID)
{
	uint pid = ID / 3;
	uint vid = ID % 3;
	Vertex vWorld = VerticesRO[pid].v[vid];

	HS_Input o;
	o.f3Position = vWorld.vPosition;
	o.f3Normal = vWorld.vNormal;
	return o;
}

DS_Output SurfaceNoTessVS(uint ID : SV_VertexID)
{
	uint pid = ID / 3;
	uint vid = ID % 3;
	Vertex vWorld = VerticesRO[pid].v[vid];

	DS_Output o;
	o.f3Position = vWorld.vPosition;
	o.f3Normal = vWorld.vNormal;
	o.f4Position = mul( float4( vWorld.vPosition.xyz, 1.0 ), g_mViewProjection[0] );
	return o;
}

struct PS_OUTPUT {
	float4 pos : SV_Target0;
	float4 norm : SV_Target1;
};

PS_OUTPUT SurfacePS(DS_Output input)
{
	PS_OUTPUT o;
	o.pos = float4(input.f3Position, 1.0f);
	o.norm = float4(input.f3Normal, 1.0f);
	return o;
}

float AreaSqr(float3 v0, float3 v1, float3 v2) 
{
	float3 e0 = v1 - v0;
	float3 e2 = v2 - v0;
	float3 a = cross(e0, e2);
	return dot(a, a);
}

[maxvertexcount(6)]
void SimpleCausticGS(triangle VS_OUTPUT2 In[3], inout TriangleStream<VS_OUTPUT3> SpriteStream)
{
	VS_OUTPUT3 o;
	//Refractive
	if((In[0].iavail & CT_REFRA) &&
		(In[1].iavail & CT_REFRA) &&
		(In[2].iavail & CT_REFRA)) 
	{
		float a0 = AreaSqr(In[0].f3PositionOri, In[1].f3PositionOri, In[2].f3PositionOri);
		float a1 = AreaSqr(In[0].f3PositionCurRefra, In[1].f3PositionCurRefra, In[2].f3PositionCurRefra);
		float s = sqrt(a0 / a1);
		o.vLightDir.w = s;
		o.iType = CT_REFRA;

		for(int i = 0; i < 3; i++) 
		{
			o.vLightDir.xyz = In[i].f3PositionOri - In[i].f3PositionCurRefra;
			o.vPosition = mul(float4(In[i].f3PositionCurRefra, 1.0f), g_mViewProjection[1]);
			o.f3Normal = In[i].f3Normal;
			SpriteStream.Append(o);
		}
		SpriteStream.RestartStrip();
	}

	//Reflective
	if((In[0].iavail & CT_REFLE) &&
		(In[1].iavail & CT_REFLE) &&
		(In[2].iavail & CT_REFLE)) 
	{
		float a0 = AreaSqr(In[0].f3PositionOri, In[1].f3PositionOri, In[2].f3PositionOri);
		float a1 = AreaSqr(In[0].f3PositionCurRefle, In[1].f3PositionCurRefle, In[2].f3PositionCurRefle);
		float s = sqrt(a0 / a1);
		o.vLightDir.w = s;
		o.iType = CT_REFLE;

		for(int i = 0; i < 3; i++) 
		{
			o.vLightDir.xyz = In[i].f3PositionOri - In[i].f3PositionCurRefle;
			o.vPosition = mul(float4(In[i].f3PositionCurRefle, 1.0f), g_mViewProjection[1]);
			o.f3Normal = In[i].f3Normal;
			SpriteStream.Append(o);
		}
		SpriteStream.RestartStrip();
	}
}

float4 SimpleCausticPS(VS_OUTPUT3 input) : SV_Target
{
	float3 norm = normalize(input.f3Normal);
	float c = input.vLightDir.w * saturate(dot(normalize(input.vLightDir.xyz), float3(0, 1, 0))) * saturate(dot(norm, ldir));
	float3 refl = normalize(reflect(-ldir, norm));
	float wf = FresnelApprox( refl, norm );

	c *= (input.iType == CT_REFLE) ? wf : (1 - wf);

	return float4(bodycolor * c, 1.0f) * PlaneColor;
}

struct VVSOutput {
	float4 Pos  : SV_Position;
	float2 tex	: TEXCOORD0;
};

float KernelFunc(float dist2, float sigma) 
{
	return exp(-dist2 * sigma);
}

float4 BilateralXPS(VVSOutput input) : SV_Target0 
{
	int2 iTex = input.Pos.xy;

	float4 p_pos = PositionMap[iTex];
	clip(p_pos.w - 0.5f);

	float3 p_norm = NormalMap[iTex].xyz;

	const float h_sq = g_fSmoothlen * g_fSmoothlen;
	const float sigma = rcp(h_sq);

	float3 norm = 0;

	for(int i = -15; i <= 15; i++) {
		int2 nTex = iTex + int2(i, 0);
		float4 n_pos = PositionMap[nTex];
		float3 n_norm = NormalMap[nTex].xyz;
		float3 diff = n_pos.xyz - p_pos.xyz;
		float dist2 = dot(diff, diff);
		float w = KernelFunc(dist2, sigma) * saturate(dot(p_norm, n_norm.xyz)) * n_pos.w;
		norm += n_norm * w;
	}

	return float4(normalize(norm.xyz), p_pos.w);
}

float4 BilateralYPS(VVSOutput input) : SV_Target0
{
	int2 iTex = input.Pos.xy;

	float4 p_pos = PositionMap[iTex];
	clip(p_pos.w - 0.5f);

	float3 p_norm = NormalMap[iTex].xyz;

	const float h_sq = g_fSmoothlen * g_fSmoothlen;
	const float sigma = rcp(h_sq);

	float3 norm = 0;

	for(int i = -15; i <= 15; i++) {
		int2 nTex = iTex + int2(0, i);
		float4 n_pos = PositionMap[nTex];
		float3 n_norm = NormalMap[nTex].xyz;
		float3 diff = n_pos.xyz - p_pos.xyz;
		float dist2 = dot(diff, diff);
		float w = KernelFunc(dist2, sigma) * saturate(dot(p_norm, n_norm.xyz)) * n_pos.w;
		norm += n_norm * w;
	}

	return float4(normalize(norm.xyz), p_pos.w);
}

float2 intersectTex(float3 pos) 
{
	float4 vpos1 = mul(float4(pos, 1.0f), g_mViewProjection[1]);
	float2 spos = vpos1.xy / vpos1.w;
	spos.y = -spos.y;
	spos = spos * 0.5f + 0.5f;
	return spos;
}

float3 SampleWorld(float3 v, float3 r) 
{
	float dln = dot(r, float3(0,1,0));
	float3 c = 0;
	if(dln < 0) 
	{
		float2 cTex = intersectTex(intersectP(v, r));
		float4 cCaustic = LightMap.Sample( g_samLinear, cTex );
		if(cCaustic.w < 0.001f || dot(cCaustic.xyz, float3(1,1,1)) < 0.05f) 
		{
			c = EnvMap.Sample( g_samLinear, r ).xyz;
		} else 
		{
			c = cCaustic.xyz;
		}
	} else 
	{
		c = EnvMap.Sample( g_samLinear, r ).xyz;
	}
	return c;
}

float4 ColorPS(VVSOutput input) : SV_Target
{
	int2 iTex = input.Pos.xy;

	float4 p_norm = NormalMap[iTex];
	float3 p_pos = PositionMap[iTex].xyz;

	clip(p_norm.w - 0.5f);

	float3 norm = normalize(p_norm.xyz);

	float3 eyedir = normalize(p_pos - g_fEyePos.xyz);

/*	float c = saturate(dot(ldir, norm));
	c = pow(c, 0.6f) * 0.3f + c * 0.4f + 0.3f;*/

	float3 refl = normalize(reflect(eyedir, norm));
	float3 refra = normalize(refract(eyedir, norm, 1 / 1.3333f));
	float wf = FresnelApprox( refl, norm );

	float3 cReflect = SampleWorld(p_pos, refl) * wf;
	float3 cRefract = SampleWorld(p_pos, refra) * (1.0f - wf);

	float3 envc = 0;
	float sp = saturate(dot(refl, ldir));
	sp = pow(sp, 400.0f) * 4.0f;
	envc += sp;

	float3 fc = envc + cReflect + cRefract * bodycolor;
	return float4(fc, 1.0f);
}

float4 GrayPS(VVSOutput input) : SV_Target
{
	int2 iTex = input.Pos.xy;

	float4 p_norm = NormalMap[iTex];

	clip(p_norm.w - 0.5f);

	float3 norm = normalize(p_norm.xyz);
	
	return float4(norm * 0.5f + 0.5f, 1.0f);//saturate(dot(norm, ldir.xyz)) * 0.85f + 0.15f;
}

float3 Uncharted2Tonemap(float3 x)
{
	const float A = 0.15;
	const float B = 0.50;
	const float C = 0.10;
	const float D = 0.20;
	const float E = 0.02;
	const float F = 0.30;

	return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

float4 TonemapPS(VVSOutput input) : SV_Target
{
	const float W = 11.2;
	int2 iTex = input.Pos.xy;
	float3 texColor = ColorMap[iTex].xyz;

	//texColor *= 16;  // Hardcoded Exposure Adjustment

	float ExposureBias = 6.0f;

	float3 curr = Uncharted2Tonemap(ExposureBias * texColor);

	float3 whiteScale = 1.0f / Uncharted2Tonemap(W);

	float3 color = curr * whiteScale;

	return float4(color,1);
}

const static float g_fMapWidth = 0.5f;
const static float g_fMapLength = 1.0f;
const static float g_fMapEdge = g_fMapWidth * 0.5f;
const static float2 g_vGround[] =
{
	{-g_fMapEdge,				g_fMapLength + g_fMapEdge	},
	{g_fMapWidth + g_fMapEdge,	g_fMapLength + g_fMapEdge	},
	{g_fMapWidth + g_fMapEdge,	-g_fMapEdge					},
	{-g_fMapEdge,				g_fMapLength + g_fMapEdge	},
	{g_fMapWidth + g_fMapEdge,	-g_fMapEdge					},
	{-g_fMapEdge,				-g_fMapEdge					}
};

VS_OUTPUT4 PlaneVS(uint ID : SV_VertexID)
{
	VS_OUTPUT4 o;
	float2 vw = g_vGround[ID];
	o.vPosition = mul(float4(vw.x, 0, vw.y, 1.0f), g_mViewProjection[0]);
	o.tex = vw;
	return o;
}

float4 PlanePS(VS_OUTPUT4 input) : SV_Target
{
	float3 pos = float3(input.tex.x, 0, input.tex.y);
	return LightMap.Sample(g_samLinear, intersectTex(pos));
}

VS_OUTPUT4 PlaneVS1(uint ID : SV_VertexID)
{
	VS_OUTPUT4 o;
	float2 vw = g_vGround[ID];
	o.vPosition = mul(float4(vw.x, 0, vw.y, 1.0f), g_mViewProjection[1]);
	o.tex = vw;
	return o;
}

float4 PlanePaintPS(VS_OUTPUT4 input) : SV_Target
{
	float3 pos = float3(input.tex.x, 0, input.tex.y);
	float3 h = normalize(ldir + normalize(g_fEyePos.xyz - pos));
	
//	float spec = pow(saturate(dot(h, float3(0, 1, 0))), 4.0f);
	float4 c = PlaneColor * (saturate(dot(ldir, float3(0, 1, 0))));
	c.w = 1.0f;
	return c;
}

const static float3 box_vertices[] =
{
    {  -1.0f, 1.0f, -1.0f  },
    {  1.0f, 1.0f, -1.0f   },
    {  1.0f, 1.0f, 1.0f    },
    {  -1.0f, 1.0f, 1.0f   },
    {  -1.0f, -1.0f, -1.0f },
    {  1.0f, -1.0f, -1.0f  },
    {  1.0f, -1.0f, 1.0f   },
    {  -1.0f, -1.0f, 1.0f  },
};

const static int box_indices[] =
{
    1,3,0,
    1,2,3,

    5,0,4,
    5,1,0,

    4,3,7,
    4,0,3,

    6,1,5,
    6,2,1,

    7,2,6,
    7,3,2,

    4,6,5,
    4,7,6,
};

VS_OUTPUT5 EnvVS(uint ID : SV_VertexID)
{
	VS_OUTPUT5 o;
	float3 vW = box_vertices[box_indices[ID]] * 50.0f;
	o.vPosition = mul(float4(vW, 1.0f), g_mViewProjection[0]);
	o.wPos = normalize(vW);
	return o;
}

float4 EnvPS(VS_OUTPUT5 input) : SV_Target
{
	float3 r = normalize(input.wPos);
	return float4(EnvMap.Sample( g_samLinear, r ).xyz, 1.0f);
}