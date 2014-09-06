//--------------------------------------------------------------------------------------
// File: FluidCS11.hlsl
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Smoothed Particle Hydrodynamics Algorithm Based Upon:
// Particle-Based Fluid Simulation for Interactive Applications
// Matthias Müller
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Optimized Grid Algorithm Based Upon:
// Broad-Phase Collision Detection with CUDA
// Scott Le Grand
//--------------------------------------------------------------------------------------
#ifndef _DEBUG
#define TRY_CUDA
#endif

struct Particle
{
    float3 position;
    float3 velocity;
	float pressure;
};

struct MINMAX {
	float3 fmin;
	float3 fmax;
};

struct ParticleForces
{
    float3 acceleration;
};

struct ParticleDensity
{
    float density;
};

struct symMat
{
	float4 c0;
	float2 c1;
};

cbuffer cbSimulationConstants : register( b0 )
{
    uint g_iNumParticles;
    float g_fTimeStep;
    float g_fSmoothlen;
    float g_fPressureStiffness;
	float g_fPressureGamma;
    float g_fRestDensity;
    float g_fDensityCoef;
	float g_fDeltaCoef;
    float g_fBeta;
	float g_fDelta;
    float g_fGradPressureCoef;
    float g_fLapViscosityCoef;
    float g_fWallStiffness;

    float4 g_vGravity;
    float4 g_vGridDim;
	float4 g_vGridDim2;
	float4 g_vGridDim3;
	float4 g_vGridDim4;
    float4 g_vPlanes[6];
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
    float g_fSmoothlen2;
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

uint3 calcGridPos(uint i)
{
	uint3 gridPos;
	gridPos.x = i & gridSizeMask.x;
	gridPos.y = (i >> gridSizeShift.y) & gridSizeMask.y;
	gridPos.z = (i >> gridSizeShift.z) & gridSizeMask.z;
	return gridPos;
}

uint calcMCGridIndexInternal(uint3 p) {
	uint3 ip = p << gridSizeShift.xyz;
	return ip.z | ip.y | ip.x;
}

uint calcMCGridIndex(float3 p) 
{
	uint3 ip = (uint3)(p * g_vGridDim4.xyz + 1.0f);
	return calcMCGridIndexInternal(ip);
}

//--------------------------------------------------------------------------------------
// Fluid Simulation
//--------------------------------------------------------------------------------------

#define SIMULATION_BLOCK_SIZE 512

#define NUM_GRID_DIM_X 512
#define NUM_GRID_DIM_Y 32
#define NUM_GRID_DIM_Z 512


// Constants
#define M_SQRT3    1.73205081f   // sqrt(3)
#define FLT_EPSILON     1.192092896e-07f

// Macros
#define SQR(x)      ((x)*(x))                        // x^2 

// ----------------------------------------------------------------------------
void dsyevc3(float3x3 A, inout float3 w)
	// ----------------------------------------------------------------------------
	// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
	// analytical algorithm.
	// Only the diagonal and upper triangular parts of A are accessed. The access
	// is read-only.
	// ----------------------------------------------------------------------------
	// Parameters:
	//   A: The symmetric input matrix
	//   w: Storage buffer for eigenvalues
	// ----------------------------------------------------------------------------
	// Return value:
	//   0: Success
	//  -1: Error
	// ----------------------------------------------------------------------------
{
	float m, c1, c0;

	// Determine coefficients of characteristic poynomial. We write
	//       | a   d   f  |
	//  A =  | d*  b   e  |
	//       | f*  e*  c  |
	float de = A[0][1] * A[1][2];                                    // d * e
	float dd = SQR(A[0][1]);                                         // d^2
	float ee = SQR(A[1][2]);                                         // e^2
	float ff = SQR(A[0][2]);                                         // f^2
	m  = A[0][0] + A[1][1] + A[2][2];
	c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
		- (dd + ee + ff);
	c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
	- 2.0f * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

	float p, sqrt_p, q, c, s, phi;
	p = SQR(m) - 3.0f*c1;
	q = m*(p - (3.0f/2.0f)*c1) - (27.0f/2.0f)*c0;
	sqrt_p = sqrt(abs(p));

	phi = 27.0f * ( 0.25f*SQR(c1)*(p - c1) + c0*(q + 27.0f/4.0f*c0));
	phi = (1.0f/3.0f) * atan2(sqrt(abs(phi)), q);

	c = sqrt_p*cos(phi);
	s = (1.0f/M_SQRT3)*sqrt_p*sin(phi);

	w[1]  = (1.0f/3.0f)*(m - c);
	w[2]  = w[1] + s;
	w[0]  = w[1] + c;
	w[1] -= s;
}

// ----------------------------------------------------------------------------
void dsyevv3(float3x3 A, inout float3x3 Q, inout float3 w)
	// ----------------------------------------------------------------------------
	// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
	// matrix A using Cardano's method for the eigenvalues and an analytical
	// method based on vector cross products for the eigenvectors.
	// Only the diagonal and upper triangular parts of A need to contain meaningful
	// values. However, all of A may be used as temporary storage and may hence be
	// destroyed.
	// ----------------------------------------------------------------------------
	// Parameters:
	//   A: The symmetric input matrix
	//   Q: Storage buffer for eigenvectors
	//   w: Storage buffer for eigenvalues
	// ----------------------------------------------------------------------------
	// Return value:
	//   0: Success
	//  -1: Error
	// ----------------------------------------------------------------------------
	// Dependencies:
	//   dsyevc3()
	// ----------------------------------------------------------------------------
	// Version history:
	//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
	//     (according to the documentation, only the upper triangular part needs
	//     to be filled)
	//   v1.0f: First released version
	// ----------------------------------------------------------------------------
{
#ifndef EVALS_ONLY
	float norm;          // Squared norm or inverse norm of current eigenvector
	float n0, n1;        // Norm of first and second columns of A
	float n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
	float thresh;        // Small number used as threshold for floating point comparisons
	float error;         // Estimated maximum roundoff error in some steps
	float wmax;          // The eigenvalue of maximum modulus
	float f, t;          // Intermediate storage
	int i, j;             // Loop counters
#endif

	// Calculate eigenvalues
	dsyevc3(A, w);

#ifndef EVALS_ONLY
	wmax = abs(w[0]);
	if ((t=abs(w[1])) > wmax)
		wmax = t;
	if ((t=abs(w[2])) > wmax)
		wmax = t;
	thresh = SQR(8.0f * FLT_EPSILON * wmax);

	// Prepare calculation of eigenvectors
	n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
	n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
	Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
	Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
	Q[2][1] = SQR(A[0][1]);

	// Calculate first eigenvector by the formula
	//   v[0] = (A - w[0]).e1 x (A - w[0]).e2
	A[0][0] -= w[0];
	A[1][1] -= w[0];
	Q[0][0] = Q[0][1] + A[0][2]*w[0];
	Q[1][0] = Q[1][1] + A[1][2]*w[0];
	Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
	norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
	n0      = n0tmp + SQR(A[0][0]);
	n1      = n1tmp + SQR(A[1][1]);
	error   = n0 * n1;

	if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
	{
		Q[0][0] = 1.0f;
		Q[1][0] = 0.0f;
		Q[2][0] = 0.0f;
	}
	else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
	{
		Q[0][0] = 0.0f;
		Q[1][0] = 1.0f;
		Q[2][0] = 0.0f;
	}
	else if (norm < SQR(64.0f * FLT_EPSILON) * error)
	{                         // If angle between A[0] and A[1] is too small, don't use
		t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
		f = -A[0][0] / A[0][1];
		if (SQR(A[1][1]) > t)
		{
			t = SQR(A[1][1]);
			f = -A[0][1] / A[1][1];
		}
		if (SQR(A[1][2]) > t)
			f = -A[0][2] / A[1][2];
		norm    = 1.0f/sqrt(1 + SQR(f));
		Q[0][0] = norm;
		Q[1][0] = f * norm;
		Q[2][0] = 0.0f;
	}
	else                      // This is the standard branch
	{
		norm = sqrt(1.0f / norm);
		for (j=0; j < 3; j++)
			Q[j][0] = Q[j][0] * norm;
	}


	// Prepare calculation of second eigenvector
	t = w[0] - w[1];
	if (abs(t) > 8.0f * FLT_EPSILON * wmax)
	{
		// For non-degenerate eigenvalue, calculate second eigenvector by the formula
		//   v[1] = (A - w[1]).e1 x (A - w[1]).e2
		A[0][0] += t;
		A[1][1] += t;
		Q[0][1]  = Q[0][1] + A[0][2]*w[1];
		Q[1][1]  = Q[1][1] + A[1][2]*w[1];
		Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
		norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
		n0       = n0tmp + SQR(A[0][0]);
		n1       = n1tmp + SQR(A[1][1]);
		error    = n0 * n1;

		if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
		{
			Q[0][1] = 1.0f;
			Q[1][1] = 0.0f;
			Q[2][1] = 0.0f;
		}
		else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
		{
			Q[0][1] = 0.0f;
			Q[1][1] = 1.0f;
			Q[2][1] = 0.0f;
		}
		else if (norm < SQR(64.0f * FLT_EPSILON) * error)
		{                       // If angle between A[0] and A[1] is too small, don't use
			t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
			f = -A[0][0] / A[0][1];
			if (SQR(A[1][1]) > t)
			{
				t = SQR(A[1][1]);
				f = -A[0][1] / A[1][1];
			}
			if (SQR(A[1][2]) > t)
				f = -A[0][2] / A[1][2];
			norm    = 1.0f/sqrt(1 + SQR(f));
			Q[0][1] = norm;
			Q[1][1] = f * norm;
			Q[2][1] = 0.0f;
		}
		else
		{
			norm = sqrt(1.0f / norm);
			for (j=0; j < 3; j++)
				Q[j][1] = Q[j][1] * norm;
		}
	}
	else
	{
		// For degenerate eigenvalue, calculate second eigenvector according to
		//   v[1] = v[0] x (A - w[1]).e[i]
		//   
		// This would really get to complicated if we could not assume all of A to
		// contain meaningful values.
		A[1][0]  = A[0][1];
		A[2][0]  = A[0][2];
		A[2][1]  = A[1][2];
		A[0][0] += w[0];
		A[1][1] += w[0];

		[unroll]
		for (i=0; i < 3; i++)
		{
			A[i][i] -= w[1];
			n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
			if (n0 > thresh)
			{
				Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
				Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
				Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
				norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
				if (norm > SQR(256.0f * FLT_EPSILON) * n0) // Accept cross product only if the angle between
				{                                         // the two vectors was not too small
					norm = sqrt(1.0f / norm);
					for (j=0; j < 3; j++)
						Q[j][1] = Q[j][1] * norm;
					break;
				}
			}
		}

		if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
		{
			[unroll]
			for (j=0; j < 3; j++)
				if (Q[j][0] != 0.0f)                                   // Find nonzero element of v[0] ...
				{                                                     // ... and swap it with the next one
					norm          = 1.0f / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
					Q[j][1]       = Q[(j+1)%3][0] * norm;
					Q[(j+1)%3][1] = -Q[j][0] * norm;
					Q[(j+2)%3][1] = 0.0f;
					break;
				}
		}
	}


	// Calculate third eigenvector according to
	//   v[2] = v[0] x v[1]
	Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
	Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
	Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];
#endif
}



//--------------------------------------------------------------------------------------
// Structured Buffers
//--------------------------------------------------------------------------------------
RWStructuredBuffer<Particle> ParticlesRW : register( u0 );
StructuredBuffer<Particle> ParticlesRO : register( t0 );

RWStructuredBuffer<ParticleDensity> ParticlesDensityRW : register( u0 );
StructuredBuffer<ParticleDensity> ParticlesDensityRO : register( t1 );

RWStructuredBuffer<ParticleForces> ParticlesForcesRW : register( u0 );
StructuredBuffer<ParticleForces> ParticlesForcesRO : register( t2 );

RWStructuredBuffer<ParticleForces> ParticlesPressureForcesRW : register( u1 );
StructuredBuffer<ParticleForces> ParticlesPressureForcesRO : register( t5 );

RWBuffer<uint> GridRW : register( u0 );
Buffer<uint2> GridRO : register( t3 );

RWStructuredBuffer<uint2> GridIndicesRW : register( u0 );
StructuredBuffer<uint2> GridIndicesRO : register( t4 );

RWStructuredBuffer<float> ParticlesPressureFixRW : register( u2 );
StructuredBuffer<float> ParticlesPressureFixRO : register( t8 );

RWStructuredBuffer<float3> ParticlesSmoothedRW : register( u1 );
StructuredBuffer<float3> ParticlesSmoothedRO : register( t6 );

RWStructuredBuffer<float> ParticlesStretchRW : register( u0 );
StructuredBuffer<float> ParticlesStretchRO : register( t7 );

AppendStructuredBuffer<uint> ParticlesAnisoRW : register( u1 );
StructuredBuffer<uint> ParticlesAnisoRO : register( t9 );

RWBuffer<uint> ParticlesMCIdxRW : register( u0 );
Buffer<uint> ParticlesMCIdxSortedRO : register( t10 );


RWBuffer<uint> NumCellsRW : register( u0 );
Buffer<uint> NumCellsRO : register( t11 );

RWTexture3D<float>	DensityFieldRW : register( u0 );

StructuredBuffer<MINMAX> BoundingBoxRO : register( t12 );
RWBuffer<uint4> GridDimRW : register( u0 );

StructuredBuffer<float3> ParticlesSmoothedSortedRO : register( t13 );
//--------------------------------------------------------------------------------------
// Grid Construction
//--------------------------------------------------------------------------------------

// For simplicity, this sample uses a 16-bit hash based on the grid cell and
// a 16-bit particle ID to keep track of the particles while sorting
// This imposes a limitation of 64K particles and 256x256 grid work
// You could extended the implementation to support large scenarios by using a uint2

float3 GridCalculateCell0(float3 position)
{
    return clamp(position * g_vGridDim.xyz + g_vGridDim2.xyz, float3(0, 0, 0), float3(NUM_GRID_DIM_X-1, NUM_GRID_DIM_Y-1, NUM_GRID_DIM_Z-1));
}

int3 GridCalculateCell(float3 position)
{
    return (int3) (GridCalculateCell0(position)) - g_iGridMin.xyz;
}

unsigned int GridConstuctKey(uint3 xyz)
{
	// Bit pack [----Y---][----Z---][----X---]
	//              8-bit		8-bit		8-bit
    return dot(xyz.yzx, uint3(g_iGridDot.xy, 1));
}

uint3 GridDecomposeKey(uint key)
{
	// Bit pack [----Y---][----Z---][----X---]
	//              8-bit		8-bit		8-bit
	uint2 gridDim = g_iGridDim.xz * 3 + 2;
	uint y = key / (gridDim.x * gridDim.y);
	uint xz = key - y * (gridDim.x * gridDim.y);
	uint z = xz / gridDim.x;
	uint x = xz - z * gridDim.x;

    return uint3(x, y, z);
}

uint2 GridConstuctKeyValuePair(uint3 xyz, uint value)
{
    // Bit pack [----Z---][----Y---][----X---][-------------VALUE--------------]
    //             8-bit		8-bit	   8-bit				   32-bit
    return uint2(GridConstuctKey(xyz), value);
}

uint GridGetKey(uint2 keyvaluepair)
{
    return keyvaluepair.y;
}

uint GridGetValue(uint2 keyvaluepair)
{
    return keyvaluepair.x;
}


//--------------------------------------------------------------------------------------
// Build Grid
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BuildGridCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int P_ID = DTid.x; // Particle ID to operate on
    
    float3 position = ParticlesRO[P_ID].position;
	
    int3 grid_xyz = GridCalculateCell( position );
    
    uint2 result = GridConstuctKeyValuePair((uint3)grid_xyz, P_ID);

	GridRW[P_ID * 2] = result.y;
	GridRW[P_ID * 2 + 1] = result.x;
}


//--------------------------------------------------------------------------------------
// Build Grid Indices
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void ClearGridIndicesCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    GridIndicesRW[DTid.x] = uint2(0, 0);
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BuildGridIndicesCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int G_ID = DTid.x; // Grid ID to operate on
    unsigned int G_ID_PREV = (G_ID == 0)? g_iNumParticles : G_ID; G_ID_PREV--;
    unsigned int G_ID_NEXT = G_ID + 1; if (G_ID_NEXT == g_iNumParticles) { G_ID_NEXT = 0; }
    
    unsigned int cell = GridGetKey( GridRO[G_ID] );
    unsigned int cell_prev = GridGetKey( GridRO[G_ID_PREV] );
    unsigned int cell_next = GridGetKey( GridRO[G_ID_NEXT] );
    if (cell != cell_prev)
    {
        // I'm the start of a cell
        GridIndicesRW[cell].x = G_ID;
    }
    if (cell != cell_next)
    {
        // I'm the end of a cell
        GridIndicesRW[cell].y = G_ID + 1;
    }
}


//--------------------------------------------------------------------------------------
// Rearrange Particles
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void RearrangeParticlesCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int ID = DTid.x; // Particle ID to operate on
    const unsigned int G_ID = GridGetValue( GridRO[ ID ] );
    ParticlesRW[ID] = ParticlesRO[ G_ID ];
	ParticlesSmoothedRW[ID] = ParticlesSmoothedRO[ G_ID ];
}


//--------------------------------------------------------------------------------------
// Density Calculation
//--------------------------------------------------------------------------------------

float CalculateDensity(float r_sq)
{
    const float h_sq = g_fSmoothlen * g_fSmoothlen;
    // Implements this equation:
    // W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
    // g_fDensityCoef = fParticleMass * 315.0f / (64.0f * PI * fSmoothlen^9)
    return g_fDensityCoef * (h_sq - r_sq) * (h_sq - r_sq) * (h_sq - r_sq);
}

float CalculateBSpline(float r_sq)
{
	const float h_sq = g_fSmoothlen * g_fSmoothlen;

	float u2 = r_sq / h_sq;
	float u = sqrt(u2);
	return (u > 2.0f) ? 0 : 
		(
		(u < 1.0f) ? 
		(1.0f - 1.5f * u2 + 0.75f * u2 * u) :
		(0.25f * (2.0f - u) * (2.0f - u) * (2.0f - u))
		) * (1.0f / 3.14159f);
}

float CalculateLapSmooth(float r_sq)
{
    const float h_sq = g_fSmoothlen * g_fSmoothlen * 4.0f;
	return 1.0f - pow(r_sq / h_sq, 1.5f);
}


float4 CalculateDelta(float r_sq, float3 diff)
{
	const float h_sq = g_fSmoothlen * g_fSmoothlen;
	// Implements this equation:
	// W_poly6(r, h) = 315 / (64 * pi * h^9) * (h^2 - r^2)^3
	// GRAD( W_poly6(r, h) ) = -945 / (32 * pi * h ^ 9) * (h^2 - r^2)^2 * r
	// g_fDeltaCoef = -945 / (32 * pi * fSmoothlen ^ 9)
	float3 gradw = g_fDeltaCoef * (h_sq - r_sq) * (h_sq - r_sq) * diff;
	return float4(gradw, dot(gradw, gradw));
}

//--------------------------------------------------------------------------------------
// Optimized Grid + Sort Algorithm
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void DensityCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int P_ID = DTid.x;
    const float h_sq = g_fSmoothlen * g_fSmoothlen;
    float3 P_position = ParticlesRO[P_ID].position;
    
#ifdef UPDATE_DELTA    
	float4 delta = 0;
#endif
#ifdef LAPLACIAN_SMOOTH
	float4 P_newpos = 0;
#endif
	float density = 0;

    // Calculate the density based on neighbors from the 8 adjacent cells + current cell
    int3 G_XY = GridCalculateCell( P_position );
	
	for (int Z = max(G_XY.z - 1, 0) ; Z <= min(G_XY.z + 1, g_iGridDim.z-1) ; Z++)
	{
		for (int Y = max(G_XY.y - 1, 0) ; Y <= min(G_XY.y + 1, g_iGridDim.y-1) ; Y++)
		{
			for (int X = max(G_XY.x - 1, 0) ; X <= min(G_XY.x + 1, g_iGridDim.x-1) ; X++)
			{
				unsigned int G_CELL = GridConstuctKey(uint3(X, Y, Z));
				uint2 G_START_END = GridIndicesRO[G_CELL];
				for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
				{
					float3 N_position = ParticlesRO[N_ID].position;

					float3 diff = N_position - P_position;
					float r_sq = dot(diff, diff);
#ifdef LAPLACIAN_SMOOTH
					if (r_sq < h_sq)
					{
						P_newpos += float4(N_position, 1.0f) * CalculateLapSmooth(r_sq);
					}
#endif
					if (r_sq < h_sq)
					{
						density += CalculateDensity(r_sq);
#ifdef UPDATE_DELTA    
						delta += CalculateDelta(r_sq, diff);		
#endif
					}
				}
			}
		}
	}

	ParticlesDensityRW[P_ID].density = density;

#ifdef LAPLACIAN_SMOOTH
	
	P_newpos.xyz /= P_newpos.w;

	P_newpos.xyz = min(max(P_newpos.xyz, 0), float3(g_vPlanes[3].w, g_vPlanes[4].w, g_vPlanes[5].w));

	ParticlesSmoothedRW[P_ID] = P_newpos.xyz;
#endif
#ifdef PRESSURE_FIX
	ParticlesPressureFixRW[P_ID] = max(0, ParticlesPressureFixRW[P_ID] + (density - g_fRestDensity));
#endif
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void CovCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	const unsigned int P_ID = DTid.x;
	const float h_sq = g_fSmoothlen * g_fSmoothlen;
	float3 P_position = ParticlesSmoothedRO[P_ID];

	float wb = 0;
	float c[6] = {0, 0, 0, 0, 0, 0};

	// Calculate the density based on neighbors from the 8 adjacent cells + current cell
	int3 G_XY = GridCalculateCell( P_position );

	uint nParticles = 0;

	for (int Z = max(G_XY.z - 1, 0) ; Z <= min(G_XY.z + 1, g_iGridDim.z-1) ; Z++)
	{
		for (int Y = max(G_XY.y - 1, 0) ; Y <= min(G_XY.y + 1, g_iGridDim.y-1) ; Y++)
		{
			for (int X = max(G_XY.x - 1, 0) ; X <= min(G_XY.x + 1, g_iGridDim.x-1) ; X++)
			{
				unsigned int G_CELL = GridConstuctKey(uint3(X, Y, Z));
				uint2 G_START_END = GridIndicesRO[G_CELL];
				for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
				{
					float3 N_position = ParticlesRO[N_ID].position;

					float3 diff = N_position - P_position;
					float r_sq = dot(diff, diff);

					if (r_sq < h_sq)
					{
						float w = CalculateLapSmooth(r_sq);
						c[0] += diff.x * diff.x * w;
						c[1] += diff.x * diff.y * w;
						c[2] += diff.x * diff.z * w;
						c[3] += diff.y * diff.y * w;
						c[4] += diff.y * diff.z * w;
						c[5] += diff.z * diff.z * w;
						wb += w;

						nParticles++;
					}
				}
			}
		}
	}

	if(nParticles < 20) {
		ParticlesStretchRW[P_ID] = 1.0f;
	} else {
		[unroll]
		for(int i = 0; i < 6; i++)
			c[i] = c[i] / wb * 700.0f;

		float3x3 A = {
			c[0], c[1], c[2],
			c[1], c[3], c[4],
			c[2], c[4], c[5]
		};

		float3 sigma;

		dsyevc3(A, sigma);

		float maxsig = max(sigma.z, max(sigma.x, sigma.y));
		float minsig = min(sigma.z, min(sigma.x, sigma.y));
		float midsig = sigma.x + sigma.y + sigma.z - maxsig - minsig;


		float g = (midsig == 0) ? 1.0f : (maxsig / midsig);

		g = max(1.0f, min(4.0f, g));

		g = (4.0f - g) / 3.0f;

		float u = (1.0f - g * g);

		float f = (1.0f - u * u * u);

		ParticlesStretchRW[P_ID] = rcp(f * f);
	}		
}
//--------------------------------------------------------------------------------------
// Force Calculation
//--------------------------------------------------------------------------------------

float CalculatePressure(float density)
{
    // Implements this equation:
    // Pressure = B * ((rho / rho_0)^y  - 1)
    return g_fPressureStiffness * max(pow(abs(density / g_fRestDensity), g_fPressureGamma) - 1, 0);
}

float3 CalculateGradPressure(float r, float P_pressure, float N_pressure, float N_density, float3 diff)
{
    const float h = g_fSmoothlen;
    float avg_pressure = 0.5f * (N_pressure + P_pressure);
    // Implements this equation:
    // W_spkiey(r, h) = 15 / (pi * h^6) * (h - r)^3
    // GRAD( W_spikey(r, h) ) = -45 / (pi * h^6) * (h - r)^2
    // g_fGradPressureCoef = fParticleMass * -45.0f / (PI * fSmoothlen^6)
    return g_fGradPressureCoef * avg_pressure / N_density * (h - r) * (h - r) / r * (diff);
}

float3 CalculateLapVelocity(float r, float3 P_velocity, float3 N_velocity, float N_density)
{
    const float h = g_fSmoothlen;
    float3 vel_diff = (N_velocity - P_velocity);
    // Implements this equation:
    // W_viscosity(r, h) = 15 / (2 * pi * h^3) * (-r^3 / (2 * h^3) + r^2 / h^2 + h / (2 * r) - 1)
    // LAPLACIAN( W_viscosity(r, h) ) = 45 / (pi * h^6) * (h - r)
    // g_fLapViscosityCoef = fParticleMass * fViscosity * 45.0f / (PI * fSmoothlen^6)
    return g_fLapViscosityCoef / N_density * (h - r) * vel_diff;
}


//--------------------------------------------------------------------------------------
// Optimized Grid + Sort Algorithm
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void ForceCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int P_ID = DTid.x; // Particle ID to operate on
    
    float3 P_position = ParticlesRO[P_ID].position;
    float P_density = ParticlesDensityRO[P_ID].density;
#ifndef PREDICT_PRESSURE
	float3 P_velocity = ParticlesRO[P_ID].velocity;
#endif

#ifdef PREDICT_PRESSURE
	float P_pressure = ParticlesPressureFixRO[P_ID];
#else
#ifdef USE_PREVIOUS_PRESSURE
	float P_np = 0;//CalculatePressure(P_density);
	float P_pressure = 
		ParticlesRO[P_ID].pressure * 0.4f;
#else
    float P_pressure = CalculatePressure(P_density);
#endif
#endif
    
    const float h_sq = g_fSmoothlen * g_fSmoothlen;
    
    float3 acceleration = 0;
	float3 accelerationPressure = 0;
    
    // Calculate the acceleration based on neighbors from the 8 adjacent cells + current cell
    int3 G_XY = GridCalculateCell( P_position );
	for (int Z = max(G_XY.z - 1, 0) ; Z <= min(G_XY.z + 1, g_iGridDim.z-1) ; Z++)
	{
		for (int Y = max(G_XY.y - 1, 0) ; Y <= min(G_XY.y + 1, g_iGridDim.y-1) ; Y++)
		{
			for (int X = max(G_XY.x - 1, 0) ; X <= min(G_XY.x + 1, g_iGridDim.x-1) ; X++)
			{
				unsigned int G_CELL = GridConstuctKey(uint3(X, Y, Z));
				uint2 G_START_END = GridIndicesRO[G_CELL];
				for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
				{
					float3 N_position = ParticlesRO[N_ID].position;
                
					float3 diff = N_position - P_position;
					float r_sq = dot(diff, diff);
					if (r_sq < h_sq && P_ID != N_ID)
					{
						float N_density = ParticlesDensityRO[N_ID].density;
#ifndef PREDICT_PRESSURE
						float3 N_velocity = ParticlesRO[N_ID].velocity;
#endif
#ifdef PREDICT_PRESSURE
						float N_pressure = ParticlesPressureFixRO[N_ID];
#else
#ifdef USE_PREVIOUS_PRESSURE
						float N_np = 0;//CalculatePressure(N_density);
						float N_pressure = 
							ParticlesRO[N_ID].pressure * 0.45f;
#else
						float N_pressure = CalculatePressure(N_density);
#endif
#endif
						float r = sqrt(r_sq);

						// Pressure Term
						accelerationPressure += CalculateGradPressure(r, P_pressure, N_pressure, N_density, diff);
	#ifndef PREDICT_PRESSURE                    
						// Viscosity Term
						acceleration += CalculateLapVelocity(r, P_velocity, N_velocity, N_density);
	#endif
					}
				}
			}
		}
	}

	accelerationPressure /= P_density;
#ifndef PREDICT_PRESSURE  
	acceleration /= P_density;

	// Apply the forces from the map walls
	[unroll]
	for (unsigned int i = 0 ; i < 6 ; i++)
	{
		float dist = dot(float4(P_position, 1.0f), g_vPlanes[i]);
		acceleration += min(dist, 0) * -g_fWallStiffness * g_vPlanes[i].xyz;
	}

	// Apply gravity
	/*const float3 wCenter = {0.35f, 0.15f, 0.5f};
	float dwc = max(0.01f, distance(P_position, wCenter));
	float u = 0.005f / (dwc * dwc);

	acceleration += u * normalize(wCenter - P_position);*/
	acceleration += g_vGravity.xyz;

    ParticlesForcesRW[P_ID].acceleration = acceleration;

	ParticlesPressureFixRW[P_ID] = P_pressure;
#endif
	ParticlesPressureForcesRW[P_ID].acceleration = accelerationPressure;

}

//--------------------------------------------------------------------------------------
// Integration
//--------------------------------------------------------------------------------------

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void IntegrateCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
    const unsigned int P_ID = DTid.x; // Particle ID to operate on
    
    float3 position = ParticlesRO[P_ID].position;
    float3 velocity = ParticlesRO[P_ID].velocity;
    float3 acceleration = ParticlesForcesRO[P_ID].acceleration;
    float3 accelerationPressure = ParticlesPressureForcesRO[P_ID].acceleration;
    
//     // Integrate
	velocity += g_fTimeStep * (acceleration + accelerationPressure);
	position += g_fTimeStep * velocity;
    
    // Update
    ParticlesRW[P_ID].position = position;
    ParticlesRW[P_ID].velocity = velocity;
	ParticlesRW[P_ID].pressure = ParticlesPressureFixRO[P_ID];
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void XSPHCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	ParticlesSmoothedRW[DTid.x] = (ParticlesSmoothedSortedRO[DTid.x] + ParticlesSmoothedRO[DTid.x]) * 0.5f;
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void FieldCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	const unsigned int P_ID = DTid.x;

	if(P_ID >= (uint) g_iGridDot.w)
		return;

	int3 gpos = (int3) GridDecomposeKey(P_ID);

	const float h_sq = g_fSmoothlen * g_fSmoothlen * 4.0f;

	float3 P_position = (((float3) gpos * 0.33333333f + g_iGridMin.xyz) - 0.33333333f) * g_fSmoothlen;

	float density = 0;

	// Calculate the density based on neighbors from the 8 adjacent cells + current cell
	int3 G_XY = GridCalculateCell( P_position );

	for (int Z = max(G_XY.z - 1, 0) ; Z <= min(G_XY.z + 1, g_iGridDim.z-1) ; Z++)
	{
		for (int Y = max(G_XY.y - 1, 0) ; Y <= min(G_XY.y + 1, g_iGridDim.y-1) ; Y++)
		{
			for (int X = max(G_XY.x - 1, 0) ; X <= min(G_XY.x + 1, g_iGridDim.x-1) ; X++)
			{
				unsigned int G_CELL = GridConstuctKey(uint3(X, Y, Z));
				uint2 G_START_END = GridIndicesRO[G_CELL];
				for (unsigned int N_ID = G_START_END.x ; N_ID < G_START_END.y ; N_ID++)
				{
					float3 N_position = ParticlesSmoothedRO[N_ID];

					float f = ParticlesStretchRO[N_ID];

					float3 diff = N_position - P_position;
					float r_sq = dot(diff, diff) * f;

					if (r_sq < h_sq)
					{
						density += CalculateBSpline(r_sq);
					}
				}
			}
		}
	}

	DensityFieldRW[gpos.xzy + 1] = density;
}

[numthreads(1, 1, 1)]
void BoundingBoxCS_Grid( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	MINMAX iMM = BoundingBoxRO[0];

	uint3 iMin = (uint3) GridCalculateCell0(iMM.fmin);
	uint3 iMax = (uint3) GridCalculateCell0(iMM.fmax);

	uint3 iSize = iMax - iMin + 1;
	uint iLength = iSize.z * iSize.y * iSize.x;

	GridDimRW[0] = uint4(iMin, 0);
	GridDimRW[1] = uint4(iSize, iLength);
	GridDimRW[2] = uint4(iSize.x*iSize.z, iSize.x, (iSize.z * 3 + 3) * (iSize.y * 3 + 3) * (iSize.x * 3 + 3), (iSize.z * 3 + 2) * (iSize.y * 3 + 2) * (iSize.x * 3 + 2));
}