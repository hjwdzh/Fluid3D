
//--------------------------------------------------------------------------------------
// Reduction
//--------------------------------------------------------------------------------------
#define SIMULATION_BLOCK_SIZE 512

struct MINMAX {
	float3 fmin;
	float3 fmax;
};

groupshared MINMAX delta_shared[SIMULATION_BLOCK_SIZE];

void ReductionFunc(inout MINMAX a, MINMAX b) {
	a.fmin = min(a.fmin, b.fmin);
	a.fmax = max(a.fmax, b.fmax);
}

#define MIN(a, b) (a < b ? a : b)

void ReductionInternal( uint3 Gid, uint3 DTid, uint3 GTid, uint GI, RWStructuredBuffer<MINMAX> OutputBuffer ) 
{
	const unsigned int P_ID = DTid.x; // Particle ID to operate on
	const unsigned int G_ID = GTid.x;

	GroupMemoryBarrierWithGroupSync();

#if (SIMULATION_BLOCK_SIZE > 512)
	if(G_ID < 512) ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 512]);
	GroupMemoryBarrierWithGroupSync();
#endif

#if (SIMULATION_BLOCK_SIZE > 256)
	if(G_ID < 256) ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 256]);
	GroupMemoryBarrierWithGroupSync();
#endif

#if (SIMULATION_BLOCK_SIZE > 128)
	if(G_ID < 128) ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 128]);
	GroupMemoryBarrierWithGroupSync();
#endif

#if (SIMULATION_BLOCK_SIZE > 64)
	if(G_ID <  64) ReductionFunc(delta_shared[G_ID], delta_shared[G_ID +  64]);
	GroupMemoryBarrierWithGroupSync();
#endif

	if(G_ID < MIN(SIMULATION_BLOCK_SIZE / 2, 32))
	{
#if (SIMULATION_BLOCK_SIZE > 32)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 32]);
		GroupMemoryBarrier();
#endif
#if (SIMULATION_BLOCK_SIZE > 16)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 16]);
		GroupMemoryBarrier();
#endif
#if (SIMULATION_BLOCK_SIZE > 8)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 8]);
		GroupMemoryBarrier();
#endif
#if (SIMULATION_BLOCK_SIZE > 4)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 4]);
		GroupMemoryBarrier();
#endif
#if (SIMULATION_BLOCK_SIZE > 2)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 2]);
		GroupMemoryBarrier();
#endif
#if (SIMULATION_BLOCK_SIZE > 1)
		ReductionFunc(delta_shared[G_ID], delta_shared[G_ID + 1]);
#endif
	}

	if(G_ID == 0) {
		OutputBuffer[Gid.x] = delta_shared[0];
	}
}

struct Particle
{
    float3 position;
    float3 velocity;
	float pressure;
};

StructuredBuffer<Particle> ParticlesRO : register( t0 );
RWStructuredBuffer<MINMAX> BoundingBoxRW : register( u0 );
StructuredBuffer<MINMAX> BoundingBoxRO : register( t12 );
Buffer<uint> DispatchRO : register( t11 );

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BoundingBoxCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	uint i = DTid.x;
	uint numElements = DispatchRO[3];
	if(i >= numElements)
		i = numElements;

	float3 pos = ParticlesRO[i].position;
	delta_shared[GTid.x].fmin = delta_shared[GTid.x].fmax = pos;

	ReductionInternal(Gid, DTid, GTid, GI, BoundingBoxRW);
}

[numthreads(SIMULATION_BLOCK_SIZE, 1, 1)]
void BoundingBoxPingPongCS( uint3 Gid : SV_GroupID, uint3 DTid : SV_DispatchThreadID, uint3 GTid : SV_GroupThreadID, uint GI : SV_GroupIndex )
{
	uint i = DTid.x;
	uint numElements = DispatchRO[3];
	if(i >= numElements)
		i = numElements;

	delta_shared[GTid.x] = BoundingBoxRO[i];

	ReductionInternal(Gid, DTid, GTid, GI, BoundingBoxRW);
}