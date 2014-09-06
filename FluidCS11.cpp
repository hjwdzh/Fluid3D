//--------------------------------------------------------------------------------------
// File: FluidCS11.cpp
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

#include "DXUT.h"
#include "DXUTcamera.h"
#include "resource.h"
#include "WaitDlg.h"
#include "DirectXTK\DDSTextureLoader.h"
#ifndef TRY_CUDA
struct uint4
{
	UINT x, y, z, w;
};
#endif

struct Particle
{
    DirectX::XMFLOAT3 vPosition;
    DirectX::XMFLOAT3 vVelocity;
	FLOAT fPressure;
};

struct MINMAX
{
    DirectX::XMFLOAT3 fMin;
    DirectX::XMFLOAT3 fMax;	
};

struct Vertex
{
	DirectX::XMFLOAT3 vPosition;
	DirectX::XMFLOAT3 vNormal;
};

struct symMat
{
	DirectX::XMFLOAT4 c0;
	DirectX::XMFLOAT2 c1;
};

struct Triangle
{
	Vertex v[3];
};

struct ParticleDensity
{
    FLOAT fDensity;
};

struct ParticleForces
{
    DirectX::XMFLOAT3 vAcceleration;
};

struct UINT2
{
    UINT x;
    UINT y;
};

inline UINT iDivUp(UINT a, UINT b) {
	return a / b + ((a % b) != 0);
}

inline float sqrt3(float x) {
	return powf(x, 1.0f / 3.0f);
}
//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------

// Compute Shader Constants
// Grid cell key size for sorting, 8-bits for x and y
const UINT NUM_GRID_DIM_X = 512;
const UINT NUM_GRID_DIM_Y = 512;
const UINT NUM_GRID_DIM_Z = 512;
const UINT NUM_GRID_INDICES = NUM_GRID_DIM_X * NUM_GRID_DIM_Y * NUM_GRID_DIM_Z;

// Numthreads size for the simulation
const UINT SIMULATION_BLOCK_SIZE = 512;
const UINT MC_SIMULATION_BLOCK_SIZE = 128;
const UINT NEIGHBOR_SEARCH = (3*3*3);

// Numthreads size for the sort
const UINT BITONIC_BLOCK_SIZE = 512;
const UINT TRANSPOSE_BLOCK_SIZE = 16;

const UINT N_RELAX = 8;
const UINT GRIDSIZELOG2[3] = {6, 6, 6};

UINT nGridIndices = NUM_GRID_INDICES;
UINT nGridDimX = NUM_GRID_DIM_X;
UINT nGridDimY = NUM_GRID_DIM_Y;
UINT nGridDimZ = NUM_GRID_DIM_Z;

UINT nGridMinX = 0;
UINT nGridMinY = 0;
UINT nGridMinZ = 0;

// For this sample, only use power-of-2 numbers >= 8K and <= 64K
// The algorithm can be extended to support any number of particles
// But to keep the sample simple, we do not implement boundary conditions to handle it
const UINT NUM_PARTICLES_8K = 8 * 1024;
const UINT NUM_PARTICLES_16K = 16 * 1024;
const UINT NUM_PARTICLES_32K = 32 * 1024;
const UINT NUM_PARTICLES_64K = 64 * 1024;
const UINT NUM_PARTICLES_128K = 128 * 1024;
const UINT NUM_PARTICLES_256K = 256 * 1024;
const UINT NUM_PARTICLES_512K = 512 * 1024;
const UINT NUM_PARTICLES_1M = 1024 * 1024;
UINT g_iNumParticles = NUM_PARTICLES_64K;
FLOAT g_fIsoValue = 1.0f;

// Particle Properties
// These will control how the fluid behaves
FLOAT g_fInitialParticleSpacing = 0.0045f;
FLOAT g_fSmoothlen = 0.012f;
FLOAT g_fPressureStiffness = 2000.0f;
FLOAT g_fRestDensity = 1000.0f;
FLOAT g_fPressureGamma = 3.0f;
FLOAT g_fParticleMass = 0.0002f;
FLOAT g_fViscosity = 0.1f;
FLOAT g_fMaxAllowableTimeStep = 0.0075f;
FLOAT g_fParticleRenderSize = 0.00125f;
FLOAT g_fParticleAspectRatio = 1.0f;
FLOAT g_fDelta = 10.0f;

// Gravity Directions
const DirectX::XMFLOAT4A GRAVITY_DOWN(0, -0.5f, 0, 0);
const DirectX::XMFLOAT4A GRAVITY_UP(0, 0.5f, 0, 0);
const DirectX::XMFLOAT4A GRAVITY_LEFT(-0.5f, 0, 0, 0);
const DirectX::XMFLOAT4A GRAVITY_RIGHT(0.5f, 0, 0, 0);
DirectX::XMFLOAT4A g_vGravity = GRAVITY_DOWN;

// Map Size
// These values should not be larger than NUM_GRID_DIM * fSmoothlen
// Since the map must be divided up into fSmoothlen sized grid cells
// And the grid cell is used as a 16-bit sort key, 8-bits for x and y
FLOAT g_fMapHeight = 0.3f;
FLOAT g_fMapWidth = 0.5f;
FLOAT g_fMapLength = 1.0f;

BOOL g_bUpdateDelta = FALSE;
BOOL g_bRecalculatePressure = TRUE;
UINT g_iFrameCounter = 0;

// Map Wall Collision Planes
FLOAT g_fWallStiffness = 3000.0f;
DirectX::XMFLOAT4A g_vPlanes[6] = {
    DirectX::XMFLOAT4A(1, 0, 0, 0),
    DirectX::XMFLOAT4A(0, 1, 0, 0),
    DirectX::XMFLOAT4A(0, 0, 1, 0),
    DirectX::XMFLOAT4A(-1, 0, 0, g_fMapWidth),
    DirectX::XMFLOAT4A(0, -1, 0, g_fMapHeight),
    DirectX::XMFLOAT4A(0, 0, -1, g_fMapLength),
};

// Simulation Algorithm
enum eSimulationMode
{
    SIM_MODE_FULL,
    SIM_MODE_VISUALIZE,
    SIM_MODE_PARTICLE,
	SIM_MODE_CAUSTICONLY,
	SIM_MODE_GRAYSURF,
	SIM_MODE_TESSSURF,
	SIM_MODE_TESSSURFSMOOTHED,
	SIM_MODE_COUNT
};

eSimulationMode g_eSimMode = SIM_MODE_FULL;

const static LPCTSTR g_pModeHint[] = 
{
	L"Full Effects",
	L"Visualization",
	L"Particles",
	L"Caustic & Shadow Only",
	L"No Tessellation",
	L"Tessellated Only",
	L"Tessellated & Bilateral Smoothed",
	NULL
};

//--------------------------------------------------------------------------------------
// Direct3D11 Global variables
//--------------------------------------------------------------------------------------
ID3D11ShaderResourceView* const     g_pNullSRV = NULL;       // Helper to Clear SRVs
ID3D11UnorderedAccessView* const    g_pNullUAV = NULL;       // Helper to Clear UAVs
ID3D11Buffer* const                 g_pNullBuffer = NULL;    // Helper to Clear Buffers
UINT                                g_iNullUINT = 0;         // Helper to Clear Buffers
ID3D11SamplerState* const			g_pNullSampler = NULL;	 // Helper to Clear Sampler States


// Cameras
CFirstPersonCamera					g_Camera;
CFirstPersonCamera					g_CausticCamera;
CFirstPersonCamera*					g_Cameras[] =
{
	&g_Camera,
	&g_CausticCamera
};
UINT								g_iSelCamera = 0;

// Shaders
ID3D11VertexShader*                 g_pParticleVS = NULL;
ID3D11GeometryShader*               g_pParticleGS = NULL;
ID3D11PixelShader*                  g_pParticlePS = NULL;

ID3D11VertexShader*                 g_pBoundingBoxVS = NULL;
ID3D11PixelShader*                  g_pBoundingBoxPS = NULL;
ID3D11PixelShader*                  g_pRayMarchingPS = NULL;

ID3D11VertexShader*                 g_pSurfaceVS = NULL;
ID3D11VertexShader*                 g_pSurfaceNoTessVS = NULL;
ID3D11PixelShader*                  g_pSurfacePS = NULL;
ID3D11VertexShader*                 g_pPlaneVS = NULL;
ID3D11VertexShader*                 g_pPlaneVS1 = NULL;
ID3D11PixelShader*                  g_pPlanePS = NULL;
ID3D11VertexShader*                 g_pEnvVS = NULL;
ID3D11PixelShader*                  g_pEnvPS = NULL;
ID3D11PixelShader*                  g_pPlanePaintPS = NULL;

ID3D11VertexShader*                 g_pSimpleCausticVS = NULL;
ID3D11PixelShader*                  g_pSimpleCausticPS = NULL;
ID3D11GeometryShader*               g_pSimpleCausticGS = NULL;
ID3D11VertexShader*                 g_pSimpleShadowVS = NULL;
ID3D11PixelShader*                  g_pSimpleShadowPS = NULL;

ID3D11PixelShader*                  g_pBilateralXPS = NULL;
ID3D11PixelShader*                  g_pBilateralYPS = NULL;
ID3D11HullShader*					g_pSurfaceHS = NULL;
ID3D11DomainShader*                 g_pSurfaceDS = NULL;
ID3D11PixelShader*                  g_pColorPS = NULL;
ID3D11PixelShader*                  g_pGrayPS = NULL;
ID3D11PixelShader*                  g_pFXAAPS = NULL;
ID3D11PixelShader*                  g_pTonemapPS = NULL;

ID3D11VertexShader*                 g_pVisualizeVS = NULL;
ID3D11PixelShader*                  g_pVisualizePS = NULL;

ID3D11ComputeShader*                g_pBuildGridCS = NULL;
ID3D11ComputeShader*                g_pClearGridIndicesCS = NULL;
ID3D11ComputeShader*                g_pBuildGridIndicesCS = NULL;
ID3D11ComputeShader*                g_pRearrangeParticlesCS = NULL;
ID3D11ComputeShader*                g_pDensity_GridCS = NULL;
ID3D11ComputeShader*                g_pCov_GridCS = NULL;
ID3D11ComputeShader*                g_pField_GridCS = NULL;
ID3D11ComputeShader*                g_pXSPH_GridCS = NULL;
ID3D11ComputeShader*                g_pDensity_Grid_SmoothCS = NULL;
ID3D11ComputeShader*                g_pDensity_Grid_DeltaCS = NULL;
ID3D11ComputeShader*                g_pDensity_Grid_PressureCS = NULL;
ID3D11ComputeShader*                g_pForce_GridCS = NULL;
ID3D11ComputeShader*                g_pBoundingBox_GridCS = NULL;
ID3D11ComputeShader*                g_pForce_Grid_PreviousCS = NULL;
ID3D11ComputeShader*                g_pForce_Grid_PredictCS = NULL;
ID3D11ComputeShader*                g_pIntegrateCS = NULL;
ID3D11ComputeShader*                g_pBoundingBoxCS = NULL;
ID3D11ComputeShader*                g_pBoundingBoxPingPongCS = NULL;
ID3D11ComputeShader*				g_pMarchingCubeCS = NULL;
ID3D11ComputeShader*				g_pNumVertsAdjustCS = NULL;
ID3D11ComputeShader*				g_pNumCellsAdjustCS = NULL;
ID3D11ComputeShader*				g_pParticleInjectAdjustCS = NULL;
ID3D11ComputeShader*				g_pExpandCellCS = NULL;
ID3D11ComputeShader*				g_pPickCellCS = NULL;

ID3D11ComputeShader*                g_pSortBitonic = NULL;
ID3D11ComputeShader*                g_pSortTranspose = NULL;

ID3D11ComputeShader*                g_pSortBitonicUint = NULL;
ID3D11ComputeShader*                g_pSortTransposeUint = NULL;

// Structured Buffers
ID3D11Buffer*                       g_pParticles = NULL;
ID3D11ShaderResourceView*           g_pParticlesSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticlesUAV = NULL;

ID3D11Buffer*                       g_pParticlesSmoothed[3] = {NULL};
ID3D11ShaderResourceView*           g_pParticlesSmoothedSRV[3] = {NULL};
ID3D11UnorderedAccessView*          g_pParticlesSmoothedUAV[3] = {NULL};

BOOL								g_iNewPartGen = 0;

ID3D11Buffer*                       g_pBoundingBoxBuffer = NULL;
ID3D11ShaderResourceView*           g_pBoundingBoxBufferSRV = NULL;
ID3D11UnorderedAccessView*          g_pBoundingBoxBufferUAV = NULL;

ID3D11Buffer*                       g_pGridDimBuffer = NULL;
ID3D11ShaderResourceView*           g_pGridDimBufferSRV = NULL;
ID3D11UnorderedAccessView*          g_pGridDimBufferUAV = NULL;
#ifdef TRY_CUDA
cudaGraphicsResource_t				g_pGridDimBufferGR = NULL;
#endif

ID3D11Buffer*                       g_pBoundingBoxBufferPingPong = NULL;
ID3D11ShaderResourceView*           g_pBoundingBoxBufferPingPongSRV = NULL;
ID3D11UnorderedAccessView*          g_pBoundingBoxBufferPingPongUAV = NULL;

ID3D11Buffer*                       g_pSortedParticles = NULL;
ID3D11ShaderResourceView*           g_pSortedParticlesSRV = NULL;
ID3D11UnorderedAccessView*          g_pSortedParticlesUAV = NULL;

ID3D11Buffer*                       g_pParticleDensity = NULL;
ID3D11ShaderResourceView*           g_pParticleDensitySRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleDensityUAV = NULL;

ID3D11Buffer*                       g_pParticleDelta[2] = {NULL};
ID3D11ShaderResourceView*           g_pParticleDeltaSRV[2] = {NULL};
ID3D11UnorderedAccessView*          g_pParticleDeltaUAV[2] = {NULL};

ID3D11Buffer*                       g_pParticleForces = NULL;
ID3D11ShaderResourceView*           g_pParticleForcesSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleForcesUAV = NULL;

ID3D11Buffer*                       g_pParticleCovMat = NULL;
ID3D11ShaderResourceView*           g_pParticleCovMatSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleCovMatUAV = NULL;

ID3D11Buffer*                       g_pParticleAniso = NULL;
ID3D11ShaderResourceView*           g_pParticleAnisoSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleAnisoUAV = NULL;

ID3D11Buffer*                       g_pParticleAnisoSorted = NULL;
ID3D11ShaderResourceView*           g_pParticleAnisoSortedSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleAnisoSortedUAV = NULL;
#ifdef TRY_CUDA
cudaGraphicsResource_t				g_pParticleAnisoSortedGR = NULL;
#endif

ID3D11Buffer*                       g_pParticleAnisoSortedPingPong = NULL;
ID3D11ShaderResourceView*           g_pParticleAnisoSortedPingPongSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticleAnisoSortedPingPongUAV = NULL;

ID3D11Buffer*                       g_pParticlePressureForces = NULL;
ID3D11ShaderResourceView*           g_pParticlePressureForcesSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticlePressureForcesUAV = NULL;

ID3D11Buffer*                       g_pParticlePressure = NULL;
ID3D11ShaderResourceView*           g_pParticlePressureSRV = NULL;
ID3D11UnorderedAccessView*          g_pParticlePressureUAV = NULL;

ID3D11Buffer*                       g_pGrid = NULL;
ID3D11ShaderResourceView*           g_pGridSRV = NULL;
ID3D11UnorderedAccessView*          g_pGridUAV = NULL;
#ifdef TRY_CUDA
cudaGraphicsResource_t				g_pGridGR = NULL;
#endif

ID3D11Buffer*                       g_pGridPingPong = NULL;
ID3D11ShaderResourceView*           g_pGridPingPongSRV = NULL;
ID3D11UnorderedAccessView*          g_pGridPingPongUAV = NULL;

ID3D11Buffer*                       g_pGridIndices = NULL;
ID3D11ShaderResourceView*           g_pGridIndicesSRV = NULL;
ID3D11UnorderedAccessView*          g_pGridIndicesUAV = NULL;

ID3D11BlendState*					g_pBSAlpha = NULL;
ID3D11BlendState*					g_pBSAddition = NULL;
ID3D11RasterizerState*				g_pRSFrontCull = NULL;

// Marching Cube Tables
ID3D11Buffer*						g_pReconstructBuffer = NULL;
ID3D11ShaderResourceView*			g_pReconstructBufferSRV = NULL;
ID3D11UnorderedAccessView*			g_pReconstructBufferUAV = NULL;
ID3D11Buffer*						g_pDrawIndirectBuffer = NULL;
ID3D11Buffer*						g_pDrawIndirectBufferStaging = NULL;
ID3D11UnorderedAccessView*			g_pDrawIndirectBufferUAV = NULL;
ID3D11ShaderResourceView*			g_pDrawIndirectBufferSRV = NULL;
#ifdef TRY_CUDA
cudaGraphicsResource_t				g_pDrawIndirectBufferGR = NULL;
UINT*								g_pDrawIndirectMapped = NULL;
UINT*								g_pDrawIndirectMappedDevPtr = NULL;
#endif

// Density Field
ID3D11Texture3D*					g_pDensityField = NULL;
ID3D11UnorderedAccessView*			g_pDensityFieldUAV = NULL;
ID3D11ShaderResourceView*			g_pDensityFieldSRV = NULL;

ID3D11RasterizerState*				g_pRSNoCull = NULL;
ID3D11SamplerState*					g_pSSField = NULL;

// Env Map
ID3D11ShaderResourceView*			g_pEnvMapSRV = NULL;
ID3D11Texture2D*					g_pTexHDR = NULL;
ID3D11ShaderResourceView*			g_pTexHDRSRV = NULL;
ID3D11RenderTargetView*				g_pTexHDRRTV = NULL;
ID3D11Texture2D*					g_pTexPosition = NULL;
ID3D11ShaderResourceView*			g_pTexPositionSRV = NULL;
ID3D11RenderTargetView*				g_pTexPositionRTV = NULL;
ID3D11Texture2D*					g_pTexPositionBack = NULL;
ID3D11ShaderResourceView*			g_pTexPositionBackSRV = NULL;
ID3D11RenderTargetView*				g_pTexPositionBackRTV = NULL;
ID3D11Texture2D*					g_pTexNormal = NULL;
ID3D11ShaderResourceView*			g_pTexNormalSRV = NULL;
ID3D11RenderTargetView*				g_pTexNormalRTV = NULL;
ID3D11Texture2D*					g_pTexNormalBack = NULL;
ID3D11ShaderResourceView*			g_pTexNormalBackSRV = NULL;
ID3D11RenderTargetView*				g_pTexNormalBackRTV = NULL;

#define CAUSTIC_MAP_SIZE 4096
UINT g_iCMWidth = CAUSTIC_MAP_SIZE * g_fMapLength / g_fMapWidth;
UINT g_iCMHeight = CAUSTIC_MAP_SIZE;
ID3D11Texture2D*					g_pTexCaustic = NULL;
ID3D11ShaderResourceView*			g_pTexCausticSRV = NULL;
ID3D11RenderTargetView*				g_pTexCausticRTV = NULL;
ID3D11Texture2D*					g_pTexCausticFiltered = NULL;
ID3D11ShaderResourceView*			g_pTexCausticFilteredSRV = NULL;
ID3D11RenderTargetView*				g_pTexCausticFilteredRTV = NULL;

BOOL								g_bAdvance = TRUE;
BOOL								g_bRotating = FALSE;

#define _DECLSPEC_ALIGN_16_ __declspec(align(16))

// Constant Buffer Layout
#pragma warning(push)
#pragma warning(disable:4324) // structure was padded due to __declspec(align())
_DECLSPEC_ALIGN_16_ struct CBSimulationConstants
{
    UINT iNumParticles;
    FLOAT fTimeStep;
    FLOAT fSmoothlen;
    FLOAT fPressureStiffness;
	FLOAT fPressureGamma;
    FLOAT fRestDensity;
    FLOAT fDensityCoef;
    FLOAT fDeltaCoef;
    FLOAT fBeta;
	FLOAT fDelta;
    FLOAT fGradPressureCoef;
    FLOAT fLapViscosityCoef;
    FLOAT fWallStiffness;
    
    DirectX::XMFLOAT4A vGravity;
    DirectX::XMFLOAT4A vGridDim;
	DirectX::XMFLOAT4A vGridDim2;
	DirectX::XMFLOAT4A vGridDim3;
	DirectX::XMFLOAT4A vGridDim4;

    DirectX::XMFLOAT4A vPlanes[6];
};

_DECLSPEC_ALIGN_16_ struct CBRenderConstants
{
    DirectX::XMFLOAT4X4 mViewProjection[2];
    DirectX::XMFLOAT4X4 mView;
	DirectX::XMFLOAT4A iVolSlicing;
	DirectX::XMFLOAT4A fTessFactor;
	DirectX::XMFLOAT4A fEyePos;
	DirectX::XMFLOAT4A fRMAssist;
	DirectX::XMFLOAT4A fRMAssist2;
	FLOAT fSmoothlen;
    FLOAT fParticleSize;
	FLOAT fParticleAspectRatio;
	FLOAT fInvGridSize;
};

_DECLSPEC_ALIGN_16_ struct SortCB
{
    UINT iLevel;
    UINT iLevelMask;
    UINT iWidth;
    UINT iHeight;
};

_DECLSPEC_ALIGN_16_ struct CBMarchingCubesConstants
{
	UINT gridSize[4];
	UINT gridSizeShift[4];
	UINT gridSizeMask[4];
	DirectX::XMFLOAT4A voxelSize;
};

_DECLSPEC_ALIGN_16_ struct CBGridDimConstants
{
	uint4 gridMin;
	uint4 gridDim;
	uint4 gridDot;
};
#pragma warning(pop)

// Constant Buffers
ID3D11Buffer*                       g_pcbSimulationConstants = NULL;
ID3D11Buffer*                       g_pcbRenderConstants = NULL;
ID3D11Buffer*                       g_pSortCB = NULL;
ID3D11Buffer*						g_pcbMarchingCubesConstants = NULL;
ID3D11Buffer*						g_pcbGridDimConstants = NULL;

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );

bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext );

HRESULT CreateSimulationBuffers( ID3D11Device* pd3dDevice );
void InitApp();
void RenderText();
void UpdateMCParas( ID3D11DeviceContext* pd3dImmediateContext );
void UpdateRenderParas( ID3D11DeviceContext* pd3dImmediateContext );

// For Marching Cubes
HRESULT CreateMCBuffers( ID3D11Device* pd3dDevice );
void DestroyMCBuffers();

template<class T>
VOID CheckBuffer(ID3D11Buffer* pBuffer) {
#ifdef _DEBUG
	DXUTGetD3D11DeviceContext()->Flush();
	D3D11_BUFFER_DESC bufdesc;
	pBuffer->GetDesc(&bufdesc);
	bufdesc.BindFlags = 0;
	bufdesc.Usage = D3D11_USAGE_STAGING;
	bufdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
	bufdesc.StructureByteStride = 0;
	bufdesc.MiscFlags = 0;

	ID3D11Buffer* pStag;
	DXUTGetD3D11Device()->CreateBuffer(&bufdesc, NULL, &pStag);
	DXUTGetD3D11DeviceContext()->CopyResource(pStag, pBuffer);
	D3D11_MAPPED_SUBRESOURCE ms;
	DXUTGetD3D11DeviceContext()->Map(pStag, 0, D3D11_MAP_READ_WRITE, 0, &ms);

	T* pData = (T*)ms.pData;

	DXUTGetD3D11DeviceContext()->Unmap(pStag, 0);
	pStag->Release();
#endif
}
//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{

    // DXUT will create and use the best device (either D3D10 or D3D11) 
    // that is available on the system depending on which D3D callbacks are set below
	AllocConsole();
	AttachConsole(GetCurrentProcessId());
	freopen("CONIN$", "r+t", stdin);
	freopen("CONOUT$", "w+t", stdout);

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackFrameMove( OnFrameMove );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

	POINT iSelRes[] = {
		{640, 360},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		{3200, 1800},
		{3840, 2160}
	};

	POINT iScr = {
		GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)
	};

	int i = 0;
	for(; i < ARRAYSIZE(iSelRes) && iScr.x > iSelRes[i].x && iScr.y > iSelRes[i].y; i++);
	i = max(0, i - 1);

    InitApp();
    DXUTInit( true, true ); // Parse the command line, show msgboxes on error, and an extra cmd line param to force REF for now
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
	DXUTCreateWindow( L"PCISPH 3D" );
	DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, iSelRes[i].x, iSelRes[i].y );
    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    // Initialize dialogs
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{

	// Update the camera's position based on user input 
	if(g_bRotating) {
		DirectX::XMVECTOR vLookAt = DirectX::XMVectorSet(g_fMapWidth * 0.5f, g_fMapHeight * 0.5f, g_fMapLength * 0.5f, 1.0f);
		DirectX::XMVECTOR vEyePt = *g_Camera.GetEyePt();
		DirectX::XMVECTOR vDir = DirectX::XMVectorSubtract(vEyePt, vLookAt);
		DirectX::XMMATRIX mRot = DirectX::XMMatrixRotationY(fElapsedTime * 0.05f);
		vDir = DirectX::XMVector3TransformCoord(vDir, mRot);
		vEyePt = DirectX::XMVectorAdd(vLookAt, vDir);

		g_Camera.SetViewParams(vEyePt, vLookAt);
		g_Camera.FrameMove( fElapsedTime );
	} else {
		g_Camera.FrameMove( fElapsedTime );
	}
	g_CausticCamera.FrameMove( fElapsedTime );
}

//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
#ifdef _DEBUG
	pDeviceSettings->d3d11.CreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

    return true;
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{

}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
	// Pass all remaining windows messages to camera so it can respond to user input

	g_Cameras[g_iSelCamera]->HandleMessages( hWnd, uMsg, wParam, lParam );

	TCHAR chBuf[250];

	switch (uMsg)
	{
	case WM_KEYDOWN:
		switch(wParam) {
		case 'C':
			printf("%f %f %f %f %f %f\n", 
				g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[0],
				g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[1],
				g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[2],
				g_Cameras[g_iSelCamera]->GetLookAtPt()->m128_f32[0],
				g_Cameras[g_iSelCamera]->GetLookAtPt()->m128_f32[1],
				g_Cameras[g_iSelCamera]->GetLookAtPt()->m128_f32[2]);
			break;
		case 'N':
			g_iSelCamera = (g_iSelCamera + 1) % ARRAYSIZE(g_Cameras);
			break;
		case 'O':
			g_bAdvance = !g_bAdvance;
			break;
		case 'R':
			g_bRotating = !g_bRotating;
			break;
		case 'M':
			g_eSimMode = (eSimulationMode)((g_eSimMode + 1) % SIM_MODE_COUNT);
			swprintf_s<250>(chBuf, L"PCISPH 3D - %s", g_pModeHint[g_eSimMode]);
			SetWindowText(DXUTGetHWND(), chBuf);
			break;
		}
	break;
	default:
		break;
	}

    return 0;
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    if ( DeviceInfo->ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x == FALSE )
        return false;

    return true;
}

//--------------------------------------------------------------------------------------
// Helper for compiling shaders with D3DX11
//--------------------------------------------------------------------------------------
HRESULT CompileShaderFromFile( WCHAR* szFileName, LPCSTR szEntryPoint, LPCSTR szShaderModel, ID3DBlob** ppBlobOut, const D3D_SHADER_MACRO* pDefines = NULL )
{
    HRESULT hr = S_OK;

    // find the file
	WCHAR* str = szFileName;

    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_WARNINGS_ARE_ERRORS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3DCOMPILE_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    // compile the shader
    ID3DBlob* pErrorBlob = NULL;
    hr = D3DCompileFromFile( str, pDefines, NULL, szEntryPoint, szShaderModel, dwShaderFlags, 0, ppBlobOut, &pErrorBlob );
    if( FAILED(hr) )
    {
        if( pErrorBlob != NULL )
            MessageBoxA(0, (char*)pErrorBlob->GetBufferPointer(), 0, 0 );
        SAFE_RELEASE( pErrorBlob );
        return hr;
    }
    SAFE_RELEASE( pErrorBlob );

    return S_OK;   
}


//--------------------------------------------------------------------------------------
// Helper for creating constant buffers
//--------------------------------------------------------------------------------------
template <class T>
HRESULT CreateConstantBuffer(ID3D11Device* pd3dDevice, ID3D11Buffer** ppCB)
{
    HRESULT hr = S_OK;

    D3D11_BUFFER_DESC Desc;
    Desc.Usage = D3D11_USAGE_DEFAULT;
    Desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    Desc.CPUAccessFlags = 0;
    Desc.MiscFlags = 0;
    Desc.ByteWidth = sizeof( T );
    V_RETURN( pd3dDevice->CreateBuffer( &Desc, NULL, ppCB ) );

    return hr;
}

//--------------------------------------------------------------------------------------
// Helper for creating structured buffers with an SRV and UAV
//--------------------------------------------------------------------------------------
template <class T>
HRESULT CreateStructuredBuffer(ID3D11Device* pd3dDevice, UINT iNumElements, ID3D11Buffer** ppBuffer, 
	ID3D11ShaderResourceView** ppSRV, ID3D11UnorderedAccessView** ppUAV, const T* pInitialData = NULL, 
	const BOOL bAppend = FALSE
#ifdef TRY_CUDA
	, cudaGraphicsResource_t* ppGR = NULL
#endif
	)
{
    HRESULT hr = S_OK;

    // Create SB
	if(ppBuffer) {
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = iNumElements * sizeof(T);
		bufferDesc.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bufferDesc.StructureByteStride = sizeof(T);

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer ) );
	}

    // Create SRV
	if(ppBuffer && ppSRV) {
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		ZeroMemory( &srvDesc, sizeof(srvDesc) );
		srvDesc.Format = DXGI_FORMAT_UNKNOWN;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		srvDesc.Buffer.ElementWidth = iNumElements;
		V_RETURN( pd3dDevice->CreateShaderResourceView( *ppBuffer, &srvDesc, ppSRV ) );
	}

    // Create UAV
	if(ppBuffer && ppUAV) {
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = DXGI_FORMAT_UNKNOWN;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = iNumElements;
		if(bAppend) uavDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_APPEND;
		V_RETURN( pd3dDevice->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV ) );
	}

#ifdef TRY_CUDA
	// Create Graphics Resource
	if(DXUTIsCUDAvailable() && ppGR) {
		V_CUDA(cudaGraphicsD3D11RegisterResource(ppGR, *ppBuffer, 0));
	}
#endif

    return hr;
}

HRESULT CreateTypedBuffer(ID3D11Device* pd3dDevice, DXGI_FORMAT format, DXGI_FORMAT formatUAV, UINT iStride, UINT iNumElements, UINT iNumElementsUAV, ID3D11Buffer** ppBuffer, 
	ID3D11ShaderResourceView** ppSRV, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData = NULL
#ifdef TRY_CUDA
	, cudaGraphicsResource_t* ppGR = NULL
#endif
	)
{
    HRESULT hr = S_OK;

    // Create SB
	if(ppBuffer) {
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = iNumElements * iStride;
		bufferDesc.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = 0;
		bufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer ) );
	}

    // Create SRV
	if(ppBuffer && ppSRV) {
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		ZeroMemory( &srvDesc, sizeof(srvDesc) );
		srvDesc.Format = format;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		srvDesc.Buffer.ElementWidth = iNumElements;
		V_RETURN( pd3dDevice->CreateShaderResourceView( *ppBuffer, &srvDesc, ppSRV ) );
	}

    // Create UAV
	if(ppBuffer && ppUAV) {
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = formatUAV;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = iNumElementsUAV;
		V_RETURN( pd3dDevice->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV ) );
	}

#ifdef TRY_CUDA
	// Create Graphics Resource
	if(DXUTIsCUDAvailable() && ppGR) {
		V_CUDA(cudaGraphicsD3D11RegisterResource(ppGR, *ppBuffer, 0));
	}
#endif

    return hr;
}


HRESULT CreateTypedTexture2D(ID3D11Device* pd3dDevice, DXGI_FORMAT format, DXGI_FORMAT formatUAV, UINT iWidth, UINT iHeight, ID3D11Texture2D** ppTex, 
	ID3D11ShaderResourceView** ppSRV, ID3D11RenderTargetView** ppRTV, ID3D11UnorderedAccessView** ppUAV, const void* pInitialData = NULL, UINT iDepth = 1
#ifdef TRY_CUDA
	, cudaGraphicsResource_t* ppGR = NULL
#endif
	)
{
    HRESULT hr = S_OK;

    // Create SB
	if(ppTex) {
		D3D11_TEXTURE2D_DESC texDesc;
		ZeroMemory( &texDesc, sizeof(texDesc) );
		texDesc.ArraySize = iDepth;
		texDesc.CPUAccessFlags = 0;
		texDesc.Format = format;
		texDesc.Height = iHeight;
		texDesc.MipLevels = 1;
		texDesc.SampleDesc.Count = 1;
		texDesc.SampleDesc.Quality = 0;
		texDesc.Width = iWidth;
		texDesc.Usage = D3D11_USAGE_DEFAULT;
		texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | (ppUAV ? D3D11_BIND_UNORDERED_ACCESS : 0);
		texDesc.MiscFlags = 0;

		D3D11_SUBRESOURCE_DATA texInitData;
		ZeroMemory( &texInitData, sizeof(texInitData) );
		texInitData.pSysMem = pInitialData;
		V_RETURN( pd3dDevice->CreateTexture2D( &texDesc, (pInitialData)? &texInitData : NULL, ppTex ) );
	}

    // Create SRV
	if(ppTex && ppSRV) {
		V_RETURN( pd3dDevice->CreateShaderResourceView( *ppTex, NULL, ppSRV ) );
	}

    // Create UAV
	if(ppUAV) {
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = formatUAV;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
		uavDesc.Texture2D.MipSlice = 0;
		V_RETURN( pd3dDevice->CreateUnorderedAccessView( *ppTex, &uavDesc, ppUAV ) );
	}
	
	if(ppRTV) {
		V_RETURN( pd3dDevice->CreateRenderTargetView( *ppTex, NULL, ppRTV ) );		
	}

#ifdef TRY_CUDA
	// Create Graphics Resource
	if(DXUTIsCUDAvailable() && ppGR) {
		V_CUDA(cudaGraphicsD3D11RegisterResource(ppGR, *ppTex, 0));
	}
#endif

    return hr;
}


HRESULT CreateDrawIndirectBuffer(ID3D11Device* pd3dDevice, UINT nElements, ID3D11Buffer** ppBuffer, ID3D11Buffer** ppBufferStaging, ID3D11UnorderedAccessView** ppUAV, ID3D11ShaderResourceView** ppSRV = NULL, const void* pInitialData = NULL, 
	const BOOL bIndexed = FALSE
#ifdef TRY_CUDA
	, cudaGraphicsResource_t* ppGR = NULL
#endif
	)
{
	HRESULT hr = S_OK;

	// Create SB
	if(ppBuffer) {
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = nElements * sizeof(UINT);
		bufferDesc.Usage = D3D11_USAGE_DEFAULT;
		bufferDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
		bufferDesc.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
		bufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBuffer ) );
	}

	if(ppSRV) {
		D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
		srvDesc.Buffer.FirstElement = 0;
		srvDesc.Buffer.NumElements = nElements;
		srvDesc.Format = DXGI_FORMAT_R32_UINT;
		srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
		V_RETURN( pd3dDevice->CreateShaderResourceView(*ppBuffer, &srvDesc, ppSRV) );
	}

	if(ppBufferStaging) {
		D3D11_BUFFER_DESC bufferDesc;
		ZeroMemory( &bufferDesc, sizeof(bufferDesc) );
		bufferDesc.ByteWidth = sizeof(UINT) * nElements;
		bufferDesc.Usage = D3D11_USAGE_STAGING;
		bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		bufferDesc.BindFlags = 0;
		bufferDesc.MiscFlags = 0;
		bufferDesc.StructureByteStride = 0;

		D3D11_SUBRESOURCE_DATA bufferInitData;
		ZeroMemory( &bufferInitData, sizeof(bufferInitData) );
		bufferInitData.pSysMem = pInitialData;
		V_RETURN( pd3dDevice->CreateBuffer( &bufferDesc, (pInitialData)? &bufferInitData : NULL, ppBufferStaging ) );
	}

	// Create UAV
	if(ppBuffer && ppUAV) {
		D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc;
		ZeroMemory( &uavDesc, sizeof(uavDesc) );
		uavDesc.Format = DXGI_FORMAT_R32_UINT;
		uavDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavDesc.Buffer.NumElements = nElements;
		V_RETURN( pd3dDevice->CreateUnorderedAccessView( *ppBuffer, &uavDesc, ppUAV ) );
	}

#ifdef TRY_CUDA
	// Create Graphics Resource
	if(DXUTIsCUDAvailable() && ppGR) {
		V_CUDA(cudaGraphicsD3D11RegisterResource(ppGR, *ppBuffer, 0));
	}
#endif

	return hr;
}

//--------------------------------------------------------------------------------------
// Create the buffers used for the simulation data
//--------------------------------------------------------------------------------------
HRESULT CreateSimulationBuffers( ID3D11Device* pd3dDevice )
{
    HRESULT hr = S_OK;

    // Destroy the old buffers in case the number of particles has changed
    SAFE_RELEASE( g_pParticles );
    SAFE_RELEASE( g_pParticlesSRV );
    SAFE_RELEASE( g_pParticlesUAV );

	SAFE_RELEASE( g_pBoundingBoxBuffer );
	SAFE_RELEASE( g_pBoundingBoxBufferSRV );
	SAFE_RELEASE( g_pBoundingBoxBufferUAV );

	SAFE_RELEASE( g_pBoundingBoxBufferPingPong );
	SAFE_RELEASE( g_pBoundingBoxBufferPingPongSRV );
	SAFE_RELEASE( g_pBoundingBoxBufferPingPongUAV );
    
    SAFE_RELEASE( g_pSortedParticles );
    SAFE_RELEASE( g_pSortedParticlesSRV );
    SAFE_RELEASE( g_pSortedParticlesUAV );

    SAFE_RELEASE( g_pParticleForces );
    SAFE_RELEASE( g_pParticleForcesSRV );
    SAFE_RELEASE( g_pParticleForcesUAV );

	SAFE_RELEASE( g_pParticlePressureForces );
	SAFE_RELEASE( g_pParticlePressureForcesSRV );
	SAFE_RELEASE( g_pParticlePressureForcesUAV );

	SAFE_RELEASE( g_pParticlePressure );
	SAFE_RELEASE( g_pParticlePressureSRV );
	SAFE_RELEASE( g_pParticlePressureUAV );
    
    SAFE_RELEASE( g_pParticleDensity );
    SAFE_RELEASE( g_pParticleDensitySRV );
    SAFE_RELEASE( g_pParticleDensityUAV );

    SAFE_RELEASE( g_pGridSRV );
    SAFE_RELEASE( g_pGridUAV );
    SAFE_RELEASE( g_pGrid );

    SAFE_RELEASE( g_pGridPingPongSRV );
    SAFE_RELEASE( g_pGridPingPongUAV );
    SAFE_RELEASE( g_pGridPingPong );

    SAFE_RELEASE( g_pGridIndicesSRV );
    SAFE_RELEASE( g_pGridIndicesUAV );
    SAFE_RELEASE( g_pGridIndices );

	SAFE_RELEASE( g_pParticleDelta[0] );
	SAFE_RELEASE( g_pParticleDeltaSRV[0] );
	SAFE_RELEASE( g_pParticleDeltaUAV[0] );

	SAFE_RELEASE( g_pParticleDelta[1] );
	SAFE_RELEASE( g_pParticleDeltaSRV[1] );
	SAFE_RELEASE( g_pParticleDeltaUAV[1] );

	SAFE_RELEASE( g_pDensityField );
	SAFE_RELEASE( g_pDensityFieldSRV );
	SAFE_RELEASE( g_pDensityFieldUAV );

	SAFE_RELEASE( g_pParticleCovMat );
	SAFE_RELEASE( g_pParticleCovMatSRV );
	SAFE_RELEASE( g_pParticleCovMatUAV );

	SAFE_RELEASE( g_pParticleAniso );
	SAFE_RELEASE( g_pParticleAnisoSRV );
	SAFE_RELEASE( g_pParticleAnisoUAV );

	DestroyMCBuffers();

    // Create the initial particle positions
    // This is only used to populate the GPU buffers on creation
	float StartX = g_fMapWidth * 0.25f;
	float StartY = 0;//g_fMapHeight * 0.1f;
	float StartZ = g_fMapLength * 0.25f;
	float volSingle = g_fParticleMass / g_fRestDensity;
	g_fInitialParticleSpacing = sqrt3(volSingle);
    const UINT iStartingWidth = (UINT)sqrt3( (FLOAT)g_iNumParticles );
	const UINT iSW2 = iStartingWidth * iStartingWidth;
    Particle* particles = new Particle[ g_iNumParticles ];
    ZeroMemory( particles, sizeof(Particle) * g_iNumParticles );
    for ( UINT i = 0 ; i < g_iNumParticles ; i++ )
    {
		UINT s = i % iSW2;
        // Arrange the particles in a nice cube
        UINT x = s % iStartingWidth;
        UINT y = s / iStartingWidth;
		UINT z = i / iSW2;
        particles[ i ].vPosition = DirectX::XMFLOAT3( g_fInitialParticleSpacing * (FLOAT)x + StartX, g_fInitialParticleSpacing * (FLOAT)y + StartY,  g_fInitialParticleSpacing * (FLOAT)z + StartZ);
    }

    // Create Structured Buffers
    V_RETURN( CreateStructuredBuffer< Particle >( pd3dDevice, g_iNumParticles, &g_pParticles, &g_pParticlesSRV, &g_pParticlesUAV, particles ) );
    DXUT_SetDebugName( g_pParticles, "Particles" );
    DXUT_SetDebugName( g_pParticlesSRV, "Particles SRV" );
    DXUT_SetDebugName( g_pParticlesUAV, "Particles UAV" );

	    // Create Structured Buffers
	for( UINT i = 0; i < 3; i++ ) {
		V_RETURN( CreateStructuredBuffer< DirectX::XMFLOAT3 >( pd3dDevice, g_iNumParticles, g_pParticlesSmoothed + i, g_pParticlesSmoothedSRV + i, g_pParticlesSmoothedUAV + i ) );
		DXUT_SetDebugName( g_pParticlesSmoothed[i], "Particles" );
		DXUT_SetDebugName( g_pParticlesSmoothedSRV[i], "Particles SRV" );
		DXUT_SetDebugName( g_pParticlesSmoothedUAV[i], "Particles UAV" );
	}

	V_RETURN( CreateStructuredBuffer< MINMAX >( pd3dDevice, g_iNumParticles, &g_pBoundingBoxBuffer, &g_pBoundingBoxBufferSRV, &g_pBoundingBoxBufferUAV, NULL ) );
	DXUT_SetDebugName( g_pBoundingBoxBuffer,    "Particles Bounding Box" );
	DXUT_SetDebugName( g_pBoundingBoxBufferSRV, "Particles Bounding Box SRV" );
	DXUT_SetDebugName( g_pBoundingBoxBufferUAV, "Particles Bounding Box UAV" );

	V_RETURN( CreateStructuredBuffer< MINMAX >( pd3dDevice, g_iNumParticles, &g_pBoundingBoxBufferPingPong, &g_pBoundingBoxBufferPingPongSRV, &g_pBoundingBoxBufferPingPongUAV, NULL ) );
	DXUT_SetDebugName( g_pBoundingBoxBufferPingPong,    "Particles Bounding Box" );
	DXUT_SetDebugName( g_pBoundingBoxBufferPingPongSRV, "Particles Bounding Box SRV" );
	DXUT_SetDebugName( g_pBoundingBoxBufferPingPongUAV, "Particles Bounding Box UAV" );

    V_RETURN( CreateStructuredBuffer< Particle >( pd3dDevice, g_iNumParticles, &g_pSortedParticles, &g_pSortedParticlesSRV, &g_pSortedParticlesUAV, particles ) );
    DXUT_SetDebugName( g_pSortedParticles, "Sorted" );
    DXUT_SetDebugName( g_pSortedParticlesSRV, "Sorted SRV" );
    DXUT_SetDebugName( g_pSortedParticlesUAV, "Sorted UAV" );

    V_RETURN( CreateStructuredBuffer< ParticleForces >( pd3dDevice, g_iNumParticles, &g_pParticleForces, &g_pParticleForcesSRV, &g_pParticleForcesUAV ) );
    DXUT_SetDebugName( g_pParticleForces, "Forces" );
    DXUT_SetDebugName( g_pParticleForcesSRV, "Forces SRV" );
    DXUT_SetDebugName( g_pParticleForcesUAV, "Forces UAV" );

	V_RETURN( CreateStructuredBuffer< ParticleForces >( pd3dDevice, g_iNumParticles, &g_pParticlePressureForces, &g_pParticlePressureForcesSRV, &g_pParticlePressureForcesUAV ) );
	DXUT_SetDebugName( g_pParticlePressureForces, "Pressure Forces" );
	DXUT_SetDebugName( g_pParticlePressureForcesSRV, "Pressure Forces SRV" );
	DXUT_SetDebugName( g_pParticlePressureForcesUAV, "Pressure Forces UAV" );

    V_RETURN( CreateStructuredBuffer< ParticleDensity >( pd3dDevice, g_iNumParticles, &g_pParticleDensity, &g_pParticleDensitySRV, &g_pParticleDensityUAV ) );
    DXUT_SetDebugName( g_pParticleDensity, "Density" );
    DXUT_SetDebugName( g_pParticleDensitySRV, "Density SRV" );
    DXUT_SetDebugName( g_pParticleDensityUAV, "Density UAV" );

	V_RETURN( CreateStructuredBuffer< FLOAT >( pd3dDevice, g_iNumParticles, &g_pParticleCovMat, &g_pParticleCovMatSRV, &g_pParticleCovMatUAV ) );
	DXUT_SetDebugName( g_pParticleCovMat, "Stretch" );
	DXUT_SetDebugName( g_pParticleCovMatSRV, "Stretch SRV" );
	DXUT_SetDebugName( g_pParticleCovMatUAV, "Stretch UAV" );

	V_RETURN( CreateStructuredBuffer< FLOAT >( pd3dDevice, g_iNumParticles, &g_pParticlePressure, &g_pParticlePressureSRV, &g_pParticlePressureUAV ) );
	DXUT_SetDebugName( g_pParticlePressure, "Pressure" );
	DXUT_SetDebugName( g_pParticlePressureSRV, "Pressure SRV" );
	DXUT_SetDebugName( g_pParticlePressureUAV, "Pressure UAV" );

	hr = CreateTypedBuffer( pd3dDevice, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32_UINT, sizeof(UINT) * 2, g_iNumParticles, g_iNumParticles * 2, &g_pGrid, &g_pGridSRV, &g_pGridUAV
#ifdef TRY_CUDA
	,0, &g_pGridGR
#endif
	);
	V_RETURN(hr);
    DXUT_SetDebugName( g_pGrid, "Grid" );
    DXUT_SetDebugName( g_pGridSRV, "Grid SRV" );
    DXUT_SetDebugName( g_pGridUAV, "Grid UAV" );

	hr = CreateTypedBuffer( pd3dDevice, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_UINT, sizeof(uint4), sizeof(CBGridDimConstants) / sizeof(uint4), sizeof(CBGridDimConstants) / sizeof(uint4), &g_pGridDimBuffer, &g_pGridDimBufferSRV, &g_pGridDimBufferUAV
#ifdef TRY_CUDA
	,0, &g_pGridDimBufferGR
#endif
	);
	V_RETURN(hr);
    DXUT_SetDebugName( g_pGridDimBuffer, "Grid" );
    DXUT_SetDebugName( g_pGridDimBufferSRV, "Grid SRV" );
    DXUT_SetDebugName( g_pGridDimBufferUAV, "Grid UAV" );

	CreateConstantBuffer<CBGridDimConstants>( pd3dDevice, &g_pcbGridDimConstants );

    V_RETURN( CreateTypedBuffer( pd3dDevice, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32_UINT, sizeof(UINT) * 2, g_iNumParticles, g_iNumParticles * 2, &g_pGridPingPong, &g_pGridPingPongSRV, &g_pGridPingPongUAV ) );
    DXUT_SetDebugName( g_pGridPingPong, "PingPong" );
    DXUT_SetDebugName( g_pGridPingPongSRV, "PingPong SRV" );
    DXUT_SetDebugName( g_pGridPingPongUAV, "PingPong UAV" );

#ifdef TRY_CUDA
	if(!DXUTIsCUDAvailable()) {
#endif
		V_RETURN( CreateStructuredBuffer< UINT2 >( pd3dDevice, NUM_GRID_INDICES, &g_pGridIndices, &g_pGridIndicesSRV, &g_pGridIndicesUAV ) );
		DXUT_SetDebugName( g_pGridIndices, "Indices" );
		DXUT_SetDebugName( g_pGridIndicesSRV, "Indices SRV" );
		DXUT_SetDebugName( g_pGridIndicesUAV, "Indices UAV" );
#ifdef TRY_CUDA
	}
#endif

    delete[] particles;

	D3D11_TEXTURE3D_DESC t3desc;
	t3desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
	t3desc.CPUAccessFlags = 0;
	t3desc.Depth = 1 << GRIDSIZELOG2[2];
	t3desc.Format = DXGI_FORMAT_R32_FLOAT;
	t3desc.Height = 1 << GRIDSIZELOG2[1];
	t3desc.MipLevels = 1;//min(min(GRIDSIZELOG2[0], GRIDSIZELOG2[1]), GRIDSIZELOG2[2]);
	t3desc.MiscFlags = 0;//D3D11_RESOURCE_MISC_GENERATE_MIPS;
	t3desc.Usage = D3D11_USAGE_DEFAULT;
	t3desc.Width = 1 << GRIDSIZELOG2[0];
	V_RETURN(pd3dDevice->CreateTexture3D(&t3desc, NULL, &g_pDensityField));
	V_RETURN(pd3dDevice->CreateUnorderedAccessView(g_pDensityField, NULL, &g_pDensityFieldUAV));
	V_RETURN(pd3dDevice->CreateShaderResourceView(g_pDensityField, NULL, &g_pDensityFieldSRV));

	V_RETURN(CreateMCBuffers(pd3dDevice));

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
    HRESULT hr;

    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

    // Compile the Shaders
    ID3DBlob* pBlob = NULL;

    // Rendering Shaders
    V_RETURN( CompileShaderFromFile( L"ParticleRender.hlsl", "ParticleVS", "vs_4_0", &pBlob ) );
    V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pParticleVS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticleVS, "ParticleVS" );

    V_RETURN( CompileShaderFromFile( L"ParticleRender.hlsl", "ParticleGS", "gs_4_0", &pBlob ) );
    V_RETURN( pd3dDevice->CreateGeometryShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pParticleGS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticleGS, "ParticleGS" );

    V_RETURN( CompileShaderFromFile( L"ParticleRender.hlsl", "ParticlePS", "ps_4_0", &pBlob ) );
    V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pParticlePS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pParticlePS, "ParticlePS" );

	V_RETURN( CompileShaderFromFile( L"VolumeVisualizer.hlsl", "VisualizeVS", "vs_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pVisualizeVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pVisualizeVS, "VisualizeVS" );

	V_RETURN( CompileShaderFromFile( L"VolumeVisualizer.hlsl", "VisualizePS", "ps_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pVisualizePS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pVisualizePS, "VisualizePS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SurfaceNoTessVS", "vs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSurfaceNoTessVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSurfaceNoTessVS, "SurfaceNoTessVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SurfaceVS", "vs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSurfaceVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSurfaceVS, "SurfaceVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SurfacePS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSurfacePS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSurfacePS, "SurfacePS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "PlaneVS", "vs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPlaneVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pPlaneVS, "PlaneVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "PlaneVS1", "vs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPlaneVS1 ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pPlaneVS1, "PlaneVS1" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "PlanePS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPlanePS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pPlanePS, "PlanePS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "EnvVS", "vs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pEnvVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pEnvVS, "EnvVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "EnvPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pEnvPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pEnvPS, "EnvPS" );


	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "PlanePaintPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pPlanePaintPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pPlanePaintPS, "PlanePaintPS" );

	
	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SimpleCausticVS", "vs_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSimpleCausticVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSimpleCausticVS, "SimpleCausticVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SimpleShadowVS", "vs_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateVertexShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSimpleShadowVS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSimpleShadowVS, "SimpleShadowVS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SimpleCausticGS", "gs_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateGeometryShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSimpleCausticGS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSimpleCausticGS, "SimpleCausticGS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SimpleCausticPS", "ps_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSimpleCausticPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSimpleCausticPS, "SimpleCausticPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SimpleShadowPS", "ps_4_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSimpleShadowPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSimpleShadowPS, "SimpleShadowPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "ColorPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pColorPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pColorPS, "ColorPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "GrayPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pGrayPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pGrayPS, "GrayPS" );

	V_RETURN( CompileShaderFromFile( L"fxaa.hlsl", "FXAAPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pFXAAPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pFXAAPS, "FXAAPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "TonemapPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pTonemapPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pTonemapPS, "TonemapPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "BilateralXPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBilateralXPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pBilateralXPS, "BilateralXPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "BilateralYPS", "ps_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreatePixelShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBilateralYPS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pBilateralYPS, "BilateralYPS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SurfaceHS", "hs_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateHullShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSurfaceHS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSurfaceHS, "SurfaceHS" );

	V_RETURN( CompileShaderFromFile( L"SurfaceRender.hlsl", "SurfaceDS", "ds_5_0", &pBlob ) );
	V_RETURN( pd3dDevice->CreateDomainShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSurfaceDS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pSurfaceDS, "SurfaceDS" );

    // Compute Shaders
    const char* CSTarget = (pd3dDevice->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0)? "cs_5_0" : "cs_4_0";
    
    CWaitDlg CompilingShadersDlg;
    CompilingShadersDlg.ShowDialog( L"Compiling Shaders..." );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "IntegrateCS", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pIntegrateCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pIntegrateCS, "IntegrateCS" );

	D3D_SHADER_MACRO pDeltaMacro[] = 
	{
		"UPDATE_DELTA", "1",
		NULL, NULL
	};

	D3D_SHADER_MACRO pSmoothMacro[] = 
	{
		"LAPLACIAN_SMOOTH", "1",
		NULL, NULL
	};

	D3D_SHADER_MACRO pPressureMacro[] = 
	{
		"PRESSURE_FIX", "1",
		NULL, NULL
	};

	D3D_SHADER_MACRO pPredictMacro[] = 
	{
		"PREDICT_PRESSURE", "1",
		NULL, NULL
	};

	D3D_SHADER_MACRO pPreviousMacro[] = 
	{
		"USE_PREVIOUS_PRESSURE", "1",
		NULL, NULL
	};

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "DensityCS_Grid", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pDensity_GridCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pDensity_GridCS, "DensityCS_Grid" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "DensityCS_Grid", CSTarget, &pBlob, pSmoothMacro ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pDensity_Grid_SmoothCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pDensity_Grid_SmoothCS, "DensityCS_Grid_Smooth" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "DensityCS_Grid", CSTarget, &pBlob, pDeltaMacro ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pDensity_Grid_DeltaCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pDensity_Grid_DeltaCS, "DensityCS_Grid_Delta" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "DensityCS_Grid", CSTarget, &pBlob, pPressureMacro ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pDensity_Grid_PressureCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pDensity_Grid_PressureCS, "DensityCS_Grid_Pressure" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "CovCS_Grid", CSTarget, &pBlob) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pCov_GridCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pCov_GridCS, "CovCS_Smooth" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "FieldCS_Grid", CSTarget, &pBlob) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pField_GridCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pField_GridCS, "FieldCS_Grid" );
	
	V_RETURN( CompileShaderFromFile( L"Reduction.hlsl", "BoundingBoxCS", CSTarget, &pBlob ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBoundingBoxCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pBoundingBoxCS, "BoundingBoxCS" );

	V_RETURN( CompileShaderFromFile( L"Reduction.hlsl", "BoundingBoxPingPongCS", CSTarget, &pBlob ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBoundingBoxPingPongCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pBoundingBoxPingPongCS, "BoundingBoxPingPongCS" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "ForceCS_Grid", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pForce_GridCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pForce_GridCS, "ForceCS_Grid" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "XSPHCS_Grid", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pXSPH_GridCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pXSPH_GridCS, "XSPHCS_Grid" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "BoundingBoxCS_Grid", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBoundingBox_GridCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pBoundingBox_GridCS, "BoundingBoxCS_Grid" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "ForceCS_Grid", CSTarget, &pBlob, pPreviousMacro ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pForce_Grid_PreviousCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pForce_Grid_PreviousCS, "ForceCS_Grid_Previous" );

	V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "ForceCS_Grid", CSTarget, &pBlob, pPredictMacro ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pForce_Grid_PredictCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pForce_Grid_PredictCS, "ForceCS_Grid_Predict" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "BuildGridCS", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBuildGridCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pBuildGridCS, "BuildGridCS" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "ClearGridIndicesCS", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pClearGridIndicesCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pClearGridIndicesCS, "ClearGridIndicesCS" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "BuildGridIndicesCS", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pBuildGridIndicesCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pBuildGridIndicesCS, "BuildGridIndicesCS" );

    V_RETURN( CompileShaderFromFile( L"FluidCS11.hlsl", "RearrangeParticlesCS", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pRearrangeParticlesCS ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pRearrangeParticlesCS, "RearrangeParticlesCS" );

    // Sort Shaders
    V_RETURN( CompileShaderFromFile( L"ComputeShaderSort11.hlsl", "BitonicSort", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSortBitonic ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pSortBitonic, "BitonicSort" );

    V_RETURN( CompileShaderFromFile( L"ComputeShaderSort11.hlsl", "MatrixTranspose", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSortTranspose ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pSortTranspose, "MatrixTranspose" );

	V_RETURN( CompileShaderFromFile( L"ComputeShaderSortUint11.hlsl", "BitonicSort", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSortBitonicUint ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pSortBitonicUint, "BitonicSortUint" );

    V_RETURN( CompileShaderFromFile( L"ComputeShaderSortUint11.hlsl", "MatrixTranspose", CSTarget, &pBlob ) );
    V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pSortTransposeUint ) );
    SAFE_RELEASE( pBlob );
    DXUT_SetDebugName( g_pSortTransposeUint, "MatrixTransposeUint" );

	//Marching Cubes Shaders
	V_RETURN( CompileShaderFromFile( L"MarchingCubes.hlsl", "MarchingCubesCS", CSTarget, &pBlob ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pMarchingCubeCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pMarchingCubeCS, "MarchingCubesCS" );

	V_RETURN( CompileShaderFromFile( L"MarchingCubes.hlsl", "AdjustNumVertsCS", CSTarget, &pBlob ) );
	V_RETURN( pd3dDevice->CreateComputeShader( pBlob->GetBufferPointer(), pBlob->GetBufferSize(), NULL, &g_pNumVertsAdjustCS ) );
	SAFE_RELEASE( pBlob );
	DXUT_SetDebugName( g_pNumVertsAdjustCS, "AdjustNumVertsCS" );

    CompilingShadersDlg.DestroyDialog();

    // Create the Simulation Buffers
    V_RETURN( CreateSimulationBuffers( pd3dDevice ) );

    // Create Constant Buffers
    V_RETURN( CreateConstantBuffer< CBSimulationConstants >( pd3dDevice, &g_pcbSimulationConstants ) );
    V_RETURN( CreateConstantBuffer< CBRenderConstants >( pd3dDevice, &g_pcbRenderConstants ) );
    V_RETURN( CreateConstantBuffer< CBMarchingCubesConstants >( pd3dDevice, &g_pcbMarchingCubesConstants ) );
    V_RETURN( CreateConstantBuffer< SortCB >( pd3dDevice, &g_pSortCB ) );

    DXUT_SetDebugName( g_pcbSimulationConstants, "Simluation" );
    DXUT_SetDebugName( g_pcbRenderConstants, "Render" );
    DXUT_SetDebugName( g_pcbMarchingCubesConstants, "MarchingCubes" );
    DXUT_SetDebugName( g_pSortCB, "Sort" );


	//Initialize Camera
	FLOAT fCamH = 2.4142f * g_fMapWidth + 0.05f;
	DirectX::XMVECTOR vecEye = { g_fMapWidth * 0.5f, fCamH, g_fMapLength * 0.5f, 0 };
	DirectX::XMVECTOR vecAt = { g_fMapWidth * 0.5f + 0.00001f, fCamH - 1.0f,  g_fMapLength * 0.5f, 0 };
	DirectX::XMVECTOR vecEye0 = { -0.385594,0.442761,1.474109, 0 };
	DirectX::XMVECTOR vecAt0 = {  0.182296, -0.097536, 0.853158, 0 };
	for(UINT i = 0; i < ARRAYSIZE(g_Cameras); i++) {
		g_Cameras[i]->SetViewParams(vecEye, vecAt);
		g_Cameras[i]->SetRotateButtons(true, false, false);
		g_Cameras[i]->SetScalers(0.01f, 1.0f);
	}

	g_Camera.SetViewParams(vecEye0, vecAt0);

	//Initialize BS
	D3D11_BLEND_DESC bdesc;
	ZeroMemory(&bdesc, sizeof(D3D11_BLEND_DESC));
	bdesc.RenderTarget[0].BlendEnable = TRUE;
	bdesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
	bdesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
	bdesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE;
	bdesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
	bdesc.RenderTarget[0].RenderTargetWriteMask = 0xf;
	bdesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
	bdesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
	V_RETURN( pd3dDevice->CreateBlendState(&bdesc, &g_pBSAddition) );

	bdesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
	bdesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
	V_RETURN( pd3dDevice->CreateBlendState(&bdesc, &g_pBSAlpha) );

/*	D3D11_BLEND_DESC1 bdesc1;
	ZeroMemory(&bdesc1, sizeof(D3D11_BLEND_DESC1));
	bdesc1.RenderTarget[0].LogicOpEnable = TRUE;
	bdesc1.RenderTarget[0].LogicOp = D3D11_LOGIC_OP_XOR;
	bdesc1.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;*/
//	V_RETURN( ((ID3D11Device1*)pd3dDevice)->CreateBlendState1(&bdesc1, &g_pBSXor));


	D3D11_RASTERIZER_DESC rsdesc;
	ZeroMemory(&rsdesc, sizeof(D3D11_RASTERIZER_DESC));
	rsdesc.AntialiasedLineEnable = TRUE;
	rsdesc.CullMode = D3D11_CULL_NONE;
	rsdesc.DepthBias = 0;
	rsdesc.DepthBiasClamp = 0;
	rsdesc.DepthClipEnable = TRUE;
	rsdesc.FillMode = D3D11_FILL_SOLID;
	rsdesc.FrontCounterClockwise = FALSE;
	rsdesc.MultisampleEnable = FALSE;
	rsdesc.ScissorEnable = FALSE;
	rsdesc.SlopeScaledDepthBias = 0;
	V_RETURN( pd3dDevice->CreateRasterizerState(&rsdesc, &g_pRSNoCull) );

	rsdesc.CullMode = D3D11_CULL_FRONT;
	V_RETURN( pd3dDevice->CreateRasterizerState(&rsdesc, &g_pRSFrontCull) );

	D3D11_SAMPLER_DESC samDesc;
	samDesc.AddressU = D3D11_TEXTURE_ADDRESS_BORDER;
	samDesc.AddressV = D3D11_TEXTURE_ADDRESS_BORDER;
	samDesc.AddressW = D3D11_TEXTURE_ADDRESS_BORDER;
	samDesc.BorderColor[0] = samDesc.BorderColor[1] = samDesc.BorderColor[2] = samDesc.BorderColor[3] = 0;
	samDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
	samDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
	samDesc.MaxAnisotropy = 1;
	samDesc.MaxLOD = FLT_MAX;
	samDesc.MinLOD = -FLT_MAX;
	samDesc.MipLODBias = 0;
	V_RETURN( pd3dDevice->CreateSamplerState(&samDesc, &g_pSSField) );
	pd3dImmediateContext->CSSetSamplers(0, 1, &g_pSSField);
	pd3dImmediateContext->PSSetSamplers(0, 1, &g_pSSField);
	pd3dImmediateContext->VSSetSamplers(0, 1, &g_pSSField);

	V_RETURN(DirectX::CreateDDSTextureFromFile(pd3dDevice, 
		L"Media\\uffizi_cross.dds", NULL, &g_pEnvMapSRV));

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

	float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
	g_fParticleAspectRatio = fAspectRatio;

	g_Camera.SetProjParams( DirectX::XM_PIDIV4, fAspectRatio, 0.01f, 100.0f );
	g_CausticCamera.SetProjParams( DirectX::XM_PIDIV4, g_fMapLength / g_fMapWidth, 0.01f, 20.0f );

	SAFE_RELEASE(g_pTexPosition);
	SAFE_RELEASE(g_pTexPositionSRV);
	SAFE_RELEASE(g_pTexPositionRTV);

	SAFE_RELEASE(g_pTexPositionBack);
	SAFE_RELEASE(g_pTexPositionBackSRV);
	SAFE_RELEASE(g_pTexPositionBackRTV);

	SAFE_RELEASE(g_pTexNormal);
	SAFE_RELEASE(g_pTexNormalSRV);
	SAFE_RELEASE(g_pTexNormalRTV);

	SAFE_RELEASE(g_pTexNormalBack);
	SAFE_RELEASE(g_pTexNormalBackSRV);
	SAFE_RELEASE(g_pTexNormalBackRTV);

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, &g_pTexPosition, &g_pTexPositionSRV, &g_pTexPositionRTV, NULL) );
	DXUT_SetDebugName( g_pTexPosition, "TEX_POSITION" );
	DXUT_SetDebugName( g_pTexPositionSRV, "TEX_POSITION_SRV" );
	DXUT_SetDebugName( g_pTexPositionRTV, "TEX_POSITION_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, &g_pTexHDR, &g_pTexHDRSRV, &g_pTexHDRRTV, NULL) );
	DXUT_SetDebugName( g_pTexHDR, "TEX_HDR" );
	DXUT_SetDebugName( g_pTexHDRSRV, "TEX_HDR_SRV" );
	DXUT_SetDebugName( g_pTexHDRRTV, "TEX_HDR_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, &g_pTexPositionBack, &g_pTexPositionBackSRV, &g_pTexPositionBackRTV, NULL) );
	DXUT_SetDebugName( g_pTexPositionBack, "TEX_POSITION_BACK" );
	DXUT_SetDebugName( g_pTexPositionBackSRV, "TEX_POSITION_BACK_SRV" );
	DXUT_SetDebugName( g_pTexPositionBackRTV, "TEX_POSITION_BACK_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, &g_pTexNormal, &g_pTexNormalSRV, &g_pTexNormalRTV, NULL) );
	DXUT_SetDebugName( g_pTexNormal, "TEX_Normal" );
	DXUT_SetDebugName( g_pTexNormalSRV, "TEX_Normal_SRV" );
	DXUT_SetDebugName( g_pTexNormalRTV, "TEX_Normal_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, &g_pTexNormalBack, &g_pTexNormalBackSRV, &g_pTexNormalBackRTV, NULL) );
	DXUT_SetDebugName( g_pTexNormalBack, "TEX_Normal_BACK" );
	DXUT_SetDebugName( g_pTexNormalBackSRV, "TEX_Normal_BACK_SRV" );
	DXUT_SetDebugName( g_pTexNormalBackRTV, "TEX_Normal_BACK_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	g_iCMWidth, g_iCMHeight, &g_pTexCaustic, &g_pTexCausticSRV, &g_pTexCausticRTV, NULL) );
	DXUT_SetDebugName( g_pTexCaustic,    "TEX_Caustic" );
	DXUT_SetDebugName( g_pTexCausticSRV, "TEX_Caustic_SRV" );
	DXUT_SetDebugName( g_pTexCausticRTV, "TEX_Caustic_RTV" );

	V_RETURN( CreateTypedTexture2D( 
	pd3dDevice, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 
	g_iCMWidth, g_iCMHeight, &g_pTexCausticFiltered, &g_pTexCausticFilteredSRV, &g_pTexCausticFilteredRTV, NULL) );
	DXUT_SetDebugName( g_pTexCausticFiltered,    "TEX_CausticFiltered" );
	DXUT_SetDebugName( g_pTexCausticFilteredSRV, "TEX_CausticFiltered_SRV" );
	DXUT_SetDebugName( g_pTexCausticFilteredRTV, "TEX_CausticFiltered_RTV" );
    return S_OK;
}

void GPUBoundingBox(ID3D11DeviceContext* pd3dImmediateContext, ID3D11ShaderResourceView* inInitialSRV,
	ID3D11UnorderedAccessView* inUAV, ID3D11ShaderResourceView* inSRV,
	ID3D11UnorderedAccessView* tempUAV, ID3D11ShaderResourceView* tempSRV, UINT numElements) 
{
	


	ID3D11UnorderedAccessView* pUAVs[] = {
		inUAV,
		tempUAV
	};

	ID3D11ShaderResourceView* pSRVs[] = {
		inSRV,
		tempSRV
	};

	UINT nblock0 = numElements;
	UINT b = 1;
	ID3D11ShaderResourceView* pSRVClr = NULL;

	pd3dImmediateContext->CSSetShaderResources( 11, 1, &g_pDrawIndirectBufferSRV ); 

	{
		nblock0 = iDivUp(nblock0, SIMULATION_BLOCK_SIZE);
		UINT pDispatch[4] = {nblock0, 1, 1, numElements};
		pd3dImmediateContext->UpdateSubresource(g_pDrawIndirectBuffer, 0, NULL, pDispatch, 0, 0);
		pd3dImmediateContext->CSSetShaderResources( 12, 1, &pSRVClr );
		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pUAVs[b], NULL );
		pd3dImmediateContext->CSSetShaderResources( 0, 1, &inInitialSRV );
		pd3dImmediateContext->CSSetShader( g_pBoundingBoxCS, NULL, 0 );
		pd3dImmediateContext->DispatchIndirect(g_pDrawIndirectBuffer, 0);
	}

	while(nblock0 > 1)
	{
		b = !b;
		numElements = nblock0;
		nblock0 = iDivUp(nblock0, SIMULATION_BLOCK_SIZE);
		UINT pDispatch[4] = {nblock0, 1, 1, numElements};
		pd3dImmediateContext->UpdateSubresource(g_pDrawIndirectBuffer, 0, NULL, pDispatch, 0, 0);
		pd3dImmediateContext->CSSetShaderResources( 12, 1, &pSRVClr );
		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pUAVs[b], NULL );
		pd3dImmediateContext->CSSetShaderResources( 12, 1, &pSRVs[!b] );
		pd3dImmediateContext->CSSetShader( g_pBoundingBoxPingPongCS, NULL, 0 );
		pd3dImmediateContext->Dispatch( nblock0, 1, 1 );
	}

	pd3dImmediateContext->CSSetShaderResources( 11, 1, &g_pNullSRV ); 

	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pGridDimBufferUAV, NULL );
	pd3dImmediateContext->CSSetShaderResources( 12, 1, &pSRVs[b] );
	pd3dImmediateContext->CSSetShader(g_pBoundingBox_GridCS, NULL, 0);
	pd3dImmediateContext->Dispatch( 1, 1, 1 );

	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pNullUAV, NULL );
	pd3dImmediateContext->CopyResource(g_pcbGridDimConstants, g_pGridDimBuffer);
	pd3dImmediateContext->CSSetConstantBuffers(4, 1, &g_pcbGridDimConstants);

#ifdef TRY_CUDA
	if(DXUTIsCUDAvailable()) 
	{
		cudaGraphicsMapResources(1, &g_pGridDimBufferGR);

		UINT* devPtr;
		size_t s;
		cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &s, g_pGridDimBufferGR);

		cudaMemcpyAsync(g_pDrawIndirectMappedDevPtr, devPtr, sizeof(CBGridDimConstants), cudaMemcpyDeviceToDevice);

		cudaGraphicsUnmapResources(1, &g_pGridDimBufferGR);
	}
#else
	pd3dImmediateContext->CopySubresourceRegion(g_pDrawIndirectBufferStaging, 0, 0, 0, 0, g_pGridDimBuffer, 0, NULL);
#endif
}

FLOAT ReadFloat(ID3D11DeviceContext* pd3dImmediateContext, ID3D11Buffer* pBuffer) 
{

	D3D11_BUFFER_DESC bufdesc;
	pBuffer->GetDesc(&bufdesc);
	bufdesc.BindFlags = 0;
	bufdesc.Usage = D3D11_USAGE_STAGING;
	bufdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
	bufdesc.StructureByteStride = 0;
	bufdesc.MiscFlags = 0;

	ID3D11Buffer* pStag;
	DXUTGetD3D11Device()->CreateBuffer(&bufdesc, NULL, &pStag);
	DXUTGetD3D11DeviceContext()->CopyResource(pStag, pBuffer);
	D3D11_MAPPED_SUBRESOURCE ms;
	DXUTGetD3D11DeviceContext()->Map(pStag, 0, D3D11_MAP_READ_WRITE, 0, &ms);

	FLOAT re = *(FLOAT*)ms.pData;

	DXUTGetD3D11DeviceContext()->Unmap(pStag, 0);
	pStag->Release();

	return re;
}

#ifdef TRY_CUDA
HRESULT GPUSort(UINT nElements, BOOL isDouble, cudaGraphicsResource_t inGR) {
	HRESULT hr = S_OK;

	V_CUDA(cudaGraphicsMapResources(1, &inGR));

	void* devPtr = NULL;
	size_t s;
	V_CUDA(cudaGraphicsResourceGetMappedPointer(&devPtr, &s, inGR));

	if(!devPtr)
		MessageBoxA(0, "No devPtr!", 0, 0);

	if(isDouble) {
		GPUSortThrustDouble(devPtr, nElements);
	} else {
		GPUSortThrustUint(devPtr, nElements);
	}

	V_CUDA(cudaGraphicsUnmapResources(1, &inGR));

	return hr;
}

#endif

//--------------------------------------------------------------------------------------
// GPU Bitonic Sort
// For more information, please see the ComputeShaderSort11 sample
//--------------------------------------------------------------------------------------
HRESULT GPUSort(ID3D11DeviceContext* pd3dImmediateContext, UINT nElements, BOOL isDouble,
             ID3D11UnorderedAccessView* inUAV, ID3D11ShaderResourceView* inSRV,
             ID3D11UnorderedAccessView* tempUAV, ID3D11ShaderResourceView* tempSRV)
{
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pSortCB );

	const UINT NUM_ELEMENTS = nElements;
    const UINT MATRIX_WIDTH = BITONIC_BLOCK_SIZE;
    const UINT MATRIX_HEIGHT = NUM_ELEMENTS / BITONIC_BLOCK_SIZE;

    // Sort the data
    // First sort the rows for the levels <= to the block size
    for( UINT level = 2 ; level <= BITONIC_BLOCK_SIZE ; level <<= 1 )
    {
        SortCB constants = { level, level, MATRIX_HEIGHT, MATRIX_WIDTH };
        pd3dImmediateContext->UpdateSubresource( g_pSortCB, 0, NULL, &constants, 0, 0 );

        // Sort the row data
        UINT UAVInitialCounts = 0;
        pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &inUAV, &UAVInitialCounts );
		pd3dImmediateContext->CSSetShader( isDouble ? g_pSortBitonic : g_pSortBitonicUint, NULL, 0 );
        pd3dImmediateContext->Dispatch( NUM_ELEMENTS / BITONIC_BLOCK_SIZE, 1, 1 );
    }

    // Then sort the rows and columns for the levels > than the block size
    // Transpose. Sort the Columns. Transpose. Sort the Rows.

	for( UINT level = (BITONIC_BLOCK_SIZE << 1) ; level <= NUM_ELEMENTS ; level <<= 1 )
    {
        SortCB constants1 = { (level / BITONIC_BLOCK_SIZE), (level & ~NUM_ELEMENTS) / BITONIC_BLOCK_SIZE, MATRIX_WIDTH, MATRIX_HEIGHT };
        pd3dImmediateContext->UpdateSubresource( g_pSortCB, 0, NULL, &constants1, 0, 0 );

        // Transpose the data from buffer 1 into buffer 2
        ID3D11ShaderResourceView* pViewNULL = NULL;
        UINT UAVInitialCounts = 0;
        pd3dImmediateContext->CSSetShaderResources( 0, 1, &pViewNULL );
        pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &tempUAV, &UAVInitialCounts );
        pd3dImmediateContext->CSSetShaderResources( 0, 1, &inSRV );
		pd3dImmediateContext->CSSetShader( isDouble ? g_pSortTranspose : g_pSortTransposeUint, NULL, 0 );
        pd3dImmediateContext->Dispatch( MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE, MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE, 1 );

        // Sort the transposed column data
        pd3dImmediateContext->CSSetShader( isDouble ? g_pSortBitonic : g_pSortBitonicUint, NULL, 0 );
        pd3dImmediateContext->Dispatch( NUM_ELEMENTS / BITONIC_BLOCK_SIZE, 1, 1 );

        SortCB constants2 = { BITONIC_BLOCK_SIZE, level, MATRIX_HEIGHT, MATRIX_WIDTH };
        pd3dImmediateContext->UpdateSubresource( g_pSortCB, 0, NULL, &constants2, 0, 0 );

        // Transpose the data from buffer 2 back into buffer 1
        pd3dImmediateContext->CSSetShaderResources( 0, 1, &pViewNULL );
        pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &inUAV, &UAVInitialCounts );
        pd3dImmediateContext->CSSetShaderResources( 0, 1, &tempSRV );
        pd3dImmediateContext->CSSetShader( isDouble ? g_pSortTranspose : g_pSortTransposeUint, NULL, 0 );
        pd3dImmediateContext->Dispatch( MATRIX_HEIGHT / TRANSPOSE_BLOCK_SIZE, MATRIX_WIDTH / TRANSPOSE_BLOCK_SIZE, 1 );

        // Sort the row data
        pd3dImmediateContext->CSSetShader( isDouble ? g_pSortBitonic : g_pSortBitonicUint, NULL, 0 );
        pd3dImmediateContext->Dispatch( NUM_ELEMENTS / BITONIC_BLOCK_SIZE, 1, 1 );
    }
	return S_OK;
}

//--------------------------------------------------------------------------------------
// GPU Fluid Simulation - Optimized Algorithm using a Grid + Sort
// Algorithm Overview:
//    Build Grid: For every particle, calculate a hash based on the grid cell it is in
//    Sort Grid: Sort all of the particles based on the grid ID hash
//        Particles in the same cell will now be adjacent in memory
//    Build Grid Indices: Located the start and end offsets for each cell
//    Rearrange: Rearrange the particles into the same order as the grid for easy lookup
//    Density, Force, Integrate: Perform the normal fluid simulation algorithm
//        Except now, only calculate particles from the 8 adjacent cells + current cell
//--------------------------------------------------------------------------------------
void SimulateFluid_Grid( ID3D11DeviceContext* pd3dImmediateContext )
{
    UINT UAVInitialCounts = 0;

	SAFE_RELEASE(g_pDensityField);
	SAFE_RELEASE(g_pDensityFieldUAV);
	SAFE_RELEASE(g_pDensityFieldSRV);

    // Setup
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbSimulationConstants );

	GPUBoundingBox(
		pd3dImmediateContext, 
		g_pParticlesSRV, 
		g_pBoundingBoxBufferUAV, 
		g_pBoundingBoxBufferSRV, 
		g_pBoundingBoxBufferPingPongUAV, 
		g_pBoundingBoxBufferPingPongSRV, 
		g_iNumParticles);

    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pGridUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pParticlesSRV );

    // Build Grid
    pd3dImmediateContext->CSSetShader( g_pBuildGridCS, NULL, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

#ifdef TRY_CUDA
	if(DXUTIsCUDAvailable())
		GPUSort(g_iNumParticles, TRUE, g_pGridGR);
	else
#endif
    // Sort Grid
    GPUSort(pd3dImmediateContext, g_iNumParticles, TRUE, g_pGridUAV, g_pGridSRV, g_pGridPingPongUAV, g_pGridPingPongSRV);

	SAFE_RELEASE(g_pGridIndices);
	SAFE_RELEASE(g_pGridIndicesSRV);
	SAFE_RELEASE(g_pGridIndicesUAV);
#ifdef TRY_CUDA
	if(DXUTIsCUDAvailable()) {
		cudaDeviceSynchronize();
		nGridIndices = g_pDrawIndirectMapped[7];
		nGridMinX = g_pDrawIndirectMapped[0];
		nGridMinY = g_pDrawIndirectMapped[1];
		nGridMinZ = g_pDrawIndirectMapped[2];
		nGridDimX = g_pDrawIndirectMapped[4];
		nGridDimY = g_pDrawIndirectMapped[5];
		nGridDimZ = g_pDrawIndirectMapped[6];
	}
#else
	D3D11_MAPPED_SUBRESOURCE ms;
	pd3dImmediateContext->Map(g_pDrawIndirectBufferStaging, 0, D3D11_MAP_READ, 0, &ms);
	UINT* pData = (UINT*)ms.pData;
	nGridIndices = pData[7];
	nGridMinX = pData[0];
	nGridMinY = pData[1];
	nGridMinZ = pData[2];
	nGridDimX = pData[4];
	nGridDimY = pData[5];
	nGridDimZ = pData[6];
	pd3dImmediateContext->Unmap(g_pDrawIndirectBufferStaging, 0);
#endif
	CreateStructuredBuffer< UINT2 >( DXUTGetD3D11Device(), nGridIndices, &g_pGridIndices, &g_pGridIndicesSRV, &g_pGridIndicesUAV );

    // Setup
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbSimulationConstants );
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pGridIndicesUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 3, 1, &g_pGridSRV );

    // Build Grid Indices
	UINT pUINTClr[4] = {0};
	pd3dImmediateContext->ClearUnorderedAccessViewUint(g_pGridIndicesUAV, pUINTClr);
    pd3dImmediateContext->CSSetShader( g_pBuildGridIndicesCS, NULL, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

    // Setup
    // Rearrange
	ID3D11UnorderedAccessView* pUAVs00[] = {
		g_pSortedParticlesUAV,
		g_pParticlesSmoothedUAV[2]
	};

    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 2, pUAVs00, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pParticlesSRV );
    pd3dImmediateContext->CSSetShaderResources( 3, 1, &g_pGridSRV );
	pd3dImmediateContext->CSSetShaderResources( 6, 1, &g_pParticlesSmoothedSRV[!g_iNewPartGen] );
	pd3dImmediateContext->CSSetShader( g_pRearrangeParticlesCS, NULL, 0 );
    pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );
	pd3dImmediateContext->CSSetShaderResources( 6, 1, &g_pNullSRV );

    // Setup
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pNullUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1, &g_pNullUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
    pd3dImmediateContext->CSSetShaderResources( 3, 1, &g_pGridSRV );
    pd3dImmediateContext->CSSetShaderResources( 4, 1, &g_pGridIndicesSRV );

	ID3D11UnorderedAccessView* pUAVs0[] = {
		g_pParticleDensityUAV,
		g_pParticlesSmoothedUAV[g_iNewPartGen]
	};

	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 2, pUAVs0, &UAVInitialCounts );
	pd3dImmediateContext->CSSetShader( g_pDensity_Grid_SmoothCS, NULL, 0 );
	pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

	if(g_iFrameCounter == 0) {
		pd3dImmediateContext->CopyResource(g_pParticlesSmoothed[!g_iNewPartGen], g_pParticlesSmoothed[g_iNewPartGen]);
	} else {
		ID3D11UnorderedAccessView* pUAVs005[] = {
			NULL,
			g_pParticlesSmoothedUAV[!g_iNewPartGen]
		};
		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 2, pUAVs005, &UAVInitialCounts );
		pd3dImmediateContext->CSSetShaderResources(6, 1, &g_pParticlesSmoothedSRV[g_iNewPartGen]);
		pd3dImmediateContext->CSSetShaderResources(13, 1, &g_pParticlesSmoothedSRV[2]);
		pd3dImmediateContext->CSSetShader(g_pXSPH_GridCS, NULL, 0);
		pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );
		pd3dImmediateContext->CSSetShaderResources(13, 1, &g_pNullSRV);
	}

	g_iNewPartGen = !g_iNewPartGen;

	ID3D11UnorderedAccessView* pUAVs01[] = {
		g_pParticleCovMatUAV,
		NULL
	};

	UINT UAVICs[2] = {0};

	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 2, pUAVs01, UAVICs );
	pd3dImmediateContext->CSSetShader( g_pCov_GridCS, NULL, 0 );
	pd3dImmediateContext->CSSetShaderResources( 6, 1, &g_pParticlesSmoothedSRV[g_iNewPartGen] );
	pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

	ID3D11UnorderedAccessView* pUAVs1[] = {
		g_pParticleForcesUAV,
		g_pParticlePressureForcesUAV,
		g_pParticlePressureUAV
	};

	if(g_iFrameCounter % 10 == 0)
		g_bRecalculatePressure = TRUE;

	// Force
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 3, pUAVs1, &UAVInitialCounts );

    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbSimulationConstants );
	pd3dImmediateContext->CSSetShaderResources( 1, 1, &g_pParticleDensitySRV );
    pd3dImmediateContext->CSSetShaderResources( 3, 1, &g_pGridSRV );
    pd3dImmediateContext->CSSetShaderResources( 4, 1, &g_pGridIndicesSRV );

	pd3dImmediateContext->CSSetShader( 
		g_bRecalculatePressure ? g_pForce_GridCS : g_pForce_Grid_PreviousCS, NULL, 0 );
	pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

	if(g_bRecalculatePressure)
		g_bRecalculatePressure = FALSE;

	// Predictive-Correction
	UINT nrelax = (g_iFrameCounter % 10 == 0) ? N_RELAX : (N_RELAX / 2);

	for(UINT i = 0; i < nrelax; i++) {
		ID3D11UnorderedAccessView* pUAVs21[] = {
			g_pParticlesUAV,
			NULL,
			NULL
		};

		// Integrate
		pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 3, pUAVs21, &UAVInitialCounts );
		pd3dImmediateContext->CSSetShaderResources( 2, 1, &g_pParticleForcesSRV );
		pd3dImmediateContext->CSSetShaderResources( 5, 1, &g_pParticlePressureForcesSRV );
		pd3dImmediateContext->CSSetShaderResources( 8, 1, &g_pParticlePressureSRV );
		pd3dImmediateContext->CSSetShader( g_pIntegrateCS, NULL, 0 );
		pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

		pd3dImmediateContext->CSSetShaderResources( 8, 1, &g_pNullSRV );

		// Density & Pressure Fix
		ID3D11UnorderedAccessView* pUAVs22[] = {
			NULL,
			NULL,
			g_pParticlePressureUAV
		};

		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 3, pUAVs22, &UAVInitialCounts );
		pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pParticlesSRV );
		pd3dImmediateContext->CSSetShader( g_pDensity_Grid_PressureCS, NULL, 0 );
		pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

		// Pressure Force
		ID3D11UnorderedAccessView* pUAVs1[] = {
			NULL,
			g_pParticlePressureForcesUAV,
			NULL
		};

		pd3dImmediateContext->CSSetShaderResources( 5, 1, &g_pNullSRV );
		// Force
		pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 3, pUAVs1, &UAVInitialCounts );
		pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
		pd3dImmediateContext->CSSetShaderResources( 1, 1, &g_pParticleDensitySRV );
		pd3dImmediateContext->CSSetShaderResources( 8, 1, &g_pParticlePressureSRV );
		pd3dImmediateContext->CSSetShader( g_pForce_Grid_PredictCS, NULL, 0 );
		pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );


	}

	ID3D11UnorderedAccessView* pUAVs2[] = {
		g_pParticlesUAV,
		NULL,
		NULL
	};

	// Integrate
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pSortedParticlesSRV );
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 3, pUAVs2, &UAVInitialCounts );
	pd3dImmediateContext->CSSetShaderResources( 2, 1, &g_pParticleForcesSRV );
	pd3dImmediateContext->CSSetShaderResources( 5, 1, &g_pParticlePressureForcesSRV );
	pd3dImmediateContext->CSSetShaderResources( 8, 1, &g_pParticlePressureSRV );
	pd3dImmediateContext->CSSetShader( g_pIntegrateCS, NULL, 0 );
	pd3dImmediateContext->Dispatch( g_iNumParticles / SIMULATION_BLOCK_SIZE, 1, 1 );

	ID3D11Device* pd3dDevice = DXUTGetD3D11Device();

	D3D11_TEXTURE3D_DESC t3desc;
	t3desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;
	t3desc.CPUAccessFlags = 0;
	t3desc.Depth = nGridDimY * 3 + 4;
	t3desc.Format = DXGI_FORMAT_R32_FLOAT;
	t3desc.Height = nGridDimZ * 3 + 4;
	t3desc.MipLevels = max(1, (UINT)(logf(min(min(nGridDimX, nGridDimY), nGridDimZ) * 2 + 4) / logf(2.0f)));
	t3desc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
	t3desc.Usage = D3D11_USAGE_DEFAULT;
	t3desc.Width = nGridDimX * 3 + 4;
	pd3dDevice->CreateTexture3D(&t3desc, NULL, &g_pDensityField);
	pd3dDevice->CreateUnorderedAccessView(g_pDensityField, NULL, &g_pDensityFieldUAV);
	pd3dDevice->CreateShaderResourceView(g_pDensityField, NULL, &g_pDensityFieldSRV);
/*
	FLOAT vClr[4] = {0};
	pd3dImmediateContext->ClearUnorderedAccessViewFloat(g_pDensityFieldUAV, vClr);*/
}


//--------------------------------------------------------------------------------------
// GPU Fluid Simulation
//--------------------------------------------------------------------------------------
void SimulateFluid( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
    UINT UAVInitialCounts = 0;

    // Update per-frame variables
    CBSimulationConstants pData = {};

    // Simulation Constants
    pData.iNumParticles = g_iNumParticles;
    // Clamp the time step when the simulation runs slowly to prevent numerical explosion
    pData.fTimeStep = min( g_fMaxAllowableTimeStep, fElapsedTime );
    pData.fSmoothlen = g_fSmoothlen;
	pData.fPressureGamma = g_fPressureGamma;
    pData.fPressureStiffness = g_fRestDensity / g_fPressureGamma;
    pData.fRestDensity = g_fRestDensity;
    pData.fDensityCoef = g_fParticleMass * 315.0f / (64.0f * DirectX::XM_PI * pow(g_fSmoothlen, 9));
	pData.fDeltaCoef = -945.0f / (64.0f * DirectX::XM_PI * pow(g_fSmoothlen, 9));
	pData.fBeta = g_fRestDensity * g_fRestDensity / (g_fParticleMass * g_fParticleMass * g_fMaxAllowableTimeStep * g_fMaxAllowableTimeStep * 2.0f);
    pData.fDelta = g_fDelta;
	pData.fGradPressureCoef = g_fParticleMass * -45.0f / (DirectX::XM_PI * pow(g_fSmoothlen, 6));
    pData.fLapViscosityCoef = g_fParticleMass * g_fViscosity * 45.0f / (DirectX::XM_PI * pow(g_fSmoothlen, 6));

    pData.vGravity = g_vGravity;
    
    // Cells are spaced the size of the smoothing length search radius
    // That way we only need to search the 8 adjacent cells + current cell
    pData.vGridDim.x = 1.0f / g_fSmoothlen;
    pData.vGridDim.y = 1.0f / g_fSmoothlen;
    pData.vGridDim.z = 1.0f / g_fSmoothlen;
    pData.vGridDim.w = (FLOAT)(1 << GRIDSIZELOG2[0]) / (FLOAT)((1 << GRIDSIZELOG2[0]) - 2);
	pData.vGridDim3.x = g_vPlanes[3].w  / (FLOAT)((1 << GRIDSIZELOG2[0]) - 2);
	pData.vGridDim3.y = g_vPlanes[4].w  / (FLOAT)((1 << GRIDSIZELOG2[1]) - 2);
	pData.vGridDim3.z = g_vPlanes[5].w  / (FLOAT)((1 << GRIDSIZELOG2[2]) - 2);
	pData.vGridDim3.w = 0;
	pData.vGridDim4.x = 1.0f / pData.vGridDim3.x;
	pData.vGridDim4.y = 1.0f / pData.vGridDim3.y;
	pData.vGridDim4.z = 1.0f / pData.vGridDim3.z;
	pData.vGridDim4.w = 0;

	pData.vGridDim2.x = pData.vGridDim2.y = pData.vGridDim2.z = 0;

    // Collision information for the map
    pData.fWallStiffness = g_fWallStiffness;

	for(int i = 0; i < 6; i++)
		pData.vPlanes[i] = g_vPlanes[i];

    pd3dImmediateContext->UpdateSubresource( g_pcbSimulationConstants, 0, NULL, &pData, 0, 0 );

	// Update per-frame variables
	UpdateMCParas(pd3dImmediateContext);

	pd3dImmediateContext->CSSetConstantBuffers(1, 1, &g_pcbMarchingCubesConstants);

    SimulateFluid_Grid( pd3dImmediateContext );
	UpdateRenderParas( pd3dImmediateContext );

    // Unset
    pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pNullUAV, &UAVInitialCounts );
    pd3dImmediateContext->CSSetShaderResources( 0, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 1, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 2, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 3, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 4, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 5, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 6, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 7, 1, &g_pNullSRV );
    pd3dImmediateContext->CSSetShaderResources( 8, 1, &g_pNullSRV );

	g_iFrameCounter++;
}

void UpdateMCParas( ID3D11DeviceContext* pd3dImmediateContext )
{
	CBMarchingCubesConstants pData = {};

	// Simulation Constants
	pData.gridSize[0] = 1 << GRIDSIZELOG2[0];
	pData.gridSize[1] = 1 << GRIDSIZELOG2[1];
	pData.gridSize[2] = 1 << GRIDSIZELOG2[2];
	pData.gridSize[3] = pData.gridSize[0] * pData.gridSize[1] * pData.gridSize[2];
	pData.gridSizeMask[0] = pData.gridSize[0] - 1;
	pData.gridSizeMask[1] = pData.gridSize[1] - 1;
	pData.gridSizeMask[2] = pData.gridSize[2] - 1;
	pData.gridSizeMask[3] = 0;
	pData.gridSizeShift[0] = 0;
	pData.gridSizeShift[1] = GRIDSIZELOG2[0];
	pData.gridSizeShift[2] = GRIDSIZELOG2[0] + GRIDSIZELOG2[1];
	pData.gridSizeShift[3] = 0;
	pData.voxelSize.x = 2.0f / pData.gridSize[0];
	pData.voxelSize.y = 2.0f / pData.gridSize[1];
	pData.voxelSize.z = 2.0f / pData.gridSize[2];
	pData.voxelSize.w = g_fIsoValue;

	pd3dImmediateContext->UpdateSubresource( g_pcbMarchingCubesConstants, 0, NULL, &pData, 0, 0 );
}

void MarchingCubes( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{

	UINT UAVInitialCounts = 0;
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pReconstructBufferUAV, &UAVInitialCounts );
	pd3dImmediateContext->CSSetShaderResources(0, 1, &g_pDensityFieldSRV);
	pd3dImmediateContext->CSSetShader( g_pMarchingCubeCS, NULL, 0 );

	UINT numVoxels = (nGridDimX * 3 + 3) * (nGridDimY * 3 + 3) * (nGridDimZ * 3 + 3);

	pd3dImmediateContext->CSSetShaderResources(11, 1, &g_pDrawIndirectBufferSRV);
	pd3dImmediateContext->CSSetShaderResources(9, 1, &g_pParticleAnisoSRV);
	pd3dImmediateContext->CSSetConstantBuffers(0, 1, &g_pcbRenderConstants);
	pd3dImmediateContext->CSSetConstantBuffers(1, 1, &g_pcbMarchingCubesConstants);
	pd3dImmediateContext->CSSetConstantBuffers(4, 1, &g_pcbGridDimConstants);
	pd3dImmediateContext->Dispatch(iDivUp(numVoxels, MC_SIMULATION_BLOCK_SIZE), 1, 1);

	pd3dImmediateContext->CopyStructureCount( g_pDrawIndirectBuffer, 0, g_pReconstructBufferUAV);
	pd3dImmediateContext->CSSetShaderResources(11, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pDrawIndirectBufferUAV, &UAVInitialCounts);
	pd3dImmediateContext->CSSetShader( g_pNumVertsAdjustCS, NULL, 0);
	pd3dImmediateContext->Dispatch(1, 1, 1);

	ID3D11Buffer* pBufNull= NULL;
	pd3dImmediateContext->CSSetSamplers(0, 1, &g_pNullSampler);
	pd3dImmediateContext->CSSetConstantBuffers(0, 1, &pBufNull);
	pd3dImmediateContext->CSSetConstantBuffers(1, 1, &pBufNull);
	pd3dImmediateContext->CSSetConstantBuffers(4, 1, &pBufNull);
	pd3dImmediateContext->CSSetShaderResources(9, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetShaderResources(11, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pNullUAV, &UAVInitialCounts);
}

void UpdateRenderParas( ID3D11DeviceContext* pd3dImmediateContext ) 
{
	DirectX::XMMATRIX mView, mView1;
	DirectX::XMMATRIX mProj, mProj1;
	DirectX::XMMATRIX mWorldViewProjection, mWorldViewProjection1;

	// Get the projection & view matrix from the camera class
	mProj = *(DirectX::XMMATRIX*)g_Cameras[g_iSelCamera]->GetProjMatrix();
	mView = *(DirectX::XMMATRIX*)g_Cameras[g_iSelCamera]->GetViewMatrix();

	mWorldViewProjection = mView * mProj;

	mProj1 = *(DirectX::XMMATRIX*)g_CausticCamera.GetProjMatrix();
	mView1 = *(DirectX::XMMATRIX*)g_CausticCamera.GetViewMatrix();

	mWorldViewProjection1 = mView1 * mProj1;

	// Update Constants
	CBRenderConstants pData = {};

	DirectX::XMStoreFloat4x4( &pData.mViewProjection[0], DirectX::XMMatrixTranspose( mWorldViewProjection ) );
	DirectX::XMStoreFloat4x4( &pData.mViewProjection[1], DirectX::XMMatrixTranspose( mWorldViewProjection1 ) );
	DirectX::XMStoreFloat4x4( &pData.mView, DirectX::XMMatrixTranspose( mView ) );
	pData.fParticleSize = g_fParticleRenderSize;
	pData.fParticleAspectRatio = g_fParticleAspectRatio;
	pData.fTessFactor.x = pData.fTessFactor.y = pData.fTessFactor.z = pData.fTessFactor.w = 3.0f;
	pData.fEyePos.x = g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[0];
	pData.fEyePos.y = g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[1];
	pData.fEyePos.z = g_Cameras[g_iSelCamera]->GetEyePt()->m128_f32[2];
	pData.fEyePos.w = 1.0f;

	FLOAT aspect = (FLOAT) DXUTGetWindowWidth() / (FLOAT) DXUTGetWindowHeight();

	UINT nSx = (UINT)(sqrtf((FLOAT)(nGridDimY * 3 + 4) * aspect));
	UINT nSy = ((nGridDimY * 3 + 4) / nSx) + (UINT)((nGridDimY * 3 + 4) % nSx > 0);

	UINT nPix = DXUTGetWindowHeight() / nSy;
	FLOAT bX = (FLOAT) (nPix * nSx) / (FLOAT) DXUTGetWindowWidth();
	FLOAT bY = (FLOAT) (nPix * nSy) / (FLOAT) DXUTGetWindowHeight();

	pData.iVolSlicing.x = nSx;
	pData.iVolSlicing.y = nSy;
	pData.iVolSlicing.z = bX;
	pData.iVolSlicing.w = bY;
	pData.fInvGridSize = 1.0f / (nGridDimY * 3 + 4);
	pData.fParticleAspectRatio = g_fParticleAspectRatio;

	pData.fSmoothlen = g_fSmoothlen;

	pData.fRMAssist.x = 3.0f / (g_fSmoothlen * (FLOAT) (nGridDimX * 3 + 4));
	pData.fRMAssist.y = 3.0f / (g_fSmoothlen * (FLOAT) (nGridDimY * 3 + 4));
	pData.fRMAssist.z = 3.0f / (g_fSmoothlen * (FLOAT) (nGridDimZ * 3 + 4));
	pData.fRMAssist.w = 0;

	pData.fRMAssist2.x = (1.0f - (FLOAT) nGridMinX) / (FLOAT) (nGridDimX * 3 + 4) * 3.0f;
	pData.fRMAssist2.y = (1.0f - (FLOAT) nGridMinY) / (FLOAT) (nGridDimY * 3 + 4) * 3.0f;
	pData.fRMAssist2.z = (1.0f - (FLOAT) nGridMinZ) / (FLOAT) (nGridDimZ * 3 + 4) * 3.0f;
	pData.fRMAssist2.w = 0;

	pd3dImmediateContext->UpdateSubresource( g_pcbRenderConstants, 0, NULL, &pData, 0, 0 );
}
//--------------------------------------------------------------------------------------
// GPU Fluid Rendering
//--------------------------------------------------------------------------------------
void RenderFluid( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
    // Set the shaders
    pd3dImmediateContext->VSSetShader( g_pParticleVS, NULL, 0 );
    pd3dImmediateContext->GSSetShader( g_pParticleGS, NULL, 0 );
    pd3dImmediateContext->PSSetShader( g_pParticlePS, NULL, 0 );

    // Set the constant buffers
    pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
    pd3dImmediateContext->GSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
    pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );

    // Setup the particles buffer and IA
    pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pParticlesSmoothedSRV[g_iNewPartGen] );
    pd3dImmediateContext->VSSetShaderResources( 2, 1, &g_pParticleAnisoSRV );

    pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pNullBuffer, &g_iNullUINT, &g_iNullUINT );
    pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_POINTLIST );
	pd3dImmediateContext->OMSetBlendState(g_pBSAlpha, NULL, -1U);
    // Draw the mesh
	float ClearColor[4] = { 0.05f, 0.05f, 0.05f, 0.0f };
 	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
 	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );
	pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, NULL);

    pd3dImmediateContext->Draw( g_iNumParticles, 0 );
	pd3dImmediateContext->OMSetBlendState(NULL, NULL, -1U);
    // Unset the particles buffer
    pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pNullSRV );
    pd3dImmediateContext->VSSetShaderResources( 1, 1, &g_pNullSRV );
    pd3dImmediateContext->VSSetShaderResources( 2, 1, &g_pNullSRV );
}

void InjectParticles( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
	UINT UAVInitialCounts = 0;

	pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbSimulationConstants );
	pd3dImmediateContext->CSSetConstantBuffers( 1, 1, &g_pcbMarchingCubesConstants );

	// Generate Density Field
	pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &g_pDensityFieldUAV, &UAVInitialCounts );
	pd3dImmediateContext->CSSetShaderResources( 4, 1, &g_pGridIndicesSRV );
	pd3dImmediateContext->CSSetShaderResources( 6, 1, &g_pParticlesSmoothedSRV[g_iNewPartGen] );
	pd3dImmediateContext->CSSetShaderResources( 7, 1, &g_pParticleCovMatSRV );
	pd3dImmediateContext->CSSetShaderResources( 9, 1, &g_pParticleAnisoSRV );
	pd3dImmediateContext->CSSetShaderResources( 11, 1, &g_pDrawIndirectBufferSRV );
	pd3dImmediateContext->CSSetShader( g_pField_GridCS, NULL, 0 );	

	pd3dImmediateContext->Dispatch(iDivUp((nGridDimX * 3 + 2) * (nGridDimY * 3 + 2) * (nGridDimZ * 3 + 2), SIMULATION_BLOCK_SIZE), 1, 1);

	pd3dImmediateContext->CSSetShaderResources(4, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetShaderResources(6, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetShaderResources(7, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetShaderResources(9, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetShaderResources(11, 1, &g_pNullSRV);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pNullUAV, &UAVInitialCounts);

	pd3dImmediateContext->GenerateMips(g_pDensityFieldSRV);
}

void RenderCaustics( ID3D11DeviceContext* pd3dImmediateContext ) 
{
	pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexCausticRTV, NULL);
	FLOAT clrz[4] = {0};
	pd3dImmediateContext->ClearRenderTargetView(g_pTexCausticRTV, clrz);	
	
	D3D11_VIEWPORT vp;
	vp.Width = g_iCMWidth;
	vp.Height = g_iCMHeight;
	vp.MinDepth = 0;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	pd3dImmediateContext->RSSetViewports(1, &vp);

	pd3dImmediateContext->VSSetShader(g_pPlaneVS1, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_pPlanePaintPS, NULL, 0);

	pd3dImmediateContext->Draw(6, 0);

	pd3dImmediateContext->RSSetState(g_pRSNoCull);
	pd3dImmediateContext->VSSetShader( g_pSimpleShadowVS, NULL, 0 );
	pd3dImmediateContext->PSSetShader( g_pSimpleShadowPS, NULL, 0 );
	pd3dImmediateContext->DrawInstancedIndirect(g_pDrawIndirectBuffer, 0);

	pd3dImmediateContext->OMSetBlendState(g_pBSAddition, NULL, -1U);
	pd3dImmediateContext->VSSetShader( g_pSimpleCausticVS, NULL, 0 );
	pd3dImmediateContext->GSSetShader( g_pSimpleCausticGS, NULL, 0 );
	pd3dImmediateContext->PSSetShader( g_pSimpleCausticPS, NULL, 0 );

	pd3dImmediateContext->DrawInstancedIndirect(g_pDrawIndirectBuffer, 0);

	pd3dImmediateContext->RSSetState(NULL);
	pd3dImmediateContext->GSSetShader( NULL, NULL, 0 );
	pd3dImmediateContext->OMSetBlendState(NULL, NULL, -1U);

	pd3dImmediateContext->VSSetShader( g_pVisualizeVS, NULL, 0 );
	pd3dImmediateContext->PSSetShader( g_pFXAAPS, NULL, 0 );
	pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexCausticFilteredRTV, NULL);
	pd3dImmediateContext->PSSetShaderResources(20, 1, &g_pTexCausticSRV);
	pd3dImmediateContext->Draw(6, 0);
}

void BeginScreenRendering( ID3D11DeviceContext* pd3dImmediateContext )
{
	FLOAT clrz[4] = {0.05f, 0.05f, 0.05f, 0};
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0, 0 );
	pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexHDRRTV, pDSV);	
	pd3dImmediateContext->ClearRenderTargetView(g_pTexHDRRTV, clrz);
	
	D3D11_VIEWPORT vp;
	vp.MinDepth = 0;
	vp.MaxDepth = 1.0f;
	vp.TopLeftX = 0;
	vp.TopLeftY = 0;
	vp.Width = DXUTGetWindowWidth();
	vp.Height = DXUTGetWindowHeight();
	pd3dImmediateContext->RSSetViewports(1, &vp);
}

void RenderEnvPlane( ID3D11DeviceContext* pd3dImmediateContext )
{
	//Render EnvMap & Plane
	pd3dImmediateContext->VSSetShader(g_pEnvVS, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_pEnvPS, NULL, 0);
	pd3dImmediateContext->Draw(36, 0);

	pd3dImmediateContext->VSSetShader(g_pPlaneVS, NULL, 0);
	pd3dImmediateContext->PSSetShader(g_pPlanePS, NULL, 0);
	pd3dImmediateContext->PSSetShaderResources(20, 1, &g_pTexCausticFilteredSRV);

	pd3dImmediateContext->Draw(6, 0);
}

void RenderWaterBody( ID3D11DeviceContext* pd3dImmediateContext, BOOL bBilateralSmoothed = TRUE, BOOL bTessellated = TRUE )
{
	ID3D11DepthStencilView* pDSV = DXUTGetD3D11DepthStencilView();
	// Render the Water Body
	// Set the shaders
	pd3dImmediateContext->GSSetShader( NULL, NULL, 0 );

	if(bTessellated) {
		pd3dImmediateContext->VSSetShader( g_pSurfaceVS, NULL, 0 );
		pd3dImmediateContext->HSSetShader(g_pSurfaceHS, NULL, 0);
		pd3dImmediateContext->DSSetShader(g_pSurfaceDS, NULL, 0);
		pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST );
		pd3dImmediateContext->RSSetState(NULL);
	} else {
		pd3dImmediateContext->VSSetShader( g_pSurfaceNoTessVS, NULL, 0 );
		pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
		pd3dImmediateContext->RSSetState(g_pRSNoCull);
	}
	
	pd3dImmediateContext->PSSetShader( g_pSurfacePS, NULL, 0 );

	// Set the constant buffers
	pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->HSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->DSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );

	// Setup the particles buffer and IA
	pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pReconstructBufferSRV );
	pd3dImmediateContext->PSSetShaderResources( 1, 1, &g_pEnvMapSRV );
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pNullBuffer, &g_iNullUINT, &g_iNullUINT );

	ID3D11RenderTargetView* pRTVs[] = {
		g_pTexPositionRTV,
		g_pTexNormalRTV
	};


	FLOAT clrZero[4] = {0};
	pd3dImmediateContext->ClearRenderTargetView( g_pTexPositionRTV, clrZero );
	pd3dImmediateContext->ClearRenderTargetView( g_pTexNormalRTV, clrZero );
	pd3dImmediateContext->OMSetRenderTargets(2, pRTVs, pDSV);
	pd3dImmediateContext->DrawInstancedIndirect(g_pDrawIndirectBuffer, 0);
	pd3dImmediateContext->RSSetState(NULL);

	ID3D11ShaderResourceView* pSRVs[] = {
		g_pTexPositionSRV,
		g_pTexNormalSRV
	};

	pd3dImmediateContext->VSSetShader(g_pVisualizeVS, NULL, 0);
	pd3dImmediateContext->HSSetShader(NULL, NULL, 0);
	pd3dImmediateContext->DSSetShader(NULL, NULL, 0);
	pd3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	
	if(bBilateralSmoothed) {
		ID3D11RenderTargetView* pRTVs2[] = {
			g_pTexNormalBackRTV,
			NULL
		};
		//Bilateral Filter the Normal
		pd3dImmediateContext->PSSetShader(g_pBilateralXPS, NULL, 0);
		pd3dImmediateContext->OMSetRenderTargets(2, pRTVs2, NULL);
		pd3dImmediateContext->PSSetShaderResources(2, 2, pSRVs);
		pd3dImmediateContext->Draw(6, 0);

		pd3dImmediateContext->PSSetShaderResources(3, 1, &g_pNullSRV);
		pd3dImmediateContext->PSSetShader(g_pBilateralYPS, NULL, 0);
		pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexNormalRTV, NULL);
		pd3dImmediateContext->PSSetShaderResources(3, 1, &g_pTexNormalBackSRV);
		pd3dImmediateContext->Draw(6, 0);
	}
	//Composite the Water Body
	pd3dImmediateContext->PSSetShader((g_eSimMode == SIM_MODE_FULL) ? g_pColorPS : g_pGrayPS, NULL, 0);
	pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexHDRRTV, NULL);
	pd3dImmediateContext->PSSetShaderResources(2, 2, pSRVs);
	pd3dImmediateContext->Draw(6, 0);

	pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pNullSRV );
	pd3dImmediateContext->PSSetShaderResources( 1, 1, &g_pNullSRV );
	pd3dImmediateContext->PSSetShaderResources( 2, 1, &g_pNullSRV ); 
	pd3dImmediateContext->PSSetShaderResources( 3, 1, &g_pNullSRV ); 
}

void PostProcessing( ID3D11DeviceContext* pd3dImmediateContext )
{
	
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	pd3dImmediateContext->VSSetShader(g_pVisualizeVS, NULL, 0);
	pd3dImmediateContext->OMSetRenderTargets(1, &g_pTexPositionRTV, NULL);	
	pd3dImmediateContext->PSSetShader(g_pTonemapPS, NULL, 0);
	pd3dImmediateContext->PSSetShaderResources(4, 1, &g_pTexHDRSRV);
	pd3dImmediateContext->Draw(6, 0);

	pd3dImmediateContext->PSSetShader(g_pFXAAPS, NULL, 0);
	pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, NULL);
	pd3dImmediateContext->PSSetShaderResources(20, 1, &g_pTexPositionSRV);
	pd3dImmediateContext->Draw(6, 0);
}
//--------------------------------------------------------------------------------------
// GPU Surface Rendering
//--------------------------------------------------------------------------------------
void RenderSurface( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{

	// Set the constant buffers
	pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->GSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );

	// Setup the particles buffer and IA
	pd3dImmediateContext->VSSetShaderResources( 0, 1, &g_pReconstructBufferSRV );
	pd3dImmediateContext->PSSetShaderResources( 1, 1, &g_pEnvMapSRV );
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pNullBuffer, &g_iNullUINT, &g_iNullUINT );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

	//Render Caustic Light-Map
	switch (g_eSimMode)
	{
	case SIM_MODE_FULL:
		RenderCaustics(pd3dImmediateContext);
		BeginScreenRendering(pd3dImmediateContext);
		RenderEnvPlane(pd3dImmediateContext);
		RenderWaterBody(pd3dImmediateContext);
		break;
	case SIM_MODE_CAUSTICONLY:
		RenderCaustics(pd3dImmediateContext);
		BeginScreenRendering(pd3dImmediateContext);
		RenderEnvPlane(pd3dImmediateContext);
		break;
	case SIM_MODE_GRAYSURF:	
		BeginScreenRendering(pd3dImmediateContext);
		RenderWaterBody(pd3dImmediateContext, FALSE, FALSE);
		break;	
	case SIM_MODE_TESSSURF:	
		BeginScreenRendering(pd3dImmediateContext);
		RenderWaterBody(pd3dImmediateContext, FALSE);
		break;	
	case SIM_MODE_TESSSURFSMOOTHED:
		BeginScreenRendering(pd3dImmediateContext);
		RenderWaterBody(pd3dImmediateContext);
		break;
	default:
		break;
	}
	
	PostProcessing(pd3dImmediateContext);

	ID3D11ShaderResourceView* pSRVClr[20] = {NULL};
	pd3dImmediateContext->PSSetShaderResources( 0, 20, pSRVClr ); 
	pd3dImmediateContext->HSSetShader(NULL, NULL, 0);
	pd3dImmediateContext->DSSetShader(NULL, NULL, 0);
	pd3dImmediateContext->GSSetShader(NULL, NULL, 0);
}

void VisualizeVolume( ID3D11DeviceContext* pd3dImmediateContext, float fElapsedTime )
{
	// Set the shaders
	pd3dImmediateContext->VSSetShader( g_pVisualizeVS, NULL, 0 );
	pd3dImmediateContext->GSSetShader( NULL, NULL, 0 );
	pd3dImmediateContext->PSSetShader( g_pVisualizePS, NULL, 0 );

	// Set the constant buffers
	pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );
	pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbRenderConstants );

	// Setup the particles buffer and IA
	pd3dImmediateContext->PSSetShaderResources(0, 1, &g_pDensityFieldSRV);
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, &g_pNullBuffer, &g_iNullUINT, &g_iNullUINT );
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

	// Draw the mesh
	float ClearColor[4] = { 0.05f, 0.05f, 0.05f, 0.0f };
	ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );

	pd3dImmediateContext->OMSetRenderTargets(1, &pRTV, NULL);

	pd3dImmediateContext->Draw(6, 0);

	// Unset the particles buffer
	pd3dImmediateContext->PSSetShaderResources( 0, 1, &g_pNullSRV );
}

//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  float fElapsedTime, void* pUserContext )
{
    // If the settings dialog is being shown, then render it instead of rendering the app's scene

	if(g_bAdvance)
		g_fMaxAllowableTimeStep = 0.0075f;
	else
		g_fMaxAllowableTimeStep = 0.0f;
	SimulateFluid( pd3dImmediateContext, fElapsedTime );

	switch (g_eSimMode)
	{
	case SIM_MODE_FULL:	case SIM_MODE_GRAYSURF:	case SIM_MODE_TESSSURF:	case SIM_MODE_TESSSURFSMOOTHED: case SIM_MODE_CAUSTICONLY:
		InjectParticles( pd3dImmediateContext, fElapsedTime );
		MarchingCubes(pd3dImmediateContext, fElapsedTime);
		RenderSurface(pd3dImmediateContext, fElapsedTime);
		break;
	case SIM_MODE_VISUALIZE:
		InjectParticles( pd3dImmediateContext, fElapsedTime );
		VisualizeVolume( pd3dImmediateContext, fElapsedTime );
		break;
	case SIM_MODE_PARTICLE:
		RenderFluid( pd3dImmediateContext, fElapsedTime );
		break;
	case SIM_MODE_COUNT:
		throw;
	default:
		break;
	}


/*	Render the HUD*/
	DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
    RenderText();
	

	DXUT_EndPerfEvent();
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
	SAFE_RELEASE(g_pTexPosition);
	SAFE_RELEASE(g_pTexPositionSRV);
	SAFE_RELEASE(g_pTexPositionRTV);

	SAFE_RELEASE(g_pTexHDR);
	SAFE_RELEASE(g_pTexHDRSRV);
	SAFE_RELEASE(g_pTexHDRRTV);

	SAFE_RELEASE(g_pTexCaustic);
	SAFE_RELEASE(g_pTexCausticSRV);
	SAFE_RELEASE(g_pTexCausticRTV);

	SAFE_RELEASE(g_pTexCausticFiltered);
	SAFE_RELEASE(g_pTexCausticFilteredSRV);
	SAFE_RELEASE(g_pTexCausticFilteredRTV);

	SAFE_RELEASE(g_pTexPositionBack);
	SAFE_RELEASE(g_pTexPositionBackSRV);
	SAFE_RELEASE(g_pTexPositionBackRTV);

	SAFE_RELEASE(g_pTexNormal);
	SAFE_RELEASE(g_pTexNormalSRV);
	SAFE_RELEASE(g_pTexNormalRTV);

	SAFE_RELEASE(g_pTexNormalBack);
	SAFE_RELEASE(g_pTexNormalBackSRV);
	SAFE_RELEASE(g_pTexNormalBackRTV);
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D10CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{

    SAFE_RELEASE( g_pcbSimulationConstants );
    SAFE_RELEASE( g_pcbRenderConstants );
    SAFE_RELEASE( g_pcbMarchingCubesConstants );
    SAFE_RELEASE( g_pSortCB );
	SAFE_RELEASE( g_pcbGridDimConstants );

    SAFE_RELEASE( g_pParticleVS );
    SAFE_RELEASE( g_pParticleGS );
    SAFE_RELEASE( g_pParticlePS );

	SAFE_RELEASE( g_pRayMarchingPS );
	SAFE_RELEASE( g_pBoundingBoxPS );
	SAFE_RELEASE( g_pBoundingBoxVS );

	SAFE_RELEASE( g_pSurfaceVS );
	SAFE_RELEASE( g_pSurfaceNoTessVS );
	SAFE_RELEASE( g_pSurfacePS );
	SAFE_RELEASE( g_pFXAAPS );
	SAFE_RELEASE( g_pPlaneVS );
	SAFE_RELEASE( g_pPlaneVS1 );
	SAFE_RELEASE( g_pPlanePS );
	SAFE_RELEASE( g_pEnvVS );
	SAFE_RELEASE( g_pEnvPS );
	SAFE_RELEASE( g_pPlanePaintPS );
	SAFE_RELEASE( g_pSimpleCausticVS );
	SAFE_RELEASE( g_pSimpleCausticPS );
	SAFE_RELEASE( g_pSimpleShadowVS );
	SAFE_RELEASE( g_pSimpleShadowPS );
	SAFE_RELEASE( g_pSimpleCausticGS );
	SAFE_RELEASE( g_pSurfaceHS );
	SAFE_RELEASE( g_pSurfaceDS );
	SAFE_RELEASE( g_pBilateralXPS );
	SAFE_RELEASE( g_pBilateralYPS );

    SAFE_RELEASE( g_pIntegrateCS );

    SAFE_RELEASE( g_pDensity_GridCS );
    SAFE_RELEASE( g_pDensity_Grid_DeltaCS );
    SAFE_RELEASE( g_pDensity_Grid_PressureCS );
	SAFE_RELEASE( g_pDensity_Grid_SmoothCS );
    SAFE_RELEASE( g_pForce_GridCS );
    SAFE_RELEASE( g_pXSPH_GridCS );
	SAFE_RELEASE( g_pForce_Grid_PreviousCS );
    SAFE_RELEASE( g_pForce_Grid_PredictCS );
    SAFE_RELEASE( g_pBuildGridCS );
    SAFE_RELEASE( g_pClearGridIndicesCS );
    SAFE_RELEASE( g_pBuildGridIndicesCS );
    SAFE_RELEASE( g_pRearrangeParticlesCS );
    SAFE_RELEASE( g_pSortBitonic );
    SAFE_RELEASE( g_pSortTranspose );
	SAFE_RELEASE( g_pSortBitonicUint );
    SAFE_RELEASE( g_pSortTransposeUint );
	SAFE_RELEASE( g_pBoundingBoxCS );
	SAFE_RELEASE( g_pBoundingBoxPingPongCS );
	SAFE_RELEASE( g_pMarchingCubeCS );
	SAFE_RELEASE( g_pNumVertsAdjustCS );
	SAFE_RELEASE( g_pNumCellsAdjustCS );
	SAFE_RELEASE( g_pParticleInjectAdjustCS );
	SAFE_RELEASE( g_pExpandCellCS );
	SAFE_RELEASE( g_pPickCellCS );
	SAFE_RELEASE( g_pColorPS );
	SAFE_RELEASE( g_pTonemapPS );
	SAFE_RELEASE( g_pBSAddition );
	SAFE_RELEASE( g_pBSAlpha );

	SAFE_RELEASE( g_pField_GridCS );

    SAFE_RELEASE( g_pParticles );
    SAFE_RELEASE( g_pParticlesSRV );
    SAFE_RELEASE( g_pParticlesUAV );
 
	for(UINT i = 0; i < 3; i++) {
		SAFE_RELEASE( g_pParticlesSmoothed[i] );
		SAFE_RELEASE( g_pParticlesSmoothedSRV[i] );
		SAFE_RELEASE( g_pParticlesSmoothedUAV[i] );   
	}

    SAFE_RELEASE( g_pSortedParticles );
    SAFE_RELEASE( g_pSortedParticlesSRV );
    SAFE_RELEASE( g_pSortedParticlesUAV );

    SAFE_RELEASE( g_pParticleForces );
    SAFE_RELEASE( g_pParticleForcesSRV );
    SAFE_RELEASE( g_pParticleForcesUAV );

	SAFE_RELEASE( g_pParticlePressureForces );
	SAFE_RELEASE( g_pParticlePressureForcesSRV );
	SAFE_RELEASE( g_pParticlePressureForcesUAV );
    
    SAFE_RELEASE( g_pParticleDensity );
    SAFE_RELEASE( g_pParticleDensitySRV );
    SAFE_RELEASE( g_pParticleDensityUAV );

	SAFE_RELEASE( g_pParticleDelta[0] );
	SAFE_RELEASE( g_pParticleDeltaSRV[0] );
	SAFE_RELEASE( g_pParticleDeltaUAV[0] );

	SAFE_RELEASE( g_pParticleDelta[1] );
	SAFE_RELEASE( g_pParticleDeltaSRV[1] );
	SAFE_RELEASE( g_pParticleDeltaUAV[1] );

    SAFE_RELEASE( g_pGridSRV );
    SAFE_RELEASE( g_pGridUAV );
    SAFE_RELEASE( g_pGrid );

    SAFE_RELEASE( g_pGridDimBufferSRV );
    SAFE_RELEASE( g_pGridDimBufferUAV );
    SAFE_RELEASE( g_pGridDimBuffer );

    SAFE_RELEASE( g_pGridPingPongSRV );
    SAFE_RELEASE( g_pGridPingPongUAV );
    SAFE_RELEASE( g_pGridPingPong );

    SAFE_RELEASE( g_pGridIndicesSRV );
    SAFE_RELEASE( g_pGridIndicesUAV );
    SAFE_RELEASE( g_pGridIndices );

	SAFE_RELEASE( g_pBoundingBoxBuffer );
	SAFE_RELEASE( g_pBoundingBoxBufferSRV );
	SAFE_RELEASE( g_pBoundingBoxBufferUAV );

	SAFE_RELEASE( g_pBoundingBoxBufferPingPong );
	SAFE_RELEASE( g_pBoundingBoxBufferPingPongSRV );
	SAFE_RELEASE( g_pBoundingBoxBufferPingPongUAV );

	SAFE_RELEASE( g_pParticlePressure );
	SAFE_RELEASE( g_pParticlePressureSRV );
	SAFE_RELEASE( g_pParticlePressureUAV );

	SAFE_RELEASE( g_pBSAlpha );
	SAFE_RELEASE( g_pBSAddition );
	SAFE_RELEASE( g_pCov_GridCS );

	SAFE_RELEASE( g_pDensityFieldSRV );
	SAFE_RELEASE( g_pDensityFieldUAV );
	SAFE_RELEASE( g_pDensityField );

	SAFE_RELEASE( g_pParticleCovMat );
	SAFE_RELEASE( g_pParticleCovMatSRV );
	SAFE_RELEASE( g_pParticleCovMatUAV );

	SAFE_RELEASE( g_pParticleAniso );
	SAFE_RELEASE( g_pParticleAnisoSRV );
	SAFE_RELEASE( g_pParticleAnisoUAV );

	SAFE_RELEASE( g_pParticleAnisoSorted );
	SAFE_RELEASE( g_pParticleAnisoSortedSRV );
	SAFE_RELEASE( g_pParticleAnisoSortedUAV );

	SAFE_RELEASE( g_pParticleAnisoSortedPingPong );
	SAFE_RELEASE( g_pParticleAnisoSortedPingPongSRV );
	SAFE_RELEASE( g_pParticleAnisoSortedPingPongUAV );

	SAFE_RELEASE( g_pVisualizeVS );
	SAFE_RELEASE( g_pVisualizePS );
	SAFE_RELEASE( g_pRSNoCull );
	SAFE_RELEASE( g_pRSFrontCull );
	SAFE_RELEASE( g_pSSField );

	SAFE_RELEASE( g_pEnvMapSRV );
//	SAFE_RELEASE( g_pBSXor );
	SAFE_RELEASE( g_pBoundingBox_GridCS );
	SAFE_RELEASE( g_pColorPS );
	SAFE_RELEASE( g_pTonemapPS );
#ifdef TRY_CUDA
	if( g_pGridGR ) cudaGraphicsUnregisterResource( g_pGridGR );
	if( g_pGridDimBufferGR ) cudaGraphicsUnregisterResource( g_pGridDimBufferGR );
	if( g_pParticleAnisoSortedGR ) cudaGraphicsUnregisterResource( g_pParticleAnisoSortedGR );
#endif

	DestroyMCBuffers();
}

HRESULT CreateMCBuffers( ID3D11Device* pd3dDevice )
{
	HRESULT hr = S_OK;

	DestroyMCBuffers();

	V_RETURN( CreateStructuredBuffer< Triangle >( pd3dDevice, 
		7168000, &g_pReconstructBuffer, &g_pReconstructBufferSRV, &g_pReconstructBufferUAV, NULL, TRUE ) );
	DXUT_SetDebugName( g_pReconstructBuffer, "Reconstructed Surface Buffer" );
	DXUT_SetDebugName( g_pReconstructBufferUAV, "Reconstructed Surface Buffer UAV" );

	hr =  CreateDrawIndirectBuffer( pd3dDevice, 32, &g_pDrawIndirectBuffer, &g_pDrawIndirectBufferStaging, &g_pDrawIndirectBufferUAV, &g_pDrawIndirectBufferSRV
#ifdef TRY_CUDA
		, NULL, 0, &g_pDrawIndirectBufferGR
#endif
		) ;
	V_RETURN( hr );
	DXUT_SetDebugName( g_pDrawIndirectBuffer, "Draw Indirect Buffer" );
	DXUT_SetDebugName( g_pDrawIndirectBufferUAV, "Draw Indirect Buffer UAV" );

#ifdef TRY_CUDA
	if(DXUTIsCUDAvailable()) {
		cudaHostAlloc( (void**) &g_pDrawIndirectMapped, sizeof(UINT) * 32, cudaHostAllocMapped );
		cudaHostGetDevicePointer( &g_pDrawIndirectMappedDevPtr, g_pDrawIndirectMapped, 0 );
	}
#endif

	return hr;
}

void DestroyMCBuffers()
{
	SAFE_RELEASE( g_pReconstructBuffer );
	SAFE_RELEASE( g_pReconstructBufferSRV );
	SAFE_RELEASE( g_pReconstructBufferUAV );

	SAFE_RELEASE( g_pDrawIndirectBuffer );
	SAFE_RELEASE( g_pDrawIndirectBufferStaging );
	SAFE_RELEASE( g_pDrawIndirectBufferUAV );
	SAFE_RELEASE( g_pDrawIndirectBufferSRV );
#ifdef TRY_CUDA
	if(g_pDrawIndirectBufferGR) cudaGraphicsUnregisterResource(g_pDrawIndirectBufferGR);
	if(g_pDrawIndirectMapped) cudaFreeHost(g_pDrawIndirectMapped);
#endif 
}
