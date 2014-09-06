To run the CUDA-assisted version under WIN7+VS2012 platform, please follow the below steps:
1. Install lastesed version of CUDA (at least cuda v5) on you computer.
2. Copy the cuda file C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\BuildCustomizations¡± to C:\Program Files (x86)\MSBuild\Microsoft.Cpp\v4.0\V110\BuildCustomizations. Notice the v110 file difference.
3. Follow the introduction on "http://blog.norture.com/2012/10/gpu-parallel-programming-in-vs2012-with-nvidia-cuda/ " to set the CUDA on your computer.
4. Open project and use 'build only' to build the 'ThrustWapper' sub-project first.
5. Compile whole project. 

*If you have no CUDA installed, you can also comment the "TRY_CUDA" term in DXUTCommon.h, and just ignore the ThrustWrapper project in the solution.

If you have any problem you can contact fyun@acm.org / zjz19900719@gmail.com

Keys:
W,S,A,D,Q,E - Moving Camera
O - Start/Stop Simulation
C - Print Camera Info
R - Start/Stop Camera Automatically Rotation
M - Switching Rendering Mode between:
	Full Effects
	Visualization
	Particles
	Caustic & Shadow Only
	No Tessellation
	Tessellated Only
	Tessellated & Bilateral Smoothed
