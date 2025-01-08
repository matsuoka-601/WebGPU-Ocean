# webgpu-ocean
A real-time 3d fluid simulation implemented in WebGPU. Works on your browsers which support WebGPU. 

[Try demo here!](https://webgpu-ocean.netlify.app/)

![webgpu-ocean-demo](https://github.com/user-attachments/assets/c284d0a8-297e-44f8-a70b-abffbfd3e150)

The following are the characteristics of the simulation.
- The simulation is based on **Smoothed Particle Hydrodynamics (SPH)** described in [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf) by Müller et al.
  - Compute shader is used for implementing SPH. 
- **Screen-Space Rendering** described in [GDC 2010 slide](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf) is used for real-time rendering of the fluid.
  - Vertex/fragment shader is used for implementing Screen-Space Rendering.
- For **fast neighborhood search on GPU**, an algorithm described in [FAST FIXED-RADIUS NEAREST NEIGHBORS: INTERACTIVE MILLION-PARTICLE FLUIDS](https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf) is used. 

## 100,000 particles mode
To reduce the load on the browser, the maximum number of particles in the simulation is limited to 40,000 by default. But with a decent GPU, it's possible to simulate ~100,000 particles in real-time. To enable simulating 100,000 particles, make 5th button visible by editing the html using developer tool. 

Below is the simulation of 100,000 particles. The simulation is run on a laptop with RTX 3060 laptop.

https://github.com/user-attachments/assets/03913ab7-a27f-4701-a5a6-95c977c1825a

## TODO
- Implement MLS-MPM ⇒ **Currently implementing in `mls-mpm` branch and it will soon be available!**
  - Currently, the bottleneck of the simulation is the neighborhood search in SPH. Therefore, implementing MLS-MPM would allow us to handle even larger real-time simulation (with > 100,000 particles?) since it doesn't require neighborhood search.
  - Now I'm actively learning MLS-MPM. But it will be harder than learning classical SPH, so any help would be appreciated :)
- Implement a rendering method described in [Unified Spray, Foam and Bubbles for Particle-Based Fluids](https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf)
  - This would make the simulation look more spectacular!
- Use better rendering method with less artifacts like [Narrow-Range Filter](https://dl.acm.org/doi/10.1145/3203201)
  - Currently, there are some artifacts derived from bilateral filter in rendered fluid. Using Narrow-Range Filter would reduce those artifacts.
