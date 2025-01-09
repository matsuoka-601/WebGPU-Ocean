# webgpu-ocean
A real-time 3d fluid simulation implemented in WebGPU. Works on your browsers which support WebGPU. 

[Try demo here!](https://webgpu-ocean.netlify.app/)

![webgpu-ocean-demo](https://github.com/user-attachments/assets/5b008b16-7d46-4e09-af21-d70f6fa2ec20)

The following are the characteristics of the simulation.
- **Moving Least Squares Material Point Method (MLS-MPM)** is implemented for the simulation. This algorithm enabled real-time simulation with **~100,000 particles on integrated graphics** and **~300,000 particles on a decent GPU.**
  - [nialltl's article](https://nialltl.neocities.org/articles/mpm_guide) helped a lot when implementing MLS-MPM. Huge thanks for them!
- **Smoothed Particle Hydrodynamics (SPH)** based on [Particle-Based Fluid Simulation for Interactive Applications](https://matthias-research.github.io/pages/publications/sca03.pdf) by Müller et al. is also implemented.
  - You can enable SPH simulation by clicking "SPH" button on the top right.
  - For **fast neighborhood search on GPU**, an algorithm described in [FAST FIXED-RADIUS NEAREST NEIGHBORS: INTERACTIVE MILLION-PARTICLE FLUIDS](https://ramakarl.com/pdfs/2014_Hoetzlein_FastFixedRadius_Neighbors.pdf) is used. 
- **Screen-Space Rendering** described in [GDC 2010 slide](https://developer.download.nvidia.com/presentations/2010/gdc/Direct3D_Effects.pdf) is used for real-time rendering of the fluid.
  - Vertex/fragment shader is used for implementing Screen-Space Rendering.
## Implementation details of MLS-MPM
Initially the simulation in this project was based on **Smoothed Particle Hydrodynamics (SPH)**. However, since the neighborhood search is really expensive, the maximum number of particles that can be simulated in real-time was 30,000 at best. So I decided to implement **Moving Least Squares Material Point Method (MLS-MPM)** which is completely free from neighborhood search. The results were very good, enabling real-time simulations of **~100,000 particles on integrated graphics** and **~300,000 particles on decent GPUs.**

My implementation of MLS-MPM is based on [nialltl's article](https://nialltl.neocities.org/articles/mpm_guide). According to the article, vanilla implementation of MPM method is not suitable for real-time simulation since the inaccuracy of the estimate of the volume forces the timestep to be small. To tackle this problem, the article suggests recalculating volume every simulation step. This technique is very effective for setting a high timestep and currently requires only **2 simulation steps per frame.**

Implementing 3D version of nialltl's MLS-MPM in WebGPU was relatively straightforward, but there was one difficult point : **P2G stage.** In the P2G stage, it's required to scatter particle data to grids in parallel. The most standard way to do this in WebGPU is using `atomicAdd`. However, since `atomicAdd` exists only for 32bit integers, it's impossible to directly use it to scatter data which is held as floating-point number. To avoid this problem, **fixed-point number** is used. That is, the data itself is hold as integers and multiplied by a constant (e.g. `1e-7`) to recover floating-point numbers. This way it's possible to use `atomicAdd` for scattering particle data to grids. (I discovered this technique in [pbmpm](https://github.com/electronicarts/pbmpm) repository, a reference implementation for [A Position Based Material Point Method (SIGGRAPH 2024)](https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf) by Chris Lewin. )
## TODO
- ~~Implement MLS-MPM~~ ⇒ **Done**
  - Currently, the bottleneck of the simulation is the neighborhood search in SPH. Therefore, implementing MLS-MPM would allow us to handle even larger real-time simulation (with > 100,000 particles?) since it doesn't require neighborhood search.
  - Now I'm actively learning MLS-MPM. But it will be harder than learning classical SPH, so any help would be appreciated :)
- Implement a rendering method described in [Unified Spray, Foam and Bubbles for Particle-Based Fluids](https://cg.informatik.uni-freiburg.de/publications/2012_CGI_sprayFoamBubbles.pdf)
  - This would make the simulation look more spectacular!
- Use better rendering method with less artifacts like [Narrow-Range Filter](https://dl.acm.org/doi/10.1145/3203201)
  - Currently, there are some artifacts derived from bilateral filter in rendered fluid. Using Narrow-Range Filter would reduce those artifacts.
