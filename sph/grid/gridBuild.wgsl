struct Particle {
    position: vec3f, 
    velocity: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct Environment {
    xGrids: i32, 
    yGrids: i32, 
    zGrids: i32, 
    cellSize: f32, 
    xHalf: f32, 
    yHalf: f32, 
    zHalf: f32, 
    offset: f32, 
}

struct SPHParams {
    mass: f32, 
    kernelRadius: f32, 
    kernelRadiusPow2: f32, 
    kernelRadiusPow5: f32, 
    kernelRadiusPow6: f32,  
    kernelRadiusPow9: f32, 
    dt: f32, 
    stiffness: f32, 
    nearStiffness: f32, 
    restDensity: f32, 
    viscosity: f32, 
    n: u32
}

@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> particleCellOffset : array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(3) var<uniform> env: Environment;
@group(0) @binding(4) var<uniform> params: SPHParams;

fn cellId(position: vec3f) -> i32 {
    let xi: i32 = i32(floor((position.x + env.xHalf + env.offset) / env.cellSize));
    let yi: i32 = i32(floor((position.y + env.yHalf + env.offset) / env.cellSize));
    let zi: i32 = i32(floor((position.z + env.zHalf + env.offset) / env.cellSize));

    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
  if (id.x < params.n)
  {
    let cellID: i32 = cellId(particles[id.x].position);
    // TODO : 変える
    if (cellID < env.xGrids * env.yGrids * env.zGrids) { 
      particleCellOffset[id.x] = atomicAdd(&cellParticleCount[cellID], 1u);
    }
  }
}