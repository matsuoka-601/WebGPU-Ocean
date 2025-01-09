struct Particle {
    position: vec3f, 
    v: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

struct RealBoxSize {
  xHalf: f32, 
  yHalf: f32, 
  zHalf: f32, 
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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> realBoxSize: RealBoxSize;
@group(0) @binding(2) var<uniform> params: SPHParams;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < params.n) {
    // avoid zero division
    if (particles[id.x].density != 0.) {
      var a = particles[id.x].force / particles[id.x].density;

      let xPlusDist = realBoxSize.xHalf - particles[id.x].position.x;
      let xMinusDist = realBoxSize.xHalf + particles[id.x].position.x;
      let yPlusDist = realBoxSize.yHalf - particles[id.x].position.y;
      let yMinusDist = realBoxSize.yHalf + particles[id.x].position.y;
      let zPlusDist = realBoxSize.zHalf - particles[id.x].position.z;
      let zMinusDist = realBoxSize.zHalf + particles[id.x].position.z;

      let wallStiffness = 8000.;

      let xPlusForce = vec3f(1., 0., 0.) * wallStiffness * min(xPlusDist, 0.);
      let xMinusForce = vec3f(-1., 0., 0.) * wallStiffness * min(xMinusDist, 0.);
      let yPlusForce = vec3f(0., 1., 0.) * wallStiffness * min(yPlusDist, 0.);
      let yMinusForce = vec3f(0., -1., 0.) * wallStiffness * min(yMinusDist, 0.);
      let zPlusForce = vec3f(0., 0., 1.) * wallStiffness * min(zPlusDist, 0.);
      let zMinusForce = vec3f(0., 0., -1.) * wallStiffness * min(zMinusDist, 0.);

      let xForce = xPlusForce + xMinusForce;
      let yForce = yPlusForce + yMinusForce;
      let zForce = zPlusForce + zMinusForce;

      a += xForce + yForce + zForce;
      particles[id.x].v += params.dt * a;
      particles[id.x].position += params.dt * particles[id.x].v;
    }
  }
}