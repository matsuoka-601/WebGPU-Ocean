struct Particle {
    position: vec3f, 
    v: vec3f, 
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

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> sortedParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;
@group(0) @binding(3) var<uniform> env: Environment;
@group(0) @binding(4) var<uniform> params: SPHParams;

fn densityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow6); // pow 使うと遅いかも
    let d = params.kernelRadius - r;
    return scale * d * d;
}

fn nearDensityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow5); // 直す
    let a = params.kernelRadiusPow9;
    let d = params.kernelRadius - r;
    return scale * d * d;
}

fn viscosityKernelLaplacian(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * params.kernelRadiusPow6);
    // let dd = kernelRadius * kernelRadius - r * r;
    let d = params.kernelRadius - r;
    return scale * d;
}

fn cellPosition(v: vec3f) -> vec3i {
    let xi = i32(floor((v.x + env.xHalf + env.offset) / env.cellSize));
    let yi = i32(floor((v.y + env.yHalf + env.offset) / env.cellSize));
    let zi = i32(floor((v.z + env.zHalf + env.offset) / env.cellSize));
    return vec3i(xi, yi, zi);
}

fn cellNumberFromId(xi: i32, yi: i32, zi: i32) -> i32 {
    return xi + yi * env.xGrids + zi * env.xGrids * env.yGrids;
}

@compute @workgroup_size(64)
fn computeForce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < params.n) {
        let n = params.n;
        let density_i = particles[id.x].density;
        let nearDensity_i = particles[id.x].nearDensity;
        let pos_i = particles[id.x].position;
        var fPress = vec3(0.0, 0.0, 0.0);
        var fVisc = vec3(0.0, 0.0, 0.0);

        let v = cellPosition(pos_i);
        if (v.x < env.xGrids && 0 <= v.x && 
            v.y < env.yGrids && 0 <= v.y && 
            v.z < env.zGrids && 0 <= v.z) 
        {
            if (v.x < env.xGrids && v.y < env.yGrids && v.z < env.zGrids) {
                for (var dz = max(-1, -v.z); dz <= min(1, env.zGrids - v.z - 1); dz++) {
                    for (var dy = max(-1, -v.y); dy <= min(1, env.yGrids - v.y - 1); dy++) {
                        let dxMin = max(-1, -v.x);
                        let dxMax = min(1, env.xGrids - v.x - 1);
                        let startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz);
                        let endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz);
                        let start = prefixSum[startCellNum];
                        let end = prefixSum[endCellNum + 1];
                        for (var j = start; j < end; j++) {
                            let density_j = sortedParticles[j].density;
                            let nearDensity_j = sortedParticles[j].nearDensity;
                            let pos_j = sortedParticles[j].position;
                            let r2 = dot(pos_i - pos_j, pos_i - pos_j); 
                            if (density_j == 0. || nearDensity_j == 0.) {
                                continue;
                            }
                            if (r2 < params.kernelRadiusPow2 && 1e-64 < r2) {
                                let r = sqrt(r2);
                                let pressure_i = max(-0., params.stiffness * (density_i - params.restDensity));
                                let pressure_j = max(-0., params.stiffness * (density_j - params.restDensity));
                                let nearPressure_i = params.nearStiffness * nearDensity_i;
                                let nearPressure_j = params.nearStiffness * nearDensity_j;
                                let sharedPressure = (pressure_i + pressure_j) / 2.0;
                                let nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
                                let dir = normalize(pos_j - pos_i);
                                fPress += -params.mass * sharedPressure * dir * densityKernelGradient(r) / density_j;
                                fPress += -params.mass * nearSharedPressure * dir * nearDensityKernelGradient(r) / nearDensity_j;
                                let relativeSpeed = sortedParticles[j].v - particles[id.x].v;
                                fVisc += params.mass * relativeSpeed * viscosityKernelLaplacian(r) / density_j;
                            }
                        }
                    }
                }
            }
        }

        // // var cnt2 = 0.;
        // for (var j = 0u; j < n; j = j + 1) {
        //     if (id.x == j) {
        //         continue;
        //     }
        //     let density_j = particles[j].density;
        //     let nearDensity_j = particles[j].nearDensity;
        //     let pos_j = particles[j].position;
        //     let r2 = dot(pos_i - pos_j, pos_i - pos_j); 
        //     if (r2 < params.kernelRadiusPow2 && 1e-64 < r2) {
        //         let r = sqrt(r2);
        //         let pressure_i = params.stiffness * (density_i - params.restDensity);
        //         let pressure_j = params.stiffness * (density_j - params.restDensity);
        //         let nearPressure_i = params.nearStiffness * nearDensity_i;
        //         let nearPressure_j = params.nearStiffness * nearDensity_j;
        //         let sharedPressure = (pressure_i + pressure_j) / 2.0;
        //         let nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
        //         let dir = normalize(pos_j - pos_i);
        //         fPress += -params.mass * sharedPressure * dir * densityKernelGradient(r) / density_j;
        //         fPress += -params.mass * nearSharedPressure * dir * nearDensityKernelGradient(r) / nearDensity_j;
        //         let relativeSpeed = particles[j].v - particles[id.x].v;
        //         fVisc += params.mass * relativeSpeed * viscosityKernelLaplacian(r) / density_j;
        //     }
        // }

        fVisc *= params.viscosity;
        let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
        particles[id.x].force = fPress + fVisc + fGrv;
    }
}