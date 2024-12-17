struct Particle {
    position: vec3f, 
    velocity: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> sortedParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> prefixSum: array<u32>;

override kernelRadius: f32;
override kernelRadiusPow2: f32;
override kernelRadiusPow5: f32;
override kernelRadiusPow6: f32;
override kernelRadiusPow9: f32;
override stiffness: f32;
override nearStiffness: f32;
override restDensity: f32;
override mass: f32;
override viscosity: f32;

override xGrids: i32;
override yGrids: i32;
override zGrids: i32;
override cellSize: f32;
override xHalf: f32;
override yHalf: f32;
override zHalf: f32;
override offset: f32;

fn densityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 15.0 / (3.1415926535 * kernelRadiusPow5); // pow 使うと遅いかも
    let d = kernelRadius - r;
    return scale * d;
}

fn nearDensityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * kernelRadiusPow6);
    let d = kernelRadius - r;
    return scale * d * d;
}

fn viscosityKernelLaplacian(r: f32) -> f32 {
    let scale: f32 = 315.0 / (64.0 * 3.1415926535 * kernelRadiusPow9);
    let dd = kernelRadius * kernelRadius - r * r;
    return scale * dd * dd * dd;
}

fn cellPosition(v: vec3f) -> vec3i {
    let xi = i32(floor((v.x + xHalf + offset) / cellSize));
    let yi = i32(floor((v.y + yHalf + offset) / cellSize));
    let zi = i32(floor((v.z + zHalf + offset) / cellSize));
    return vec3i(xi, yi, zi);
}

fn cellNumberFromId(xi: i32, yi: i32, zi: i32) -> i32 {
    return xi + yi * xGrids + zi * xGrids * yGrids;
}

@compute @workgroup_size(64)
fn computeForce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        let n = arrayLength(&particles);
        let density_i = particles[id.x].density;
        let nearDensity_i = particles[id.x].nearDensity;
        let pos_i = particles[id.x].position;
        var fPress = vec3(0.0, 0.0, 0.0);
        var fVisc = vec3(0.0, 0.0, 0.0);

        let v = cellPosition(pos_i);
        if (v.x < xGrids && v.y < yGrids && v.z < zGrids) {
            for (var dz = max(-1, -v.z); dz <= min(1, zGrids - v.z - 1); dz++) {
                for (var dy = max(-1, -v.y); dy <= min(1, yGrids - v.y - 1); dy++) {
                    let dxMin = max(-1, -v.x);
                    let dxMax = min(1, xGrids - v.x - 1);
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
                        if (r2 < kernelRadiusPow2 && 1e-64 < r2) {
                            let r = sqrt(r2);
                            let pressure_i = stiffness * (density_i - restDensity);
                            let pressure_j = stiffness * (density_j - restDensity);
                            let nearPressure_i = nearStiffness * nearDensity_i;
                            let nearPressure_j = nearStiffness * nearDensity_j;
                            let sharedPressure = (pressure_i + pressure_j) / 2.0;
                            let nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
                            let dir = normalize(pos_j - pos_i);
                            fPress += -mass * sharedPressure * dir * densityKernelGradient(r) / density_j;
                            fPress += -mass * nearSharedPressure * dir * nearDensityKernelGradient(r) / nearDensity_j;
                            let relativeSpeed = sortedParticles[j].velocity - particles[id.x].velocity;
                            fVisc += mass * relativeSpeed * viscosityKernelLaplacian(r) / density_j;
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
        //     if (r2 < kernelRadiusPow2 && 1e-64 < r2) {
        //         let r = sqrt(r2);
        //         let pressure_i = stiffness * (density_i - restDensity);
        //         let pressure_j = stiffness * (density_j - restDensity);
        //         let nearPressure_i = nearStiffness * nearDensity_i;
        //         let nearPressure_j = nearStiffness * nearDensity_j;
        //         let sharedPressure = (pressure_i + pressure_j) / 2.0;
        //         let nearSharedPressure = (nearPressure_i + nearPressure_j) / 2.0;
        //         let dir = normalize(pos_j - pos_i);
        //         fPress += -mass * sharedPressure * dir * densityKernelGradient(r) / density_j;
        //         fPress += -mass * nearSharedPressure * dir * nearDensityKernelGradient(r) / nearDensity_j;
        //         let relativeSpeed = particles[j].velocity - particles[id.x].velocity;
        //         fVisc += mass * relativeSpeed * viscosityKernelLaplacian(r) / density_j;
        //     }
        // }

        fVisc *= viscosity;
        let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
        particles[id.x].force = fPress + fVisc + fGrv;
    }
}