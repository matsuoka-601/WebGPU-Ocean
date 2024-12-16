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

override mass: f32; 
override kernelRadius: f32;
override kernelRadiusPow2: f32;
override kernelRadiusPow5: f32;
override kernelRadiusPow6: f32; 

override xGrids: i32;
override yGrids: i32;
override zGrids: i32;
override gridCount: u32;
override cellSize: f32;
override xHalf: f32;
override yHalf: f32;
override zHalf: f32;
override offset: f32;

fn nearDensityKernel(r: f32) -> f32 {
    let scale = 15.0 / (3.1415926535 * kernelRadiusPow6);
    let d = kernelRadius - r;
    return scale * d * d * d;
}

fn densityKernel(r: f32) -> f32 {
    let scale = 15.0 / (2 * 3.1415926535 * kernelRadiusPow5);
    let d = kernelRadius - r;
    return scale * d * d;
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
fn computeDensity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        particles[id.x].density = 0.0;
        particles[id.x].nearDensity = 0.0;
        let n = arrayLength(&particles);
        let pos_i = particles[id.x].position;


        let v = cellPosition(pos_i);
        for (var dz = max(-1, -v.z); dz <= min(1, zGrids - v.z - 1); dz++) {
            for (var dy = max(-1, -v.y); dy <= min(1, yGrids - v.y - 1); dy++) {
                let dxMin = max(-1, -v.x);
                let dxMax = min(1, xGrids - v.x - 1);
                let startCellNum = cellNumberFromId(v.x + dxMin, v.y + dy, v.z + dz);
                let endCellNum = cellNumberFromId(v.x + dxMax, v.y + dy, v.z + dz);
                let start = prefixSum[startCellNum];
                let end = prefixSum[endCellNum + 1];
                if (start < gridCount && end < gridCount) {
                    for (var j = start; j < end; j++) {
                        let pos_j = sortedParticles[j].position;
                        let r2 = dot(pos_i - pos_j, pos_i - pos_j);
                        if (r2 < kernelRadiusPow2) {
                            particles[id.x].density += mass * densityKernel(sqrt(r2));
                            particles[id.x].nearDensity += mass * nearDensityKernel(sqrt(r2));
                        }
                    }
                }
            }
        }
        
        // for (var j = 0u; j < n; j = j + 1) {
        //     let pos_j = particles[j].position;
        //     let r2 = dot(pos_i - pos_j, pos_i - pos_j);
        //     if (r2 < kernelRadiusPow2) {
        //         particles[id.x].density += mass * densityKernel(sqrt(r2));
        //         particles[id.x].nearDensity += mass * nearDensityKernel(sqrt(r2));
        //     }
        // }
    }
}