struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}
struct Cell {
    vx: atomic<i32>, 
    vy: atomic<i32>, 
    vz: atomic<i32>, 
    mass: atomic<i32>, 
}

override fixed_point_multiplier: f32; 

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}


@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> numParticles: u32;

@compute @workgroup_size(64)
fn p2g_1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < numParticles) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        let C: mat3x3f = particle.C;

        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                            cell_idx.x + f32(gx) - 1., 
                            cell_idx.y + f32(gy) - 1.,
                            cell_idx.z + f32(gz) - 1.  
                        );
                    let cell_dist = (cell_x + 0.5f) - particle.position;

                    let Q: vec3f = C * cell_dist;

                    let mass_contrib: f32 = weight * 1.0; // assuming particle.mass = 1.0
                    let vel_contrib: vec3f = mass_contrib * (particle.v + Q);
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    atomicAdd(&cells[cell_index].mass, encodeFixedPoint(mass_contrib));
                    atomicAdd(&cells[cell_index].vx, encodeFixedPoint(vel_contrib.x));
                    atomicAdd(&cells[cell_index].vy, encodeFixedPoint(vel_contrib.y));
                    atomicAdd(&cells[cell_index].vz, encodeFixedPoint(vel_contrib.z));
                }
            }
        }
    }
}