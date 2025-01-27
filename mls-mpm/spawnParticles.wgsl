struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> init_box_size: vec3f;
@group(0) @binding(2) var<uniform> numParticles: i32;

@compute @workgroup_size(1)
fn spawn() {
    let dx: f32 = 0.6;
    let center: vec3f = init_box_size / 2;
    let beg: vec3f = vec3f(5);
    let base: vec3f = beg + vec3f(4.5 * dx, 4.5 * dx, 0);
    let vScale: f32 = 0.5;

    let dummy = numParticles;

    for (var i = 0; i < 10; i++) {
        for (var j = 0; j < 10; j++) {
            var offset = 10 * i + j;
            let pos = beg + vec3f(f32(i), f32(j), 0) * dx;
            particles[(numParticles - 1) - offset].position = pos;
            let vDir = normalize(center - pos);
            particles[(numParticles - 1) - offset].v = vDir * vScale; // 一定
            particles[(numParticles - 1) - offset].C = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
        }
    }
}