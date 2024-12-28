struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 
override grid_res: i32;

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}


@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        if (cells[id.x].mass > 0) { // 0 との比較は普通にしてよい
            var float_v: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx), 
                decodeFixedPoint(cells[id.x].vy), 
                decodeFixedPoint(cells[id.x].vz)
            );
            float_v /= decodeFixedPoint(cells[id.x].mass);
            cells[id.x].vx = encodeFixedPoint(float_v.x);
            cells[id.x].vy = encodeFixedPoint(float_v.y + -0.3 * dt);
            cells[id.x].vz = encodeFixedPoint(float_v.z);

            var x: i32 = i32(id.x) / grid_res / grid_res;
            var y: i32 = (i32(id.x) / grid_res) % grid_res;
            var z: i32 = i32(id.x) % grid_res;
            if (x < 2 || x > grid_res - 3) { cells[id.x].vx = 0; }
            if (y < 2 || y > grid_res - 3) { cells[id.x].vy = 0; }
            if (z < 2 || z > grid_res - 3) { cells[id.x].vz = 0; }
        }
    }
}