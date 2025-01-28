struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}
struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;
@group(0) @binding(3) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(4) var depthTexture: texture_2d<f32>;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;

    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        let uv = vec2f(0.0, 0.5);
        let iuv = uv * vec2f(896, 469);
        let depth: f32 = abs(textureLoad(depthTexture, vec2u(iuv), 0).x);
        var mouseCellIndex: u32 = 1000000000; // 適当な invalid 値
        var forceDir = vec3f(0.);
        if (depth < 1e5) {
            let mouseViewPos = getViewPosFromTexCoord(uv, iuv);
            let mouseWorldPos = uniforms.inv_view_matrix * vec4f(mouseViewPos, 1.); // 位置なので 1
            let mouseCellPos: vec3i = vec3i(floor(mouseWorldPos).xyz);
            mouseCellIndex =    u32(mouseCellPos.x) * u32(init_box_size.y) * u32(init_box_size.z) + 
                                u32(mouseCellPos.y) * u32(init_box_size.z) + 
                                u32(mouseCellPos.z);
            let center = real_box_size / 2;
            forceDir = normalize(mouseWorldPos.xyz - center);
        }

        let dt = dt;

        if (cells[id.x].mass > 0) { // 0 との比較は普通にしてよい
            var float_v: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx), 
                decodeFixedPoint(cells[id.x].vy), 
                decodeFixedPoint(cells[id.x].vz)
            );
            float_v /= decodeFixedPoint(cells[id.x].mass);
            
            if (mouseCellIndex == id.x) { 
                cells[id.x].vx = encodeFixedPoint(float_v.x); 
                cells[id.x].vy = encodeFixedPoint(float_v.y); 
                cells[id.x].vz = encodeFixedPoint(float_v.z + 100.); 
            } else {
                cells[id.x].vx = encodeFixedPoint(float_v.x);
                cells[id.x].vy = encodeFixedPoint(float_v.y); 
                cells[id.x].vz = encodeFixedPoint(float_v.z);
            } 

            var x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
            var y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
            var z: i32 = i32(id.x) % i32(init_box_size.z);
            // 整数を ceil したら，その整数に一致するかは確認する必要があり
            if (x < 2 || x > i32(ceil(real_box_size.x) - 3)) { cells[id.x].vx = 0; } 
            if (y < 2 || y > i32(ceil(real_box_size.y) - 3)) { cells[id.x].vy = 0; }
            if (z < 2 || z > i32(ceil(real_box_size.z) - 3)) { cells[id.x].vz = 0; }
        }
    }
}