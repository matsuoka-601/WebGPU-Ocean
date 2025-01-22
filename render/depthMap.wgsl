struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
}

struct FragmentOutput {
    @location(0) frag_color: vec4f, 
    @builtin(frag_depth) frag_depth: f32, 
}

struct RenderUniforms {
    texel_size: vec2f, 
    sphere_size: f32, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct PosVel {
    position: vec3f, 
    v: vec3f, 
}

@group(0) @binding(0) var<storage> particles: array<PosVel>;
@group(0) @binding(1) var<uniform> uniforms: RenderUniforms;

// assuming center is origin
fn computeStretchedVertex(position: vec2f, velocity_dir: vec2f, strength: f32) -> vec2f {
    // velocity_dir is obtained by normalizing velocity and set z element 0
    let stretch_offset: vec2f = dot(velocity_dir, position) * velocity_dir;
    return position + stretch_offset * strength;
}

fn area(v1: vec2f, v2: vec2f, v3: vec2f, v4: vec2f) -> f32 {
    let ab = v2 - v1;
    let ad = v4 - v1;
    let s = abs(ab.x * ad.y - ab.y * ad.x);
    return s;
}

fn scaleQuad(vel: vec2f, r: f32, strength: f32) -> f32 {
    let s1: f32 = r * r;
    let v1 = computeStretchedVertex(vec2f(0.5 * r, 0.5 * r), vel, strength);
    let v2 = computeStretchedVertex(vec2f(-0.5 * r, 0.5 * r), vel, strength);
    let v3 = computeStretchedVertex(vec2f(-0.5 * r, -0.5 * r), vel, strength);
    let v4 = computeStretchedVertex(vec2f(0.5 * r, -0.5 * r), vel, strength);
    let s2: f32 = area(v1, v2, v3, v4);
    return sqrt(s1 / s2);
}


@vertex
fn vs(    
    @builtin(vertex_index) vertex_index: u32, 
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    var corner_positions = array(
        vec2( 0.5,  0.5),
        vec2( 0.5, -0.5),
        vec2(-0.5, -0.5),
        vec2( 0.5,  0.5),
        vec2(-0.5, -0.5),
        vec2(-0.5,  0.5),
    );

    var corner = vec3(corner_positions[vertex_index] * uniforms.sphere_size, 0.0);
    let projected_velocity = (uniforms.view_matrix * vec4f(particles[instance_index].v, 0.0)).xy;
    let strength = 1.0;
    let stretched_position = computeStretchedVertex(corner_positions[vertex_index] * uniforms.sphere_size, projected_velocity, strength);
    corner = vec3(stretched_position, 0.0) * scaleQuad(projected_velocity, uniforms.sphere_size, strength);


    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    return VertexOutput(out_position, uv, view_position);
}

@fragment
fn fs(input: FragmentInput) -> FragmentOutput {
    var out: FragmentOutput;

    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard;
    }
    var normalz = sqrt(1.0 - r2);
    var normal = vec3(normalxy, normalz);

    var radius = uniforms.sphere_size / 2;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;

    out.frag_color = vec4(real_view_pos.z, 0., 0., 1.);
    return out;
}