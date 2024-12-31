(function(){const g=document.createElement("link").relList;if(g&&g.supports&&g.supports("modulepreload"))return;for(const b of document.querySelectorAll('link[rel="modulepreload"]'))E(b);new MutationObserver(b=>{for(const I of b)if(I.type==="childList")for(const L of I.addedNodes)L.tagName==="LINK"&&L.rel==="modulepreload"&&E(L)}).observe(document,{childList:!0,subtree:!0});function U(b){const I={};return b.integrity&&(I.integrity=b.integrity),b.referrerPolicy&&(I.referrerPolicy=b.referrerPolicy),b.crossOrigin==="use-credentials"?I.credentials="include":b.crossOrigin==="anonymous"?I.credentials="omit":I.credentials="same-origin",I}function E(b){if(b.ep)return;b.ep=!0;const I=U(b);fetch(b.href,I)}})();var zt=`struct VertexOutput {
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

struct Uniforms {
    size: f32, 
    view_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
}

struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

@group(0) @binding(0) var<storage> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

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

    
    
    let corner = vec3(corner_positions[vertex_index] * uniforms.size, 0.0);
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

    var radius = uniforms.size / 2;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;

    
    out.frag_color = vec4(real_view_pos.z, 0., 0., 1.);
    return out;
}`,Pt=`@group(0) @binding(1) var texture: texture_2d<f32>;

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var r = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    
    return vec4(0, r, r, 1.0);
}`,bt=`@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}

override depth_threshold: f32;  
override projected_particle_constant: f32; 
override max_filter_size: f32;
struct FilterUniforms {
    blur_dir: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    
    if (depth >= 1e4 || depth <= 0.) {
        return vec4f(vec3f(depth), 1.);
    }

    
    var filter_size: i32 = min(i32(max_filter_size), i32(ceil(projected_particle_constant / depth)));

    
    var sigma: f32 = f32(filter_size) / 3.0; 
    var two_sigma: f32 = 2.0 * sigma * sigma;
    var sigma_depth: f32 = depth_threshold / 3.0;
    var two_sigma_depth: f32 = 2.0 * sigma_depth * sigma_depth;

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * uniforms.blur_dir), 0).r);
        

        var rr: f32 = dot(coords, coords);
        var w: f32 = exp(-rr / two_sigma);

        var r_depth: f32 = sampled_depth - depth;
        var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);
        sum += sampled_depth * w * wd;
        wsum += w * wd;
    }

    
    
    
    

    
    

    
    

    
    
    
    

    sum /= wsum;
    
    
    

    return vec4f(sum, 0., 0., 1.);
}`,At=`@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FluidUniforms;
@group(0) @binding(3) var thickness_texture: texture_2d<f32>;
@group(0) @binding(4) var envmap_texture: texture_cube<f32>;

struct FluidUniforms {
    texel_size: vec2f, 
    inv_projection_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
    view_matrix: mat4x4f, 
    inv_view_matrix: mat4x4f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

fn computeViewPosFromUVDepth(tex_coord: vec2f, depth: f32) -> vec3f {
    var ndc: vec4f = vec4f(tex_coord.x * 2.0 - 1.0, 1.0 - 2.0 * tex_coord.y, 0.0, 1.0);
    
    ndc.z = -uniforms.projection_matrix[2].z + uniforms.projection_matrix[3].z / depth;
    ndc.w = 1.0;

    var eye_pos: vec4f = uniforms.inv_projection_matrix * ndc;

    return eye_pos.xyz / eye_pos.w;
}

fn getViewPosFromTexCoord(tex_coord: vec2f, iuv: vec2f) -> vec3f {
    var depth: f32 = abs(textureLoad(texture, vec2u(iuv), 0).x);
    return computeViewPosFromUVDepth(tex_coord, depth);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    let bgColor: vec3f = vec3f(0.8, 0.8, 0.8);

    if (depth >= 1e4 || depth <= 0.) {
        return vec4f(bgColor, 1.);
    }

    var viewPos: vec3f = computeViewPosFromUVDepth(input.uv, depth); 

    var ddx: vec3f = getViewPosFromTexCoord(input.uv + vec2f(uniforms.texel_size.x, 0.), input.iuv + vec2f(1.0, 0.0)) - viewPos; 
    var ddy: vec3f = getViewPosFromTexCoord(input.uv + vec2f(0., uniforms.texel_size.y), input.iuv + vec2f(0.0, 1.0)) - viewPos; 
    var ddx2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(-uniforms.texel_size.x, 0.), input.iuv + vec2f(-1.0, 0.0));
    var ddy2: vec3f = viewPos - getViewPosFromTexCoord(input.uv + vec2f(0., -uniforms.texel_size.y), input.iuv + vec2f(0.0, -1.0));

    if (abs(ddx.z) > abs(ddx2.z)) {
        ddx = ddx2; 
    }
    if (abs(ddy.z) > abs(ddy2.z)) {
        ddy = ddy2;
    }

    var normal: vec3f = -normalize(cross(ddx, ddy)); 
    var rayDir = normalize(viewPos);
    var lightDir = normalize((uniforms.view_matrix * vec4f(1, 1, 1, 0.)).xyz);
    var H: vec3f        = normalize(lightDir - rayDir);
    var specular: f32   = pow(max(0.0, dot(H, normal)), 250.);
    var diffuse: f32  = max(0.0, dot(lightDir, normal)) * 1.0;

    var density = 1.5; 
    
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    var diffuseColor = vec3f(0.085, 0.6375, 0.9);
    var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor)); 
    var refractionColor: vec3f = bgColor * transmittance;

    let F0 = 0.02;
    var fresnel: f32 = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0., 0.3);

    var reflectionDir: vec3f = reflect(rayDir, normal);
    var reflectionDirWorld: vec3f = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    var reflectionColor: vec3f = textureSampleLevel(envmap_texture, texture_sampler, reflectionDirWorld, 0.).rgb; 
    var finalColor = 1.0 * specular + mix(refractionColor, reflectionColor, fresnel);

    return vec4f(finalColor, 1.0);

    

    
    
    
    
    
    
    
    
    
    
    
}`,Tt=`struct VertexOutput {
  @builtin(position) position : vec4f,
  @location(0) uv : vec2f,
  @location(1) iuv : vec2f,
}

override screenWidth: f32;
override screenHeight: f32;

@vertex
fn vs(@builtin(vertex_index) vertex_index : u32) -> VertexOutput {
    var out: VertexOutput;

    var pos = array(
        vec2( 1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2(-1.0,  1.0),
    );

    var uv = array(
        vec2(1.0, 0.0),
        vec2(1.0, 1.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(0.0, 0.0),
    );

    out.position = vec4(pos[vertex_index], 0.0, 1.0);
    out.uv = uv[vertex_index];
    out.iuv = out.uv * vec2f(screenWidth, screenHeight);

    return out;
}`,Bt=`struct Uniforms {
    size: f32, 
    view_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
}

struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
}

struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

@group(0) @binding(0) var<storage> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

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

    
    
    let corner = vec3(corner_positions[vertex_index] * uniforms.size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    return VertexOutput(out_position, uv);
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var normalxy: vec2f = input.uv * 2.0 - 1.0;
    var r2: f32 = dot(normalxy, normalxy);
    if (r2 > 1.0) {
        discard;
    }
    var thickness: f32 = sqrt(1.0 - r2);
    let particle_alpha = 0.05;

    return vec4f(vec3f(particle_alpha * thickness), 1.0);
}`,Ft=`@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}

struct FilterUniforms {
    blur_dir: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    
    var thickness: f32 = textureLoad(texture, vec2u(input.iuv), 0).r;
    if (thickness == 0.) {
        return vec4f(0., 0., 0., 1.);
    }

    
    var filter_size: i32 = 30; 
    var sigma: f32 = f32(filter_size) / 3.0;
    var two_sigma: f32 = 2.0 * sigma * sigma;

    var sum = 0.;
    var wsum = 0.;

    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_thickness: f32 = textureLoad(texture, vec2u(input.iuv + uniforms.blur_dir * coords), 0).r;

        var w: f32 = exp(-coords.x * coords.x / two_sigma);

        sum += sampled_thickness * w;
        wsum += w;
    }

    sum /= wsum;

    return vec4f(sum, 0., 0., 1.);
}`,St=`struct VertexOutput {
    @builtin(position) position: vec4f, 
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
    @location(2) speed: f32, 
}

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) view_position: vec3f, 
    @location(2) speed: f32, 
}

struct FragmentOutput {
    @location(0) frag_color: vec4f, 
    @builtin(frag_depth) frag_depth: f32, 
}

struct Uniforms {
    size: f32, 
    view_matrix: mat4x4f, 
    projection_matrix: mat4x4f, 
}

struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

@group(0) @binding(0) var<storage> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

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

    let corner = vec3(corner_positions[vertex_index] * uniforms.size, 0.0);
    let uv = corner_positions[vertex_index] + 0.5;

    let real_position = particles[instance_index].position;
    let view_position = (uniforms.view_matrix * vec4f(real_position, 1.0)).xyz;

    let out_position = uniforms.projection_matrix * vec4f(view_position + corner, 1.0);

    let speed = sqrt(dot(particles[instance_index].v, particles[instance_index].v));

    return VertexOutput(out_position, uv, view_position, speed);
}

fn value_to_color(value: f32) -> vec3<f32> {
    
    let col0 = vec3f(0, 0.4, 0.8);
    let col1 = vec3f(35, 161, 165) / 256;
    let col2 = vec3f(95, 254, 150) / 256;
    let col3 = vec3f(243, 250, 49) / 256;
    let col4 = vec3f(255, 165, 0) / 256;

    if (0 <= value && value < 0.25) {
        let t = value / 0.25;
        return mix(col0, col1, t);
    } else if (0.25 <= value && value < 0.50) {
        let t = (value - 0.25) / 0.25;
        return mix(col1, col2, t);
    } else if (0.50 <= value && value < 0.75) {
        let t = (value - 0.50) / 0.25;
        return mix(col2, col3, t);
    } else {
        let t = (value - 0.75) / 0.25;
        return mix(col3, col4, t);
    }

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

    var radius = uniforms.size / 2;
    var real_view_pos: vec4f = vec4f(input.view_position + normal * radius, 1.0);
    var clip_space_pos: vec4f = uniforms.projection_matrix * real_view_pos;
    out.frag_depth = clip_space_pos.z / clip_space_pos.w;

    var diffuse: f32 = max(0.0, dot(normal, normalize(vec3(1.0, 1.0, 1.0))));
    var color: vec3f = value_to_color(input.speed / 1.5);

    out.frag_color = vec4(color * diffuse, 1.);
    return out;
}`,It=`struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        cells[id.x].mass = 0;
        cells[id.x].vx = 0;
        cells[id.x].vy = 0;
        cells[id.x].vz = 0;
    }
}`,Gt=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
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

@compute @workgroup_size(64)
fn p2g_1(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
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

                    let mass_contrib: f32 = weight * 1.0; 
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
}`,Ut=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}
struct Cell {
    vx: atomic<i32>, 
    vy: atomic<i32>, 
    vz: atomic<i32>, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override stiffness: f32;
override rest_density: f32;
override dynamic_viscosity: f32;
override dt: f32;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

@compute @workgroup_size(64)
fn p2g_2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        var density: f32 = 0.;
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                            cell_idx.x + f32(gx) - 1., 
                            cell_idx.y + f32(gy) - 1.,
                            cell_idx.z + f32(gz) - 1.  
                        );
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    density += decodeFixedPoint(cells[cell_index].mass) * weight;
                }
            }
        }

        let volume: f32 = 1.0 / density; 

        let pressure: f32 = max(-0.0, stiffness * (pow(density / rest_density, 5.) - 1));

        var stress: mat3x3f = mat3x3f(-pressure, 0, 0, 0, -pressure, 0, 0, 0, -pressure);
        let dudv: mat3x3f = particle.C;
        let strain: mat3x3f = dudv + transpose(dudv);
        stress += dynamic_viscosity * strain;

        let eq_16_term0 = -volume * 4 * stress * dt;

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
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    let momentum: vec3f = eq_16_term0 * weight * cell_dist;
                    atomicAdd(&cells[cell_index].vx, encodeFixedPoint(momentum.x));
                    atomicAdd(&cells[cell_index].vy, encodeFixedPoint(momentum.y));
                    atomicAdd(&cells[cell_index].vz, encodeFixedPoint(momentum.z));
                }
            }
        }
    }
}`,Et=`struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 

@group(0) @binding(0) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(1) var<uniform> real_box_size: vec3f;
@group(0) @binding(2) var<uniform> init_box_size: vec3f;

fn encodeFixedPoint(floating_point: f32) -> i32 {
	return i32(floating_point * fixed_point_multiplier);
}
fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@compute @workgroup_size(64)
fn updateGrid(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&cells)) {
        if (cells[id.x].mass > 0) { 
            var float_v: vec3f = vec3f(
                decodeFixedPoint(cells[id.x].vx), 
                decodeFixedPoint(cells[id.x].vy), 
                decodeFixedPoint(cells[id.x].vz)
            );
            float_v /= decodeFixedPoint(cells[id.x].mass);
            cells[id.x].vx = encodeFixedPoint(float_v.x);
            cells[id.x].vy = encodeFixedPoint(float_v.y + -0.3 * dt);
            cells[id.x].vz = encodeFixedPoint(float_v.z);

            var x: i32 = i32(id.x) / i32(init_box_size.z) / i32(init_box_size.y);
            var y: i32 = (i32(id.x) / i32(init_box_size.z)) % i32(init_box_size.y);
            var z: i32 = i32(id.x) % i32(init_box_size.z);
            
            if (x < 2 || x > i32(ceil(real_box_size.x) - 3)) { cells[id.x].vx = 0; } 
            if (y < 2 || y > i32(ceil(real_box_size.y) - 3)) { cells[id.x].vy = 0; }
            if (z < 2 || z > i32(ceil(real_box_size.z) - 3)) { cells[id.x].vz = 0; }
        }
    }
}`,Vt=`struct Particle {
    position: vec3f, 
    v: vec3f, 
    C: mat3x3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}
struct Cell {
    vx: i32, 
    vy: i32, 
    vz: i32, 
    mass: i32, 
}

override fixed_point_multiplier: f32; 
override dt: f32; 

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read> cells: array<Cell>;
@group(0) @binding(2) var<uniform> real_box_size: vec3f;
@group(0) @binding(3) var<uniform> init_box_size: vec3f;

fn decodeFixedPoint(fixed_point: i32) -> f32 {
	return f32(fixed_point) / fixed_point_multiplier;
}

@compute @workgroup_size(64)
fn g2p(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        particles[id.x].v = vec3f(0.);
        var weights: array<vec3f, 3>;

        let particle = particles[id.x];
        let cell_idx: vec3f = floor(particle.position);
        let cell_diff: vec3f = particle.position - (cell_idx + 0.5f);
        weights[0] = 0.5f * (0.5f - cell_diff) * (0.5f - cell_diff);
        weights[1] = 0.75f - cell_diff * cell_diff;
        weights[2] = 0.5f * (0.5f + cell_diff) * (0.5f + cell_diff);

        var B: mat3x3f = mat3x3f(vec3f(0.), vec3f(0.), vec3f(0.));
        for (var gx = 0; gx < 3; gx++) {
            for (var gy = 0; gy < 3; gy++) {
                for (var gz = 0; gz < 3; gz++) {
                    let weight: f32 = weights[gx].x * weights[gy].y * weights[gz].z;
                    let cell_x: vec3f = vec3f(
                        cell_idx.x + f32(gx) - 1., 
                        cell_idx.y + f32(gy) - 1.,
                        cell_idx.z + f32(gz) - 1.  
                    );
                    let cell_dist: vec3f = (cell_x + 0.5f) - particle.position;
                    let cell_index: i32 = 
                        i32(cell_x.x) * i32(init_box_size.y) * i32(init_box_size.z) + 
                        i32(cell_x.y) * i32(init_box_size.z) + 
                        i32(cell_x.z);
                    let weighted_velocity: vec3f = vec3f(
                        decodeFixedPoint(cells[cell_index].vx), 
                        decodeFixedPoint(cells[cell_index].vy), 
                        decodeFixedPoint(cells[cell_index].vz)
                    ) * weight;
                    let term: mat3x3f = mat3x3f(
                        weighted_velocity * cell_dist.x, 
                        weighted_velocity * cell_dist.y, 
                        weighted_velocity * cell_dist.z
                    );

                    B += term;

                    particles[id.x].v += weighted_velocity;
                }
            }
        }

        particles[id.x].C = B * 4.0f;
        particles[id.x].position += particles[id.x].v * dt;
        particles[id.x].position = vec3f(
            clamp(particles[id.x].position.x, 1., real_box_size.x - 2.), 
            clamp(particles[id.x].position.y, 1., real_box_size.y - 2.), 
            clamp(particles[id.x].position.z, 1., real_box_size.z - 2.)
        );
        
        let k = 2.0;
        let wall_stiffness = 0.2;
        let x_n: vec3f = particles[id.x].position + particles[id.x].v * dt * k;
        let wall_min: vec3f = vec3f(3.);
        let wall_max: vec3f = real_box_size - 4.;
        if (x_n.x < wall_min.x) { particles[id.x].v.x += wall_stiffness * (wall_min.x - x_n.x); }
        if (x_n.x > wall_max.x) { particles[id.x].v.x += wall_stiffness * (wall_max.x - x_n.x); }
        if (x_n.y < wall_min.y) { particles[id.x].v.y += wall_stiffness * (wall_min.y - x_n.y); }
        if (x_n.y > wall_max.y) { particles[id.x].v.y += wall_stiffness * (wall_max.y - x_n.y); }
        if (x_n.z < wall_min.z) { particles[id.x].v.z += wall_stiffness * (wall_min.z - x_n.z); }
        if (x_n.z > wall_max.z) { particles[id.x].v.z += wall_stiffness * (wall_max.z - x_n.z); }
    }
}`;function qt(f,g){return class extends f{constructor(...U){super(...U),g(this)}}}const Ot=qt(Array,f=>f.fill(0));let S=1e-6;function Nt(f){function g(n=0,r=0){const t=new f(2);return n!==void 0&&(t[0]=n,r!==void 0&&(t[1]=r)),t}const U=g;function E(n,r,t){const o=t??new f(2);return o[0]=n,o[1]=r,o}function b(n,r){const t=r??new f(2);return t[0]=Math.ceil(n[0]),t[1]=Math.ceil(n[1]),t}function I(n,r){const t=r??new f(2);return t[0]=Math.floor(n[0]),t[1]=Math.floor(n[1]),t}function L(n,r){const t=r??new f(2);return t[0]=Math.round(n[0]),t[1]=Math.round(n[1]),t}function Z(n,r=0,t=1,o){const a=o??new f(2);return a[0]=Math.min(t,Math.max(r,n[0])),a[1]=Math.min(t,Math.max(r,n[1])),a}function j(n,r,t){const o=t??new f(2);return o[0]=n[0]+r[0],o[1]=n[1]+r[1],o}function X(n,r,t,o){const a=o??new f(2);return a[0]=n[0]+r[0]*t,a[1]=n[1]+r[1]*t,a}function dn(n,r){const t=n[0],o=n[1],a=r[0],x=r[1],y=Math.sqrt(t*t+o*o),c=Math.sqrt(a*a+x*x),u=y*c,w=u&&bn(n,r)/u;return Math.acos(w)}function H(n,r,t){const o=t??new f(2);return o[0]=n[0]-r[0],o[1]=n[1]-r[1],o}const Fn=H;function pn(n,r){return Math.abs(n[0]-r[0])<S&&Math.abs(n[1]-r[1])<S}function zn(n,r){return n[0]===r[0]&&n[1]===r[1]}function Pn(n,r,t,o){const a=o??new f(2);return a[0]=n[0]+t*(r[0]-n[0]),a[1]=n[1]+t*(r[1]-n[1]),a}function Sn(n,r,t,o){const a=o??new f(2);return a[0]=n[0]+t[0]*(r[0]-n[0]),a[1]=n[1]+t[1]*(r[1]-n[1]),a}function N(n,r,t){const o=t??new f(2);return o[0]=Math.max(n[0],r[0]),o[1]=Math.max(n[1],r[1]),o}function wn(n,r,t){const o=t??new f(2);return o[0]=Math.min(n[0],r[0]),o[1]=Math.min(n[1],r[1]),o}function tn(n,r,t){const o=t??new f(2);return o[0]=n[0]*r,o[1]=n[1]*r,o}const Nn=tn;function vn(n,r,t){const o=t??new f(2);return o[0]=n[0]/r,o[1]=n[1]/r,o}function xn(n,r){const t=r??new f(2);return t[0]=1/n[0],t[1]=1/n[1],t}const In=xn;function en(n,r,t){const o=t??new f(3),a=n[0]*r[1]-n[1]*r[0];return o[0]=0,o[1]=0,o[2]=a,o}function bn(n,r){return n[0]*r[0]+n[1]*r[1]}function k(n){const r=n[0],t=n[1];return Math.sqrt(r*r+t*t)}const An=k;function G(n){const r=n[0],t=n[1];return r*r+t*t}const V=G;function F(n,r){const t=n[0]-r[0],o=n[1]-r[1];return Math.sqrt(t*t+o*o)}const Gn=F;function on(n,r){const t=n[0]-r[0],o=n[1]-r[1];return t*t+o*o}const Un=on;function sn(n,r){const t=r??new f(2),o=n[0],a=n[1],x=Math.sqrt(o*o+a*a);return x>1e-5?(t[0]=o/x,t[1]=a/x):(t[0]=0,t[1]=0),t}function En(n,r){const t=r??new f(2);return t[0]=-n[0],t[1]=-n[1],t}function q(n,r){const t=r??new f(2);return t[0]=n[0],t[1]=n[1],t}const gn=q;function Tn(n,r,t){const o=t??new f(2);return o[0]=n[0]*r[0],o[1]=n[1]*r[1],o}const Vn=Tn;function Bn(n,r,t){const o=t??new f(2);return o[0]=n[0]/r[0],o[1]=n[1]/r[1],o}const rn=Bn;function qn(n=1,r){const t=r??new f(2),o=Math.random()*2*Math.PI;return t[0]=Math.cos(o)*n,t[1]=Math.sin(o)*n,t}function i(n){const r=n??new f(2);return r[0]=0,r[1]=0,r}function d(n,r,t){const o=t??new f(2),a=n[0],x=n[1];return o[0]=a*r[0]+x*r[4]+r[12],o[1]=a*r[1]+x*r[5]+r[13],o}function e(n,r,t){const o=t??new f(2),a=n[0],x=n[1];return o[0]=r[0]*a+r[4]*x+r[8],o[1]=r[1]*a+r[5]*x+r[9],o}function s(n,r,t,o){const a=o??new f(2),x=n[0]-r[0],y=n[1]-r[1],c=Math.sin(t),u=Math.cos(t);return a[0]=x*u-y*c+r[0],a[1]=x*c+y*u+r[1],a}function l(n,r,t){const o=t??new f(2);return sn(n,o),tn(o,r,o)}function p(n,r,t){const o=t??new f(2);return k(n)>r?l(n,r,o):q(n,o)}function _(n,r,t){const o=t??new f(2);return Pn(n,r,.5,o)}return{create:g,fromValues:U,set:E,ceil:b,floor:I,round:L,clamp:Z,add:j,addScaled:X,angle:dn,subtract:H,sub:Fn,equalsApproximately:pn,equals:zn,lerp:Pn,lerpV:Sn,max:N,min:wn,mulScalar:tn,scale:Nn,divScalar:vn,inverse:xn,invert:In,cross:en,dot:bn,length:k,len:An,lengthSq:G,lenSq:V,distance:F,dist:Gn,distanceSq:on,distSq:Un,normalize:sn,negate:En,copy:q,clone:gn,multiply:Tn,mul:Vn,divide:Bn,div:rn,random:qn,zero:i,transformMat4:d,transformMat3:e,rotate:s,setLength:l,truncate:p,midpoint:_}}const wt=new Map;function mt(f){let g=wt.get(f);return g||(g=Nt(f),wt.set(f,g)),g}function Lt(f){function g(c,u,w){const v=new f(3);return c!==void 0&&(v[0]=c,u!==void 0&&(v[1]=u,w!==void 0&&(v[2]=w))),v}const U=g;function E(c,u,w,v){const h=v??new f(3);return h[0]=c,h[1]=u,h[2]=w,h}function b(c,u){const w=u??new f(3);return w[0]=Math.ceil(c[0]),w[1]=Math.ceil(c[1]),w[2]=Math.ceil(c[2]),w}function I(c,u){const w=u??new f(3);return w[0]=Math.floor(c[0]),w[1]=Math.floor(c[1]),w[2]=Math.floor(c[2]),w}function L(c,u){const w=u??new f(3);return w[0]=Math.round(c[0]),w[1]=Math.round(c[1]),w[2]=Math.round(c[2]),w}function Z(c,u=0,w=1,v){const h=v??new f(3);return h[0]=Math.min(w,Math.max(u,c[0])),h[1]=Math.min(w,Math.max(u,c[1])),h[2]=Math.min(w,Math.max(u,c[2])),h}function j(c,u,w){const v=w??new f(3);return v[0]=c[0]+u[0],v[1]=c[1]+u[1],v[2]=c[2]+u[2],v}function X(c,u,w,v){const h=v??new f(3);return h[0]=c[0]+u[0]*w,h[1]=c[1]+u[1]*w,h[2]=c[2]+u[2]*w,h}function dn(c,u){const w=c[0],v=c[1],h=c[2],m=u[0],M=u[1],D=u[2],A=Math.sqrt(w*w+v*v+h*h),z=Math.sqrt(m*m+M*M+D*D),P=A*z,T=P&&bn(c,u)/P;return Math.acos(T)}function H(c,u,w){const v=w??new f(3);return v[0]=c[0]-u[0],v[1]=c[1]-u[1],v[2]=c[2]-u[2],v}const Fn=H;function pn(c,u){return Math.abs(c[0]-u[0])<S&&Math.abs(c[1]-u[1])<S&&Math.abs(c[2]-u[2])<S}function zn(c,u){return c[0]===u[0]&&c[1]===u[1]&&c[2]===u[2]}function Pn(c,u,w,v){const h=v??new f(3);return h[0]=c[0]+w*(u[0]-c[0]),h[1]=c[1]+w*(u[1]-c[1]),h[2]=c[2]+w*(u[2]-c[2]),h}function Sn(c,u,w,v){const h=v??new f(3);return h[0]=c[0]+w[0]*(u[0]-c[0]),h[1]=c[1]+w[1]*(u[1]-c[1]),h[2]=c[2]+w[2]*(u[2]-c[2]),h}function N(c,u,w){const v=w??new f(3);return v[0]=Math.max(c[0],u[0]),v[1]=Math.max(c[1],u[1]),v[2]=Math.max(c[2],u[2]),v}function wn(c,u,w){const v=w??new f(3);return v[0]=Math.min(c[0],u[0]),v[1]=Math.min(c[1],u[1]),v[2]=Math.min(c[2],u[2]),v}function tn(c,u,w){const v=w??new f(3);return v[0]=c[0]*u,v[1]=c[1]*u,v[2]=c[2]*u,v}const Nn=tn;function vn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u,v[1]=c[1]/u,v[2]=c[2]/u,v}function xn(c,u){const w=u??new f(3);return w[0]=1/c[0],w[1]=1/c[1],w[2]=1/c[2],w}const In=xn;function en(c,u,w){const v=w??new f(3),h=c[2]*u[0]-c[0]*u[2],m=c[0]*u[1]-c[1]*u[0];return v[0]=c[1]*u[2]-c[2]*u[1],v[1]=h,v[2]=m,v}function bn(c,u){return c[0]*u[0]+c[1]*u[1]+c[2]*u[2]}function k(c){const u=c[0],w=c[1],v=c[2];return Math.sqrt(u*u+w*w+v*v)}const An=k;function G(c){const u=c[0],w=c[1],v=c[2];return u*u+w*w+v*v}const V=G;function F(c,u){const w=c[0]-u[0],v=c[1]-u[1],h=c[2]-u[2];return Math.sqrt(w*w+v*v+h*h)}const Gn=F;function on(c,u){const w=c[0]-u[0],v=c[1]-u[1],h=c[2]-u[2];return w*w+v*v+h*h}const Un=on;function sn(c,u){const w=u??new f(3),v=c[0],h=c[1],m=c[2],M=Math.sqrt(v*v+h*h+m*m);return M>1e-5?(w[0]=v/M,w[1]=h/M,w[2]=m/M):(w[0]=0,w[1]=0,w[2]=0),w}function En(c,u){const w=u??new f(3);return w[0]=-c[0],w[1]=-c[1],w[2]=-c[2],w}function q(c,u){const w=u??new f(3);return w[0]=c[0],w[1]=c[1],w[2]=c[2],w}const gn=q;function Tn(c,u,w){const v=w??new f(3);return v[0]=c[0]*u[0],v[1]=c[1]*u[1],v[2]=c[2]*u[2],v}const Vn=Tn;function Bn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u[0],v[1]=c[1]/u[1],v[2]=c[2]/u[2],v}const rn=Bn;function qn(c=1,u){const w=u??new f(3),v=Math.random()*2*Math.PI,h=Math.random()*2-1,m=Math.sqrt(1-h*h)*c;return w[0]=Math.cos(v)*m,w[1]=Math.sin(v)*m,w[2]=h*c,w}function i(c){const u=c??new f(3);return u[0]=0,u[1]=0,u[2]=0,u}function d(c,u,w){const v=w??new f(3),h=c[0],m=c[1],M=c[2],D=u[3]*h+u[7]*m+u[11]*M+u[15]||1;return v[0]=(u[0]*h+u[4]*m+u[8]*M+u[12])/D,v[1]=(u[1]*h+u[5]*m+u[9]*M+u[13])/D,v[2]=(u[2]*h+u[6]*m+u[10]*M+u[14])/D,v}function e(c,u,w){const v=w??new f(3),h=c[0],m=c[1],M=c[2];return v[0]=h*u[0*4+0]+m*u[1*4+0]+M*u[2*4+0],v[1]=h*u[0*4+1]+m*u[1*4+1]+M*u[2*4+1],v[2]=h*u[0*4+2]+m*u[1*4+2]+M*u[2*4+2],v}function s(c,u,w){const v=w??new f(3),h=c[0],m=c[1],M=c[2];return v[0]=h*u[0]+m*u[4]+M*u[8],v[1]=h*u[1]+m*u[5]+M*u[9],v[2]=h*u[2]+m*u[6]+M*u[10],v}function l(c,u,w){const v=w??new f(3),h=u[0],m=u[1],M=u[2],D=u[3]*2,A=c[0],z=c[1],P=c[2],T=m*P-M*z,B=M*A-h*P,O=h*z-m*A;return v[0]=A+T*D+(m*O-M*B)*2,v[1]=z+B*D+(M*T-h*O)*2,v[2]=P+O*D+(h*B-m*T)*2,v}function p(c,u){const w=u??new f(3);return w[0]=c[12],w[1]=c[13],w[2]=c[14],w}function _(c,u,w){const v=w??new f(3),h=u*4;return v[0]=c[h+0],v[1]=c[h+1],v[2]=c[h+2],v}function n(c,u){const w=u??new f(3),v=c[0],h=c[1],m=c[2],M=c[4],D=c[5],A=c[6],z=c[8],P=c[9],T=c[10];return w[0]=Math.sqrt(v*v+h*h+m*m),w[1]=Math.sqrt(M*M+D*D+A*A),w[2]=Math.sqrt(z*z+P*P+T*T),w}function r(c,u,w,v){const h=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0],M[1]=m[1]*Math.cos(w)-m[2]*Math.sin(w),M[2]=m[1]*Math.sin(w)+m[2]*Math.cos(w),h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function t(c,u,w,v){const h=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[2]*Math.sin(w)+m[0]*Math.cos(w),M[1]=m[1],M[2]=m[2]*Math.cos(w)-m[0]*Math.sin(w),h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function o(c,u,w,v){const h=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0]*Math.cos(w)-m[1]*Math.sin(w),M[1]=m[0]*Math.sin(w)+m[1]*Math.cos(w),M[2]=m[2],h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function a(c,u,w){const v=w??new f(3);return sn(c,v),tn(v,u,v)}function x(c,u,w){const v=w??new f(3);return k(c)>u?a(c,u,v):q(c,v)}function y(c,u,w){const v=w??new f(3);return Pn(c,u,.5,v)}return{create:g,fromValues:U,set:E,ceil:b,floor:I,round:L,clamp:Z,add:j,addScaled:X,angle:dn,subtract:H,sub:Fn,equalsApproximately:pn,equals:zn,lerp:Pn,lerpV:Sn,max:N,min:wn,mulScalar:tn,scale:Nn,divScalar:vn,inverse:xn,invert:In,cross:en,dot:bn,length:k,len:An,lengthSq:G,lenSq:V,distance:F,dist:Gn,distanceSq:on,distSq:Un,normalize:sn,negate:En,copy:q,clone:gn,multiply:Tn,mul:Vn,divide:Bn,div:rn,random:qn,zero:i,transformMat4:d,transformMat4Upper3x3:e,transformMat3:s,transformQuat:l,getTranslation:p,getAxis:_,getScaling:n,rotateX:r,rotateY:t,rotateZ:o,setLength:a,truncate:x,midpoint:y}}const xt=new Map;function ut(f){let g=xt.get(f);return g||(g=Lt(f),xt.set(f,g)),g}function kt(f){const g=mt(f),U=ut(f);function E(i,d,e,s,l,p,_,n,r){const t=new f(12);return t[3]=0,t[7]=0,t[11]=0,i!==void 0&&(t[0]=i,d!==void 0&&(t[1]=d,e!==void 0&&(t[2]=e,s!==void 0&&(t[4]=s,l!==void 0&&(t[5]=l,p!==void 0&&(t[6]=p,_!==void 0&&(t[8]=_,n!==void 0&&(t[9]=n,r!==void 0&&(t[10]=r))))))))),t}function b(i,d,e,s,l,p,_,n,r,t){const o=t??new f(12);return o[0]=i,o[1]=d,o[2]=e,o[3]=0,o[4]=s,o[5]=l,o[6]=p,o[7]=0,o[8]=_,o[9]=n,o[10]=r,o[11]=0,o}function I(i,d){const e=d??new f(12);return e[0]=i[0],e[1]=i[1],e[2]=i[2],e[3]=0,e[4]=i[4],e[5]=i[5],e[6]=i[6],e[7]=0,e[8]=i[8],e[9]=i[9],e[10]=i[10],e[11]=0,e}function L(i,d){const e=d??new f(12),s=i[0],l=i[1],p=i[2],_=i[3],n=s+s,r=l+l,t=p+p,o=s*n,a=l*n,x=l*r,y=p*n,c=p*r,u=p*t,w=_*n,v=_*r,h=_*t;return e[0]=1-x-u,e[1]=a+h,e[2]=y-v,e[3]=0,e[4]=a-h,e[5]=1-o-u,e[6]=c+w,e[7]=0,e[8]=y+v,e[9]=c-w,e[10]=1-o-x,e[11]=0,e}function Z(i,d){const e=d??new f(12);return e[0]=-i[0],e[1]=-i[1],e[2]=-i[2],e[4]=-i[4],e[5]=-i[5],e[6]=-i[6],e[8]=-i[8],e[9]=-i[9],e[10]=-i[10],e}function j(i,d){const e=d??new f(12);return e[0]=i[0],e[1]=i[1],e[2]=i[2],e[4]=i[4],e[5]=i[5],e[6]=i[6],e[8]=i[8],e[9]=i[9],e[10]=i[10],e}const X=j;function dn(i,d){return Math.abs(i[0]-d[0])<S&&Math.abs(i[1]-d[1])<S&&Math.abs(i[2]-d[2])<S&&Math.abs(i[4]-d[4])<S&&Math.abs(i[5]-d[5])<S&&Math.abs(i[6]-d[6])<S&&Math.abs(i[8]-d[8])<S&&Math.abs(i[9]-d[9])<S&&Math.abs(i[10]-d[10])<S}function H(i,d){return i[0]===d[0]&&i[1]===d[1]&&i[2]===d[2]&&i[4]===d[4]&&i[5]===d[5]&&i[6]===d[6]&&i[8]===d[8]&&i[9]===d[9]&&i[10]===d[10]}function Fn(i){const d=i??new f(12);return d[0]=1,d[1]=0,d[2]=0,d[4]=0,d[5]=1,d[6]=0,d[8]=0,d[9]=0,d[10]=1,d}function pn(i,d){const e=d??new f(12);if(e===i){let x;return x=i[1],i[1]=i[4],i[4]=x,x=i[2],i[2]=i[8],i[8]=x,x=i[6],i[6]=i[9],i[9]=x,e}const s=i[0*4+0],l=i[0*4+1],p=i[0*4+2],_=i[1*4+0],n=i[1*4+1],r=i[1*4+2],t=i[2*4+0],o=i[2*4+1],a=i[2*4+2];return e[0]=s,e[1]=_,e[2]=t,e[4]=l,e[5]=n,e[6]=o,e[8]=p,e[9]=r,e[10]=a,e}function zn(i,d){const e=d??new f(12),s=i[0*4+0],l=i[0*4+1],p=i[0*4+2],_=i[1*4+0],n=i[1*4+1],r=i[1*4+2],t=i[2*4+0],o=i[2*4+1],a=i[2*4+2],x=a*n-r*o,y=-a*_+r*t,c=o*_-n*t,u=1/(s*x+l*y+p*c);return e[0]=x*u,e[1]=(-a*l+p*o)*u,e[2]=(r*l-p*n)*u,e[4]=y*u,e[5]=(a*s-p*t)*u,e[6]=(-r*s+p*_)*u,e[8]=c*u,e[9]=(-o*s+l*t)*u,e[10]=(n*s-l*_)*u,e}function Pn(i){const d=i[0],e=i[0*4+1],s=i[0*4+2],l=i[1*4+0],p=i[1*4+1],_=i[1*4+2],n=i[2*4+0],r=i[2*4+1],t=i[2*4+2];return d*(p*t-r*_)-l*(e*t-r*s)+n*(e*_-p*s)}const Sn=zn;function N(i,d,e){const s=e??new f(12),l=i[0],p=i[1],_=i[2],n=i[4],r=i[5],t=i[6],o=i[8],a=i[9],x=i[10],y=d[0],c=d[1],u=d[2],w=d[4],v=d[5],h=d[6],m=d[8],M=d[9],D=d[10];return s[0]=l*y+n*c+o*u,s[1]=p*y+r*c+a*u,s[2]=_*y+t*c+x*u,s[4]=l*w+n*v+o*h,s[5]=p*w+r*v+a*h,s[6]=_*w+t*v+x*h,s[8]=l*m+n*M+o*D,s[9]=p*m+r*M+a*D,s[10]=_*m+t*M+x*D,s}const wn=N;function tn(i,d,e){const s=e??Fn();return i!==s&&(s[0]=i[0],s[1]=i[1],s[2]=i[2],s[4]=i[4],s[5]=i[5],s[6]=i[6]),s[8]=d[0],s[9]=d[1],s[10]=1,s}function Nn(i,d){const e=d??g.create();return e[0]=i[8],e[1]=i[9],e}function vn(i,d,e){const s=e??g.create(),l=d*4;return s[0]=i[l+0],s[1]=i[l+1],s}function xn(i,d,e,s){const l=s===i?i:j(i,s),p=e*4;return l[p+0]=d[0],l[p+1]=d[1],l}function In(i,d){const e=d??g.create(),s=i[0],l=i[1],p=i[4],_=i[5];return e[0]=Math.sqrt(s*s+l*l),e[1]=Math.sqrt(p*p+_*_),e}function en(i,d){const e=d??U.create(),s=i[0],l=i[1],p=i[2],_=i[4],n=i[5],r=i[6],t=i[8],o=i[9],a=i[10];return e[0]=Math.sqrt(s*s+l*l+p*p),e[1]=Math.sqrt(_*_+n*n+r*r),e[2]=Math.sqrt(t*t+o*o+a*a),e}function bn(i,d){const e=d??new f(12);return e[0]=1,e[1]=0,e[2]=0,e[4]=0,e[5]=1,e[6]=0,e[8]=i[0],e[9]=i[1],e[10]=1,e}function k(i,d,e){const s=e??new f(12),l=d[0],p=d[1],_=i[0],n=i[1],r=i[2],t=i[1*4+0],o=i[1*4+1],a=i[1*4+2],x=i[2*4+0],y=i[2*4+1],c=i[2*4+2];return i!==s&&(s[0]=_,s[1]=n,s[2]=r,s[4]=t,s[5]=o,s[6]=a),s[8]=_*l+t*p+x,s[9]=n*l+o*p+y,s[10]=r*l+a*p+c,s}function An(i,d){const e=d??new f(12),s=Math.cos(i),l=Math.sin(i);return e[0]=s,e[1]=l,e[2]=0,e[4]=-l,e[5]=s,e[6]=0,e[8]=0,e[9]=0,e[10]=1,e}function G(i,d,e){const s=e??new f(12),l=i[0*4+0],p=i[0*4+1],_=i[0*4+2],n=i[1*4+0],r=i[1*4+1],t=i[1*4+2],o=Math.cos(d),a=Math.sin(d);return s[0]=o*l+a*n,s[1]=o*p+a*r,s[2]=o*_+a*t,s[4]=o*n-a*l,s[5]=o*r-a*p,s[6]=o*t-a*_,i!==s&&(s[8]=i[8],s[9]=i[9],s[10]=i[10]),s}function V(i,d){const e=d??new f(12),s=Math.cos(i),l=Math.sin(i);return e[0]=1,e[1]=0,e[2]=0,e[4]=0,e[5]=s,e[6]=l,e[8]=0,e[9]=-l,e[10]=s,e}function F(i,d,e){const s=e??new f(12),l=i[4],p=i[5],_=i[6],n=i[8],r=i[9],t=i[10],o=Math.cos(d),a=Math.sin(d);return s[4]=o*l+a*n,s[5]=o*p+a*r,s[6]=o*_+a*t,s[8]=o*n-a*l,s[9]=o*r-a*p,s[10]=o*t-a*_,i!==s&&(s[0]=i[0],s[1]=i[1],s[2]=i[2]),s}function Gn(i,d){const e=d??new f(12),s=Math.cos(i),l=Math.sin(i);return e[0]=s,e[1]=0,e[2]=-l,e[4]=0,e[5]=1,e[6]=0,e[8]=l,e[9]=0,e[10]=s,e}function on(i,d,e){const s=e??new f(12),l=i[0*4+0],p=i[0*4+1],_=i[0*4+2],n=i[2*4+0],r=i[2*4+1],t=i[2*4+2],o=Math.cos(d),a=Math.sin(d);return s[0]=o*l-a*n,s[1]=o*p-a*r,s[2]=o*_-a*t,s[8]=o*n+a*l,s[9]=o*r+a*p,s[10]=o*t+a*_,i!==s&&(s[4]=i[4],s[5]=i[5],s[6]=i[6]),s}const Un=An,sn=G;function En(i,d){const e=d??new f(12);return e[0]=i[0],e[1]=0,e[2]=0,e[4]=0,e[5]=i[1],e[6]=0,e[8]=0,e[9]=0,e[10]=1,e}function q(i,d,e){const s=e??new f(12),l=d[0],p=d[1];return s[0]=l*i[0*4+0],s[1]=l*i[0*4+1],s[2]=l*i[0*4+2],s[4]=p*i[1*4+0],s[5]=p*i[1*4+1],s[6]=p*i[1*4+2],i!==s&&(s[8]=i[8],s[9]=i[9],s[10]=i[10]),s}function gn(i,d){const e=d??new f(12);return e[0]=i[0],e[1]=0,e[2]=0,e[4]=0,e[5]=i[1],e[6]=0,e[8]=0,e[9]=0,e[10]=i[2],e}function Tn(i,d,e){const s=e??new f(12),l=d[0],p=d[1],_=d[2];return s[0]=l*i[0*4+0],s[1]=l*i[0*4+1],s[2]=l*i[0*4+2],s[4]=p*i[1*4+0],s[5]=p*i[1*4+1],s[6]=p*i[1*4+2],s[8]=_*i[2*4+0],s[9]=_*i[2*4+1],s[10]=_*i[2*4+2],s}function Vn(i,d){const e=d??new f(12);return e[0]=i,e[1]=0,e[2]=0,e[4]=0,e[5]=i,e[6]=0,e[8]=0,e[9]=0,e[10]=1,e}function Bn(i,d,e){const s=e??new f(12);return s[0]=d*i[0*4+0],s[1]=d*i[0*4+1],s[2]=d*i[0*4+2],s[4]=d*i[1*4+0],s[5]=d*i[1*4+1],s[6]=d*i[1*4+2],i!==s&&(s[8]=i[8],s[9]=i[9],s[10]=i[10]),s}function rn(i,d){const e=d??new f(12);return e[0]=i,e[1]=0,e[2]=0,e[4]=0,e[5]=i,e[6]=0,e[8]=0,e[9]=0,e[10]=i,e}function qn(i,d,e){const s=e??new f(12);return s[0]=d*i[0*4+0],s[1]=d*i[0*4+1],s[2]=d*i[0*4+2],s[4]=d*i[1*4+0],s[5]=d*i[1*4+1],s[6]=d*i[1*4+2],s[8]=d*i[2*4+0],s[9]=d*i[2*4+1],s[10]=d*i[2*4+2],s}return{clone:X,create:E,set:b,fromMat4:I,fromQuat:L,negate:Z,copy:j,equalsApproximately:dn,equals:H,identity:Fn,transpose:pn,inverse:zn,invert:Sn,determinant:Pn,mul:wn,multiply:N,setTranslation:tn,getTranslation:Nn,getAxis:vn,setAxis:xn,getScaling:In,get3DScaling:en,translation:bn,translate:k,rotation:An,rotate:G,rotationX:V,rotateX:F,rotationY:Gn,rotateY:on,rotationZ:Un,rotateZ:sn,scaling:En,scale:q,uniformScaling:Vn,uniformScale:Bn,scaling3D:gn,scale3D:Tn,uniformScaling3D:rn,uniformScale3D:qn}}const vt=new Map;function Xt(f){let g=vt.get(f);return g||(g=kt(f),vt.set(f,g)),g}function Yt(f){const g=ut(f);function U(n,r,t,o,a,x,y,c,u,w,v,h,m,M,D,A){const z=new f(16);return n!==void 0&&(z[0]=n,r!==void 0&&(z[1]=r,t!==void 0&&(z[2]=t,o!==void 0&&(z[3]=o,a!==void 0&&(z[4]=a,x!==void 0&&(z[5]=x,y!==void 0&&(z[6]=y,c!==void 0&&(z[7]=c,u!==void 0&&(z[8]=u,w!==void 0&&(z[9]=w,v!==void 0&&(z[10]=v,h!==void 0&&(z[11]=h,m!==void 0&&(z[12]=m,M!==void 0&&(z[13]=M,D!==void 0&&(z[14]=D,A!==void 0&&(z[15]=A)))))))))))))))),z}function E(n,r,t,o,a,x,y,c,u,w,v,h,m,M,D,A,z){const P=z??new f(16);return P[0]=n,P[1]=r,P[2]=t,P[3]=o,P[4]=a,P[5]=x,P[6]=y,P[7]=c,P[8]=u,P[9]=w,P[10]=v,P[11]=h,P[12]=m,P[13]=M,P[14]=D,P[15]=A,P}function b(n,r){const t=r??new f(16);return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=0,t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=0,t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function I(n,r){const t=r??new f(16),o=n[0],a=n[1],x=n[2],y=n[3],c=o+o,u=a+a,w=x+x,v=o*c,h=a*c,m=a*u,M=x*c,D=x*u,A=x*w,z=y*c,P=y*u,T=y*w;return t[0]=1-m-A,t[1]=h+T,t[2]=M-P,t[3]=0,t[4]=h-T,t[5]=1-v-A,t[6]=D+z,t[7]=0,t[8]=M+P,t[9]=D-z,t[10]=1-v-m,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function L(n,r){const t=r??new f(16);return t[0]=-n[0],t[1]=-n[1],t[2]=-n[2],t[3]=-n[3],t[4]=-n[4],t[5]=-n[5],t[6]=-n[6],t[7]=-n[7],t[8]=-n[8],t[9]=-n[9],t[10]=-n[10],t[11]=-n[11],t[12]=-n[12],t[13]=-n[13],t[14]=-n[14],t[15]=-n[15],t}function Z(n,r){const t=r??new f(16);return t[0]=n[0],t[1]=n[1],t[2]=n[2],t[3]=n[3],t[4]=n[4],t[5]=n[5],t[6]=n[6],t[7]=n[7],t[8]=n[8],t[9]=n[9],t[10]=n[10],t[11]=n[11],t[12]=n[12],t[13]=n[13],t[14]=n[14],t[15]=n[15],t}const j=Z;function X(n,r){return Math.abs(n[0]-r[0])<S&&Math.abs(n[1]-r[1])<S&&Math.abs(n[2]-r[2])<S&&Math.abs(n[3]-r[3])<S&&Math.abs(n[4]-r[4])<S&&Math.abs(n[5]-r[5])<S&&Math.abs(n[6]-r[6])<S&&Math.abs(n[7]-r[7])<S&&Math.abs(n[8]-r[8])<S&&Math.abs(n[9]-r[9])<S&&Math.abs(n[10]-r[10])<S&&Math.abs(n[11]-r[11])<S&&Math.abs(n[12]-r[12])<S&&Math.abs(n[13]-r[13])<S&&Math.abs(n[14]-r[14])<S&&Math.abs(n[15]-r[15])<S}function dn(n,r){return n[0]===r[0]&&n[1]===r[1]&&n[2]===r[2]&&n[3]===r[3]&&n[4]===r[4]&&n[5]===r[5]&&n[6]===r[6]&&n[7]===r[7]&&n[8]===r[8]&&n[9]===r[9]&&n[10]===r[10]&&n[11]===r[11]&&n[12]===r[12]&&n[13]===r[13]&&n[14]===r[14]&&n[15]===r[15]}function H(n){const r=n??new f(16);return r[0]=1,r[1]=0,r[2]=0,r[3]=0,r[4]=0,r[5]=1,r[6]=0,r[7]=0,r[8]=0,r[9]=0,r[10]=1,r[11]=0,r[12]=0,r[13]=0,r[14]=0,r[15]=1,r}function Fn(n,r){const t=r??new f(16);if(t===n){let B;return B=n[1],n[1]=n[4],n[4]=B,B=n[2],n[2]=n[8],n[8]=B,B=n[3],n[3]=n[12],n[12]=B,B=n[6],n[6]=n[9],n[9]=B,B=n[7],n[7]=n[13],n[13]=B,B=n[11],n[11]=n[14],n[14]=B,t}const o=n[0*4+0],a=n[0*4+1],x=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],h=n[2*4+0],m=n[2*4+1],M=n[2*4+2],D=n[2*4+3],A=n[3*4+0],z=n[3*4+1],P=n[3*4+2],T=n[3*4+3];return t[0]=o,t[1]=c,t[2]=h,t[3]=A,t[4]=a,t[5]=u,t[6]=m,t[7]=z,t[8]=x,t[9]=w,t[10]=M,t[11]=P,t[12]=y,t[13]=v,t[14]=D,t[15]=T,t}function pn(n,r){const t=r??new f(16),o=n[0*4+0],a=n[0*4+1],x=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],h=n[2*4+0],m=n[2*4+1],M=n[2*4+2],D=n[2*4+3],A=n[3*4+0],z=n[3*4+1],P=n[3*4+2],T=n[3*4+3],B=M*T,O=P*D,Y=w*T,R=P*v,Q=w*D,cn=M*v,an=x*T,K=P*y,un=x*D,ln=M*y,fn=x*v,$=w*y,J=h*z,C=A*m,_n=c*z,W=A*u,hn=c*m,$n=h*u,Wn=o*z,Zn=A*a,Qn=o*m,kn=h*a,Kn=o*u,Cn=c*a,Ln=B*u+R*m+Q*z-(O*u+Y*m+cn*z),nt=O*a+an*m+ln*z-(B*a+K*m+un*z),ot=Y*a+K*u+fn*z-(R*a+an*u+$*z),tt=cn*a+un*u+$*m-(Q*a+ln*u+fn*m),nn=1/(o*Ln+c*nt+h*ot+A*tt);return t[0]=nn*Ln,t[1]=nn*nt,t[2]=nn*ot,t[3]=nn*tt,t[4]=nn*(O*c+Y*h+cn*A-(B*c+R*h+Q*A)),t[5]=nn*(B*o+K*h+un*A-(O*o+an*h+ln*A)),t[6]=nn*(R*o+an*c+$*A-(Y*o+K*c+fn*A)),t[7]=nn*(Q*o+ln*c+fn*h-(cn*o+un*c+$*h)),t[8]=nn*(J*v+W*D+hn*T-(C*v+_n*D+$n*T)),t[9]=nn*(C*y+Wn*D+kn*T-(J*y+Zn*D+Qn*T)),t[10]=nn*(_n*y+Zn*v+Kn*T-(W*y+Wn*v+Cn*T)),t[11]=nn*($n*y+Qn*v+Cn*D-(hn*y+kn*v+Kn*D)),t[12]=nn*(_n*M+$n*P+C*w-(hn*P+J*w+W*M)),t[13]=nn*(Qn*P+J*x+Zn*M-(Wn*M+kn*P+C*x)),t[14]=nn*(Wn*w+Cn*P+W*x-(Kn*P+_n*x+Zn*w)),t[15]=nn*(Kn*M+hn*x+kn*w-(Qn*w+Cn*M+$n*x)),t}function zn(n){const r=n[0],t=n[0*4+1],o=n[0*4+2],a=n[0*4+3],x=n[1*4+0],y=n[1*4+1],c=n[1*4+2],u=n[1*4+3],w=n[2*4+0],v=n[2*4+1],h=n[2*4+2],m=n[2*4+3],M=n[3*4+0],D=n[3*4+1],A=n[3*4+2],z=n[3*4+3],P=h*z,T=A*m,B=c*z,O=A*u,Y=c*m,R=h*u,Q=o*z,cn=A*a,an=o*m,K=h*a,un=o*u,ln=c*a,fn=P*y+O*v+Y*D-(T*y+B*v+R*D),$=T*t+Q*v+K*D-(P*t+cn*v+an*D),J=B*t+cn*y+un*D-(O*t+Q*y+ln*D),C=R*t+an*y+ln*v-(Y*t+K*y+un*v);return r*fn+x*$+w*J+M*C}const Pn=pn;function Sn(n,r,t){const o=t??new f(16),a=n[0],x=n[1],y=n[2],c=n[3],u=n[4],w=n[5],v=n[6],h=n[7],m=n[8],M=n[9],D=n[10],A=n[11],z=n[12],P=n[13],T=n[14],B=n[15],O=r[0],Y=r[1],R=r[2],Q=r[3],cn=r[4],an=r[5],K=r[6],un=r[7],ln=r[8],fn=r[9],$=r[10],J=r[11],C=r[12],_n=r[13],W=r[14],hn=r[15];return o[0]=a*O+u*Y+m*R+z*Q,o[1]=x*O+w*Y+M*R+P*Q,o[2]=y*O+v*Y+D*R+T*Q,o[3]=c*O+h*Y+A*R+B*Q,o[4]=a*cn+u*an+m*K+z*un,o[5]=x*cn+w*an+M*K+P*un,o[6]=y*cn+v*an+D*K+T*un,o[7]=c*cn+h*an+A*K+B*un,o[8]=a*ln+u*fn+m*$+z*J,o[9]=x*ln+w*fn+M*$+P*J,o[10]=y*ln+v*fn+D*$+T*J,o[11]=c*ln+h*fn+A*$+B*J,o[12]=a*C+u*_n+m*W+z*hn,o[13]=x*C+w*_n+M*W+P*hn,o[14]=y*C+v*_n+D*W+T*hn,o[15]=c*C+h*_n+A*W+B*hn,o}const N=Sn;function wn(n,r,t){const o=t??H();return n!==o&&(o[0]=n[0],o[1]=n[1],o[2]=n[2],o[3]=n[3],o[4]=n[4],o[5]=n[5],o[6]=n[6],o[7]=n[7],o[8]=n[8],o[9]=n[9],o[10]=n[10],o[11]=n[11]),o[12]=r[0],o[13]=r[1],o[14]=r[2],o[15]=1,o}function tn(n,r){const t=r??g.create();return t[0]=n[12],t[1]=n[13],t[2]=n[14],t}function Nn(n,r,t){const o=t??g.create(),a=r*4;return o[0]=n[a+0],o[1]=n[a+1],o[2]=n[a+2],o}function vn(n,r,t,o){const a=o===n?o:Z(n,o),x=t*4;return a[x+0]=r[0],a[x+1]=r[1],a[x+2]=r[2],a}function xn(n,r){const t=r??g.create(),o=n[0],a=n[1],x=n[2],y=n[4],c=n[5],u=n[6],w=n[8],v=n[9],h=n[10];return t[0]=Math.sqrt(o*o+a*a+x*x),t[1]=Math.sqrt(y*y+c*c+u*u),t[2]=Math.sqrt(w*w+v*v+h*h),t}function In(n,r,t,o,a){const x=a??new f(16),y=Math.tan(Math.PI*.5-.5*n);if(x[0]=y/r,x[1]=0,x[2]=0,x[3]=0,x[4]=0,x[5]=y,x[6]=0,x[7]=0,x[8]=0,x[9]=0,x[11]=-1,x[12]=0,x[13]=0,x[15]=0,Number.isFinite(o)){const c=1/(t-o);x[10]=o*c,x[14]=o*t*c}else x[10]=-1,x[14]=-t;return x}function en(n,r,t,o=1/0,a){const x=a??new f(16),y=1/Math.tan(n*.5);if(x[0]=y/r,x[1]=0,x[2]=0,x[3]=0,x[4]=0,x[5]=y,x[6]=0,x[7]=0,x[8]=0,x[9]=0,x[11]=-1,x[12]=0,x[13]=0,x[15]=0,o===1/0)x[10]=0,x[14]=t;else{const c=1/(o-t);x[10]=t*c,x[14]=o*t*c}return x}function bn(n,r,t,o,a,x,y){const c=y??new f(16);return c[0]=2/(r-n),c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2/(o-t),c[6]=0,c[7]=0,c[8]=0,c[9]=0,c[10]=1/(a-x),c[11]=0,c[12]=(r+n)/(n-r),c[13]=(o+t)/(t-o),c[14]=a/(a-x),c[15]=1,c}function k(n,r,t,o,a,x,y){const c=y??new f(16),u=r-n,w=o-t,v=a-x;return c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+r)/u,c[9]=(o+t)/w,c[10]=x/v,c[11]=-1,c[12]=0,c[13]=0,c[14]=a*x/v,c[15]=0,c}function An(n,r,t,o,a,x=1/0,y){const c=y??new f(16),u=r-n,w=o-t;if(c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+r)/u,c[9]=(o+t)/w,c[11]=-1,c[12]=0,c[13]=0,c[15]=0,x===1/0)c[10]=0,c[14]=a;else{const v=1/(x-a);c[10]=a*v,c[14]=x*a*v}return c}const G=g.create(),V=g.create(),F=g.create();function Gn(n,r,t,o){const a=o??new f(16);return g.normalize(g.subtract(r,n,F),F),g.normalize(g.cross(t,F,G),G),g.normalize(g.cross(F,G,V),V),a[0]=G[0],a[1]=G[1],a[2]=G[2],a[3]=0,a[4]=V[0],a[5]=V[1],a[6]=V[2],a[7]=0,a[8]=F[0],a[9]=F[1],a[10]=F[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function on(n,r,t,o){const a=o??new f(16);return g.normalize(g.subtract(n,r,F),F),g.normalize(g.cross(t,F,G),G),g.normalize(g.cross(F,G,V),V),a[0]=G[0],a[1]=G[1],a[2]=G[2],a[3]=0,a[4]=V[0],a[5]=V[1],a[6]=V[2],a[7]=0,a[8]=F[0],a[9]=F[1],a[10]=F[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function Un(n,r,t,o){const a=o??new f(16);return g.normalize(g.subtract(n,r,F),F),g.normalize(g.cross(t,F,G),G),g.normalize(g.cross(F,G,V),V),a[0]=G[0],a[1]=V[0],a[2]=F[0],a[3]=0,a[4]=G[1],a[5]=V[1],a[6]=F[1],a[7]=0,a[8]=G[2],a[9]=V[2],a[10]=F[2],a[11]=0,a[12]=-(G[0]*n[0]+G[1]*n[1]+G[2]*n[2]),a[13]=-(V[0]*n[0]+V[1]*n[1]+V[2]*n[2]),a[14]=-(F[0]*n[0]+F[1]*n[1]+F[2]*n[2]),a[15]=1,a}function sn(n,r){const t=r??new f(16);return t[0]=1,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=1,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=1,t[11]=0,t[12]=n[0],t[13]=n[1],t[14]=n[2],t[15]=1,t}function En(n,r,t){const o=t??new f(16),a=r[0],x=r[1],y=r[2],c=n[0],u=n[1],w=n[2],v=n[3],h=n[1*4+0],m=n[1*4+1],M=n[1*4+2],D=n[1*4+3],A=n[2*4+0],z=n[2*4+1],P=n[2*4+2],T=n[2*4+3],B=n[3*4+0],O=n[3*4+1],Y=n[3*4+2],R=n[3*4+3];return n!==o&&(o[0]=c,o[1]=u,o[2]=w,o[3]=v,o[4]=h,o[5]=m,o[6]=M,o[7]=D,o[8]=A,o[9]=z,o[10]=P,o[11]=T),o[12]=c*a+h*x+A*y+B,o[13]=u*a+m*x+z*y+O,o[14]=w*a+M*x+P*y+Y,o[15]=v*a+D*x+T*y+R,o}function q(n,r){const t=r??new f(16),o=Math.cos(n),a=Math.sin(n);return t[0]=1,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=o,t[6]=a,t[7]=0,t[8]=0,t[9]=-a,t[10]=o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function gn(n,r,t){const o=t??new f(16),a=n[4],x=n[5],y=n[6],c=n[7],u=n[8],w=n[9],v=n[10],h=n[11],m=Math.cos(r),M=Math.sin(r);return o[4]=m*a+M*u,o[5]=m*x+M*w,o[6]=m*y+M*v,o[7]=m*c+M*h,o[8]=m*u-M*a,o[9]=m*w-M*x,o[10]=m*v-M*y,o[11]=m*h-M*c,n!==o&&(o[0]=n[0],o[1]=n[1],o[2]=n[2],o[3]=n[3],o[12]=n[12],o[13]=n[13],o[14]=n[14],o[15]=n[15]),o}function Tn(n,r){const t=r??new f(16),o=Math.cos(n),a=Math.sin(n);return t[0]=o,t[1]=0,t[2]=-a,t[3]=0,t[4]=0,t[5]=1,t[6]=0,t[7]=0,t[8]=a,t[9]=0,t[10]=o,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function Vn(n,r,t){const o=t??new f(16),a=n[0*4+0],x=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[2*4+0],w=n[2*4+1],v=n[2*4+2],h=n[2*4+3],m=Math.cos(r),M=Math.sin(r);return o[0]=m*a-M*u,o[1]=m*x-M*w,o[2]=m*y-M*v,o[3]=m*c-M*h,o[8]=m*u+M*a,o[9]=m*w+M*x,o[10]=m*v+M*y,o[11]=m*h+M*c,n!==o&&(o[4]=n[4],o[5]=n[5],o[6]=n[6],o[7]=n[7],o[12]=n[12],o[13]=n[13],o[14]=n[14],o[15]=n[15]),o}function Bn(n,r){const t=r??new f(16),o=Math.cos(n),a=Math.sin(n);return t[0]=o,t[1]=a,t[2]=0,t[3]=0,t[4]=-a,t[5]=o,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=1,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function rn(n,r,t){const o=t??new f(16),a=n[0*4+0],x=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[1*4+0],w=n[1*4+1],v=n[1*4+2],h=n[1*4+3],m=Math.cos(r),M=Math.sin(r);return o[0]=m*a+M*u,o[1]=m*x+M*w,o[2]=m*y+M*v,o[3]=m*c+M*h,o[4]=m*u-M*a,o[5]=m*w-M*x,o[6]=m*v-M*y,o[7]=m*h-M*c,n!==o&&(o[8]=n[8],o[9]=n[9],o[10]=n[10],o[11]=n[11],o[12]=n[12],o[13]=n[13],o[14]=n[14],o[15]=n[15]),o}function qn(n,r,t){const o=t??new f(16);let a=n[0],x=n[1],y=n[2];const c=Math.sqrt(a*a+x*x+y*y);a/=c,x/=c,y/=c;const u=a*a,w=x*x,v=y*y,h=Math.cos(r),m=Math.sin(r),M=1-h;return o[0]=u+(1-u)*h,o[1]=a*x*M+y*m,o[2]=a*y*M-x*m,o[3]=0,o[4]=a*x*M-y*m,o[5]=w+(1-w)*h,o[6]=x*y*M+a*m,o[7]=0,o[8]=a*y*M+x*m,o[9]=x*y*M-a*m,o[10]=v+(1-v)*h,o[11]=0,o[12]=0,o[13]=0,o[14]=0,o[15]=1,o}const i=qn;function d(n,r,t,o){const a=o??new f(16);let x=r[0],y=r[1],c=r[2];const u=Math.sqrt(x*x+y*y+c*c);x/=u,y/=u,c/=u;const w=x*x,v=y*y,h=c*c,m=Math.cos(t),M=Math.sin(t),D=1-m,A=w+(1-w)*m,z=x*y*D+c*M,P=x*c*D-y*M,T=x*y*D-c*M,B=v+(1-v)*m,O=y*c*D+x*M,Y=x*c*D+y*M,R=y*c*D-x*M,Q=h+(1-h)*m,cn=n[0],an=n[1],K=n[2],un=n[3],ln=n[4],fn=n[5],$=n[6],J=n[7],C=n[8],_n=n[9],W=n[10],hn=n[11];return a[0]=A*cn+z*ln+P*C,a[1]=A*an+z*fn+P*_n,a[2]=A*K+z*$+P*W,a[3]=A*un+z*J+P*hn,a[4]=T*cn+B*ln+O*C,a[5]=T*an+B*fn+O*_n,a[6]=T*K+B*$+O*W,a[7]=T*un+B*J+O*hn,a[8]=Y*cn+R*ln+Q*C,a[9]=Y*an+R*fn+Q*_n,a[10]=Y*K+R*$+Q*W,a[11]=Y*un+R*J+Q*hn,n!==a&&(a[12]=n[12],a[13]=n[13],a[14]=n[14],a[15]=n[15]),a}const e=d;function s(n,r){const t=r??new f(16);return t[0]=n[0],t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=n[1],t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=n[2],t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function l(n,r,t){const o=t??new f(16),a=r[0],x=r[1],y=r[2];return o[0]=a*n[0*4+0],o[1]=a*n[0*4+1],o[2]=a*n[0*4+2],o[3]=a*n[0*4+3],o[4]=x*n[1*4+0],o[5]=x*n[1*4+1],o[6]=x*n[1*4+2],o[7]=x*n[1*4+3],o[8]=y*n[2*4+0],o[9]=y*n[2*4+1],o[10]=y*n[2*4+2],o[11]=y*n[2*4+3],n!==o&&(o[12]=n[12],o[13]=n[13],o[14]=n[14],o[15]=n[15]),o}function p(n,r){const t=r??new f(16);return t[0]=n,t[1]=0,t[2]=0,t[3]=0,t[4]=0,t[5]=n,t[6]=0,t[7]=0,t[8]=0,t[9]=0,t[10]=n,t[11]=0,t[12]=0,t[13]=0,t[14]=0,t[15]=1,t}function _(n,r,t){const o=t??new f(16);return o[0]=r*n[0*4+0],o[1]=r*n[0*4+1],o[2]=r*n[0*4+2],o[3]=r*n[0*4+3],o[4]=r*n[1*4+0],o[5]=r*n[1*4+1],o[6]=r*n[1*4+2],o[7]=r*n[1*4+3],o[8]=r*n[2*4+0],o[9]=r*n[2*4+1],o[10]=r*n[2*4+2],o[11]=r*n[2*4+3],n!==o&&(o[12]=n[12],o[13]=n[13],o[14]=n[14],o[15]=n[15]),o}return{create:U,set:E,fromMat3:b,fromQuat:I,negate:L,copy:Z,clone:j,equalsApproximately:X,equals:dn,identity:H,transpose:Fn,inverse:pn,determinant:zn,invert:Pn,multiply:Sn,mul:N,setTranslation:wn,getTranslation:tn,getAxis:Nn,setAxis:vn,getScaling:xn,perspective:In,perspectiveReverseZ:en,ortho:bn,frustum:k,frustumReverseZ:An,aim:Gn,cameraAim:on,lookAt:Un,translation:sn,translate:En,rotationX:q,rotateX:gn,rotationY:Tn,rotateY:Vn,rotationZ:Bn,rotateZ:rn,axisRotation:qn,rotation:i,axisRotate:d,rotate:e,scaling:s,scale:l,uniformScaling:p,uniformScale:_}}const gt=new Map;function Rt(f){let g=gt.get(f);return g||(g=Yt(f),gt.set(f,g)),g}function jt(f){const g=ut(f);function U(i,d,e,s){const l=new f(4);return i!==void 0&&(l[0]=i,d!==void 0&&(l[1]=d,e!==void 0&&(l[2]=e,s!==void 0&&(l[3]=s)))),l}const E=U;function b(i,d,e,s,l){const p=l??new f(4);return p[0]=i,p[1]=d,p[2]=e,p[3]=s,p}function I(i,d,e){const s=e??new f(4),l=d*.5,p=Math.sin(l);return s[0]=p*i[0],s[1]=p*i[1],s[2]=p*i[2],s[3]=Math.cos(l),s}function L(i,d){const e=d??g.create(3),s=Math.acos(i[3])*2,l=Math.sin(s*.5);return l>S?(e[0]=i[0]/l,e[1]=i[1]/l,e[2]=i[2]/l):(e[0]=1,e[1]=0,e[2]=0),{angle:s,axis:e}}function Z(i,d){const e=k(i,d);return Math.acos(2*e*e-1)}function j(i,d,e){const s=e??new f(4),l=i[0],p=i[1],_=i[2],n=i[3],r=d[0],t=d[1],o=d[2],a=d[3];return s[0]=l*a+n*r+p*o-_*t,s[1]=p*a+n*t+_*r-l*o,s[2]=_*a+n*o+l*t-p*r,s[3]=n*a-l*r-p*t-_*o,s}const X=j;function dn(i,d,e){const s=e??new f(4),l=d*.5,p=i[0],_=i[1],n=i[2],r=i[3],t=Math.sin(l),o=Math.cos(l);return s[0]=p*o+r*t,s[1]=_*o+n*t,s[2]=n*o-_*t,s[3]=r*o-p*t,s}function H(i,d,e){const s=e??new f(4),l=d*.5,p=i[0],_=i[1],n=i[2],r=i[3],t=Math.sin(l),o=Math.cos(l);return s[0]=p*o-n*t,s[1]=_*o+r*t,s[2]=n*o+p*t,s[3]=r*o-_*t,s}function Fn(i,d,e){const s=e??new f(4),l=d*.5,p=i[0],_=i[1],n=i[2],r=i[3],t=Math.sin(l),o=Math.cos(l);return s[0]=p*o+_*t,s[1]=_*o-p*t,s[2]=n*o+r*t,s[3]=r*o-n*t,s}function pn(i,d,e,s){const l=s??new f(4),p=i[0],_=i[1],n=i[2],r=i[3];let t=d[0],o=d[1],a=d[2],x=d[3],y=p*t+_*o+n*a+r*x;y<0&&(y=-y,t=-t,o=-o,a=-a,x=-x);let c,u;if(1-y>S){const w=Math.acos(y),v=Math.sin(w);c=Math.sin((1-e)*w)/v,u=Math.sin(e*w)/v}else c=1-e,u=e;return l[0]=c*p+u*t,l[1]=c*_+u*o,l[2]=c*n+u*a,l[3]=c*r+u*x,l}function zn(i,d){const e=d??new f(4),s=i[0],l=i[1],p=i[2],_=i[3],n=s*s+l*l+p*p+_*_,r=n?1/n:0;return e[0]=-s*r,e[1]=-l*r,e[2]=-p*r,e[3]=_*r,e}function Pn(i,d){const e=d??new f(4);return e[0]=-i[0],e[1]=-i[1],e[2]=-i[2],e[3]=i[3],e}function Sn(i,d){const e=d??new f(4),s=i[0]+i[5]+i[10];if(s>0){const l=Math.sqrt(s+1);e[3]=.5*l;const p=.5/l;e[0]=(i[6]-i[9])*p,e[1]=(i[8]-i[2])*p,e[2]=(i[1]-i[4])*p}else{let l=0;i[5]>i[0]&&(l=1),i[10]>i[l*4+l]&&(l=2);const p=(l+1)%3,_=(l+2)%3,n=Math.sqrt(i[l*4+l]-i[p*4+p]-i[_*4+_]+1);e[l]=.5*n;const r=.5/n;e[3]=(i[p*4+_]-i[_*4+p])*r,e[p]=(i[p*4+l]+i[l*4+p])*r,e[_]=(i[_*4+l]+i[l*4+_])*r}return e}function N(i,d,e,s,l){const p=l??new f(4),_=i*.5,n=d*.5,r=e*.5,t=Math.sin(_),o=Math.cos(_),a=Math.sin(n),x=Math.cos(n),y=Math.sin(r),c=Math.cos(r);switch(s){case"xyz":p[0]=t*x*c+o*a*y,p[1]=o*a*c-t*x*y,p[2]=o*x*y+t*a*c,p[3]=o*x*c-t*a*y;break;case"xzy":p[0]=t*x*c-o*a*y,p[1]=o*a*c-t*x*y,p[2]=o*x*y+t*a*c,p[3]=o*x*c+t*a*y;break;case"yxz":p[0]=t*x*c+o*a*y,p[1]=o*a*c-t*x*y,p[2]=o*x*y-t*a*c,p[3]=o*x*c+t*a*y;break;case"yzx":p[0]=t*x*c+o*a*y,p[1]=o*a*c+t*x*y,p[2]=o*x*y-t*a*c,p[3]=o*x*c-t*a*y;break;case"zxy":p[0]=t*x*c-o*a*y,p[1]=o*a*c+t*x*y,p[2]=o*x*y+t*a*c,p[3]=o*x*c-t*a*y;break;case"zyx":p[0]=t*x*c-o*a*y,p[1]=o*a*c+t*x*y,p[2]=o*x*y-t*a*c,p[3]=o*x*c+t*a*y;break;default:throw new Error(`Unknown rotation order: ${s}`)}return p}function wn(i,d){const e=d??new f(4);return e[0]=i[0],e[1]=i[1],e[2]=i[2],e[3]=i[3],e}const tn=wn;function Nn(i,d,e){const s=e??new f(4);return s[0]=i[0]+d[0],s[1]=i[1]+d[1],s[2]=i[2]+d[2],s[3]=i[3]+d[3],s}function vn(i,d,e){const s=e??new f(4);return s[0]=i[0]-d[0],s[1]=i[1]-d[1],s[2]=i[2]-d[2],s[3]=i[3]-d[3],s}const xn=vn;function In(i,d,e){const s=e??new f(4);return s[0]=i[0]*d,s[1]=i[1]*d,s[2]=i[2]*d,s[3]=i[3]*d,s}const en=In;function bn(i,d,e){const s=e??new f(4);return s[0]=i[0]/d,s[1]=i[1]/d,s[2]=i[2]/d,s[3]=i[3]/d,s}function k(i,d){return i[0]*d[0]+i[1]*d[1]+i[2]*d[2]+i[3]*d[3]}function An(i,d,e,s){const l=s??new f(4);return l[0]=i[0]+e*(d[0]-i[0]),l[1]=i[1]+e*(d[1]-i[1]),l[2]=i[2]+e*(d[2]-i[2]),l[3]=i[3]+e*(d[3]-i[3]),l}function G(i){const d=i[0],e=i[1],s=i[2],l=i[3];return Math.sqrt(d*d+e*e+s*s+l*l)}const V=G;function F(i){const d=i[0],e=i[1],s=i[2],l=i[3];return d*d+e*e+s*s+l*l}const Gn=F;function on(i,d){const e=d??new f(4),s=i[0],l=i[1],p=i[2],_=i[3],n=Math.sqrt(s*s+l*l+p*p+_*_);return n>1e-5?(e[0]=s/n,e[1]=l/n,e[2]=p/n,e[3]=_/n):(e[0]=0,e[1]=0,e[2]=0,e[3]=1),e}function Un(i,d){return Math.abs(i[0]-d[0])<S&&Math.abs(i[1]-d[1])<S&&Math.abs(i[2]-d[2])<S&&Math.abs(i[3]-d[3])<S}function sn(i,d){return i[0]===d[0]&&i[1]===d[1]&&i[2]===d[2]&&i[3]===d[3]}function En(i){const d=i??new f(4);return d[0]=0,d[1]=0,d[2]=0,d[3]=1,d}const q=g.create(),gn=g.create(),Tn=g.create();function Vn(i,d,e){const s=e??new f(4),l=g.dot(i,d);return l<-.999999?(g.cross(gn,i,q),g.len(q)<1e-6&&g.cross(Tn,i,q),g.normalize(q,q),I(q,Math.PI,s),s):l>.999999?(s[0]=0,s[1]=0,s[2]=0,s[3]=1,s):(g.cross(i,d,q),s[0]=q[0],s[1]=q[1],s[2]=q[2],s[3]=1+l,on(s,s))}const Bn=new f(4),rn=new f(4);function qn(i,d,e,s,l,p){const _=p??new f(4);return pn(i,s,l,Bn),pn(d,e,l,rn),pn(Bn,rn,2*l*(1-l),_),_}return{create:U,fromValues:E,set:b,fromAxisAngle:I,toAxisAngle:L,angle:Z,multiply:j,mul:X,rotateX:dn,rotateY:H,rotateZ:Fn,slerp:pn,inverse:zn,conjugate:Pn,fromMat:Sn,fromEuler:N,copy:wn,clone:tn,add:Nn,subtract:vn,sub:xn,mulScalar:In,scale:en,divScalar:bn,dot:k,lerp:An,length:G,len:V,lengthSq:F,lenSq:Gn,normalize:on,equalsApproximately:Un,equals:sn,identity:En,rotationTo:Vn,sqlerp:qn}}const _t=new Map;function Ht(f){let g=_t.get(f);return g||(g=jt(f),_t.set(f,g)),g}function $t(f){function g(e,s,l,p){const _=new f(4);return e!==void 0&&(_[0]=e,s!==void 0&&(_[1]=s,l!==void 0&&(_[2]=l,p!==void 0&&(_[3]=p)))),_}const U=g;function E(e,s,l,p,_){const n=_??new f(4);return n[0]=e,n[1]=s,n[2]=l,n[3]=p,n}function b(e,s){const l=s??new f(4);return l[0]=Math.ceil(e[0]),l[1]=Math.ceil(e[1]),l[2]=Math.ceil(e[2]),l[3]=Math.ceil(e[3]),l}function I(e,s){const l=s??new f(4);return l[0]=Math.floor(e[0]),l[1]=Math.floor(e[1]),l[2]=Math.floor(e[2]),l[3]=Math.floor(e[3]),l}function L(e,s){const l=s??new f(4);return l[0]=Math.round(e[0]),l[1]=Math.round(e[1]),l[2]=Math.round(e[2]),l[3]=Math.round(e[3]),l}function Z(e,s=0,l=1,p){const _=p??new f(4);return _[0]=Math.min(l,Math.max(s,e[0])),_[1]=Math.min(l,Math.max(s,e[1])),_[2]=Math.min(l,Math.max(s,e[2])),_[3]=Math.min(l,Math.max(s,e[3])),_}function j(e,s,l){const p=l??new f(4);return p[0]=e[0]+s[0],p[1]=e[1]+s[1],p[2]=e[2]+s[2],p[3]=e[3]+s[3],p}function X(e,s,l,p){const _=p??new f(4);return _[0]=e[0]+s[0]*l,_[1]=e[1]+s[1]*l,_[2]=e[2]+s[2]*l,_[3]=e[3]+s[3]*l,_}function dn(e,s,l){const p=l??new f(4);return p[0]=e[0]-s[0],p[1]=e[1]-s[1],p[2]=e[2]-s[2],p[3]=e[3]-s[3],p}const H=dn;function Fn(e,s){return Math.abs(e[0]-s[0])<S&&Math.abs(e[1]-s[1])<S&&Math.abs(e[2]-s[2])<S&&Math.abs(e[3]-s[3])<S}function pn(e,s){return e[0]===s[0]&&e[1]===s[1]&&e[2]===s[2]&&e[3]===s[3]}function zn(e,s,l,p){const _=p??new f(4);return _[0]=e[0]+l*(s[0]-e[0]),_[1]=e[1]+l*(s[1]-e[1]),_[2]=e[2]+l*(s[2]-e[2]),_[3]=e[3]+l*(s[3]-e[3]),_}function Pn(e,s,l,p){const _=p??new f(4);return _[0]=e[0]+l[0]*(s[0]-e[0]),_[1]=e[1]+l[1]*(s[1]-e[1]),_[2]=e[2]+l[2]*(s[2]-e[2]),_[3]=e[3]+l[3]*(s[3]-e[3]),_}function Sn(e,s,l){const p=l??new f(4);return p[0]=Math.max(e[0],s[0]),p[1]=Math.max(e[1],s[1]),p[2]=Math.max(e[2],s[2]),p[3]=Math.max(e[3],s[3]),p}function N(e,s,l){const p=l??new f(4);return p[0]=Math.min(e[0],s[0]),p[1]=Math.min(e[1],s[1]),p[2]=Math.min(e[2],s[2]),p[3]=Math.min(e[3],s[3]),p}function wn(e,s,l){const p=l??new f(4);return p[0]=e[0]*s,p[1]=e[1]*s,p[2]=e[2]*s,p[3]=e[3]*s,p}const tn=wn;function Nn(e,s,l){const p=l??new f(4);return p[0]=e[0]/s,p[1]=e[1]/s,p[2]=e[2]/s,p[3]=e[3]/s,p}function vn(e,s){const l=s??new f(4);return l[0]=1/e[0],l[1]=1/e[1],l[2]=1/e[2],l[3]=1/e[3],l}const xn=vn;function In(e,s){return e[0]*s[0]+e[1]*s[1]+e[2]*s[2]+e[3]*s[3]}function en(e){const s=e[0],l=e[1],p=e[2],_=e[3];return Math.sqrt(s*s+l*l+p*p+_*_)}const bn=en;function k(e){const s=e[0],l=e[1],p=e[2],_=e[3];return s*s+l*l+p*p+_*_}const An=k;function G(e,s){const l=e[0]-s[0],p=e[1]-s[1],_=e[2]-s[2],n=e[3]-s[3];return Math.sqrt(l*l+p*p+_*_+n*n)}const V=G;function F(e,s){const l=e[0]-s[0],p=e[1]-s[1],_=e[2]-s[2],n=e[3]-s[3];return l*l+p*p+_*_+n*n}const Gn=F;function on(e,s){const l=s??new f(4),p=e[0],_=e[1],n=e[2],r=e[3],t=Math.sqrt(p*p+_*_+n*n+r*r);return t>1e-5?(l[0]=p/t,l[1]=_/t,l[2]=n/t,l[3]=r/t):(l[0]=0,l[1]=0,l[2]=0,l[3]=0),l}function Un(e,s){const l=s??new f(4);return l[0]=-e[0],l[1]=-e[1],l[2]=-e[2],l[3]=-e[3],l}function sn(e,s){const l=s??new f(4);return l[0]=e[0],l[1]=e[1],l[2]=e[2],l[3]=e[3],l}const En=sn;function q(e,s,l){const p=l??new f(4);return p[0]=e[0]*s[0],p[1]=e[1]*s[1],p[2]=e[2]*s[2],p[3]=e[3]*s[3],p}const gn=q;function Tn(e,s,l){const p=l??new f(4);return p[0]=e[0]/s[0],p[1]=e[1]/s[1],p[2]=e[2]/s[2],p[3]=e[3]/s[3],p}const Vn=Tn;function Bn(e){const s=e??new f(4);return s[0]=0,s[1]=0,s[2]=0,s[3]=0,s}function rn(e,s,l){const p=l??new f(4),_=e[0],n=e[1],r=e[2],t=e[3];return p[0]=s[0]*_+s[4]*n+s[8]*r+s[12]*t,p[1]=s[1]*_+s[5]*n+s[9]*r+s[13]*t,p[2]=s[2]*_+s[6]*n+s[10]*r+s[14]*t,p[3]=s[3]*_+s[7]*n+s[11]*r+s[15]*t,p}function qn(e,s,l){const p=l??new f(4);return on(e,p),wn(p,s,p)}function i(e,s,l){const p=l??new f(4);return en(e)>s?qn(e,s,p):sn(e,p)}function d(e,s,l){const p=l??new f(4);return zn(e,s,.5,p)}return{create:g,fromValues:U,set:E,ceil:b,floor:I,round:L,clamp:Z,add:j,addScaled:X,subtract:dn,sub:H,equalsApproximately:Fn,equals:pn,lerp:zn,lerpV:Pn,max:Sn,min:N,mulScalar:wn,scale:tn,divScalar:Nn,inverse:vn,invert:xn,dot:In,length:en,len:bn,lengthSq:k,lenSq:An,distance:G,dist:V,distanceSq:F,distSq:Gn,normalize:on,negate:Un,copy:sn,clone:En,multiply:q,mul:gn,divide:Tn,div:Vn,zero:Bn,transformMat4:rn,setLength:qn,truncate:i,midpoint:d}}const ht=new Map;function Wt(f){let g=ht.get(f);return g||(g=$t(f),ht.set(f,g)),g}function dt(f,g,U,E,b,I){return{mat3:Xt(f),mat4:Rt(g),quat:Ht(U),vec2:mt(E),vec3:ut(b),vec4:Wt(I)}}const{mat3:ie,mat4:On,quat:se,vec2:re,vec3:ce,vec4:ae}=dt(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);dt(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array);dt(Ot,Array,Array,Array,Array,Array);const Mt=4e5,at=112,Zt=16,Qt=64,Kt=64,Jt=64;let Hn=0;function Ct(f){let g=new ArrayBuffer(at*Mt);const U=.55;for(let L=3;L<f[1]*.4;L+=U)for(let Z=3;Z<f[0]-5;Z+=U)for(let j=3;j<f[2]/2;j+=U){const X=at*Hn,dn={position:new Float32Array(g,X+0,3),v:new Float32Array(g,X+16,3),C:new Float32Array(g,X+32,12),force:new Float32Array(g,X+80,3),density:new Float32Array(g,X+92,1),nearDensity:new Float32Array(g,X+96,1)},H=2*Math.random();dn.position.set([Z+H,L+H,j+H]),Hn++}let E=new ArrayBuffer(at*Hn);const b=new Uint8Array(g),I=new Uint8Array(E);return I.set(b.subarray(0,I.length)),E}async function ne(){const f=document.querySelector("canvas");if(!navigator.gpu)throw alert("WebGPU is not supported on your browser."),new Error;const g=await navigator.gpu.requestAdapter();if(!g)throw alert("Adapter is not available."),new Error;const U=await g.requestDevice(),E=f.getContext("webgpu");if(!E)throw new Error;let b=1;f.width=b*f.clientWidth,f.height=b*f.clientHeight;const I=navigator.gpu.getPreferredCanvasFormat();return E.configure({device:U,format:I}),{canvas:f,device:U,presentationFormat:I,context:E}}function te(f,g){const U=f.clientWidth/f.clientHeight,E=On.perspective(g,U,.1,500),b=On.identity();return{projection:E,view:b}}function ee(f,g,U,E){var b=On.identity();On.translate(b,E,b),On.rotateY(b,U,b),On.rotateX(b,g,b),On.translate(b,[0,0,f],b);var I=On.multiply(b,[0,0,0,1]);return On.lookAt([I[0],I[1],I[2]],E,[0,1,0])}const Dt=.6,yt=2*Dt;async function oe(){const{canvas:f,device:g,presentationFormat:U,context:E}=await ne();E.configure({device:g,format:U,alphaMode:"premultiplied"});const b=g.createShaderModule({code:Tt}),I=g.createShaderModule({code:zt}),L=g.createShaderModule({code:Pt}),Z=g.createShaderModule({code:bt}),j=g.createShaderModule({code:At}),X=g.createShaderModule({code:St}),dn=g.createShaderModule({code:Bt}),H=g.createShaderModule({code:Ft}),Fn=g.createShaderModule({code:It}),pn=g.createShaderModule({code:Gt}),zn=g.createShaderModule({code:Ut}),Pn=g.createShaderModule({code:Et}),Sn=g.createShaderModule({code:Vt}),N={stiffness:6,restDensity:4,dynamic_viscosity:.03,dt:.15,fixed_point_multiplier:1e7},wn=Qt*Kt*Jt,tn=g.createRenderPipeline({label:"circles pipeline",layout:"auto",vertex:{module:I},fragment:{module:I,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),Nn=g.createRenderPipeline({label:"ball pipeline",layout:"auto",vertex:{module:X},fragment:{module:X,targets:[{format:U}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),vn=90*Math.PI/180,{projection:xn,view:In}=te(f,vn),en={screenHeight:f.height,screenWidth:f.width},bn={depth_threshold:Dt*10,max_filter_size:100,projected_particle_constant:10*yt*.05*(f.height/2)/Math.tan(vn/2)},k=g.createRenderPipeline({label:"filter pipeline",layout:"auto",vertex:{module:b,constants:en},fragment:{module:Z,constants:bn,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"}}),An=g.createRenderPipeline({label:"fluid rendering pipeline",layout:"auto",vertex:{module:b,constants:en},fragment:{module:j,targets:[{format:U}]},primitive:{topology:"triangle-list",cullMode:"none"}}),G=g.createRenderPipeline({label:"thickness pipeline",layout:"auto",vertex:{module:dn},fragment:{module:dn,targets:[{format:"r16float",writeMask:GPUColorWrite.RED,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}}),V=g.createRenderPipeline({label:"thickness filter pipeline",layout:"auto",vertex:{module:b,constants:en},fragment:{module:H,targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),F=g.createRenderPipeline({label:"show pipeline",layout:"auto",vertex:{module:b,constants:en},fragment:{module:L,targets:[{format:U}]},primitive:{topology:"triangle-list",cullMode:"none"}}),Gn=g.createComputePipeline({label:"clear grid pipeline",layout:"auto",compute:{module:Fn}}),on=g.createComputePipeline({label:"p2g 1 pipeline",layout:"auto",compute:{module:pn,constants:{fixed_point_multiplier:N.fixed_point_multiplier}}}),Un=g.createComputePipeline({label:"p2g 2 pipeline",layout:"auto",compute:{module:zn,constants:{fixed_point_multiplier:N.fixed_point_multiplier,stiffness:N.stiffness,rest_density:N.restDensity,dynamic_viscosity:N.dynamic_viscosity,dt:N.dt}}}),sn=g.createComputePipeline({label:"update grid pipeline",layout:"auto",compute:{module:Pn,constants:{fixed_point_multiplier:N.fixed_point_multiplier,dt:N.dt}}}),En=g.createComputePipeline({label:"g2p pipeline",layout:"auto",compute:{module:Sn,constants:{fixed_point_multiplier:N.fixed_point_multiplier,dt:N.dt}}}),gn=g.createTexture({label:"depth map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),Vn=g.createTexture({label:"temporary texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),rn=g.createTexture({label:"thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),i=g.createTexture({label:"temporary thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),d=g.createTexture({size:[f.width,f.height,1],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});d.createView();let e;{const mn=["cubemap/posx.png","cubemap/negx.png","cubemap/posy.png","cubemap/negy.png","cubemap/posz.png","cubemap/negz.png"].map(async Xn=>{const Yn=await fetch(Xn);return createImageBitmap(await Yn.blob())}),Mn=await Promise.all(mn);console.log(Mn[0].width,Mn[0].height),e=g.createTexture({dimension:"2d",size:[Mn[0].width,Mn[0].height,6],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let Xn=0;Xn<Mn.length;Xn++){const Yn=Mn[Xn];g.queue.copyExternalImageToTexture({source:Yn},{texture:e,origin:[0,0,Xn]},[Yn.width,Yn.height])}}const s=e.createView({dimension:"cube"}),l=g.createSampler({magFilter:"linear",minFilter:"linear"}),p=new ArrayBuffer(144),_={size:new Float32Array(p,0,1),view_matrix:new Float32Array(p,16,16),projection_matrix:new Float32Array(p,80,16)},n=new ArrayBuffer(8),r=new ArrayBuffer(8),t={blur_dir:new Float32Array(n)},o={blur_dir:new Float32Array(r)};t.blur_dir.set([1,0]),o.blur_dir.set([0,1]);const a=new ArrayBuffer(272),x={texel_size:new Float32Array(a,0,2),inv_projection_matrix:new Float32Array(a,16,16),projection_matrix:new Float32Array(a,80,16),view_matrix:new Float32Array(a,144,16),inv_view_matrix:new Float32Array(a,208,16)};x.texel_size.set([1/f.width,1/f.height]),x.projection_matrix.set(xn);const y=On.identity(),c=On.identity();On.inverse(xn,y),On.inverse(In,c),x.inv_projection_matrix.set(y),x.inv_view_matrix.set(c);const u=new ArrayBuffer(12),w=new Float32Array(u),v=new ArrayBuffer(12),h=new Float32Array(v),m=g.createBuffer({label:"particles buffer",size:at*Mt,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),M=g.createBuffer({label:"cells buffer",size:Zt*wn,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),D=g.createBuffer({label:"uniform buffer",size:p.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),A=g.createBuffer({label:"filter uniform buffer",size:n.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),z=g.createBuffer({label:"filter uniform buffer",size:r.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),P=g.createBuffer({label:"filter uniform buffer",size:a.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),T=g.createBuffer({label:"real box size buffer",size:u.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),B=g.createBuffer({label:"init box size buffer",size:v.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});g.queue.writeBuffer(A,0,n),g.queue.writeBuffer(z,0,r),g.queue.writeBuffer(P,0,a);const O=g.createBindGroup({layout:Gn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:M}}]}),Y=g.createBindGroup({layout:on.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:M}},{binding:2,resource:{buffer:B}}]}),R=g.createBindGroup({layout:Un.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:M}},{binding:2,resource:{buffer:B}}]}),Q=g.createBindGroup({layout:sn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:M}},{binding:1,resource:{buffer:T}},{binding:2,resource:{buffer:B}}]}),cn=g.createBindGroup({layout:En.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:M}},{binding:2,resource:{buffer:T}},{binding:3,resource:{buffer:B}}]});g.createBindGroup({label:"ball bind group",layout:Nn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:D}}]});const an=g.createBindGroup({label:"circle bind group",layout:tn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:D}}]});g.createBindGroup({label:"show bind group",layout:F.getBindGroupLayout(0),entries:[{binding:1,resource:rn}]});const K=[g.createBindGroup({label:"filterX bind group",layout:k.getBindGroupLayout(0),entries:[{binding:1,resource:gn},{binding:2,resource:{buffer:A}}]}),g.createBindGroup({label:"filterY bind group",layout:k.getBindGroupLayout(0),entries:[{binding:1,resource:Vn},{binding:2,resource:{buffer:z}}]})],un=g.createBindGroup({label:"fluid bind group",layout:An.getBindGroupLayout(0),entries:[{binding:0,resource:l},{binding:1,resource:gn},{binding:2,resource:{buffer:P}},{binding:3,resource:rn},{binding:4,resource:s}]}),ln=g.createBindGroup({label:"thickness bind group",layout:G.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:m}},{binding:1,resource:{buffer:D}}]}),fn=[g.createBindGroup({label:"thickness filterX bind group",layout:V.getBindGroupLayout(0),entries:[{binding:1,resource:rn},{binding:2,resource:{buffer:A}}]}),g.createBindGroup({label:"thickness filterY bind group",layout:V.getBindGroupLayout(0),entries:[{binding:1,resource:i},{binding:2,resource:{buffer:z}}]})];let $=!1,J=0,C=0,_n=-Math.PI/2,W=-Math.PI/12;const hn=.005,$n=-.99*Math.PI/2,Wn=0;let Zn=1;const Qn=[{MIN_DISTANCE:100,MAX_DISTANCE:100,INIT_DISTANCE:100},{MIN_DISTANCE:50,MAX_DISTANCE:100,INIT_DISTANCE:50},{MIN_DISTANCE:100,MAX_DISTANCE:100,INIT_DISTANCE:100},{MIN_DISTANCE:100,MAX_DISTANCE:100,INIT_DISTANCE:100},{MIN_DISTANCE:100,MAX_DISTANCE:100,INIT_DISTANCE:100}];let kn=Qn[Zn].INIT_DISTANCE;const Kn=document.getElementById("fluidCanvas");Kn.addEventListener("mousedown",yn=>{$=!0,J=yn.clientX,C=yn.clientY}),Kn.addEventListener("wheel",yn=>{yn.preventDefault();var mn=yn.deltaY;kn+=(mn>0?1:-1)*.5;const Mn=Qn[Zn];kn<Mn.MIN_DISTANCE&&(kn=Mn.MIN_DISTANCE),kn>Mn.MAX_DISTANCE&&(kn=Mn.MAX_DISTANCE)}),document.addEventListener("mousemove",yn=>{if($){const mn=yn.clientX,Mn=yn.clientY,Xn=J-mn,Yn=C-Mn;_n+=hn*Xn,W+=hn*Yn,W>Wn&&(W=Wn),W<$n&&(W=$n),J=mn,C=Mn}}),document.addEventListener("mouseup",()=>{$&&($=!1)}),document.getElementById("number-button").addEventListener("change",function(yn){const mn=yn.target;(mn==null?void 0:mn.name)==="options"&&mn.value});let Ln=[40,64,72],nt=[...Ln];const ot=Ct(Ln);g.queue.writeBuffer(m,0,ot);let tt=document.getElementById("error-reason");tt.textContent="",g.lost.then(yn=>{const mn=yn.reason?`reason: ${yn.reason}`:"unknown reason";tt.textContent=mn}),console.log(Hn);let nn=0;async function pt(){nn+=.01,performance.now();const yn={colorAttachments:[{view:gn,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:d.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}};E.getCurrentTexture().createView(),d.createView();const mn=[{colorAttachments:[{view:Vn,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:gn,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];E.getCurrentTexture().createView();const Mn={colorAttachments:[{view:E.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Xn={colorAttachments:[{view:rn,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Yn=[{colorAttachments:[{view:i,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:rn,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];_.size.set([yt]),_.projection_matrix.set(xn);const lt=ee(kn,W,_n,[Ln[0]/2,Ln[1]/4,Ln[2]/2]);_.view_matrix.set(lt),x.view_matrix.set(lt),On.inverse(lt,c),x.inv_view_matrix.set(c),nt[2]=Ln[2]*(.25*(Math.cos(3*nn)+1)+.5),w.set(nt),h.set(Ln),g.queue.writeBuffer(D,0,p),g.queue.writeBuffer(P,0,a),g.queue.writeBuffer(T,0,u),g.queue.writeBuffer(B,0,v);const ft=Math.ceil(Ln[0])*Math.ceil(Ln[1])*Math.ceil(Ln[2]);if(ft>wn)throw new Error("gridCount is bigger than maxGridCount");const Rn=g.createCommandEncoder(),Dn=Rn.beginComputePass();for(let Jn=0;Jn<2;Jn++)Dn.setBindGroup(0,O),Dn.setPipeline(Gn),Dn.dispatchWorkgroups(Math.ceil(ft/64)),Dn.setBindGroup(0,Y),Dn.setPipeline(on),Dn.dispatchWorkgroups(Math.ceil(Hn/64)),Dn.setBindGroup(0,R),Dn.setPipeline(Un),Dn.dispatchWorkgroups(Math.ceil(Hn/64)),Dn.setBindGroup(0,Q),Dn.setPipeline(sn),Dn.dispatchWorkgroups(Math.ceil(ft/64)),Dn.setBindGroup(0,cn),Dn.setPipeline(En),Dn.dispatchWorkgroups(Math.ceil(Hn/64));Dn.end();{const Jn=Rn.beginRenderPass(yn);Jn.setBindGroup(0,an),Jn.setPipeline(tn),Jn.draw(6,Hn),Jn.end();for(var et=0;et<5;et++){const rt=Rn.beginRenderPass(mn[0]);rt.setBindGroup(0,K[0]),rt.setPipeline(k),rt.draw(6),rt.end();const jn=Rn.beginRenderPass(mn[1]);jn.setBindGroup(0,K[1]),jn.setPipeline(k),jn.draw(6),jn.end()}const it=Rn.beginRenderPass(Xn);it.setBindGroup(0,ln),it.setPipeline(G),it.draw(6,Hn),it.end();for(var et=0;et<1;et++){const jn=Rn.beginRenderPass(Yn[0]);jn.setBindGroup(0,fn[0]),jn.setPipeline(V),jn.draw(6),jn.end();const ct=Rn.beginRenderPass(Yn[1]);ct.setBindGroup(0,fn[1]),ct.setPipeline(V),ct.draw(6),ct.end()}const st=Rn.beginRenderPass(Mn);st.setBindGroup(0,un),st.setPipeline(An),st.draw(6),st.end()}g.queue.submit([Rn.finish()]),performance.now(),requestAnimationFrame(pt)}requestAnimationFrame(pt)}oe();
