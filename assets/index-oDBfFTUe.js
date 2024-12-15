(function(){const g=document.createElement("link").relList;if(g&&g.supports&&g.supports("modulepreload"))return;for(const D of document.querySelectorAll('link[rel="modulepreload"]'))E(D);new MutationObserver(D=>{for(const G of D)if(G.type==="childList")for(const k of G.addedNodes)k.tagName==="LINK"&&k.rel==="modulepreload"&&E(k)}).observe(document,{childList:!0,subtree:!0});function S(D){const G={};return D.integrity&&(G.integrity=D.integrity),D.referrerPolicy&&(G.referrerPolicy=D.referrerPolicy),D.crossOrigin==="use-credentials"?G.credentials="include":D.crossOrigin==="anonymous"?G.credentials="omit":G.credentials="same-origin",G}function E(D){if(D.ep)return;D.ep=!0;const G=S(D);fetch(D.href,G)}})();var Fe=`struct Particle {
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
}`,Ne=`struct Particle {
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
    let scale: f32 = 15.0 / (3.1415926535 * kernelRadiusPow5); 
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

        var cnt1 = 0.;

        let v = cellPosition(pos_i);
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
                        cnt1 += r;
                    }
                }
            }
        }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
            fVisc *= viscosity;
            let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
            particles[id.x].force = fPress + fVisc + fGrv;
        
    }
}`,Ve=`struct Particle {
  position: vec3f, 
  velocity: vec3f, 
  force: vec3f, 
  density: f32, 
  nearDensity: f32, 
}

struct RealBoxSize {
  xHalf: f32, 
  yHalf: f32, 
  zHalf: f32, 
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> realBoxSize: RealBoxSize;

override dt: f32;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < arrayLength(&particles)) {
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
    particles[id.x].velocity += dt * a;
    particles[id.x].position += dt * particles[id.x].velocity;
  }
}`,He=`struct VertexOutput {
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
    velocity: vec3f, 
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
}`,We=`@group(0) @binding(1) var texture: texture_2d<f32>;

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var r = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    
    return vec4(0, r, r, 1.0);
}`,qe=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,Ke=`@group(0) @binding(0) var texture_sampler: sampler;
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

    
    
    if (depth >= 1e4 || depth <= 0.) {
        
        
        return vec4f(0.7, 0.7, 0.7, 1.);
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
    var lightDir = normalize((uniforms.view_matrix * vec4f(-1, -1, -1, 0.)).xyz);
    var H: vec3f        = normalize(lightDir - rayDir);
    var specular: f32   = pow(max(0.0, dot(H, normal)), 250.);
    var diffuse: f32  = max(0.0, dot(lightDir, normal)) * 1.0;

    var density = 1.5; 
    
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    var diffuseColor = vec3f(0.085, 0.6375, 0.9);
    var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor)); 
    var refractionColor: vec3f = vec3f(0.7, 0.7, 0.7) * transmittance;

    let F0 = 0.02;
    var fresnel: f32 = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0., 0.5);

    var reflectionDir: vec3f = reflect(rayDir, normal);
    var reflectionDirWorld: vec3f = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    var reflectionColor: vec3f = textureSampleLevel(envmap_texture, texture_sampler, reflectionDirWorld, 0.).rgb; 
    var finalColor = specular + mix(refractionColor, reflectionColor, fresnel);

    return vec4f(finalColor, 1.0);

    

    
    
    
    
    
    
    
    
    
    
    
}`,je=`struct VertexOutput {
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
}`,Ye=`struct Uniforms {
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
    velocity: vec3f, 
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
}`,Xe=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,Ze=`struct VertexOutput {
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
    velocity: vec3f, 
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

    let speed = sqrt(dot(particles[instance_index].velocity, particles[instance_index].velocity));

    return VertexOutput(out_position, uv, view_position, speed);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let i = floor(h * 6.0); 
    let f = h * 6.0 - i; 
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    
    if (i == 0.0) {
        r = v;
        g = t;
        b = p;
    } else if (i == 1.0) {
        r = q;
        g = v;
        b = p;
    } else if (i == 2.0) {
        r = p;
        g = v;
        b = t;
    } else if (i == 3.0) {
        r = p;
        g = q;
        b = v;
    } else if (i == 4.0) {
        r = t;
        g = p;
        b = v;
    } else if (i == 5.0) {
        r = v;
        g = p;
        b = q;
    }

    return vec3f(r, g, b); 
}

fn get_color_by_speed(speed: f32, max_speed: f32) -> vec3f {
    
    let normalized_speed = clamp(abs(speed) / max_speed, 0.0, 1.0);
    
    let hue = (1.0 - normalized_speed) * 0.7;
    let saturation = 1.0;
    let value = 1.0;

    return hsv_to_rgb(hue, saturation, value);
}

fn value_to_color(value: f32, min: f32, max: f32) -> vec3<f32> {
    
    let normalized = (value - min) / (max - min);
    
    
    let r = normalized;
    let g = (normalized - 0.5);
    let b = (1.0 - normalized);
    
    return vec3<f32>(r, g, b); 
}

fn value_to_color2(value: f32) -> vec3<f32> {
    
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

    var diffuse: f32 = max(0.0, dot(normal, normalize(vec3(1., 1.0, 1.0))));
    
    
    var color: vec3f = value_to_color2(input.speed / 1.5);
    
    
    

    
    
    
    out.frag_color = vec4(color * diffuse, 1.);
    return out;
}`,$e=`@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
    if (id.x < arrayLength(&cellParticleCount)) {
        cellParticleCount[id.x] = 0u;
    }
}`,Qe=`struct Particle {
    position: vec3f, 
    velocity: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

override xGrids: u32;
override yGrids: u32;
override cellSize: f32;
override xHalf: f32;
override yHalf: f32;
override zHalf: f32;
override offset: f32;

fn cellId(position: vec3f) -> u32 {
    let xi: u32 = u32(floor((position.x + xHalf + offset) / cellSize));
    let yi: u32 = u32(floor((position.y + yHalf + offset) / cellSize));
    let zi: u32 = u32(floor((position.z + zHalf + offset) / cellSize));

    return xi + yi * xGrids + zi * xGrids * yGrids;
}

@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> particleCellOffset : array<u32>;
@group(0) @binding(2) var<storage, read_write> particles: array<Particle>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
  if (id.x < arrayLength(&particles))
  {
    let cellID: u32 = cellId(particles[id.x].position);
    particleCellOffset[id.x] = atomicAdd(&cellParticleCount[cellID], 1u);
  }
}`,Je=`struct Particle {
    position: vec3f, 
    velocity: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

@group(0) @binding(0) var<storage, read> sourceParticles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> targetParticles: array<Particle>;
@group(0) @binding(2) var<storage, read> cellParticleCount : array<u32>;
@group(0) @binding(3) var<storage, read> particleCellOffset : array<u32>;

override xGrids: u32;
override yGrids: u32;
override cellSize: f32;
override xHalf: f32;
override yHalf: f32;
override zHalf: f32;
override offset: f32;

fn cellId(position: vec3f) -> u32 {
    let xi: u32 = u32(floor((position.x + xHalf + offset) / cellSize));
    let yi: u32 = u32(floor((position.y + yHalf + offset) / cellSize));
    let zi: u32 = u32(floor((position.z + zHalf + offset) / cellSize));

    return xi + yi * xGrids + zi * xGrids * yGrids;
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>) {
    if (id.x < arrayLength(&sourceParticles)) {
        let cellId: u32 = cellId(sourceParticles[id.x].position);
        let targetIndex = cellParticleCount[cellId + 1] - particleCellOffset[id.x] - 1;
        targetParticles[targetIndex] = sourceParticles[id.x];
    }
}`;const Ce=`

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    temp[ELM_TID]     = select(items[ELM_GID], 0, ELM_GID >= ELEMENT_COUNT);
    temp[ELM_TID + 1] = select(items[ELM_GID + 1], 0, ELM_GID + 1 >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        let last_offset = ITEMS_PER_WORKGROUP - 1;

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (ELM_GID >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID] = temp[ELM_TID];

    if (ELM_GID + 1 >= ELEMENT_COUNT) {
        return;
    }
    items[ELM_GID + 1] = temp[ELM_TID + 1];
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`,nt=`

@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

override WORKGROUP_SIZE_X: u32;
override WORKGROUP_SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

const NUM_BANKS: u32 = 32;
const LOG_NUM_BANKS: u32 = 5;

fn get_offset(offset: u32) -> u32 {
    // return offset >> LOG_NUM_BANKS; // Conflict-free
    return (offset >> NUM_BANKS) + (offset >> (2 * LOG_NUM_BANKS)); // Zero bank conflict
}

var<workgroup> temp: array<u32, ITEMS_PER_WORKGROUP*2>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn reduce_downsweep(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID
    
    let ELM_TID = TID * 2; // Element pair local ID
    let ELM_GID = GID * 2; // Element pair global ID
    
    // Load input to shared memory
    let ai: u32 = TID;
    let bi: u32 = TID + (ITEMS_PER_WORKGROUP >> 1);
    let s_ai = ai + get_offset(ai);
    let s_bi = bi + get_offset(bi);
    let g_ai = ai + WID * 2;
    let g_bi = bi + WID * 2;
    temp[s_ai] = select(items[g_ai], 0, g_ai >= ELEMENT_COUNT);
    temp[s_bi] = select(items[g_bi], 0, g_bi >= ELEMENT_COUNT);

    var offset: u32 = 1;

    // Up-sweep (reduce) phase
    for (var d: u32 = ITEMS_PER_WORKGROUP >> 1; d > 0; d >>= 1) {
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);
            temp[bi] += temp[ai];
        }

        offset *= 2;
    }

    // Save workgroup sum and clear last element
    if (TID == 0) {
        var last_offset = ITEMS_PER_WORKGROUP - 1;
        last_offset += get_offset(last_offset);

        blockSums[WORKGROUP_ID] = temp[last_offset];
        temp[last_offset] = 0;
    }

    // Down-sweep phase
    for (var d: u32 = 1; d < ITEMS_PER_WORKGROUP; d *= 2) {
        offset >>= 1;
        workgroupBarrier();

        if (TID < d) {
            var ai: u32 = offset * (ELM_TID + 1) - 1;
            var bi: u32 = offset * (ELM_TID + 2) - 1;
            ai += get_offset(ai);
            bi += get_offset(bi);

            let t: u32 = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    // Copy result from shared memory to global memory
    if (g_ai < ELEMENT_COUNT) {
        items[g_ai] = temp[s_ai];
    }
    if (g_bi < ELEMENT_COUNT) {
        items[g_bi] = temp[s_bi];
    }
}

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn add_block_sums(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) w_dim: vec3<u32>,
    @builtin(local_invocation_index) TID: u32, // Local thread ID
) {
    let WORKGROUP_ID = w_id.x + w_id.y * w_dim.x;
    let WID = WORKGROUP_ID * THREADS_PER_WORKGROUP;
    let GID = WID + TID; // Global thread ID

    let ELM_ID = GID * 2;

    if (ELM_ID >= ELEMENT_COUNT) {
        return;
    }

    let blockSum = blockSums[WORKGROUP_ID];

    items[ELM_ID] += blockSum;

    if (ELM_ID + 1 >= ELEMENT_COUNT) {
        return;
    }

    items[ELM_ID + 1] += blockSum;
}`;function et(f,g){const S={x:g,y:1};if(g>f.limits.maxComputeWorkgroupsPerDimension){const E=Math.floor(Math.sqrt(g)),D=Math.ceil(g/E);S.x=E,S.y=D}return S}class tt{constructor({device:g,data:S,count:E,workgroup_size:D={x:16,y:16},avoid_bank_conflicts:G=!1}){if(this.device=g,this.workgroup_size=D,this.threads_per_workgroup=D.x*D.y,this.items_per_workgroup=2*this.threads_per_workgroup,Math.log2(this.threads_per_workgroup)%1!==0)throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`);this.pipelines=[],this.shaderModule=this.device.createShaderModule({label:"prefix-sum",code:G?nt:Ce}),this.create_pass_recursive(S,E)}create_pass_recursive(g,S){const E=Math.ceil(S/this.items_per_workgroup),D=et(this.device,E),G=this.device.createBuffer({label:"prefix-sum-block-sum",size:E*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),k=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),q=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:k,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:G}}]}),A=this.device.createPipelineLayout({bindGroupLayouts:[k]}),Z=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:A,compute:{module:this.shaderModule,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ITEMS_PER_WORKGROUP:this.items_per_workgroup,ELEMENT_COUNT:S}}});if(this.pipelines.push({pipeline:Z,bindGroup:q,dispatchSize:D}),E>1){this.create_pass_recursive(G,E);const nn=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:A,compute:{module:this.shaderModule,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ELEMENT_COUNT:S}}});this.pipelines.push({pipeline:nn,bindGroup:q,dispatchSize:D})}}get_dispatch_chain(){return this.pipelines.flatMap(g=>[g.dispatchSize.x,g.dispatchSize.y,1])}dispatch(g,S,E=0){for(let D=0;D<this.pipelines.length;D++){const{pipeline:G,bindGroup:k,dispatchSize:q}=this.pipelines[D];g.setPipeline(G),g.setBindGroup(0,k),S==null?g.dispatchWorkgroups(q.x,q.y,1):g.dispatchWorkgroupsIndirect(S,E+D*3*4)}}}function rt(f,g){return class extends f{constructor(...S){super(...S),g(this)}}}const ot=rt(Array,f=>f.fill(0));let U=1e-6;function st(f){function g(n=0,i=0){const e=new f(2);return n!==void 0&&(e[0]=n,i!==void 0&&(e[1]=i)),e}const S=g;function E(n,i,e){const r=e??new f(2);return r[0]=n,r[1]=i,r}function D(n,i){const e=i??new f(2);return e[0]=Math.ceil(n[0]),e[1]=Math.ceil(n[1]),e}function G(n,i){const e=i??new f(2);return e[0]=Math.floor(n[0]),e[1]=Math.floor(n[1]),e}function k(n,i){const e=i??new f(2);return e[0]=Math.round(n[0]),e[1]=Math.round(n[1]),e}function q(n,i=0,e=1,r){const a=r??new f(2);return a[0]=Math.min(e,Math.max(i,n[0])),a[1]=Math.min(e,Math.max(i,n[1])),a}function A(n,i,e){const r=e??new f(2);return r[0]=n[0]+i[0],r[1]=n[1]+i[1],r}function Z(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+i[0]*e,a[1]=n[1]+i[1]*e,a}function nn(n,i){const e=n[0],r=n[1],a=i[0],_=i[1],y=Math.sqrt(e*e+r*r),c=Math.sqrt(a*a+_*_),u=y*c,w=u&&bn(n,i)/u;return Math.acos(w)}function an(n,i,e){const r=e??new f(2);return r[0]=n[0]-i[0],r[1]=n[1]-i[1],r}const hn=an;function pn(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U}function Pn(n,i){return n[0]===i[0]&&n[1]===i[1]}function zn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e*(i[0]-n[0]),a[1]=n[1]+e*(i[1]-n[1]),a}function Tn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e[0]*(i[0]-n[0]),a[1]=n[1]+e[1]*(i[1]-n[1]),a}function On(n,i,e){const r=e??new f(2);return r[0]=Math.max(n[0],i[0]),r[1]=Math.max(n[1],i[1]),r}function F(n,i,e){const r=e??new f(2);return r[0]=Math.min(n[0],i[0]),r[1]=Math.min(n[1],i[1]),r}function un(n,i,e){const r=e??new f(2);return r[0]=n[0]*i,r[1]=n[1]*i,r}const Un=un;function xn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i,r[1]=n[1]/i,r}function wn(n,i){const e=i??new f(2);return e[0]=1/n[0],e[1]=1/n[1],e}const Bn=wn;function ln(n,i,e){const r=e??new f(3),a=n[0]*i[1]-n[1]*i[0];return r[0]=0,r[1]=0,r[2]=a,r}function bn(n,i){return n[0]*i[0]+n[1]*i[1]}function $(n){const i=n[0],e=n[1];return Math.sqrt(i*i+e*e)}const Gn=$;function B(n){const i=n[0],e=n[1];return i*i+e*e}const L=B;function O(n,i){const e=n[0]-i[0],r=n[1]-i[1];return Math.sqrt(e*e+r*r)}const en=O;function vn(n,i){const e=n[0]-i[0],r=n[1]-i[1];return e*e+r*r}const kn=vn;function _n(n,i){const e=i??new f(2),r=n[0],a=n[1],_=Math.sqrt(r*r+a*a);return _>1e-5?(e[0]=r/_,e[1]=a/_):(e[0]=0,e[1]=0),e}function yn(n,i){const e=i??new f(2);return e[0]=-n[0],e[1]=-n[1],e}function N(n,i){const e=i??new f(2);return e[0]=n[0],e[1]=n[1],e}const mn=N;function fn(n,i,e){const r=e??new f(2);return r[0]=n[0]*i[0],r[1]=n[1]*i[1],r}const Sn=fn;function dn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i[0],r[1]=n[1]/i[1],r}const En=dn;function Mn(n=1,i){const e=i??new f(2),r=Math.random()*2*Math.PI;return e[0]=Math.cos(r)*n,e[1]=Math.sin(r)*n,e}function o(n){const i=n??new f(2);return i[0]=0,i[1]=0,i}function d(n,i,e){const r=e??new f(2),a=n[0],_=n[1];return r[0]=a*i[0]+_*i[4]+i[12],r[1]=a*i[1]+_*i[5]+i[13],r}function t(n,i,e){const r=e??new f(2),a=n[0],_=n[1];return r[0]=i[0]*a+i[4]*_+i[8],r[1]=i[1]*a+i[5]*_+i[9],r}function s(n,i,e,r){const a=r??new f(2),_=n[0]-i[0],y=n[1]-i[1],c=Math.sin(e),u=Math.cos(e);return a[0]=_*u-y*c+i[0],a[1]=_*c+y*u+i[1],a}function l(n,i,e){const r=e??new f(2);return _n(n,r),un(r,i,r)}function p(n,i,e){const r=e??new f(2);return $(n)>i?l(n,i,r):N(n,r)}function h(n,i,e){const r=e??new f(2);return zn(n,i,.5,r)}return{create:g,fromValues:S,set:E,ceil:D,floor:G,round:k,clamp:q,add:A,addScaled:Z,angle:nn,subtract:an,sub:hn,equalsApproximately:pn,equals:Pn,lerp:zn,lerpV:Tn,max:On,min:F,mulScalar:un,scale:Un,divScalar:xn,inverse:wn,invert:Bn,cross:ln,dot:bn,length:$,len:Gn,lengthSq:B,lenSq:L,distance:O,dist:en,distanceSq:vn,distSq:kn,normalize:_n,negate:yn,copy:N,clone:mn,multiply:fn,mul:Sn,divide:dn,div:En,random:Mn,zero:o,transformMat4:d,transformMat3:t,rotate:s,setLength:l,truncate:p,midpoint:h}}const Pe=new Map;function Ie(f){let g=Pe.get(f);return g||(g=st(f),Pe.set(f,g)),g}function it(f){function g(c,u,w){const v=new f(3);return c!==void 0&&(v[0]=c,u!==void 0&&(v[1]=u,w!==void 0&&(v[2]=w))),v}const S=g;function E(c,u,w,v){const x=v??new f(3);return x[0]=c,x[1]=u,x[2]=w,x}function D(c,u){const w=u??new f(3);return w[0]=Math.ceil(c[0]),w[1]=Math.ceil(c[1]),w[2]=Math.ceil(c[2]),w}function G(c,u){const w=u??new f(3);return w[0]=Math.floor(c[0]),w[1]=Math.floor(c[1]),w[2]=Math.floor(c[2]),w}function k(c,u){const w=u??new f(3);return w[0]=Math.round(c[0]),w[1]=Math.round(c[1]),w[2]=Math.round(c[2]),w}function q(c,u=0,w=1,v){const x=v??new f(3);return x[0]=Math.min(w,Math.max(u,c[0])),x[1]=Math.min(w,Math.max(u,c[1])),x[2]=Math.min(w,Math.max(u,c[2])),x}function A(c,u,w){const v=w??new f(3);return v[0]=c[0]+u[0],v[1]=c[1]+u[1],v[2]=c[2]+u[2],v}function Z(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+u[0]*w,x[1]=c[1]+u[1]*w,x[2]=c[2]+u[2]*w,x}function nn(c,u){const w=c[0],v=c[1],x=c[2],m=u[0],M=u[1],P=u[2],R=Math.sqrt(w*w+v*v+x*x),z=Math.sqrt(m*m+M*M+P*P),b=R*z,I=b&&bn(c,u)/b;return Math.acos(I)}function an(c,u,w){const v=w??new f(3);return v[0]=c[0]-u[0],v[1]=c[1]-u[1],v[2]=c[2]-u[2],v}const hn=an;function pn(c,u){return Math.abs(c[0]-u[0])<U&&Math.abs(c[1]-u[1])<U&&Math.abs(c[2]-u[2])<U}function Pn(c,u){return c[0]===u[0]&&c[1]===u[1]&&c[2]===u[2]}function zn(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+w*(u[0]-c[0]),x[1]=c[1]+w*(u[1]-c[1]),x[2]=c[2]+w*(u[2]-c[2]),x}function Tn(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+w[0]*(u[0]-c[0]),x[1]=c[1]+w[1]*(u[1]-c[1]),x[2]=c[2]+w[2]*(u[2]-c[2]),x}function On(c,u,w){const v=w??new f(3);return v[0]=Math.max(c[0],u[0]),v[1]=Math.max(c[1],u[1]),v[2]=Math.max(c[2],u[2]),v}function F(c,u,w){const v=w??new f(3);return v[0]=Math.min(c[0],u[0]),v[1]=Math.min(c[1],u[1]),v[2]=Math.min(c[2],u[2]),v}function un(c,u,w){const v=w??new f(3);return v[0]=c[0]*u,v[1]=c[1]*u,v[2]=c[2]*u,v}const Un=un;function xn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u,v[1]=c[1]/u,v[2]=c[2]/u,v}function wn(c,u){const w=u??new f(3);return w[0]=1/c[0],w[1]=1/c[1],w[2]=1/c[2],w}const Bn=wn;function ln(c,u,w){const v=w??new f(3),x=c[2]*u[0]-c[0]*u[2],m=c[0]*u[1]-c[1]*u[0];return v[0]=c[1]*u[2]-c[2]*u[1],v[1]=x,v[2]=m,v}function bn(c,u){return c[0]*u[0]+c[1]*u[1]+c[2]*u[2]}function $(c){const u=c[0],w=c[1],v=c[2];return Math.sqrt(u*u+w*w+v*v)}const Gn=$;function B(c){const u=c[0],w=c[1],v=c[2];return u*u+w*w+v*v}const L=B;function O(c,u){const w=c[0]-u[0],v=c[1]-u[1],x=c[2]-u[2];return Math.sqrt(w*w+v*v+x*x)}const en=O;function vn(c,u){const w=c[0]-u[0],v=c[1]-u[1],x=c[2]-u[2];return w*w+v*v+x*x}const kn=vn;function _n(c,u){const w=u??new f(3),v=c[0],x=c[1],m=c[2],M=Math.sqrt(v*v+x*x+m*m);return M>1e-5?(w[0]=v/M,w[1]=x/M,w[2]=m/M):(w[0]=0,w[1]=0,w[2]=0),w}function yn(c,u){const w=u??new f(3);return w[0]=-c[0],w[1]=-c[1],w[2]=-c[2],w}function N(c,u){const w=u??new f(3);return w[0]=c[0],w[1]=c[1],w[2]=c[2],w}const mn=N;function fn(c,u,w){const v=w??new f(3);return v[0]=c[0]*u[0],v[1]=c[1]*u[1],v[2]=c[2]*u[2],v}const Sn=fn;function dn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u[0],v[1]=c[1]/u[1],v[2]=c[2]/u[2],v}const En=dn;function Mn(c=1,u){const w=u??new f(3),v=Math.random()*2*Math.PI,x=Math.random()*2-1,m=Math.sqrt(1-x*x)*c;return w[0]=Math.cos(v)*m,w[1]=Math.sin(v)*m,w[2]=x*c,w}function o(c){const u=c??new f(3);return u[0]=0,u[1]=0,u[2]=0,u}function d(c,u,w){const v=w??new f(3),x=c[0],m=c[1],M=c[2],P=u[3]*x+u[7]*m+u[11]*M+u[15]||1;return v[0]=(u[0]*x+u[4]*m+u[8]*M+u[12])/P,v[1]=(u[1]*x+u[5]*m+u[9]*M+u[13])/P,v[2]=(u[2]*x+u[6]*m+u[10]*M+u[14])/P,v}function t(c,u,w){const v=w??new f(3),x=c[0],m=c[1],M=c[2];return v[0]=x*u[0*4+0]+m*u[1*4+0]+M*u[2*4+0],v[1]=x*u[0*4+1]+m*u[1*4+1]+M*u[2*4+1],v[2]=x*u[0*4+2]+m*u[1*4+2]+M*u[2*4+2],v}function s(c,u,w){const v=w??new f(3),x=c[0],m=c[1],M=c[2];return v[0]=x*u[0]+m*u[4]+M*u[8],v[1]=x*u[1]+m*u[5]+M*u[9],v[2]=x*u[2]+m*u[6]+M*u[10],v}function l(c,u,w){const v=w??new f(3),x=u[0],m=u[1],M=u[2],P=u[3]*2,R=c[0],z=c[1],b=c[2],I=m*b-M*z,T=M*R-x*b,V=x*z-m*R;return v[0]=R+I*P+(m*V-M*T)*2,v[1]=z+T*P+(M*I-x*V)*2,v[2]=b+V*P+(x*T-m*I)*2,v}function p(c,u){const w=u??new f(3);return w[0]=c[12],w[1]=c[13],w[2]=c[14],w}function h(c,u,w){const v=w??new f(3),x=u*4;return v[0]=c[x+0],v[1]=c[x+1],v[2]=c[x+2],v}function n(c,u){const w=u??new f(3),v=c[0],x=c[1],m=c[2],M=c[4],P=c[5],R=c[6],z=c[8],b=c[9],I=c[10];return w[0]=Math.sqrt(v*v+x*x+m*m),w[1]=Math.sqrt(M*M+P*P+R*R),w[2]=Math.sqrt(z*z+b*b+I*I),w}function i(c,u,w,v){const x=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0],M[1]=m[1]*Math.cos(w)-m[2]*Math.sin(w),M[2]=m[1]*Math.sin(w)+m[2]*Math.cos(w),x[0]=M[0]+u[0],x[1]=M[1]+u[1],x[2]=M[2]+u[2],x}function e(c,u,w,v){const x=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[2]*Math.sin(w)+m[0]*Math.cos(w),M[1]=m[1],M[2]=m[2]*Math.cos(w)-m[0]*Math.sin(w),x[0]=M[0]+u[0],x[1]=M[1]+u[1],x[2]=M[2]+u[2],x}function r(c,u,w,v){const x=v??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0]*Math.cos(w)-m[1]*Math.sin(w),M[1]=m[0]*Math.sin(w)+m[1]*Math.cos(w),M[2]=m[2],x[0]=M[0]+u[0],x[1]=M[1]+u[1],x[2]=M[2]+u[2],x}function a(c,u,w){const v=w??new f(3);return _n(c,v),un(v,u,v)}function _(c,u,w){const v=w??new f(3);return $(c)>u?a(c,u,v):N(c,v)}function y(c,u,w){const v=w??new f(3);return zn(c,u,.5,v)}return{create:g,fromValues:S,set:E,ceil:D,floor:G,round:k,clamp:q,add:A,addScaled:Z,angle:nn,subtract:an,sub:hn,equalsApproximately:pn,equals:Pn,lerp:zn,lerpV:Tn,max:On,min:F,mulScalar:un,scale:Un,divScalar:xn,inverse:wn,invert:Bn,cross:ln,dot:bn,length:$,len:Gn,lengthSq:B,lenSq:L,distance:O,dist:en,distanceSq:vn,distSq:kn,normalize:_n,negate:yn,copy:N,clone:mn,multiply:fn,mul:Sn,divide:dn,div:En,random:Mn,zero:o,transformMat4:d,transformMat4Upper3x3:t,transformMat3:s,transformQuat:l,getTranslation:p,getAxis:h,getScaling:n,rotateX:i,rotateY:e,rotateZ:r,setLength:a,truncate:_,midpoint:y}}const ze=new Map;function de(f){let g=ze.get(f);return g||(g=it(f),ze.set(f,g)),g}function ct(f){const g=Ie(f),S=de(f);function E(o,d,t,s,l,p,h,n,i){const e=new f(12);return e[3]=0,e[7]=0,e[11]=0,o!==void 0&&(e[0]=o,d!==void 0&&(e[1]=d,t!==void 0&&(e[2]=t,s!==void 0&&(e[4]=s,l!==void 0&&(e[5]=l,p!==void 0&&(e[6]=p,h!==void 0&&(e[8]=h,n!==void 0&&(e[9]=n,i!==void 0&&(e[10]=i))))))))),e}function D(o,d,t,s,l,p,h,n,i,e){const r=e??new f(12);return r[0]=o,r[1]=d,r[2]=t,r[3]=0,r[4]=s,r[5]=l,r[6]=p,r[7]=0,r[8]=h,r[9]=n,r[10]=i,r[11]=0,r}function G(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[3]=0,t[4]=o[4],t[5]=o[5],t[6]=o[6],t[7]=0,t[8]=o[8],t[9]=o[9],t[10]=o[10],t[11]=0,t}function k(o,d){const t=d??new f(12),s=o[0],l=o[1],p=o[2],h=o[3],n=s+s,i=l+l,e=p+p,r=s*n,a=l*n,_=l*i,y=p*n,c=p*i,u=p*e,w=h*n,v=h*i,x=h*e;return t[0]=1-_-u,t[1]=a+x,t[2]=y-v,t[3]=0,t[4]=a-x,t[5]=1-r-u,t[6]=c+w,t[7]=0,t[8]=y+v,t[9]=c-w,t[10]=1-r-_,t[11]=0,t}function q(o,d){const t=d??new f(12);return t[0]=-o[0],t[1]=-o[1],t[2]=-o[2],t[4]=-o[4],t[5]=-o[5],t[6]=-o[6],t[8]=-o[8],t[9]=-o[9],t[10]=-o[10],t}function A(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[4]=o[4],t[5]=o[5],t[6]=o[6],t[8]=o[8],t[9]=o[9],t[10]=o[10],t}const Z=A;function nn(o,d){return Math.abs(o[0]-d[0])<U&&Math.abs(o[1]-d[1])<U&&Math.abs(o[2]-d[2])<U&&Math.abs(o[4]-d[4])<U&&Math.abs(o[5]-d[5])<U&&Math.abs(o[6]-d[6])<U&&Math.abs(o[8]-d[8])<U&&Math.abs(o[9]-d[9])<U&&Math.abs(o[10]-d[10])<U}function an(o,d){return o[0]===d[0]&&o[1]===d[1]&&o[2]===d[2]&&o[4]===d[4]&&o[5]===d[5]&&o[6]===d[6]&&o[8]===d[8]&&o[9]===d[9]&&o[10]===d[10]}function hn(o){const d=o??new f(12);return d[0]=1,d[1]=0,d[2]=0,d[4]=0,d[5]=1,d[6]=0,d[8]=0,d[9]=0,d[10]=1,d}function pn(o,d){const t=d??new f(12);if(t===o){let _;return _=o[1],o[1]=o[4],o[4]=_,_=o[2],o[2]=o[8],o[8]=_,_=o[6],o[6]=o[9],o[9]=_,t}const s=o[0*4+0],l=o[0*4+1],p=o[0*4+2],h=o[1*4+0],n=o[1*4+1],i=o[1*4+2],e=o[2*4+0],r=o[2*4+1],a=o[2*4+2];return t[0]=s,t[1]=h,t[2]=e,t[4]=l,t[5]=n,t[6]=r,t[8]=p,t[9]=i,t[10]=a,t}function Pn(o,d){const t=d??new f(12),s=o[0*4+0],l=o[0*4+1],p=o[0*4+2],h=o[1*4+0],n=o[1*4+1],i=o[1*4+2],e=o[2*4+0],r=o[2*4+1],a=o[2*4+2],_=a*n-i*r,y=-a*h+i*e,c=r*h-n*e,u=1/(s*_+l*y+p*c);return t[0]=_*u,t[1]=(-a*l+p*r)*u,t[2]=(i*l-p*n)*u,t[4]=y*u,t[5]=(a*s-p*e)*u,t[6]=(-i*s+p*h)*u,t[8]=c*u,t[9]=(-r*s+l*e)*u,t[10]=(n*s-l*h)*u,t}function zn(o){const d=o[0],t=o[0*4+1],s=o[0*4+2],l=o[1*4+0],p=o[1*4+1],h=o[1*4+2],n=o[2*4+0],i=o[2*4+1],e=o[2*4+2];return d*(p*e-i*h)-l*(t*e-i*s)+n*(t*h-p*s)}const Tn=Pn;function On(o,d,t){const s=t??new f(12),l=o[0],p=o[1],h=o[2],n=o[4],i=o[5],e=o[6],r=o[8],a=o[9],_=o[10],y=d[0],c=d[1],u=d[2],w=d[4],v=d[5],x=d[6],m=d[8],M=d[9],P=d[10];return s[0]=l*y+n*c+r*u,s[1]=p*y+i*c+a*u,s[2]=h*y+e*c+_*u,s[4]=l*w+n*v+r*x,s[5]=p*w+i*v+a*x,s[6]=h*w+e*v+_*x,s[8]=l*m+n*M+r*P,s[9]=p*m+i*M+a*P,s[10]=h*m+e*M+_*P,s}const F=On;function un(o,d,t){const s=t??hn();return o!==s&&(s[0]=o[0],s[1]=o[1],s[2]=o[2],s[4]=o[4],s[5]=o[5],s[6]=o[6]),s[8]=d[0],s[9]=d[1],s[10]=1,s}function Un(o,d){const t=d??g.create();return t[0]=o[8],t[1]=o[9],t}function xn(o,d,t){const s=t??g.create(),l=d*4;return s[0]=o[l+0],s[1]=o[l+1],s}function wn(o,d,t,s){const l=s===o?o:A(o,s),p=t*4;return l[p+0]=d[0],l[p+1]=d[1],l}function Bn(o,d){const t=d??g.create(),s=o[0],l=o[1],p=o[4],h=o[5];return t[0]=Math.sqrt(s*s+l*l),t[1]=Math.sqrt(p*p+h*h),t}function ln(o,d){const t=d??S.create(),s=o[0],l=o[1],p=o[2],h=o[4],n=o[5],i=o[6],e=o[8],r=o[9],a=o[10];return t[0]=Math.sqrt(s*s+l*l+p*p),t[1]=Math.sqrt(h*h+n*n+i*i),t[2]=Math.sqrt(e*e+r*r+a*a),t}function bn(o,d){const t=d??new f(12);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=1,t[6]=0,t[8]=o[0],t[9]=o[1],t[10]=1,t}function $(o,d,t){const s=t??new f(12),l=d[0],p=d[1],h=o[0],n=o[1],i=o[2],e=o[1*4+0],r=o[1*4+1],a=o[1*4+2],_=o[2*4+0],y=o[2*4+1],c=o[2*4+2];return o!==s&&(s[0]=h,s[1]=n,s[2]=i,s[4]=e,s[5]=r,s[6]=a),s[8]=h*l+e*p+_,s[9]=n*l+r*p+y,s[10]=i*l+a*p+c,s}function Gn(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=s,t[1]=l,t[2]=0,t[4]=-l,t[5]=s,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function B(o,d,t){const s=t??new f(12),l=o[0*4+0],p=o[0*4+1],h=o[0*4+2],n=o[1*4+0],i=o[1*4+1],e=o[1*4+2],r=Math.cos(d),a=Math.sin(d);return s[0]=r*l+a*n,s[1]=r*p+a*i,s[2]=r*h+a*e,s[4]=r*n-a*l,s[5]=r*i-a*p,s[6]=r*e-a*h,o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function L(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=s,t[6]=l,t[8]=0,t[9]=-l,t[10]=s,t}function O(o,d,t){const s=t??new f(12),l=o[4],p=o[5],h=o[6],n=o[8],i=o[9],e=o[10],r=Math.cos(d),a=Math.sin(d);return s[4]=r*l+a*n,s[5]=r*p+a*i,s[6]=r*h+a*e,s[8]=r*n-a*l,s[9]=r*i-a*p,s[10]=r*e-a*h,o!==s&&(s[0]=o[0],s[1]=o[1],s[2]=o[2]),s}function en(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=s,t[1]=0,t[2]=-l,t[4]=0,t[5]=1,t[6]=0,t[8]=l,t[9]=0,t[10]=s,t}function vn(o,d,t){const s=t??new f(12),l=o[0*4+0],p=o[0*4+1],h=o[0*4+2],n=o[2*4+0],i=o[2*4+1],e=o[2*4+2],r=Math.cos(d),a=Math.sin(d);return s[0]=r*l-a*n,s[1]=r*p-a*i,s[2]=r*h-a*e,s[8]=r*n+a*l,s[9]=r*i+a*p,s[10]=r*e+a*h,o!==s&&(s[4]=o[4],s[5]=o[5],s[6]=o[6]),s}const kn=Gn,_n=B;function yn(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=0,t[2]=0,t[4]=0,t[5]=o[1],t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function N(o,d,t){const s=t??new f(12),l=d[0],p=d[1];return s[0]=l*o[0*4+0],s[1]=l*o[0*4+1],s[2]=l*o[0*4+2],s[4]=p*o[1*4+0],s[5]=p*o[1*4+1],s[6]=p*o[1*4+2],o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function mn(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=0,t[2]=0,t[4]=0,t[5]=o[1],t[6]=0,t[8]=0,t[9]=0,t[10]=o[2],t}function fn(o,d,t){const s=t??new f(12),l=d[0],p=d[1],h=d[2];return s[0]=l*o[0*4+0],s[1]=l*o[0*4+1],s[2]=l*o[0*4+2],s[4]=p*o[1*4+0],s[5]=p*o[1*4+1],s[6]=p*o[1*4+2],s[8]=h*o[2*4+0],s[9]=h*o[2*4+1],s[10]=h*o[2*4+2],s}function Sn(o,d){const t=d??new f(12);return t[0]=o,t[1]=0,t[2]=0,t[4]=0,t[5]=o,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function dn(o,d,t){const s=t??new f(12);return s[0]=d*o[0*4+0],s[1]=d*o[0*4+1],s[2]=d*o[0*4+2],s[4]=d*o[1*4+0],s[5]=d*o[1*4+1],s[6]=d*o[1*4+2],o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function En(o,d){const t=d??new f(12);return t[0]=o,t[1]=0,t[2]=0,t[4]=0,t[5]=o,t[6]=0,t[8]=0,t[9]=0,t[10]=o,t}function Mn(o,d,t){const s=t??new f(12);return s[0]=d*o[0*4+0],s[1]=d*o[0*4+1],s[2]=d*o[0*4+2],s[4]=d*o[1*4+0],s[5]=d*o[1*4+1],s[6]=d*o[1*4+2],s[8]=d*o[2*4+0],s[9]=d*o[2*4+1],s[10]=d*o[2*4+2],s}return{clone:Z,create:E,set:D,fromMat4:G,fromQuat:k,negate:q,copy:A,equalsApproximately:nn,equals:an,identity:hn,transpose:pn,inverse:Pn,invert:Tn,determinant:zn,mul:F,multiply:On,setTranslation:un,getTranslation:Un,getAxis:xn,setAxis:wn,getScaling:Bn,get3DScaling:ln,translation:bn,translate:$,rotation:Gn,rotate:B,rotationX:L,rotateX:O,rotationY:en,rotateY:vn,rotationZ:kn,rotateZ:_n,scaling:yn,scale:N,uniformScaling:Sn,uniformScale:dn,scaling3D:mn,scale3D:fn,uniformScaling3D:En,uniformScale3D:Mn}}const be=new Map;function at(f){let g=be.get(f);return g||(g=ct(f),be.set(f,g)),g}function ut(f){const g=de(f);function S(n,i,e,r,a,_,y,c,u,w,v,x,m,M,P,R){const z=new f(16);return n!==void 0&&(z[0]=n,i!==void 0&&(z[1]=i,e!==void 0&&(z[2]=e,r!==void 0&&(z[3]=r,a!==void 0&&(z[4]=a,_!==void 0&&(z[5]=_,y!==void 0&&(z[6]=y,c!==void 0&&(z[7]=c,u!==void 0&&(z[8]=u,w!==void 0&&(z[9]=w,v!==void 0&&(z[10]=v,x!==void 0&&(z[11]=x,m!==void 0&&(z[12]=m,M!==void 0&&(z[13]=M,P!==void 0&&(z[14]=P,R!==void 0&&(z[15]=R)))))))))))))))),z}function E(n,i,e,r,a,_,y,c,u,w,v,x,m,M,P,R,z){const b=z??new f(16);return b[0]=n,b[1]=i,b[2]=e,b[3]=r,b[4]=a,b[5]=_,b[6]=y,b[7]=c,b[8]=u,b[9]=w,b[10]=v,b[11]=x,b[12]=m,b[13]=M,b[14]=P,b[15]=R,b}function D(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=0,e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=0,e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function G(n,i){const e=i??new f(16),r=n[0],a=n[1],_=n[2],y=n[3],c=r+r,u=a+a,w=_+_,v=r*c,x=a*c,m=a*u,M=_*c,P=_*u,R=_*w,z=y*c,b=y*u,I=y*w;return e[0]=1-m-R,e[1]=x+I,e[2]=M-b,e[3]=0,e[4]=x-I,e[5]=1-v-R,e[6]=P+z,e[7]=0,e[8]=M+b,e[9]=P-z,e[10]=1-v-m,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function k(n,i){const e=i??new f(16);return e[0]=-n[0],e[1]=-n[1],e[2]=-n[2],e[3]=-n[3],e[4]=-n[4],e[5]=-n[5],e[6]=-n[6],e[7]=-n[7],e[8]=-n[8],e[9]=-n[9],e[10]=-n[10],e[11]=-n[11],e[12]=-n[12],e[13]=-n[13],e[14]=-n[14],e[15]=-n[15],e}function q(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],e}const A=q;function Z(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U&&Math.abs(n[2]-i[2])<U&&Math.abs(n[3]-i[3])<U&&Math.abs(n[4]-i[4])<U&&Math.abs(n[5]-i[5])<U&&Math.abs(n[6]-i[6])<U&&Math.abs(n[7]-i[7])<U&&Math.abs(n[8]-i[8])<U&&Math.abs(n[9]-i[9])<U&&Math.abs(n[10]-i[10])<U&&Math.abs(n[11]-i[11])<U&&Math.abs(n[12]-i[12])<U&&Math.abs(n[13]-i[13])<U&&Math.abs(n[14]-i[14])<U&&Math.abs(n[15]-i[15])<U}function nn(n,i){return n[0]===i[0]&&n[1]===i[1]&&n[2]===i[2]&&n[3]===i[3]&&n[4]===i[4]&&n[5]===i[5]&&n[6]===i[6]&&n[7]===i[7]&&n[8]===i[8]&&n[9]===i[9]&&n[10]===i[10]&&n[11]===i[11]&&n[12]===i[12]&&n[13]===i[13]&&n[14]===i[14]&&n[15]===i[15]}function an(n){const i=n??new f(16);return i[0]=1,i[1]=0,i[2]=0,i[3]=0,i[4]=0,i[5]=1,i[6]=0,i[7]=0,i[8]=0,i[9]=0,i[10]=1,i[11]=0,i[12]=0,i[13]=0,i[14]=0,i[15]=1,i}function hn(n,i){const e=i??new f(16);if(e===n){let T;return T=n[1],n[1]=n[4],n[4]=T,T=n[2],n[2]=n[8],n[8]=T,T=n[3],n[3]=n[12],n[12]=T,T=n[6],n[6]=n[9],n[9]=T,T=n[7],n[7]=n[13],n[13]=T,T=n[11],n[11]=n[14],n[14]=T,e}const r=n[0*4+0],a=n[0*4+1],_=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],x=n[2*4+0],m=n[2*4+1],M=n[2*4+2],P=n[2*4+3],R=n[3*4+0],z=n[3*4+1],b=n[3*4+2],I=n[3*4+3];return e[0]=r,e[1]=c,e[2]=x,e[3]=R,e[4]=a,e[5]=u,e[6]=m,e[7]=z,e[8]=_,e[9]=w,e[10]=M,e[11]=b,e[12]=y,e[13]=v,e[14]=P,e[15]=I,e}function pn(n,i){const e=i??new f(16),r=n[0*4+0],a=n[0*4+1],_=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],x=n[2*4+0],m=n[2*4+1],M=n[2*4+2],P=n[2*4+3],R=n[3*4+0],z=n[3*4+1],b=n[3*4+2],I=n[3*4+3],T=M*I,V=b*P,Y=w*I,H=b*v,X=w*P,W=M*v,K=_*I,J=b*y,tn=_*P,Q=M*y,rn=_*v,on=w*y,sn=x*z,cn=R*m,Rn=c*z,In=R*u,Dn=c*m,$n=x*u,Qn=r*z,Jn=R*a,Cn=r*m,ne=x*a,ee=r*u,te=c*a,ce=T*u+H*m+X*z-(V*u+Y*m+W*z),ae=V*a+K*m+Q*z-(T*a+J*m+tn*z),Zn=Y*a+J*u+rn*z-(H*a+K*u+on*z),re=W*a+tn*u+on*m-(X*a+Q*u+rn*m),C=1/(r*ce+c*ae+x*Zn+R*re);return e[0]=C*ce,e[1]=C*ae,e[2]=C*Zn,e[3]=C*re,e[4]=C*(V*c+Y*x+W*R-(T*c+H*x+X*R)),e[5]=C*(T*r+J*x+tn*R-(V*r+K*x+Q*R)),e[6]=C*(H*r+K*c+on*R-(Y*r+J*c+rn*R)),e[7]=C*(X*r+Q*c+rn*x-(W*r+tn*c+on*x)),e[8]=C*(sn*v+In*P+Dn*I-(cn*v+Rn*P+$n*I)),e[9]=C*(cn*y+Qn*P+ne*I-(sn*y+Jn*P+Cn*I)),e[10]=C*(Rn*y+Jn*v+ee*I-(In*y+Qn*v+te*I)),e[11]=C*($n*y+Cn*v+te*P-(Dn*y+ne*v+ee*P)),e[12]=C*(Rn*M+$n*b+cn*w-(Dn*b+sn*w+In*M)),e[13]=C*(Cn*b+sn*_+Jn*M-(Qn*M+ne*b+cn*_)),e[14]=C*(Qn*w+te*b+In*_-(ee*b+Rn*_+Jn*w)),e[15]=C*(ee*M+Dn*_+ne*w-(Cn*w+te*M+$n*_)),e}function Pn(n){const i=n[0],e=n[0*4+1],r=n[0*4+2],a=n[0*4+3],_=n[1*4+0],y=n[1*4+1],c=n[1*4+2],u=n[1*4+3],w=n[2*4+0],v=n[2*4+1],x=n[2*4+2],m=n[2*4+3],M=n[3*4+0],P=n[3*4+1],R=n[3*4+2],z=n[3*4+3],b=x*z,I=R*m,T=c*z,V=R*u,Y=c*m,H=x*u,X=r*z,W=R*a,K=r*m,J=x*a,tn=r*u,Q=c*a,rn=b*y+V*v+Y*P-(I*y+T*v+H*P),on=I*e+X*v+J*P-(b*e+W*v+K*P),sn=T*e+W*y+tn*P-(V*e+X*y+Q*P),cn=H*e+K*y+Q*v-(Y*e+J*y+tn*v);return i*rn+_*on+w*sn+M*cn}const zn=pn;function Tn(n,i,e){const r=e??new f(16),a=n[0],_=n[1],y=n[2],c=n[3],u=n[4],w=n[5],v=n[6],x=n[7],m=n[8],M=n[9],P=n[10],R=n[11],z=n[12],b=n[13],I=n[14],T=n[15],V=i[0],Y=i[1],H=i[2],X=i[3],W=i[4],K=i[5],J=i[6],tn=i[7],Q=i[8],rn=i[9],on=i[10],sn=i[11],cn=i[12],Rn=i[13],In=i[14],Dn=i[15];return r[0]=a*V+u*Y+m*H+z*X,r[1]=_*V+w*Y+M*H+b*X,r[2]=y*V+v*Y+P*H+I*X,r[3]=c*V+x*Y+R*H+T*X,r[4]=a*W+u*K+m*J+z*tn,r[5]=_*W+w*K+M*J+b*tn,r[6]=y*W+v*K+P*J+I*tn,r[7]=c*W+x*K+R*J+T*tn,r[8]=a*Q+u*rn+m*on+z*sn,r[9]=_*Q+w*rn+M*on+b*sn,r[10]=y*Q+v*rn+P*on+I*sn,r[11]=c*Q+x*rn+R*on+T*sn,r[12]=a*cn+u*Rn+m*In+z*Dn,r[13]=_*cn+w*Rn+M*In+b*Dn,r[14]=y*cn+v*Rn+P*In+I*Dn,r[15]=c*cn+x*Rn+R*In+T*Dn,r}const On=Tn;function F(n,i,e){const r=e??an();return n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11]),r[12]=i[0],r[13]=i[1],r[14]=i[2],r[15]=1,r}function un(n,i){const e=i??g.create();return e[0]=n[12],e[1]=n[13],e[2]=n[14],e}function Un(n,i,e){const r=e??g.create(),a=i*4;return r[0]=n[a+0],r[1]=n[a+1],r[2]=n[a+2],r}function xn(n,i,e,r){const a=r===n?r:q(n,r),_=e*4;return a[_+0]=i[0],a[_+1]=i[1],a[_+2]=i[2],a}function wn(n,i){const e=i??g.create(),r=n[0],a=n[1],_=n[2],y=n[4],c=n[5],u=n[6],w=n[8],v=n[9],x=n[10];return e[0]=Math.sqrt(r*r+a*a+_*_),e[1]=Math.sqrt(y*y+c*c+u*u),e[2]=Math.sqrt(w*w+v*v+x*x),e}function Bn(n,i,e,r,a){const _=a??new f(16),y=Math.tan(Math.PI*.5-.5*n);if(_[0]=y/i,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=y,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[11]=-1,_[12]=0,_[13]=0,_[15]=0,Number.isFinite(r)){const c=1/(e-r);_[10]=r*c,_[14]=r*e*c}else _[10]=-1,_[14]=-e;return _}function ln(n,i,e,r=1/0,a){const _=a??new f(16),y=1/Math.tan(n*.5);if(_[0]=y/i,_[1]=0,_[2]=0,_[3]=0,_[4]=0,_[5]=y,_[6]=0,_[7]=0,_[8]=0,_[9]=0,_[11]=-1,_[12]=0,_[13]=0,_[15]=0,r===1/0)_[10]=0,_[14]=e;else{const c=1/(r-e);_[10]=e*c,_[14]=r*e*c}return _}function bn(n,i,e,r,a,_,y){const c=y??new f(16);return c[0]=2/(i-n),c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2/(r-e),c[6]=0,c[7]=0,c[8]=0,c[9]=0,c[10]=1/(a-_),c[11]=0,c[12]=(i+n)/(n-i),c[13]=(r+e)/(e-r),c[14]=a/(a-_),c[15]=1,c}function $(n,i,e,r,a,_,y){const c=y??new f(16),u=i-n,w=r-e,v=a-_;return c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[10]=_/v,c[11]=-1,c[12]=0,c[13]=0,c[14]=a*_/v,c[15]=0,c}function Gn(n,i,e,r,a,_=1/0,y){const c=y??new f(16),u=i-n,w=r-e;if(c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[11]=-1,c[12]=0,c[13]=0,c[15]=0,_===1/0)c[10]=0,c[14]=a;else{const v=1/(_-a);c[10]=a*v,c[14]=_*a*v}return c}const B=g.create(),L=g.create(),O=g.create();function en(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(i,n,O),O),g.normalize(g.cross(e,O,B),B),g.normalize(g.cross(O,B,L),L),a[0]=B[0],a[1]=B[1],a[2]=B[2],a[3]=0,a[4]=L[0],a[5]=L[1],a[6]=L[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function vn(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(n,i,O),O),g.normalize(g.cross(e,O,B),B),g.normalize(g.cross(O,B,L),L),a[0]=B[0],a[1]=B[1],a[2]=B[2],a[3]=0,a[4]=L[0],a[5]=L[1],a[6]=L[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function kn(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(n,i,O),O),g.normalize(g.cross(e,O,B),B),g.normalize(g.cross(O,B,L),L),a[0]=B[0],a[1]=L[0],a[2]=O[0],a[3]=0,a[4]=B[1],a[5]=L[1],a[6]=O[1],a[7]=0,a[8]=B[2],a[9]=L[2],a[10]=O[2],a[11]=0,a[12]=-(B[0]*n[0]+B[1]*n[1]+B[2]*n[2]),a[13]=-(L[0]*n[0]+L[1]*n[1]+L[2]*n[2]),a[14]=-(O[0]*n[0]+O[1]*n[1]+O[2]*n[2]),a[15]=1,a}function _n(n,i){const e=i??new f(16);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=n[0],e[13]=n[1],e[14]=n[2],e[15]=1,e}function yn(n,i,e){const r=e??new f(16),a=i[0],_=i[1],y=i[2],c=n[0],u=n[1],w=n[2],v=n[3],x=n[1*4+0],m=n[1*4+1],M=n[1*4+2],P=n[1*4+3],R=n[2*4+0],z=n[2*4+1],b=n[2*4+2],I=n[2*4+3],T=n[3*4+0],V=n[3*4+1],Y=n[3*4+2],H=n[3*4+3];return n!==r&&(r[0]=c,r[1]=u,r[2]=w,r[3]=v,r[4]=x,r[5]=m,r[6]=M,r[7]=P,r[8]=R,r[9]=z,r[10]=b,r[11]=I),r[12]=c*a+x*_+R*y+T,r[13]=u*a+m*_+z*y+V,r[14]=w*a+M*_+b*y+Y,r[15]=v*a+P*_+I*y+H,r}function N(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=r,e[6]=a,e[7]=0,e[8]=0,e[9]=-a,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function mn(n,i,e){const r=e??new f(16),a=n[4],_=n[5],y=n[6],c=n[7],u=n[8],w=n[9],v=n[10],x=n[11],m=Math.cos(i),M=Math.sin(i);return r[4]=m*a+M*u,r[5]=m*_+M*w,r[6]=m*y+M*v,r[7]=m*c+M*x,r[8]=m*u-M*a,r[9]=m*w-M*_,r[10]=m*v-M*y,r[11]=m*x-M*c,n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function fn(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=0,e[2]=-a,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=a,e[9]=0,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Sn(n,i,e){const r=e??new f(16),a=n[0*4+0],_=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[2*4+0],w=n[2*4+1],v=n[2*4+2],x=n[2*4+3],m=Math.cos(i),M=Math.sin(i);return r[0]=m*a-M*u,r[1]=m*_-M*w,r[2]=m*y-M*v,r[3]=m*c-M*x,r[8]=m*u+M*a,r[9]=m*w+M*_,r[10]=m*v+M*y,r[11]=m*x+M*c,n!==r&&(r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function dn(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=a,e[2]=0,e[3]=0,e[4]=-a,e[5]=r,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function En(n,i,e){const r=e??new f(16),a=n[0*4+0],_=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[1*4+0],w=n[1*4+1],v=n[1*4+2],x=n[1*4+3],m=Math.cos(i),M=Math.sin(i);return r[0]=m*a+M*u,r[1]=m*_+M*w,r[2]=m*y+M*v,r[3]=m*c+M*x,r[4]=m*u-M*a,r[5]=m*w-M*_,r[6]=m*v-M*y,r[7]=m*x-M*c,n!==r&&(r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function Mn(n,i,e){const r=e??new f(16);let a=n[0],_=n[1],y=n[2];const c=Math.sqrt(a*a+_*_+y*y);a/=c,_/=c,y/=c;const u=a*a,w=_*_,v=y*y,x=Math.cos(i),m=Math.sin(i),M=1-x;return r[0]=u+(1-u)*x,r[1]=a*_*M+y*m,r[2]=a*y*M-_*m,r[3]=0,r[4]=a*_*M-y*m,r[5]=w+(1-w)*x,r[6]=_*y*M+a*m,r[7]=0,r[8]=a*y*M+_*m,r[9]=_*y*M-a*m,r[10]=v+(1-v)*x,r[11]=0,r[12]=0,r[13]=0,r[14]=0,r[15]=1,r}const o=Mn;function d(n,i,e,r){const a=r??new f(16);let _=i[0],y=i[1],c=i[2];const u=Math.sqrt(_*_+y*y+c*c);_/=u,y/=u,c/=u;const w=_*_,v=y*y,x=c*c,m=Math.cos(e),M=Math.sin(e),P=1-m,R=w+(1-w)*m,z=_*y*P+c*M,b=_*c*P-y*M,I=_*y*P-c*M,T=v+(1-v)*m,V=y*c*P+_*M,Y=_*c*P+y*M,H=y*c*P-_*M,X=x+(1-x)*m,W=n[0],K=n[1],J=n[2],tn=n[3],Q=n[4],rn=n[5],on=n[6],sn=n[7],cn=n[8],Rn=n[9],In=n[10],Dn=n[11];return a[0]=R*W+z*Q+b*cn,a[1]=R*K+z*rn+b*Rn,a[2]=R*J+z*on+b*In,a[3]=R*tn+z*sn+b*Dn,a[4]=I*W+T*Q+V*cn,a[5]=I*K+T*rn+V*Rn,a[6]=I*J+T*on+V*In,a[7]=I*tn+T*sn+V*Dn,a[8]=Y*W+H*Q+X*cn,a[9]=Y*K+H*rn+X*Rn,a[10]=Y*J+H*on+X*In,a[11]=Y*tn+H*sn+X*Dn,n!==a&&(a[12]=n[12],a[13]=n[13],a[14]=n[14],a[15]=n[15]),a}const t=d;function s(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n[1],e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n[2],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function l(n,i,e){const r=e??new f(16),a=i[0],_=i[1],y=i[2];return r[0]=a*n[0*4+0],r[1]=a*n[0*4+1],r[2]=a*n[0*4+2],r[3]=a*n[0*4+3],r[4]=_*n[1*4+0],r[5]=_*n[1*4+1],r[6]=_*n[1*4+2],r[7]=_*n[1*4+3],r[8]=y*n[2*4+0],r[9]=y*n[2*4+1],r[10]=y*n[2*4+2],r[11]=y*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function p(n,i){const e=i??new f(16);return e[0]=n,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function h(n,i,e){const r=e??new f(16);return r[0]=i*n[0*4+0],r[1]=i*n[0*4+1],r[2]=i*n[0*4+2],r[3]=i*n[0*4+3],r[4]=i*n[1*4+0],r[5]=i*n[1*4+1],r[6]=i*n[1*4+2],r[7]=i*n[1*4+3],r[8]=i*n[2*4+0],r[9]=i*n[2*4+1],r[10]=i*n[2*4+2],r[11]=i*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}return{create:S,set:E,fromMat3:D,fromQuat:G,negate:k,copy:q,clone:A,equalsApproximately:Z,equals:nn,identity:an,transpose:hn,inverse:pn,determinant:Pn,invert:zn,multiply:Tn,mul:On,setTranslation:F,getTranslation:un,getAxis:Un,setAxis:xn,getScaling:wn,perspective:Bn,perspectiveReverseZ:ln,ortho:bn,frustum:$,frustumReverseZ:Gn,aim:en,cameraAim:vn,lookAt:kn,translation:_n,translate:yn,rotationX:N,rotateX:mn,rotationY:fn,rotateY:Sn,rotationZ:dn,rotateZ:En,axisRotation:Mn,rotation:o,axisRotate:d,rotate:t,scaling:s,scale:l,uniformScaling:p,uniformScale:h}}const Ge=new Map;function lt(f){let g=Ge.get(f);return g||(g=ut(f),Ge.set(f,g)),g}function ft(f){const g=de(f);function S(o,d,t,s){const l=new f(4);return o!==void 0&&(l[0]=o,d!==void 0&&(l[1]=d,t!==void 0&&(l[2]=t,s!==void 0&&(l[3]=s)))),l}const E=S;function D(o,d,t,s,l){const p=l??new f(4);return p[0]=o,p[1]=d,p[2]=t,p[3]=s,p}function G(o,d,t){const s=t??new f(4),l=d*.5,p=Math.sin(l);return s[0]=p*o[0],s[1]=p*o[1],s[2]=p*o[2],s[3]=Math.cos(l),s}function k(o,d){const t=d??g.create(3),s=Math.acos(o[3])*2,l=Math.sin(s*.5);return l>U?(t[0]=o[0]/l,t[1]=o[1]/l,t[2]=o[2]/l):(t[0]=1,t[1]=0,t[2]=0),{angle:s,axis:t}}function q(o,d){const t=$(o,d);return Math.acos(2*t*t-1)}function A(o,d,t){const s=t??new f(4),l=o[0],p=o[1],h=o[2],n=o[3],i=d[0],e=d[1],r=d[2],a=d[3];return s[0]=l*a+n*i+p*r-h*e,s[1]=p*a+n*e+h*i-l*r,s[2]=h*a+n*r+l*e-p*i,s[3]=n*a-l*i-p*e-h*r,s}const Z=A;function nn(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r+i*e,s[1]=h*r+n*e,s[2]=n*r-h*e,s[3]=i*r-p*e,s}function an(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r-n*e,s[1]=h*r+i*e,s[2]=n*r+p*e,s[3]=i*r-h*e,s}function hn(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r+h*e,s[1]=h*r-p*e,s[2]=n*r+i*e,s[3]=i*r-n*e,s}function pn(o,d,t,s){const l=s??new f(4),p=o[0],h=o[1],n=o[2],i=o[3];let e=d[0],r=d[1],a=d[2],_=d[3],y=p*e+h*r+n*a+i*_;y<0&&(y=-y,e=-e,r=-r,a=-a,_=-_);let c,u;if(1-y>U){const w=Math.acos(y),v=Math.sin(w);c=Math.sin((1-t)*w)/v,u=Math.sin(t*w)/v}else c=1-t,u=t;return l[0]=c*p+u*e,l[1]=c*h+u*r,l[2]=c*n+u*a,l[3]=c*i+u*_,l}function Pn(o,d){const t=d??new f(4),s=o[0],l=o[1],p=o[2],h=o[3],n=s*s+l*l+p*p+h*h,i=n?1/n:0;return t[0]=-s*i,t[1]=-l*i,t[2]=-p*i,t[3]=h*i,t}function zn(o,d){const t=d??new f(4);return t[0]=-o[0],t[1]=-o[1],t[2]=-o[2],t[3]=o[3],t}function Tn(o,d){const t=d??new f(4),s=o[0]+o[5]+o[10];if(s>0){const l=Math.sqrt(s+1);t[3]=.5*l;const p=.5/l;t[0]=(o[6]-o[9])*p,t[1]=(o[8]-o[2])*p,t[2]=(o[1]-o[4])*p}else{let l=0;o[5]>o[0]&&(l=1),o[10]>o[l*4+l]&&(l=2);const p=(l+1)%3,h=(l+2)%3,n=Math.sqrt(o[l*4+l]-o[p*4+p]-o[h*4+h]+1);t[l]=.5*n;const i=.5/n;t[3]=(o[p*4+h]-o[h*4+p])*i,t[p]=(o[p*4+l]+o[l*4+p])*i,t[h]=(o[h*4+l]+o[l*4+h])*i}return t}function On(o,d,t,s,l){const p=l??new f(4),h=o*.5,n=d*.5,i=t*.5,e=Math.sin(h),r=Math.cos(h),a=Math.sin(n),_=Math.cos(n),y=Math.sin(i),c=Math.cos(i);switch(s){case"xyz":p[0]=e*_*c+r*a*y,p[1]=r*a*c-e*_*y,p[2]=r*_*y+e*a*c,p[3]=r*_*c-e*a*y;break;case"xzy":p[0]=e*_*c-r*a*y,p[1]=r*a*c-e*_*y,p[2]=r*_*y+e*a*c,p[3]=r*_*c+e*a*y;break;case"yxz":p[0]=e*_*c+r*a*y,p[1]=r*a*c-e*_*y,p[2]=r*_*y-e*a*c,p[3]=r*_*c+e*a*y;break;case"yzx":p[0]=e*_*c+r*a*y,p[1]=r*a*c+e*_*y,p[2]=r*_*y-e*a*c,p[3]=r*_*c-e*a*y;break;case"zxy":p[0]=e*_*c-r*a*y,p[1]=r*a*c+e*_*y,p[2]=r*_*y+e*a*c,p[3]=r*_*c-e*a*y;break;case"zyx":p[0]=e*_*c-r*a*y,p[1]=r*a*c+e*_*y,p[2]=r*_*y-e*a*c,p[3]=r*_*c+e*a*y;break;default:throw new Error(`Unknown rotation order: ${s}`)}return p}function F(o,d){const t=d??new f(4);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[3]=o[3],t}const un=F;function Un(o,d,t){const s=t??new f(4);return s[0]=o[0]+d[0],s[1]=o[1]+d[1],s[2]=o[2]+d[2],s[3]=o[3]+d[3],s}function xn(o,d,t){const s=t??new f(4);return s[0]=o[0]-d[0],s[1]=o[1]-d[1],s[2]=o[2]-d[2],s[3]=o[3]-d[3],s}const wn=xn;function Bn(o,d,t){const s=t??new f(4);return s[0]=o[0]*d,s[1]=o[1]*d,s[2]=o[2]*d,s[3]=o[3]*d,s}const ln=Bn;function bn(o,d,t){const s=t??new f(4);return s[0]=o[0]/d,s[1]=o[1]/d,s[2]=o[2]/d,s[3]=o[3]/d,s}function $(o,d){return o[0]*d[0]+o[1]*d[1]+o[2]*d[2]+o[3]*d[3]}function Gn(o,d,t,s){const l=s??new f(4);return l[0]=o[0]+t*(d[0]-o[0]),l[1]=o[1]+t*(d[1]-o[1]),l[2]=o[2]+t*(d[2]-o[2]),l[3]=o[3]+t*(d[3]-o[3]),l}function B(o){const d=o[0],t=o[1],s=o[2],l=o[3];return Math.sqrt(d*d+t*t+s*s+l*l)}const L=B;function O(o){const d=o[0],t=o[1],s=o[2],l=o[3];return d*d+t*t+s*s+l*l}const en=O;function vn(o,d){const t=d??new f(4),s=o[0],l=o[1],p=o[2],h=o[3],n=Math.sqrt(s*s+l*l+p*p+h*h);return n>1e-5?(t[0]=s/n,t[1]=l/n,t[2]=p/n,t[3]=h/n):(t[0]=0,t[1]=0,t[2]=0,t[3]=1),t}function kn(o,d){return Math.abs(o[0]-d[0])<U&&Math.abs(o[1]-d[1])<U&&Math.abs(o[2]-d[2])<U&&Math.abs(o[3]-d[3])<U}function _n(o,d){return o[0]===d[0]&&o[1]===d[1]&&o[2]===d[2]&&o[3]===d[3]}function yn(o){const d=o??new f(4);return d[0]=0,d[1]=0,d[2]=0,d[3]=1,d}const N=g.create(),mn=g.create(),fn=g.create();function Sn(o,d,t){const s=t??new f(4),l=g.dot(o,d);return l<-.999999?(g.cross(mn,o,N),g.len(N)<1e-6&&g.cross(fn,o,N),g.normalize(N,N),G(N,Math.PI,s),s):l>.999999?(s[0]=0,s[1]=0,s[2]=0,s[3]=1,s):(g.cross(o,d,N),s[0]=N[0],s[1]=N[1],s[2]=N[2],s[3]=1+l,vn(s,s))}const dn=new f(4),En=new f(4);function Mn(o,d,t,s,l,p){const h=p??new f(4);return pn(o,s,l,dn),pn(d,t,l,En),pn(dn,En,2*l*(1-l),h),h}return{create:S,fromValues:E,set:D,fromAxisAngle:G,toAxisAngle:k,angle:q,multiply:A,mul:Z,rotateX:nn,rotateY:an,rotateZ:hn,slerp:pn,inverse:Pn,conjugate:zn,fromMat:Tn,fromEuler:On,copy:F,clone:un,add:Un,subtract:xn,sub:wn,mulScalar:Bn,scale:ln,divScalar:bn,dot:$,lerp:Gn,length:B,len:L,lengthSq:O,lenSq:en,normalize:vn,equalsApproximately:kn,equals:_n,identity:yn,rotationTo:Sn,sqlerp:Mn}}const Se=new Map;function dt(f){let g=Se.get(f);return g||(g=ft(f),Se.set(f,g)),g}function pt(f){function g(t,s,l,p){const h=new f(4);return t!==void 0&&(h[0]=t,s!==void 0&&(h[1]=s,l!==void 0&&(h[2]=l,p!==void 0&&(h[3]=p)))),h}const S=g;function E(t,s,l,p,h){const n=h??new f(4);return n[0]=t,n[1]=s,n[2]=l,n[3]=p,n}function D(t,s){const l=s??new f(4);return l[0]=Math.ceil(t[0]),l[1]=Math.ceil(t[1]),l[2]=Math.ceil(t[2]),l[3]=Math.ceil(t[3]),l}function G(t,s){const l=s??new f(4);return l[0]=Math.floor(t[0]),l[1]=Math.floor(t[1]),l[2]=Math.floor(t[2]),l[3]=Math.floor(t[3]),l}function k(t,s){const l=s??new f(4);return l[0]=Math.round(t[0]),l[1]=Math.round(t[1]),l[2]=Math.round(t[2]),l[3]=Math.round(t[3]),l}function q(t,s=0,l=1,p){const h=p??new f(4);return h[0]=Math.min(l,Math.max(s,t[0])),h[1]=Math.min(l,Math.max(s,t[1])),h[2]=Math.min(l,Math.max(s,t[2])),h[3]=Math.min(l,Math.max(s,t[3])),h}function A(t,s,l){const p=l??new f(4);return p[0]=t[0]+s[0],p[1]=t[1]+s[1],p[2]=t[2]+s[2],p[3]=t[3]+s[3],p}function Z(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+s[0]*l,h[1]=t[1]+s[1]*l,h[2]=t[2]+s[2]*l,h[3]=t[3]+s[3]*l,h}function nn(t,s,l){const p=l??new f(4);return p[0]=t[0]-s[0],p[1]=t[1]-s[1],p[2]=t[2]-s[2],p[3]=t[3]-s[3],p}const an=nn;function hn(t,s){return Math.abs(t[0]-s[0])<U&&Math.abs(t[1]-s[1])<U&&Math.abs(t[2]-s[2])<U&&Math.abs(t[3]-s[3])<U}function pn(t,s){return t[0]===s[0]&&t[1]===s[1]&&t[2]===s[2]&&t[3]===s[3]}function Pn(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+l*(s[0]-t[0]),h[1]=t[1]+l*(s[1]-t[1]),h[2]=t[2]+l*(s[2]-t[2]),h[3]=t[3]+l*(s[3]-t[3]),h}function zn(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+l[0]*(s[0]-t[0]),h[1]=t[1]+l[1]*(s[1]-t[1]),h[2]=t[2]+l[2]*(s[2]-t[2]),h[3]=t[3]+l[3]*(s[3]-t[3]),h}function Tn(t,s,l){const p=l??new f(4);return p[0]=Math.max(t[0],s[0]),p[1]=Math.max(t[1],s[1]),p[2]=Math.max(t[2],s[2]),p[3]=Math.max(t[3],s[3]),p}function On(t,s,l){const p=l??new f(4);return p[0]=Math.min(t[0],s[0]),p[1]=Math.min(t[1],s[1]),p[2]=Math.min(t[2],s[2]),p[3]=Math.min(t[3],s[3]),p}function F(t,s,l){const p=l??new f(4);return p[0]=t[0]*s,p[1]=t[1]*s,p[2]=t[2]*s,p[3]=t[3]*s,p}const un=F;function Un(t,s,l){const p=l??new f(4);return p[0]=t[0]/s,p[1]=t[1]/s,p[2]=t[2]/s,p[3]=t[3]/s,p}function xn(t,s){const l=s??new f(4);return l[0]=1/t[0],l[1]=1/t[1],l[2]=1/t[2],l[3]=1/t[3],l}const wn=xn;function Bn(t,s){return t[0]*s[0]+t[1]*s[1]+t[2]*s[2]+t[3]*s[3]}function ln(t){const s=t[0],l=t[1],p=t[2],h=t[3];return Math.sqrt(s*s+l*l+p*p+h*h)}const bn=ln;function $(t){const s=t[0],l=t[1],p=t[2],h=t[3];return s*s+l*l+p*p+h*h}const Gn=$;function B(t,s){const l=t[0]-s[0],p=t[1]-s[1],h=t[2]-s[2],n=t[3]-s[3];return Math.sqrt(l*l+p*p+h*h+n*n)}const L=B;function O(t,s){const l=t[0]-s[0],p=t[1]-s[1],h=t[2]-s[2],n=t[3]-s[3];return l*l+p*p+h*h+n*n}const en=O;function vn(t,s){const l=s??new f(4),p=t[0],h=t[1],n=t[2],i=t[3],e=Math.sqrt(p*p+h*h+n*n+i*i);return e>1e-5?(l[0]=p/e,l[1]=h/e,l[2]=n/e,l[3]=i/e):(l[0]=0,l[1]=0,l[2]=0,l[3]=0),l}function kn(t,s){const l=s??new f(4);return l[0]=-t[0],l[1]=-t[1],l[2]=-t[2],l[3]=-t[3],l}function _n(t,s){const l=s??new f(4);return l[0]=t[0],l[1]=t[1],l[2]=t[2],l[3]=t[3],l}const yn=_n;function N(t,s,l){const p=l??new f(4);return p[0]=t[0]*s[0],p[1]=t[1]*s[1],p[2]=t[2]*s[2],p[3]=t[3]*s[3],p}const mn=N;function fn(t,s,l){const p=l??new f(4);return p[0]=t[0]/s[0],p[1]=t[1]/s[1],p[2]=t[2]/s[2],p[3]=t[3]/s[3],p}const Sn=fn;function dn(t){const s=t??new f(4);return s[0]=0,s[1]=0,s[2]=0,s[3]=0,s}function En(t,s,l){const p=l??new f(4),h=t[0],n=t[1],i=t[2],e=t[3];return p[0]=s[0]*h+s[4]*n+s[8]*i+s[12]*e,p[1]=s[1]*h+s[5]*n+s[9]*i+s[13]*e,p[2]=s[2]*h+s[6]*n+s[10]*i+s[14]*e,p[3]=s[3]*h+s[7]*n+s[11]*i+s[15]*e,p}function Mn(t,s,l){const p=l??new f(4);return vn(t,p),F(p,s,p)}function o(t,s,l){const p=l??new f(4);return ln(t)>s?Mn(t,s,p):_n(t,p)}function d(t,s,l){const p=l??new f(4);return Pn(t,s,.5,p)}return{create:g,fromValues:S,set:E,ceil:D,floor:G,round:k,clamp:q,add:A,addScaled:Z,subtract:nn,sub:an,equalsApproximately:hn,equals:pn,lerp:Pn,lerpV:zn,max:Tn,min:On,mulScalar:F,scale:un,divScalar:Un,inverse:xn,invert:wn,dot:Bn,length:ln,len:bn,lengthSq:$,lenSq:Gn,distance:B,dist:L,distanceSq:O,distSq:en,normalize:vn,negate:kn,copy:_n,clone:yn,multiply:N,mul:mn,divide:fn,div:Sn,zero:dn,transformMat4:En,setLength:Mn,truncate:o,midpoint:d}}const Ee=new Map;function wt(f){let g=Ee.get(f);return g||(g=pt(f),Ee.set(f,g)),g}function pe(f,g,S,E,D,G){return{mat3:at(f),mat4:lt(g),quat:dt(S),vec2:Ie(E),vec3:de(D),vec4:wt(G)}}const{mat3:mt,mat4:gn,quat:Mt,vec2:Dt,vec3:Pt,vec4:zt}=pe(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);pe(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array);pe(ot,Array,Array,Array,Array,Array);const Hn=.07,Xn=1,jn=2,Yn=1;function vt(f){let g=new ArrayBuffer(64*f);var S=0;const E=.5;for(var D=-jn;S<f;D+=E*Hn)for(var G=-.95*Xn;G<.95*Xn&&S<f;G+=E*Hn)for(var k=-.95*Yn;k<0&&S<f;k+=E*Hn){let q=1e-4*Math.random();const A=64*S;({position:new Float32Array(g,A,3),velocity:new Float32Array(g,A+16,3),force:new Float32Array(g,A+32,3),density:new Float32Array(g,A+44,1),nearDensity:new Float32Array(g,A+48,1)}).position.set([G+q,D,k]),S++}return g}async function _t(){const f=document.querySelector("canvas"),g=await navigator.gpu.requestAdapter();if(!g)throw new Error;const S=await g.requestDevice(),E=f.getContext("webgpu");if(!E)throw new Error;const{devicePixelRatio:D}=window;f.width=D*f.clientWidth,f.height=D*f.clientHeight;const G=navigator.gpu.getPreferredCanvasFormat();return E.configure({device:S,format:G}),{canvas:f,device:S,presentationFormat:G,context:E}}function gt(f,g,S,E){var D=[f[0]*2-1,1-f[1]*2,0,1];D[2]=-S[4*2+2]+S[4*3+2]/g;var G=gn.multiply(E,D),k=G[3];return[G[0]/k,G[1]/k,G[2]/k]}function ht(f,g){const S=f.clientWidth/f.clientHeight,E=gn.perspective(g,S,.1,50),D=gn.lookAt([0,0,5.5],[0,0,0],[0,1,0]),G=new Float32Array(4);G[0]=1.5,G[1]=1,G[2]=2,G[3]=1;const k=gn.multiply(D,G),q=Math.abs(k[2]),A=gn.multiply(E,k),Z=[A[0]/A[3],A[1]/A[3],A[2]/A[3]];console.log("camera: ",k),console.log("ndc: ",Z);const nn=[(Z[0]+1)/2,(1-Z[1])/2],an=gn.inverse(E),hn=gt(nn,q,E,an);return console.log(hn),{projection:E,view:D}}function xt(f,g,S,E){var D=gn.identity();gn.translate(D,E,D),gn.rotateY(D,S,D),gn.rotateX(D,g,D),gn.translate(D,[0,0,f],D);var G=gn.multiply(D,[0,0,0,1]);return gn.lookAt([G[0],G[1],G[2]],E,[0,1,0])}const Te=.025,Re=2*Te;async function yt(){const{canvas:f,device:g,presentationFormat:S,context:E}=await _t();E.configure({device:g,format:S,alphaMode:"premultiplied"});const D=g.createShaderModule({code:je}),G=g.createShaderModule({code:He}),k=g.createShaderModule({code:We}),q=g.createShaderModule({code:qe}),A=g.createShaderModule({code:Ke}),Z=g.createShaderModule({code:Ze}),nn=g.createShaderModule({code:Ye}),an=g.createShaderModule({code:Xe}),hn=g.createShaderModule({code:Fe}),pn=g.createShaderModule({code:Ne}),Pn=g.createShaderModule({code:Ve}),zn=g.createShaderModule({code:Qe}),Tn=g.createShaderModule({code:$e}),On=g.createShaderModule({code:Je}),F={kernelRadius:Hn,kernelRadiusPow2:Math.pow(Hn,2),kernelRadiusPow4:Math.pow(Hn,4),kernelRadiusPow5:Math.pow(Hn,5),kernelRadiusPow6:Math.pow(Hn,6),kernelRadiusPow9:Math.pow(Hn,9),stiffness:15,nearStiffness:1.5,mass:1,restDensity:35e3,viscosity:2e5,dt:.008,xHalf:Xn,yHalf:jn,zHalf:Yn},un=g.createRenderPipeline({label:"circles pipeline",layout:"auto",vertex:{module:G},fragment:{module:G,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),Un=g.createRenderPipeline({label:"ball pipeline",layout:"auto",vertex:{module:Z},fragment:{module:Z,targets:[{format:S}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),xn=90*Math.PI/180,{projection:wn,view:Bn}=ht(f,xn),ln={screenHeight:f.height,screenWidth:f.width},bn={depth_threshold:Te*10,max_filter_size:100,projected_particle_constant:10*Re*.05*(f.height/2)/Math.tan(xn/2)},$=g.createRenderPipeline({label:"filter pipeline",layout:"auto",vertex:{module:D,constants:ln},fragment:{module:q,constants:bn,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"}}),Gn=g.createRenderPipeline({label:"fluid rendering pipeline",layout:"auto",vertex:{module:D,constants:ln},fragment:{module:A,targets:[{format:S}]},primitive:{topology:"triangle-list",cullMode:"none"}}),B=g.createRenderPipeline({label:"thickness pipeline",layout:"auto",vertex:{module:nn},fragment:{module:nn,targets:[{format:"r16float",writeMask:GPUColorWrite.RED,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}}),L=g.createRenderPipeline({label:"thickness filter pipeline",layout:"auto",vertex:{module:D,constants:ln},fragment:{module:an,targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),O=g.createRenderPipeline({label:"show pipeline",layout:"auto",vertex:{module:D,constants:ln},fragment:{module:k,targets:[{format:S}]},primitive:{topology:"triangle-list",cullMode:"none"}}),en=1*Hn,vn=2*Xn,kn=2*jn,_n=2*Yn,yn=4*en,N=Math.ceil((vn+yn)/en),mn=Math.ceil((kn+yn)/en),fn=Math.ceil((_n+yn)/en),Sn=N*mn*fn,dn=yn/2,En={xHalf:Xn,yHalf:jn,zHalf:Yn,xGrids:N,yGrids:mn,cellSize:en,offset:dn},Mn=g.createComputePipeline({label:"grid clear pipeline",layout:"auto",compute:{module:Tn}}),o=g.createComputePipeline({label:"grid build pipeline",layout:"auto",compute:{module:zn,constants:En}}),d=g.createComputePipeline({label:"reorder pipeline",layout:"auto",compute:{module:On,constants:En}}),t=g.createComputePipeline({label:"density pipeline",layout:"auto",compute:{module:hn,constants:{kernelRadius:F.kernelRadius,kernelRadiusPow2:F.kernelRadiusPow2,kernelRadiusPow5:F.kernelRadiusPow5,kernelRadiusPow6:F.kernelRadiusPow6,mass:F.mass,xHalf:Xn,yHalf:jn,zHalf:Yn,xGrids:N,yGrids:mn,zGrids:fn,cellSize:en,offset:dn}}}),s=g.createComputePipeline({label:"force pipeline",layout:"auto",compute:{module:pn,constants:{kernelRadius:F.kernelRadius,kernelRadiusPow2:F.kernelRadiusPow2,kernelRadiusPow5:F.kernelRadiusPow5,kernelRadiusPow6:F.kernelRadiusPow6,kernelRadiusPow9:F.kernelRadiusPow9,mass:F.mass,stiffness:F.stiffness,nearStiffness:F.nearStiffness,viscosity:F.viscosity,restDensity:F.restDensity,xHalf:Xn,yHalf:jn,zHalf:Yn,xGrids:N,yGrids:mn,zGrids:fn,cellSize:en,offset:dn}}}),l=g.createComputePipeline({label:"integrate pipeline",layout:"auto",compute:{module:Pn,constants:{dt:F.dt}}}),h=g.createTexture({label:"depth map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),i=g.createTexture({label:"temporary texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),r=g.createTexture({label:"thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),_=g.createTexture({label:"temporary thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),y=g.createTexture({size:[f.width,f.height,1],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});y.createView();let c;{const qn=["cubemap/posx.jpg","cubemap/negx.jpg","cubemap/posy.jpg","cubemap/negy.jpg","cubemap/posz.jpg","cubemap/negz.jpg"].map(async Nn=>{const Vn=await fetch(Nn);return createImageBitmap(await Vn.blob())}),Fn=await Promise.all(qn);console.log(Fn[0].width,Fn[0].height),c=g.createTexture({dimension:"2d",size:[Fn[0].width,Fn[0].height,6],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let Nn=0;Nn<Fn.length;Nn++){const Vn=Fn[Nn];g.queue.copyExternalImageToTexture({source:Vn},{texture:c,origin:[0,0,Nn]},[Vn.width,Vn.height])}}const u=c.createView({dimension:"cube"}),w=g.createSampler({magFilter:"linear",minFilter:"linear"}),v=2e4,x=vt(v),m=new ArrayBuffer(144),M={size:new Float32Array(m,0,1),view_matrix:new Float32Array(m,16,16),projection_matrix:new Float32Array(m,80,16)},P=new ArrayBuffer(8),R=new ArrayBuffer(8),z={blur_dir:new Float32Array(P)},b={blur_dir:new Float32Array(R)};z.blur_dir.set([1,0]),b.blur_dir.set([0,1]);const I=new ArrayBuffer(272),T={texel_size:new Float32Array(I,0,2),inv_projection_matrix:new Float32Array(I,16,16),projection_matrix:new Float32Array(I,80,16),view_matrix:new Float32Array(I,144,16),inv_view_matrix:new Float32Array(I,208,16)};T.texel_size.set([1/f.width,1/f.height]),T.projection_matrix.set(wn);const V=gn.identity(),Y=gn.identity();gn.inverse(wn,V),gn.inverse(Bn,Y),T.inv_projection_matrix.set(V),T.inv_view_matrix.set(Y);const H=new ArrayBuffer(12),X={xHalf:new Float32Array(H,0,1),yHalf:new Float32Array(H,4,1),zHalf:new Float32Array(H,8,1)};X.xHalf.set([Xn]),X.yHalf.set([jn]),X.zHalf.set([Yn]);const W=g.createBuffer({label:"particles buffer",size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),K=g.createBuffer({label:"cell particle count buffer",size:4*(Sn+1),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),J=g.createBuffer({label:"target particles buffer",size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),tn=g.createBuffer({label:"particle cell offset buffer",size:4*v,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});g.queue.writeBuffer(W,0,x);const Q=g.createBuffer({label:"uniform buffer",size:m.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),rn=g.createBuffer({label:"filter uniform buffer",size:P.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),on=g.createBuffer({label:"filter uniform buffer",size:R.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),sn=g.createBuffer({label:"filter uniform buffer",size:I.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),cn=g.createBuffer({label:"real box size buffer",size:H.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});g.queue.writeBuffer(rn,0,P),g.queue.writeBuffer(on,0,R),g.queue.writeBuffer(sn,0,I),g.queue.writeBuffer(cn,0,H);const Rn=g.createBindGroup({layout:Mn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}}]}),In=g.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}},{binding:1,resource:{buffer:tn}},{binding:2,resource:{buffer:W}}]}),Dn=g.createBindGroup({layout:d.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}},{binding:3,resource:{buffer:tn}}]}),$n=g.createBindGroup({layout:t.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Qn=g.createBindGroup({layout:s.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Jn=g.createBindGroup({layout:l.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:cn}}]}),Cn=g.createBindGroup({label:"ball bind group",layout:Un.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:Q}}]}),ne=g.createBindGroup({label:"circle bind group",layout:un.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:Q}}]});g.createBindGroup({label:"show bind group",layout:O.getBindGroupLayout(0),entries:[{binding:1,resource:r}]});const ee=[g.createBindGroup({label:"filterX bind group",layout:$.getBindGroupLayout(0),entries:[{binding:1,resource:h},{binding:2,resource:{buffer:rn}}]}),g.createBindGroup({label:"filterY bind group",layout:$.getBindGroupLayout(0),entries:[{binding:1,resource:i},{binding:2,resource:{buffer:on}}]})],te=g.createBindGroup({label:"fluid bind group",layout:Gn.getBindGroupLayout(0),entries:[{binding:0,resource:w},{binding:1,resource:h},{binding:2,resource:{buffer:sn}},{binding:3,resource:r},{binding:4,resource:u}]}),ce=g.createBindGroup({label:"thickness bind group",layout:B.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:W}},{binding:1,resource:{buffer:Q}}]}),ae=[g.createBindGroup({label:"thickness filterX bind group",layout:L.getBindGroupLayout(0),entries:[{binding:1,resource:r},{binding:2,resource:{buffer:rn}}]}),g.createBindGroup({label:"thickness filterY bind group",layout:L.getBindGroupLayout(0),entries:[{binding:1,resource:_},{binding:2,resource:{buffer:on}}]})];let Zn=!1,re=0,C=0,we=Math.PI/4,oe=-Math.PI/12;const ve=.005,_e=-.99*Math.PI/2,ge=0,he=1,xe=3;let se=2;const ye=document.getElementById("fluidCanvas");ye.addEventListener("mousedown",Ln=>{Zn=!0,re=Ln.clientX,C=Ln.clientY}),ye.addEventListener("wheel",Ln=>{Ln.preventDefault();var qn=Ln.deltaY;se+=(qn>0?1:-1)*.05,se<he&&(se=he),se>xe&&(se=xe)}),document.addEventListener("mousemove",Ln=>{if(Zn){const qn=Ln.clientX,Fn=Ln.clientY,Nn=re-qn,Vn=C-Fn;we+=ve*Nn,oe+=ve*Vn,oe>ge&&(oe=ge),oe<_e&&(oe=_e),re=qn,C=Fn,console.log("Dragging... Delta:",Nn,Vn)}}),document.addEventListener("mouseup",()=>{Zn&&(Zn=!1,console.log("Drag ended"))});async function me(){const Ln=performance.now(),qn={colorAttachments:[{view:h,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},Fn={colorAttachments:[{view:E.getCurrentTexture().createView(),clearValue:{r:.7,g:.7,b:.7,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},Nn=[{colorAttachments:[{view:i,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:h,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];E.getCurrentTexture().createView();const Vn={colorAttachments:[{view:E.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Oe={colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Ue=[{colorAttachments:[{view:_,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];M.size.set([Re]),M.projection_matrix.set(wn);const Me=xt(se,oe,we,[0,-jn,0]);M.view_matrix.set(Me),T.view_matrix.set(Me),g.queue.writeBuffer(Q,0,m),g.queue.writeBuffer(sn,0,I);const Be=document.getElementById("slider"),ke=document.getElementById("slider-value"),Ae=document.getElementById("particle"),De=parseInt(Be.value)/200+.5;ke.textContent=De.toFixed(2),X.zHalf.set([Yn*De]),g.queue.writeBuffer(cn,0,H);const Kn=g.createCommandEncoder(),j=Kn.beginComputePass();for(let An=0;An<1;An++)j.setBindGroup(0,Rn),j.setPipeline(Mn),j.dispatchWorkgroups(Math.ceil((Sn+1)/64)),j.setBindGroup(0,In),j.setPipeline(o),j.dispatchWorkgroups(Math.ceil(v/64)),new tt({device:g,data:K,count:Sn+1}).dispatch(j),j.setBindGroup(0,Dn),j.setPipeline(d),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,$n),j.setPipeline(t),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Dn),j.setPipeline(d),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Qn),j.setPipeline(s),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Jn),j.setPipeline(l),j.dispatchWorkgroups(Math.ceil(v/64));if(j.end(),Ae.checked){const An=Kn.beginRenderPass(Fn);An.setBindGroup(0,Cn),An.setPipeline(Un),An.draw(6,v),An.end()}else{const An=Kn.beginRenderPass(qn);An.setBindGroup(0,ne),An.setPipeline(un),An.draw(6,v),An.end();for(var Wn=0;Wn<6;Wn++){const le=Kn.beginRenderPass(Nn[Wn%2]);le.setBindGroup(0,ee[Wn%2]),le.setPipeline($),le.draw(6),le.end()}const ie=Kn.beginRenderPass(Oe);ie.setBindGroup(0,ce),ie.setPipeline(B),ie.draw(6,v),ie.end();for(var Wn=0;Wn<4;Wn++){const fe=Kn.beginRenderPass(Ue[Wn%2]);fe.setBindGroup(0,ae[Wn%2]),fe.setPipeline(L),fe.draw(6),fe.end()}const ue=Kn.beginRenderPass(Vn);ue.setBindGroup(0,te),ue.setPipeline(Gn),ue.draw(6),ue.end()}g.queue.submit([Kn.finish()]);const Le=performance.now();console.log(`js: ${(Le-Ln).toFixed(1)}ms`),requestAnimationFrame(me)}requestAnimationFrame(me)}yt();
