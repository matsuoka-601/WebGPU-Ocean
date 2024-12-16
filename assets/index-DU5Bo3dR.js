(function(){const _=document.createElement("link").relList;if(_&&_.supports&&_.supports("modulepreload"))return;for(const M of document.querySelectorAll('link[rel="modulepreload"]'))S(M);new MutationObserver(M=>{for(const G of M)if(G.type==="childList")for(const A of G.addedNodes)A.tagName==="LINK"&&A.rel==="modulepreload"&&S(A)}).observe(document,{childList:!0,subtree:!0});function I(M){const G={};return M.integrity&&(G.integrity=M.integrity),M.referrerPolicy&&(G.referrerPolicy=M.referrerPolicy),M.crossOrigin==="use-credentials"?G.credentials="include":M.crossOrigin==="anonymous"?G.credentials="omit":G.credentials="same-origin",G}function S(M){if(M.ep)return;M.ep=!0;const G=I(M);fetch(M.href,G)}})();var Ne=`struct Particle {
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
        
        
        
        
        
        
        
        
        
    }
}`,Ve=`struct Particle {
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
override gridCount: u32;
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

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        fVisc *= viscosity;
        let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
        particles[id.x].force = fPress + fVisc + fGrv;
    }
}`,We=`struct Particle {
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
    
    if (particles[id.x].density != 0.) {
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
}`,qe=`@group(0) @binding(1) var texture: texture_2d<f32>;

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var r = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    
    return vec4(0, r, r, 1.0);
}`,Ke=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,je=`@group(0) @binding(0) var texture_sampler: sampler;
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

    

    
    
    
    
    
    
    
    
    
    
    
}`,Ye=`struct VertexOutput {
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
}`,Xe=`struct Uniforms {
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
}`,Ze=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,$e=`struct VertexOutput {
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
}`,Qe=`@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
    if (id.x < arrayLength(&cellParticleCount)) {
        cellParticleCount[id.x] = 0u;
    }
}`,Je=`struct Particle {
    position: vec3f, 
    velocity: vec3f, 
    force: vec3f, 
    density: f32, 
    nearDensity: f32, 
}

override xGrids: u32;
override yGrids: u32;
override gridCount: u32;
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
    if (cellID < gridCount) {
      particleCellOffset[id.x] = atomicAdd(&cellParticleCount[cellID], 1u);
    }
  }
}`,Ce=`struct Particle {
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
override gridCount: u32;
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
        if (cellId < gridCount) {
            let targetIndex = cellParticleCount[cellId + 1] - particleCellOffset[id.x] - 1;
            if (targetIndex < arrayLength(&targetParticles)) {
                targetParticles[targetIndex] = sourceParticles[id.x];
            }
        }
    }
}`;const nt=`

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
}`,et=`

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
}`;function tt(f,_){const I={x:_,y:1};if(_>f.limits.maxComputeWorkgroupsPerDimension){const S=Math.floor(Math.sqrt(_)),M=Math.ceil(_/S);I.x=S,I.y=M}return I}class rt{constructor({device:_,data:I,count:S,workgroup_size:M={x:16,y:16},avoid_bank_conflicts:G=!1}){if(this.device=_,this.workgroup_size=M,this.threads_per_workgroup=M.x*M.y,this.items_per_workgroup=2*this.threads_per_workgroup,Math.log2(this.threads_per_workgroup)%1!==0)throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`);this.pipelines=[],this.shaderModule=this.device.createShaderModule({label:"prefix-sum",code:G?et:nt}),this.create_pass_recursive(I,S)}create_pass_recursive(_,I){const S=Math.ceil(I/this.items_per_workgroup),M=tt(this.device,S),G=this.device.createBuffer({label:"prefix-sum-block-sum",size:S*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),A=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),q=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:A,entries:[{binding:0,resource:{buffer:_}},{binding:1,resource:{buffer:G}}]}),k=this.device.createPipelineLayout({bindGroupLayouts:[A]}),Z=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:k,compute:{module:this.shaderModule,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ITEMS_PER_WORKGROUP:this.items_per_workgroup,ELEMENT_COUNT:I}}});if(this.pipelines.push({pipeline:Z,bindGroup:q,dispatchSize:M}),S>1){this.create_pass_recursive(G,S);const nn=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:k,compute:{module:this.shaderModule,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ELEMENT_COUNT:I}}});this.pipelines.push({pipeline:nn,bindGroup:q,dispatchSize:M})}}get_dispatch_chain(){return this.pipelines.flatMap(_=>[_.dispatchSize.x,_.dispatchSize.y,1])}dispatch(_,I,S=0){for(let M=0;M<this.pipelines.length;M++){const{pipeline:G,bindGroup:A,dispatchSize:q}=this.pipelines[M];_.setPipeline(G),_.setBindGroup(0,A),I==null?_.dispatchWorkgroups(q.x,q.y,1):_.dispatchWorkgroupsIndirect(I,S+M*3*4)}}}function ot(f,_){return class extends f{constructor(...I){super(...I),_(this)}}}const st=ot(Array,f=>f.fill(0));let U=1e-6;function it(f){function _(n=0,i=0){const e=new f(2);return n!==void 0&&(e[0]=n,i!==void 0&&(e[1]=i)),e}const I=_;function S(n,i,e){const r=e??new f(2);return r[0]=n,r[1]=i,r}function M(n,i){const e=i??new f(2);return e[0]=Math.ceil(n[0]),e[1]=Math.ceil(n[1]),e}function G(n,i){const e=i??new f(2);return e[0]=Math.floor(n[0]),e[1]=Math.floor(n[1]),e}function A(n,i){const e=i??new f(2);return e[0]=Math.round(n[0]),e[1]=Math.round(n[1]),e}function q(n,i=0,e=1,r){const a=r??new f(2);return a[0]=Math.min(e,Math.max(i,n[0])),a[1]=Math.min(e,Math.max(i,n[1])),a}function k(n,i,e){const r=e??new f(2);return r[0]=n[0]+i[0],r[1]=n[1]+i[1],r}function Z(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+i[0]*e,a[1]=n[1]+i[1]*e,a}function nn(n,i){const e=n[0],r=n[1],a=i[0],g=i[1],y=Math.sqrt(e*e+r*r),c=Math.sqrt(a*a+g*g),u=y*c,w=u&&Gn(n,i)/u;return Math.acos(w)}function an(n,i,e){const r=e??new f(2);return r[0]=n[0]-i[0],r[1]=n[1]-i[1],r}const xn=an;function wn(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U}function zn(n,i){return n[0]===i[0]&&n[1]===i[1]}function bn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e*(i[0]-n[0]),a[1]=n[1]+e*(i[1]-n[1]),a}function Tn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e[0]*(i[0]-n[0]),a[1]=n[1]+e[1]*(i[1]-n[1]),a}function On(n,i,e){const r=e??new f(2);return r[0]=Math.max(n[0],i[0]),r[1]=Math.max(n[1],i[1]),r}function F(n,i,e){const r=e??new f(2);return r[0]=Math.min(n[0],i[0]),r[1]=Math.min(n[1],i[1]),r}function un(n,i,e){const r=e??new f(2);return r[0]=n[0]*i,r[1]=n[1]*i,r}const Un=un;function yn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i,r[1]=n[1]/i,r}function vn(n,i){const e=i??new f(2);return e[0]=1/n[0],e[1]=1/n[1],e}const Bn=vn;function ln(n,i,e){const r=e??new f(3),a=n[0]*i[1]-n[1]*i[0];return r[0]=0,r[1]=0,r[2]=a,r}function Gn(n,i){return n[0]*i[0]+n[1]*i[1]}function $(n){const i=n[0],e=n[1];return Math.sqrt(i*i+e*e)}const In=$;function B(n){const i=n[0],e=n[1];return i*i+e*e}const L=B;function O(n,i){const e=n[0]-i[0],r=n[1]-i[1];return Math.sqrt(e*e+r*r)}const en=O;function gn(n,i){const e=n[0]-i[0],r=n[1]-i[1];return e*e+r*r}const An=gn;function _n(n,i){const e=i??new f(2),r=n[0],a=n[1],g=Math.sqrt(r*r+a*a);return g>1e-5?(e[0]=r/g,e[1]=a/g):(e[0]=0,e[1]=0),e}function mn(n,i){const e=i??new f(2);return e[0]=-n[0],e[1]=-n[1],e}function N(n,i){const e=i??new f(2);return e[0]=n[0],e[1]=n[1],e}const Dn=N;function fn(n,i,e){const r=e??new f(2);return r[0]=n[0]*i[0],r[1]=n[1]*i[1],r}const dn=fn;function pn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i[0],r[1]=n[1]/i[1],r}const Sn=pn;function Mn(n=1,i){const e=i??new f(2),r=Math.random()*2*Math.PI;return e[0]=Math.cos(r)*n,e[1]=Math.sin(r)*n,e}function o(n){const i=n??new f(2);return i[0]=0,i[1]=0,i}function d(n,i,e){const r=e??new f(2),a=n[0],g=n[1];return r[0]=a*i[0]+g*i[4]+i[12],r[1]=a*i[1]+g*i[5]+i[13],r}function t(n,i,e){const r=e??new f(2),a=n[0],g=n[1];return r[0]=i[0]*a+i[4]*g+i[8],r[1]=i[1]*a+i[5]*g+i[9],r}function s(n,i,e,r){const a=r??new f(2),g=n[0]-i[0],y=n[1]-i[1],c=Math.sin(e),u=Math.cos(e);return a[0]=g*u-y*c+i[0],a[1]=g*c+y*u+i[1],a}function l(n,i,e){const r=e??new f(2);return _n(n,r),un(r,i,r)}function p(n,i,e){const r=e??new f(2);return $(n)>i?l(n,i,r):N(n,r)}function h(n,i,e){const r=e??new f(2);return bn(n,i,.5,r)}return{create:_,fromValues:I,set:S,ceil:M,floor:G,round:A,clamp:q,add:k,addScaled:Z,angle:nn,subtract:an,sub:xn,equalsApproximately:wn,equals:zn,lerp:bn,lerpV:Tn,max:On,min:F,mulScalar:un,scale:Un,divScalar:yn,inverse:vn,invert:Bn,cross:ln,dot:Gn,length:$,len:In,lengthSq:B,lenSq:L,distance:O,dist:en,distanceSq:gn,distSq:An,normalize:_n,negate:mn,copy:N,clone:Dn,multiply:fn,mul:dn,divide:pn,div:Sn,random:Mn,zero:o,transformMat4:d,transformMat3:t,rotate:s,setLength:l,truncate:p,midpoint:h}}const Pe=new Map;function Re(f){let _=Pe.get(f);return _||(_=it(f),Pe.set(f,_)),_}function ct(f){function _(c,u,w){const v=new f(3);return c!==void 0&&(v[0]=c,u!==void 0&&(v[1]=u,w!==void 0&&(v[2]=w))),v}const I=_;function S(c,u,w,v){const x=v??new f(3);return x[0]=c,x[1]=u,x[2]=w,x}function M(c,u){const w=u??new f(3);return w[0]=Math.ceil(c[0]),w[1]=Math.ceil(c[1]),w[2]=Math.ceil(c[2]),w}function G(c,u){const w=u??new f(3);return w[0]=Math.floor(c[0]),w[1]=Math.floor(c[1]),w[2]=Math.floor(c[2]),w}function A(c,u){const w=u??new f(3);return w[0]=Math.round(c[0]),w[1]=Math.round(c[1]),w[2]=Math.round(c[2]),w}function q(c,u=0,w=1,v){const x=v??new f(3);return x[0]=Math.min(w,Math.max(u,c[0])),x[1]=Math.min(w,Math.max(u,c[1])),x[2]=Math.min(w,Math.max(u,c[2])),x}function k(c,u,w){const v=w??new f(3);return v[0]=c[0]+u[0],v[1]=c[1]+u[1],v[2]=c[2]+u[2],v}function Z(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+u[0]*w,x[1]=c[1]+u[1]*w,x[2]=c[2]+u[2]*w,x}function nn(c,u){const w=c[0],v=c[1],x=c[2],m=u[0],D=u[1],P=u[2],E=Math.sqrt(w*w+v*v+x*x),z=Math.sqrt(m*m+D*D+P*P),b=E*z,R=b&&Gn(c,u)/b;return Math.acos(R)}function an(c,u,w){const v=w??new f(3);return v[0]=c[0]-u[0],v[1]=c[1]-u[1],v[2]=c[2]-u[2],v}const xn=an;function wn(c,u){return Math.abs(c[0]-u[0])<U&&Math.abs(c[1]-u[1])<U&&Math.abs(c[2]-u[2])<U}function zn(c,u){return c[0]===u[0]&&c[1]===u[1]&&c[2]===u[2]}function bn(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+w*(u[0]-c[0]),x[1]=c[1]+w*(u[1]-c[1]),x[2]=c[2]+w*(u[2]-c[2]),x}function Tn(c,u,w,v){const x=v??new f(3);return x[0]=c[0]+w[0]*(u[0]-c[0]),x[1]=c[1]+w[1]*(u[1]-c[1]),x[2]=c[2]+w[2]*(u[2]-c[2]),x}function On(c,u,w){const v=w??new f(3);return v[0]=Math.max(c[0],u[0]),v[1]=Math.max(c[1],u[1]),v[2]=Math.max(c[2],u[2]),v}function F(c,u,w){const v=w??new f(3);return v[0]=Math.min(c[0],u[0]),v[1]=Math.min(c[1],u[1]),v[2]=Math.min(c[2],u[2]),v}function un(c,u,w){const v=w??new f(3);return v[0]=c[0]*u,v[1]=c[1]*u,v[2]=c[2]*u,v}const Un=un;function yn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u,v[1]=c[1]/u,v[2]=c[2]/u,v}function vn(c,u){const w=u??new f(3);return w[0]=1/c[0],w[1]=1/c[1],w[2]=1/c[2],w}const Bn=vn;function ln(c,u,w){const v=w??new f(3),x=c[2]*u[0]-c[0]*u[2],m=c[0]*u[1]-c[1]*u[0];return v[0]=c[1]*u[2]-c[2]*u[1],v[1]=x,v[2]=m,v}function Gn(c,u){return c[0]*u[0]+c[1]*u[1]+c[2]*u[2]}function $(c){const u=c[0],w=c[1],v=c[2];return Math.sqrt(u*u+w*w+v*v)}const In=$;function B(c){const u=c[0],w=c[1],v=c[2];return u*u+w*w+v*v}const L=B;function O(c,u){const w=c[0]-u[0],v=c[1]-u[1],x=c[2]-u[2];return Math.sqrt(w*w+v*v+x*x)}const en=O;function gn(c,u){const w=c[0]-u[0],v=c[1]-u[1],x=c[2]-u[2];return w*w+v*v+x*x}const An=gn;function _n(c,u){const w=u??new f(3),v=c[0],x=c[1],m=c[2],D=Math.sqrt(v*v+x*x+m*m);return D>1e-5?(w[0]=v/D,w[1]=x/D,w[2]=m/D):(w[0]=0,w[1]=0,w[2]=0),w}function mn(c,u){const w=u??new f(3);return w[0]=-c[0],w[1]=-c[1],w[2]=-c[2],w}function N(c,u){const w=u??new f(3);return w[0]=c[0],w[1]=c[1],w[2]=c[2],w}const Dn=N;function fn(c,u,w){const v=w??new f(3);return v[0]=c[0]*u[0],v[1]=c[1]*u[1],v[2]=c[2]*u[2],v}const dn=fn;function pn(c,u,w){const v=w??new f(3);return v[0]=c[0]/u[0],v[1]=c[1]/u[1],v[2]=c[2]/u[2],v}const Sn=pn;function Mn(c=1,u){const w=u??new f(3),v=Math.random()*2*Math.PI,x=Math.random()*2-1,m=Math.sqrt(1-x*x)*c;return w[0]=Math.cos(v)*m,w[1]=Math.sin(v)*m,w[2]=x*c,w}function o(c){const u=c??new f(3);return u[0]=0,u[1]=0,u[2]=0,u}function d(c,u,w){const v=w??new f(3),x=c[0],m=c[1],D=c[2],P=u[3]*x+u[7]*m+u[11]*D+u[15]||1;return v[0]=(u[0]*x+u[4]*m+u[8]*D+u[12])/P,v[1]=(u[1]*x+u[5]*m+u[9]*D+u[13])/P,v[2]=(u[2]*x+u[6]*m+u[10]*D+u[14])/P,v}function t(c,u,w){const v=w??new f(3),x=c[0],m=c[1],D=c[2];return v[0]=x*u[0*4+0]+m*u[1*4+0]+D*u[2*4+0],v[1]=x*u[0*4+1]+m*u[1*4+1]+D*u[2*4+1],v[2]=x*u[0*4+2]+m*u[1*4+2]+D*u[2*4+2],v}function s(c,u,w){const v=w??new f(3),x=c[0],m=c[1],D=c[2];return v[0]=x*u[0]+m*u[4]+D*u[8],v[1]=x*u[1]+m*u[5]+D*u[9],v[2]=x*u[2]+m*u[6]+D*u[10],v}function l(c,u,w){const v=w??new f(3),x=u[0],m=u[1],D=u[2],P=u[3]*2,E=c[0],z=c[1],b=c[2],R=m*b-D*z,T=D*E-x*b,V=x*z-m*E;return v[0]=E+R*P+(m*V-D*T)*2,v[1]=z+T*P+(D*R-x*V)*2,v[2]=b+V*P+(x*T-m*R)*2,v}function p(c,u){const w=u??new f(3);return w[0]=c[12],w[1]=c[13],w[2]=c[14],w}function h(c,u,w){const v=w??new f(3),x=u*4;return v[0]=c[x+0],v[1]=c[x+1],v[2]=c[x+2],v}function n(c,u){const w=u??new f(3),v=c[0],x=c[1],m=c[2],D=c[4],P=c[5],E=c[6],z=c[8],b=c[9],R=c[10];return w[0]=Math.sqrt(v*v+x*x+m*m),w[1]=Math.sqrt(D*D+P*P+E*E),w[2]=Math.sqrt(z*z+b*b+R*R),w}function i(c,u,w,v){const x=v??new f(3),m=[],D=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],D[0]=m[0],D[1]=m[1]*Math.cos(w)-m[2]*Math.sin(w),D[2]=m[1]*Math.sin(w)+m[2]*Math.cos(w),x[0]=D[0]+u[0],x[1]=D[1]+u[1],x[2]=D[2]+u[2],x}function e(c,u,w,v){const x=v??new f(3),m=[],D=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],D[0]=m[2]*Math.sin(w)+m[0]*Math.cos(w),D[1]=m[1],D[2]=m[2]*Math.cos(w)-m[0]*Math.sin(w),x[0]=D[0]+u[0],x[1]=D[1]+u[1],x[2]=D[2]+u[2],x}function r(c,u,w,v){const x=v??new f(3),m=[],D=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],D[0]=m[0]*Math.cos(w)-m[1]*Math.sin(w),D[1]=m[0]*Math.sin(w)+m[1]*Math.cos(w),D[2]=m[2],x[0]=D[0]+u[0],x[1]=D[1]+u[1],x[2]=D[2]+u[2],x}function a(c,u,w){const v=w??new f(3);return _n(c,v),un(v,u,v)}function g(c,u,w){const v=w??new f(3);return $(c)>u?a(c,u,v):N(c,v)}function y(c,u,w){const v=w??new f(3);return bn(c,u,.5,v)}return{create:_,fromValues:I,set:S,ceil:M,floor:G,round:A,clamp:q,add:k,addScaled:Z,angle:nn,subtract:an,sub:xn,equalsApproximately:wn,equals:zn,lerp:bn,lerpV:Tn,max:On,min:F,mulScalar:un,scale:Un,divScalar:yn,inverse:vn,invert:Bn,cross:ln,dot:Gn,length:$,len:In,lengthSq:B,lenSq:L,distance:O,dist:en,distanceSq:gn,distSq:An,normalize:_n,negate:mn,copy:N,clone:Dn,multiply:fn,mul:dn,divide:pn,div:Sn,random:Mn,zero:o,transformMat4:d,transformMat4Upper3x3:t,transformMat3:s,transformQuat:l,getTranslation:p,getAxis:h,getScaling:n,rotateX:i,rotateY:e,rotateZ:r,setLength:a,truncate:g,midpoint:y}}const ze=new Map;function pe(f){let _=ze.get(f);return _||(_=ct(f),ze.set(f,_)),_}function at(f){const _=Re(f),I=pe(f);function S(o,d,t,s,l,p,h,n,i){const e=new f(12);return e[3]=0,e[7]=0,e[11]=0,o!==void 0&&(e[0]=o,d!==void 0&&(e[1]=d,t!==void 0&&(e[2]=t,s!==void 0&&(e[4]=s,l!==void 0&&(e[5]=l,p!==void 0&&(e[6]=p,h!==void 0&&(e[8]=h,n!==void 0&&(e[9]=n,i!==void 0&&(e[10]=i))))))))),e}function M(o,d,t,s,l,p,h,n,i,e){const r=e??new f(12);return r[0]=o,r[1]=d,r[2]=t,r[3]=0,r[4]=s,r[5]=l,r[6]=p,r[7]=0,r[8]=h,r[9]=n,r[10]=i,r[11]=0,r}function G(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[3]=0,t[4]=o[4],t[5]=o[5],t[6]=o[6],t[7]=0,t[8]=o[8],t[9]=o[9],t[10]=o[10],t[11]=0,t}function A(o,d){const t=d??new f(12),s=o[0],l=o[1],p=o[2],h=o[3],n=s+s,i=l+l,e=p+p,r=s*n,a=l*n,g=l*i,y=p*n,c=p*i,u=p*e,w=h*n,v=h*i,x=h*e;return t[0]=1-g-u,t[1]=a+x,t[2]=y-v,t[3]=0,t[4]=a-x,t[5]=1-r-u,t[6]=c+w,t[7]=0,t[8]=y+v,t[9]=c-w,t[10]=1-r-g,t[11]=0,t}function q(o,d){const t=d??new f(12);return t[0]=-o[0],t[1]=-o[1],t[2]=-o[2],t[4]=-o[4],t[5]=-o[5],t[6]=-o[6],t[8]=-o[8],t[9]=-o[9],t[10]=-o[10],t}function k(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[4]=o[4],t[5]=o[5],t[6]=o[6],t[8]=o[8],t[9]=o[9],t[10]=o[10],t}const Z=k;function nn(o,d){return Math.abs(o[0]-d[0])<U&&Math.abs(o[1]-d[1])<U&&Math.abs(o[2]-d[2])<U&&Math.abs(o[4]-d[4])<U&&Math.abs(o[5]-d[5])<U&&Math.abs(o[6]-d[6])<U&&Math.abs(o[8]-d[8])<U&&Math.abs(o[9]-d[9])<U&&Math.abs(o[10]-d[10])<U}function an(o,d){return o[0]===d[0]&&o[1]===d[1]&&o[2]===d[2]&&o[4]===d[4]&&o[5]===d[5]&&o[6]===d[6]&&o[8]===d[8]&&o[9]===d[9]&&o[10]===d[10]}function xn(o){const d=o??new f(12);return d[0]=1,d[1]=0,d[2]=0,d[4]=0,d[5]=1,d[6]=0,d[8]=0,d[9]=0,d[10]=1,d}function wn(o,d){const t=d??new f(12);if(t===o){let g;return g=o[1],o[1]=o[4],o[4]=g,g=o[2],o[2]=o[8],o[8]=g,g=o[6],o[6]=o[9],o[9]=g,t}const s=o[0*4+0],l=o[0*4+1],p=o[0*4+2],h=o[1*4+0],n=o[1*4+1],i=o[1*4+2],e=o[2*4+0],r=o[2*4+1],a=o[2*4+2];return t[0]=s,t[1]=h,t[2]=e,t[4]=l,t[5]=n,t[6]=r,t[8]=p,t[9]=i,t[10]=a,t}function zn(o,d){const t=d??new f(12),s=o[0*4+0],l=o[0*4+1],p=o[0*4+2],h=o[1*4+0],n=o[1*4+1],i=o[1*4+2],e=o[2*4+0],r=o[2*4+1],a=o[2*4+2],g=a*n-i*r,y=-a*h+i*e,c=r*h-n*e,u=1/(s*g+l*y+p*c);return t[0]=g*u,t[1]=(-a*l+p*r)*u,t[2]=(i*l-p*n)*u,t[4]=y*u,t[5]=(a*s-p*e)*u,t[6]=(-i*s+p*h)*u,t[8]=c*u,t[9]=(-r*s+l*e)*u,t[10]=(n*s-l*h)*u,t}function bn(o){const d=o[0],t=o[0*4+1],s=o[0*4+2],l=o[1*4+0],p=o[1*4+1],h=o[1*4+2],n=o[2*4+0],i=o[2*4+1],e=o[2*4+2];return d*(p*e-i*h)-l*(t*e-i*s)+n*(t*h-p*s)}const Tn=zn;function On(o,d,t){const s=t??new f(12),l=o[0],p=o[1],h=o[2],n=o[4],i=o[5],e=o[6],r=o[8],a=o[9],g=o[10],y=d[0],c=d[1],u=d[2],w=d[4],v=d[5],x=d[6],m=d[8],D=d[9],P=d[10];return s[0]=l*y+n*c+r*u,s[1]=p*y+i*c+a*u,s[2]=h*y+e*c+g*u,s[4]=l*w+n*v+r*x,s[5]=p*w+i*v+a*x,s[6]=h*w+e*v+g*x,s[8]=l*m+n*D+r*P,s[9]=p*m+i*D+a*P,s[10]=h*m+e*D+g*P,s}const F=On;function un(o,d,t){const s=t??xn();return o!==s&&(s[0]=o[0],s[1]=o[1],s[2]=o[2],s[4]=o[4],s[5]=o[5],s[6]=o[6]),s[8]=d[0],s[9]=d[1],s[10]=1,s}function Un(o,d){const t=d??_.create();return t[0]=o[8],t[1]=o[9],t}function yn(o,d,t){const s=t??_.create(),l=d*4;return s[0]=o[l+0],s[1]=o[l+1],s}function vn(o,d,t,s){const l=s===o?o:k(o,s),p=t*4;return l[p+0]=d[0],l[p+1]=d[1],l}function Bn(o,d){const t=d??_.create(),s=o[0],l=o[1],p=o[4],h=o[5];return t[0]=Math.sqrt(s*s+l*l),t[1]=Math.sqrt(p*p+h*h),t}function ln(o,d){const t=d??I.create(),s=o[0],l=o[1],p=o[2],h=o[4],n=o[5],i=o[6],e=o[8],r=o[9],a=o[10];return t[0]=Math.sqrt(s*s+l*l+p*p),t[1]=Math.sqrt(h*h+n*n+i*i),t[2]=Math.sqrt(e*e+r*r+a*a),t}function Gn(o,d){const t=d??new f(12);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=1,t[6]=0,t[8]=o[0],t[9]=o[1],t[10]=1,t}function $(o,d,t){const s=t??new f(12),l=d[0],p=d[1],h=o[0],n=o[1],i=o[2],e=o[1*4+0],r=o[1*4+1],a=o[1*4+2],g=o[2*4+0],y=o[2*4+1],c=o[2*4+2];return o!==s&&(s[0]=h,s[1]=n,s[2]=i,s[4]=e,s[5]=r,s[6]=a),s[8]=h*l+e*p+g,s[9]=n*l+r*p+y,s[10]=i*l+a*p+c,s}function In(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=s,t[1]=l,t[2]=0,t[4]=-l,t[5]=s,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function B(o,d,t){const s=t??new f(12),l=o[0*4+0],p=o[0*4+1],h=o[0*4+2],n=o[1*4+0],i=o[1*4+1],e=o[1*4+2],r=Math.cos(d),a=Math.sin(d);return s[0]=r*l+a*n,s[1]=r*p+a*i,s[2]=r*h+a*e,s[4]=r*n-a*l,s[5]=r*i-a*p,s[6]=r*e-a*h,o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function L(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=s,t[6]=l,t[8]=0,t[9]=-l,t[10]=s,t}function O(o,d,t){const s=t??new f(12),l=o[4],p=o[5],h=o[6],n=o[8],i=o[9],e=o[10],r=Math.cos(d),a=Math.sin(d);return s[4]=r*l+a*n,s[5]=r*p+a*i,s[6]=r*h+a*e,s[8]=r*n-a*l,s[9]=r*i-a*p,s[10]=r*e-a*h,o!==s&&(s[0]=o[0],s[1]=o[1],s[2]=o[2]),s}function en(o,d){const t=d??new f(12),s=Math.cos(o),l=Math.sin(o);return t[0]=s,t[1]=0,t[2]=-l,t[4]=0,t[5]=1,t[6]=0,t[8]=l,t[9]=0,t[10]=s,t}function gn(o,d,t){const s=t??new f(12),l=o[0*4+0],p=o[0*4+1],h=o[0*4+2],n=o[2*4+0],i=o[2*4+1],e=o[2*4+2],r=Math.cos(d),a=Math.sin(d);return s[0]=r*l-a*n,s[1]=r*p-a*i,s[2]=r*h-a*e,s[8]=r*n+a*l,s[9]=r*i+a*p,s[10]=r*e+a*h,o!==s&&(s[4]=o[4],s[5]=o[5],s[6]=o[6]),s}const An=In,_n=B;function mn(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=0,t[2]=0,t[4]=0,t[5]=o[1],t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function N(o,d,t){const s=t??new f(12),l=d[0],p=d[1];return s[0]=l*o[0*4+0],s[1]=l*o[0*4+1],s[2]=l*o[0*4+2],s[4]=p*o[1*4+0],s[5]=p*o[1*4+1],s[6]=p*o[1*4+2],o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function Dn(o,d){const t=d??new f(12);return t[0]=o[0],t[1]=0,t[2]=0,t[4]=0,t[5]=o[1],t[6]=0,t[8]=0,t[9]=0,t[10]=o[2],t}function fn(o,d,t){const s=t??new f(12),l=d[0],p=d[1],h=d[2];return s[0]=l*o[0*4+0],s[1]=l*o[0*4+1],s[2]=l*o[0*4+2],s[4]=p*o[1*4+0],s[5]=p*o[1*4+1],s[6]=p*o[1*4+2],s[8]=h*o[2*4+0],s[9]=h*o[2*4+1],s[10]=h*o[2*4+2],s}function dn(o,d){const t=d??new f(12);return t[0]=o,t[1]=0,t[2]=0,t[4]=0,t[5]=o,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function pn(o,d,t){const s=t??new f(12);return s[0]=d*o[0*4+0],s[1]=d*o[0*4+1],s[2]=d*o[0*4+2],s[4]=d*o[1*4+0],s[5]=d*o[1*4+1],s[6]=d*o[1*4+2],o!==s&&(s[8]=o[8],s[9]=o[9],s[10]=o[10]),s}function Sn(o,d){const t=d??new f(12);return t[0]=o,t[1]=0,t[2]=0,t[4]=0,t[5]=o,t[6]=0,t[8]=0,t[9]=0,t[10]=o,t}function Mn(o,d,t){const s=t??new f(12);return s[0]=d*o[0*4+0],s[1]=d*o[0*4+1],s[2]=d*o[0*4+2],s[4]=d*o[1*4+0],s[5]=d*o[1*4+1],s[6]=d*o[1*4+2],s[8]=d*o[2*4+0],s[9]=d*o[2*4+1],s[10]=d*o[2*4+2],s}return{clone:Z,create:S,set:M,fromMat4:G,fromQuat:A,negate:q,copy:k,equalsApproximately:nn,equals:an,identity:xn,transpose:wn,inverse:zn,invert:Tn,determinant:bn,mul:F,multiply:On,setTranslation:un,getTranslation:Un,getAxis:yn,setAxis:vn,getScaling:Bn,get3DScaling:ln,translation:Gn,translate:$,rotation:In,rotate:B,rotationX:L,rotateX:O,rotationY:en,rotateY:gn,rotationZ:An,rotateZ:_n,scaling:mn,scale:N,uniformScaling:dn,uniformScale:pn,scaling3D:Dn,scale3D:fn,uniformScaling3D:Sn,uniformScale3D:Mn}}const be=new Map;function ut(f){let _=be.get(f);return _||(_=at(f),be.set(f,_)),_}function lt(f){const _=pe(f);function I(n,i,e,r,a,g,y,c,u,w,v,x,m,D,P,E){const z=new f(16);return n!==void 0&&(z[0]=n,i!==void 0&&(z[1]=i,e!==void 0&&(z[2]=e,r!==void 0&&(z[3]=r,a!==void 0&&(z[4]=a,g!==void 0&&(z[5]=g,y!==void 0&&(z[6]=y,c!==void 0&&(z[7]=c,u!==void 0&&(z[8]=u,w!==void 0&&(z[9]=w,v!==void 0&&(z[10]=v,x!==void 0&&(z[11]=x,m!==void 0&&(z[12]=m,D!==void 0&&(z[13]=D,P!==void 0&&(z[14]=P,E!==void 0&&(z[15]=E)))))))))))))))),z}function S(n,i,e,r,a,g,y,c,u,w,v,x,m,D,P,E,z){const b=z??new f(16);return b[0]=n,b[1]=i,b[2]=e,b[3]=r,b[4]=a,b[5]=g,b[6]=y,b[7]=c,b[8]=u,b[9]=w,b[10]=v,b[11]=x,b[12]=m,b[13]=D,b[14]=P,b[15]=E,b}function M(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=0,e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=0,e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function G(n,i){const e=i??new f(16),r=n[0],a=n[1],g=n[2],y=n[3],c=r+r,u=a+a,w=g+g,v=r*c,x=a*c,m=a*u,D=g*c,P=g*u,E=g*w,z=y*c,b=y*u,R=y*w;return e[0]=1-m-E,e[1]=x+R,e[2]=D-b,e[3]=0,e[4]=x-R,e[5]=1-v-E,e[6]=P+z,e[7]=0,e[8]=D+b,e[9]=P-z,e[10]=1-v-m,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function A(n,i){const e=i??new f(16);return e[0]=-n[0],e[1]=-n[1],e[2]=-n[2],e[3]=-n[3],e[4]=-n[4],e[5]=-n[5],e[6]=-n[6],e[7]=-n[7],e[8]=-n[8],e[9]=-n[9],e[10]=-n[10],e[11]=-n[11],e[12]=-n[12],e[13]=-n[13],e[14]=-n[14],e[15]=-n[15],e}function q(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],e}const k=q;function Z(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U&&Math.abs(n[2]-i[2])<U&&Math.abs(n[3]-i[3])<U&&Math.abs(n[4]-i[4])<U&&Math.abs(n[5]-i[5])<U&&Math.abs(n[6]-i[6])<U&&Math.abs(n[7]-i[7])<U&&Math.abs(n[8]-i[8])<U&&Math.abs(n[9]-i[9])<U&&Math.abs(n[10]-i[10])<U&&Math.abs(n[11]-i[11])<U&&Math.abs(n[12]-i[12])<U&&Math.abs(n[13]-i[13])<U&&Math.abs(n[14]-i[14])<U&&Math.abs(n[15]-i[15])<U}function nn(n,i){return n[0]===i[0]&&n[1]===i[1]&&n[2]===i[2]&&n[3]===i[3]&&n[4]===i[4]&&n[5]===i[5]&&n[6]===i[6]&&n[7]===i[7]&&n[8]===i[8]&&n[9]===i[9]&&n[10]===i[10]&&n[11]===i[11]&&n[12]===i[12]&&n[13]===i[13]&&n[14]===i[14]&&n[15]===i[15]}function an(n){const i=n??new f(16);return i[0]=1,i[1]=0,i[2]=0,i[3]=0,i[4]=0,i[5]=1,i[6]=0,i[7]=0,i[8]=0,i[9]=0,i[10]=1,i[11]=0,i[12]=0,i[13]=0,i[14]=0,i[15]=1,i}function xn(n,i){const e=i??new f(16);if(e===n){let T;return T=n[1],n[1]=n[4],n[4]=T,T=n[2],n[2]=n[8],n[8]=T,T=n[3],n[3]=n[12],n[12]=T,T=n[6],n[6]=n[9],n[9]=T,T=n[7],n[7]=n[13],n[13]=T,T=n[11],n[11]=n[14],n[14]=T,e}const r=n[0*4+0],a=n[0*4+1],g=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],x=n[2*4+0],m=n[2*4+1],D=n[2*4+2],P=n[2*4+3],E=n[3*4+0],z=n[3*4+1],b=n[3*4+2],R=n[3*4+3];return e[0]=r,e[1]=c,e[2]=x,e[3]=E,e[4]=a,e[5]=u,e[6]=m,e[7]=z,e[8]=g,e[9]=w,e[10]=D,e[11]=b,e[12]=y,e[13]=v,e[14]=P,e[15]=R,e}function wn(n,i){const e=i??new f(16),r=n[0*4+0],a=n[0*4+1],g=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],v=n[1*4+3],x=n[2*4+0],m=n[2*4+1],D=n[2*4+2],P=n[2*4+3],E=n[3*4+0],z=n[3*4+1],b=n[3*4+2],R=n[3*4+3],T=D*R,V=b*P,Y=w*R,W=b*v,X=w*P,H=D*v,K=g*R,J=b*y,tn=g*P,Q=D*y,rn=g*v,on=w*y,sn=x*z,cn=E*m,En=c*z,Rn=E*u,Pn=c*m,$n=x*u,Qn=r*z,Jn=E*a,Cn=r*m,ne=x*a,ee=r*u,te=c*a,ae=T*u+W*m+X*z-(V*u+Y*m+H*z),ue=V*a+K*m+Q*z-(T*a+J*m+tn*z),Zn=Y*a+J*u+rn*z-(W*a+K*u+on*z),re=H*a+tn*u+on*m-(X*a+Q*u+rn*m),C=1/(r*ae+c*ue+x*Zn+E*re);return e[0]=C*ae,e[1]=C*ue,e[2]=C*Zn,e[3]=C*re,e[4]=C*(V*c+Y*x+H*E-(T*c+W*x+X*E)),e[5]=C*(T*r+J*x+tn*E-(V*r+K*x+Q*E)),e[6]=C*(W*r+K*c+on*E-(Y*r+J*c+rn*E)),e[7]=C*(X*r+Q*c+rn*x-(H*r+tn*c+on*x)),e[8]=C*(sn*v+Rn*P+Pn*R-(cn*v+En*P+$n*R)),e[9]=C*(cn*y+Qn*P+ne*R-(sn*y+Jn*P+Cn*R)),e[10]=C*(En*y+Jn*v+ee*R-(Rn*y+Qn*v+te*R)),e[11]=C*($n*y+Cn*v+te*P-(Pn*y+ne*v+ee*P)),e[12]=C*(En*D+$n*b+cn*w-(Pn*b+sn*w+Rn*D)),e[13]=C*(Cn*b+sn*g+Jn*D-(Qn*D+ne*b+cn*g)),e[14]=C*(Qn*w+te*b+Rn*g-(ee*b+En*g+Jn*w)),e[15]=C*(ee*D+Pn*g+ne*w-(Cn*w+te*D+$n*g)),e}function zn(n){const i=n[0],e=n[0*4+1],r=n[0*4+2],a=n[0*4+3],g=n[1*4+0],y=n[1*4+1],c=n[1*4+2],u=n[1*4+3],w=n[2*4+0],v=n[2*4+1],x=n[2*4+2],m=n[2*4+3],D=n[3*4+0],P=n[3*4+1],E=n[3*4+2],z=n[3*4+3],b=x*z,R=E*m,T=c*z,V=E*u,Y=c*m,W=x*u,X=r*z,H=E*a,K=r*m,J=x*a,tn=r*u,Q=c*a,rn=b*y+V*v+Y*P-(R*y+T*v+W*P),on=R*e+X*v+J*P-(b*e+H*v+K*P),sn=T*e+H*y+tn*P-(V*e+X*y+Q*P),cn=W*e+K*y+Q*v-(Y*e+J*y+tn*v);return i*rn+g*on+w*sn+D*cn}const bn=wn;function Tn(n,i,e){const r=e??new f(16),a=n[0],g=n[1],y=n[2],c=n[3],u=n[4],w=n[5],v=n[6],x=n[7],m=n[8],D=n[9],P=n[10],E=n[11],z=n[12],b=n[13],R=n[14],T=n[15],V=i[0],Y=i[1],W=i[2],X=i[3],H=i[4],K=i[5],J=i[6],tn=i[7],Q=i[8],rn=i[9],on=i[10],sn=i[11],cn=i[12],En=i[13],Rn=i[14],Pn=i[15];return r[0]=a*V+u*Y+m*W+z*X,r[1]=g*V+w*Y+D*W+b*X,r[2]=y*V+v*Y+P*W+R*X,r[3]=c*V+x*Y+E*W+T*X,r[4]=a*H+u*K+m*J+z*tn,r[5]=g*H+w*K+D*J+b*tn,r[6]=y*H+v*K+P*J+R*tn,r[7]=c*H+x*K+E*J+T*tn,r[8]=a*Q+u*rn+m*on+z*sn,r[9]=g*Q+w*rn+D*on+b*sn,r[10]=y*Q+v*rn+P*on+R*sn,r[11]=c*Q+x*rn+E*on+T*sn,r[12]=a*cn+u*En+m*Rn+z*Pn,r[13]=g*cn+w*En+D*Rn+b*Pn,r[14]=y*cn+v*En+P*Rn+R*Pn,r[15]=c*cn+x*En+E*Rn+T*Pn,r}const On=Tn;function F(n,i,e){const r=e??an();return n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11]),r[12]=i[0],r[13]=i[1],r[14]=i[2],r[15]=1,r}function un(n,i){const e=i??_.create();return e[0]=n[12],e[1]=n[13],e[2]=n[14],e}function Un(n,i,e){const r=e??_.create(),a=i*4;return r[0]=n[a+0],r[1]=n[a+1],r[2]=n[a+2],r}function yn(n,i,e,r){const a=r===n?r:q(n,r),g=e*4;return a[g+0]=i[0],a[g+1]=i[1],a[g+2]=i[2],a}function vn(n,i){const e=i??_.create(),r=n[0],a=n[1],g=n[2],y=n[4],c=n[5],u=n[6],w=n[8],v=n[9],x=n[10];return e[0]=Math.sqrt(r*r+a*a+g*g),e[1]=Math.sqrt(y*y+c*c+u*u),e[2]=Math.sqrt(w*w+v*v+x*x),e}function Bn(n,i,e,r,a){const g=a??new f(16),y=Math.tan(Math.PI*.5-.5*n);if(g[0]=y/i,g[1]=0,g[2]=0,g[3]=0,g[4]=0,g[5]=y,g[6]=0,g[7]=0,g[8]=0,g[9]=0,g[11]=-1,g[12]=0,g[13]=0,g[15]=0,Number.isFinite(r)){const c=1/(e-r);g[10]=r*c,g[14]=r*e*c}else g[10]=-1,g[14]=-e;return g}function ln(n,i,e,r=1/0,a){const g=a??new f(16),y=1/Math.tan(n*.5);if(g[0]=y/i,g[1]=0,g[2]=0,g[3]=0,g[4]=0,g[5]=y,g[6]=0,g[7]=0,g[8]=0,g[9]=0,g[11]=-1,g[12]=0,g[13]=0,g[15]=0,r===1/0)g[10]=0,g[14]=e;else{const c=1/(r-e);g[10]=e*c,g[14]=r*e*c}return g}function Gn(n,i,e,r,a,g,y){const c=y??new f(16);return c[0]=2/(i-n),c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2/(r-e),c[6]=0,c[7]=0,c[8]=0,c[9]=0,c[10]=1/(a-g),c[11]=0,c[12]=(i+n)/(n-i),c[13]=(r+e)/(e-r),c[14]=a/(a-g),c[15]=1,c}function $(n,i,e,r,a,g,y){const c=y??new f(16),u=i-n,w=r-e,v=a-g;return c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[10]=g/v,c[11]=-1,c[12]=0,c[13]=0,c[14]=a*g/v,c[15]=0,c}function In(n,i,e,r,a,g=1/0,y){const c=y??new f(16),u=i-n,w=r-e;if(c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[11]=-1,c[12]=0,c[13]=0,c[15]=0,g===1/0)c[10]=0,c[14]=a;else{const v=1/(g-a);c[10]=a*v,c[14]=g*a*v}return c}const B=_.create(),L=_.create(),O=_.create();function en(n,i,e,r){const a=r??new f(16);return _.normalize(_.subtract(i,n,O),O),_.normalize(_.cross(e,O,B),B),_.normalize(_.cross(O,B,L),L),a[0]=B[0],a[1]=B[1],a[2]=B[2],a[3]=0,a[4]=L[0],a[5]=L[1],a[6]=L[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function gn(n,i,e,r){const a=r??new f(16);return _.normalize(_.subtract(n,i,O),O),_.normalize(_.cross(e,O,B),B),_.normalize(_.cross(O,B,L),L),a[0]=B[0],a[1]=B[1],a[2]=B[2],a[3]=0,a[4]=L[0],a[5]=L[1],a[6]=L[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function An(n,i,e,r){const a=r??new f(16);return _.normalize(_.subtract(n,i,O),O),_.normalize(_.cross(e,O,B),B),_.normalize(_.cross(O,B,L),L),a[0]=B[0],a[1]=L[0],a[2]=O[0],a[3]=0,a[4]=B[1],a[5]=L[1],a[6]=O[1],a[7]=0,a[8]=B[2],a[9]=L[2],a[10]=O[2],a[11]=0,a[12]=-(B[0]*n[0]+B[1]*n[1]+B[2]*n[2]),a[13]=-(L[0]*n[0]+L[1]*n[1]+L[2]*n[2]),a[14]=-(O[0]*n[0]+O[1]*n[1]+O[2]*n[2]),a[15]=1,a}function _n(n,i){const e=i??new f(16);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=n[0],e[13]=n[1],e[14]=n[2],e[15]=1,e}function mn(n,i,e){const r=e??new f(16),a=i[0],g=i[1],y=i[2],c=n[0],u=n[1],w=n[2],v=n[3],x=n[1*4+0],m=n[1*4+1],D=n[1*4+2],P=n[1*4+3],E=n[2*4+0],z=n[2*4+1],b=n[2*4+2],R=n[2*4+3],T=n[3*4+0],V=n[3*4+1],Y=n[3*4+2],W=n[3*4+3];return n!==r&&(r[0]=c,r[1]=u,r[2]=w,r[3]=v,r[4]=x,r[5]=m,r[6]=D,r[7]=P,r[8]=E,r[9]=z,r[10]=b,r[11]=R),r[12]=c*a+x*g+E*y+T,r[13]=u*a+m*g+z*y+V,r[14]=w*a+D*g+b*y+Y,r[15]=v*a+P*g+R*y+W,r}function N(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=r,e[6]=a,e[7]=0,e[8]=0,e[9]=-a,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Dn(n,i,e){const r=e??new f(16),a=n[4],g=n[5],y=n[6],c=n[7],u=n[8],w=n[9],v=n[10],x=n[11],m=Math.cos(i),D=Math.sin(i);return r[4]=m*a+D*u,r[5]=m*g+D*w,r[6]=m*y+D*v,r[7]=m*c+D*x,r[8]=m*u-D*a,r[9]=m*w-D*g,r[10]=m*v-D*y,r[11]=m*x-D*c,n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function fn(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=0,e[2]=-a,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=a,e[9]=0,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function dn(n,i,e){const r=e??new f(16),a=n[0*4+0],g=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[2*4+0],w=n[2*4+1],v=n[2*4+2],x=n[2*4+3],m=Math.cos(i),D=Math.sin(i);return r[0]=m*a-D*u,r[1]=m*g-D*w,r[2]=m*y-D*v,r[3]=m*c-D*x,r[8]=m*u+D*a,r[9]=m*w+D*g,r[10]=m*v+D*y,r[11]=m*x+D*c,n!==r&&(r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function pn(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=a,e[2]=0,e[3]=0,e[4]=-a,e[5]=r,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Sn(n,i,e){const r=e??new f(16),a=n[0*4+0],g=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[1*4+0],w=n[1*4+1],v=n[1*4+2],x=n[1*4+3],m=Math.cos(i),D=Math.sin(i);return r[0]=m*a+D*u,r[1]=m*g+D*w,r[2]=m*y+D*v,r[3]=m*c+D*x,r[4]=m*u-D*a,r[5]=m*w-D*g,r[6]=m*v-D*y,r[7]=m*x-D*c,n!==r&&(r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function Mn(n,i,e){const r=e??new f(16);let a=n[0],g=n[1],y=n[2];const c=Math.sqrt(a*a+g*g+y*y);a/=c,g/=c,y/=c;const u=a*a,w=g*g,v=y*y,x=Math.cos(i),m=Math.sin(i),D=1-x;return r[0]=u+(1-u)*x,r[1]=a*g*D+y*m,r[2]=a*y*D-g*m,r[3]=0,r[4]=a*g*D-y*m,r[5]=w+(1-w)*x,r[6]=g*y*D+a*m,r[7]=0,r[8]=a*y*D+g*m,r[9]=g*y*D-a*m,r[10]=v+(1-v)*x,r[11]=0,r[12]=0,r[13]=0,r[14]=0,r[15]=1,r}const o=Mn;function d(n,i,e,r){const a=r??new f(16);let g=i[0],y=i[1],c=i[2];const u=Math.sqrt(g*g+y*y+c*c);g/=u,y/=u,c/=u;const w=g*g,v=y*y,x=c*c,m=Math.cos(e),D=Math.sin(e),P=1-m,E=w+(1-w)*m,z=g*y*P+c*D,b=g*c*P-y*D,R=g*y*P-c*D,T=v+(1-v)*m,V=y*c*P+g*D,Y=g*c*P+y*D,W=y*c*P-g*D,X=x+(1-x)*m,H=n[0],K=n[1],J=n[2],tn=n[3],Q=n[4],rn=n[5],on=n[6],sn=n[7],cn=n[8],En=n[9],Rn=n[10],Pn=n[11];return a[0]=E*H+z*Q+b*cn,a[1]=E*K+z*rn+b*En,a[2]=E*J+z*on+b*Rn,a[3]=E*tn+z*sn+b*Pn,a[4]=R*H+T*Q+V*cn,a[5]=R*K+T*rn+V*En,a[6]=R*J+T*on+V*Rn,a[7]=R*tn+T*sn+V*Pn,a[8]=Y*H+W*Q+X*cn,a[9]=Y*K+W*rn+X*En,a[10]=Y*J+W*on+X*Rn,a[11]=Y*tn+W*sn+X*Pn,n!==a&&(a[12]=n[12],a[13]=n[13],a[14]=n[14],a[15]=n[15]),a}const t=d;function s(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n[1],e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n[2],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function l(n,i,e){const r=e??new f(16),a=i[0],g=i[1],y=i[2];return r[0]=a*n[0*4+0],r[1]=a*n[0*4+1],r[2]=a*n[0*4+2],r[3]=a*n[0*4+3],r[4]=g*n[1*4+0],r[5]=g*n[1*4+1],r[6]=g*n[1*4+2],r[7]=g*n[1*4+3],r[8]=y*n[2*4+0],r[9]=y*n[2*4+1],r[10]=y*n[2*4+2],r[11]=y*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function p(n,i){const e=i??new f(16);return e[0]=n,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function h(n,i,e){const r=e??new f(16);return r[0]=i*n[0*4+0],r[1]=i*n[0*4+1],r[2]=i*n[0*4+2],r[3]=i*n[0*4+3],r[4]=i*n[1*4+0],r[5]=i*n[1*4+1],r[6]=i*n[1*4+2],r[7]=i*n[1*4+3],r[8]=i*n[2*4+0],r[9]=i*n[2*4+1],r[10]=i*n[2*4+2],r[11]=i*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}return{create:I,set:S,fromMat3:M,fromQuat:G,negate:A,copy:q,clone:k,equalsApproximately:Z,equals:nn,identity:an,transpose:xn,inverse:wn,determinant:zn,invert:bn,multiply:Tn,mul:On,setTranslation:F,getTranslation:un,getAxis:Un,setAxis:yn,getScaling:vn,perspective:Bn,perspectiveReverseZ:ln,ortho:Gn,frustum:$,frustumReverseZ:In,aim:en,cameraAim:gn,lookAt:An,translation:_n,translate:mn,rotationX:N,rotateX:Dn,rotationY:fn,rotateY:dn,rotationZ:pn,rotateZ:Sn,axisRotation:Mn,rotation:o,axisRotate:d,rotate:t,scaling:s,scale:l,uniformScaling:p,uniformScale:h}}const Ge=new Map;function ft(f){let _=Ge.get(f);return _||(_=lt(f),Ge.set(f,_)),_}function dt(f){const _=pe(f);function I(o,d,t,s){const l=new f(4);return o!==void 0&&(l[0]=o,d!==void 0&&(l[1]=d,t!==void 0&&(l[2]=t,s!==void 0&&(l[3]=s)))),l}const S=I;function M(o,d,t,s,l){const p=l??new f(4);return p[0]=o,p[1]=d,p[2]=t,p[3]=s,p}function G(o,d,t){const s=t??new f(4),l=d*.5,p=Math.sin(l);return s[0]=p*o[0],s[1]=p*o[1],s[2]=p*o[2],s[3]=Math.cos(l),s}function A(o,d){const t=d??_.create(3),s=Math.acos(o[3])*2,l=Math.sin(s*.5);return l>U?(t[0]=o[0]/l,t[1]=o[1]/l,t[2]=o[2]/l):(t[0]=1,t[1]=0,t[2]=0),{angle:s,axis:t}}function q(o,d){const t=$(o,d);return Math.acos(2*t*t-1)}function k(o,d,t){const s=t??new f(4),l=o[0],p=o[1],h=o[2],n=o[3],i=d[0],e=d[1],r=d[2],a=d[3];return s[0]=l*a+n*i+p*r-h*e,s[1]=p*a+n*e+h*i-l*r,s[2]=h*a+n*r+l*e-p*i,s[3]=n*a-l*i-p*e-h*r,s}const Z=k;function nn(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r+i*e,s[1]=h*r+n*e,s[2]=n*r-h*e,s[3]=i*r-p*e,s}function an(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r-n*e,s[1]=h*r+i*e,s[2]=n*r+p*e,s[3]=i*r-h*e,s}function xn(o,d,t){const s=t??new f(4),l=d*.5,p=o[0],h=o[1],n=o[2],i=o[3],e=Math.sin(l),r=Math.cos(l);return s[0]=p*r+h*e,s[1]=h*r-p*e,s[2]=n*r+i*e,s[3]=i*r-n*e,s}function wn(o,d,t,s){const l=s??new f(4),p=o[0],h=o[1],n=o[2],i=o[3];let e=d[0],r=d[1],a=d[2],g=d[3],y=p*e+h*r+n*a+i*g;y<0&&(y=-y,e=-e,r=-r,a=-a,g=-g);let c,u;if(1-y>U){const w=Math.acos(y),v=Math.sin(w);c=Math.sin((1-t)*w)/v,u=Math.sin(t*w)/v}else c=1-t,u=t;return l[0]=c*p+u*e,l[1]=c*h+u*r,l[2]=c*n+u*a,l[3]=c*i+u*g,l}function zn(o,d){const t=d??new f(4),s=o[0],l=o[1],p=o[2],h=o[3],n=s*s+l*l+p*p+h*h,i=n?1/n:0;return t[0]=-s*i,t[1]=-l*i,t[2]=-p*i,t[3]=h*i,t}function bn(o,d){const t=d??new f(4);return t[0]=-o[0],t[1]=-o[1],t[2]=-o[2],t[3]=o[3],t}function Tn(o,d){const t=d??new f(4),s=o[0]+o[5]+o[10];if(s>0){const l=Math.sqrt(s+1);t[3]=.5*l;const p=.5/l;t[0]=(o[6]-o[9])*p,t[1]=(o[8]-o[2])*p,t[2]=(o[1]-o[4])*p}else{let l=0;o[5]>o[0]&&(l=1),o[10]>o[l*4+l]&&(l=2);const p=(l+1)%3,h=(l+2)%3,n=Math.sqrt(o[l*4+l]-o[p*4+p]-o[h*4+h]+1);t[l]=.5*n;const i=.5/n;t[3]=(o[p*4+h]-o[h*4+p])*i,t[p]=(o[p*4+l]+o[l*4+p])*i,t[h]=(o[h*4+l]+o[l*4+h])*i}return t}function On(o,d,t,s,l){const p=l??new f(4),h=o*.5,n=d*.5,i=t*.5,e=Math.sin(h),r=Math.cos(h),a=Math.sin(n),g=Math.cos(n),y=Math.sin(i),c=Math.cos(i);switch(s){case"xyz":p[0]=e*g*c+r*a*y,p[1]=r*a*c-e*g*y,p[2]=r*g*y+e*a*c,p[3]=r*g*c-e*a*y;break;case"xzy":p[0]=e*g*c-r*a*y,p[1]=r*a*c-e*g*y,p[2]=r*g*y+e*a*c,p[3]=r*g*c+e*a*y;break;case"yxz":p[0]=e*g*c+r*a*y,p[1]=r*a*c-e*g*y,p[2]=r*g*y-e*a*c,p[3]=r*g*c+e*a*y;break;case"yzx":p[0]=e*g*c+r*a*y,p[1]=r*a*c+e*g*y,p[2]=r*g*y-e*a*c,p[3]=r*g*c-e*a*y;break;case"zxy":p[0]=e*g*c-r*a*y,p[1]=r*a*c+e*g*y,p[2]=r*g*y+e*a*c,p[3]=r*g*c-e*a*y;break;case"zyx":p[0]=e*g*c-r*a*y,p[1]=r*a*c+e*g*y,p[2]=r*g*y-e*a*c,p[3]=r*g*c+e*a*y;break;default:throw new Error(`Unknown rotation order: ${s}`)}return p}function F(o,d){const t=d??new f(4);return t[0]=o[0],t[1]=o[1],t[2]=o[2],t[3]=o[3],t}const un=F;function Un(o,d,t){const s=t??new f(4);return s[0]=o[0]+d[0],s[1]=o[1]+d[1],s[2]=o[2]+d[2],s[3]=o[3]+d[3],s}function yn(o,d,t){const s=t??new f(4);return s[0]=o[0]-d[0],s[1]=o[1]-d[1],s[2]=o[2]-d[2],s[3]=o[3]-d[3],s}const vn=yn;function Bn(o,d,t){const s=t??new f(4);return s[0]=o[0]*d,s[1]=o[1]*d,s[2]=o[2]*d,s[3]=o[3]*d,s}const ln=Bn;function Gn(o,d,t){const s=t??new f(4);return s[0]=o[0]/d,s[1]=o[1]/d,s[2]=o[2]/d,s[3]=o[3]/d,s}function $(o,d){return o[0]*d[0]+o[1]*d[1]+o[2]*d[2]+o[3]*d[3]}function In(o,d,t,s){const l=s??new f(4);return l[0]=o[0]+t*(d[0]-o[0]),l[1]=o[1]+t*(d[1]-o[1]),l[2]=o[2]+t*(d[2]-o[2]),l[3]=o[3]+t*(d[3]-o[3]),l}function B(o){const d=o[0],t=o[1],s=o[2],l=o[3];return Math.sqrt(d*d+t*t+s*s+l*l)}const L=B;function O(o){const d=o[0],t=o[1],s=o[2],l=o[3];return d*d+t*t+s*s+l*l}const en=O;function gn(o,d){const t=d??new f(4),s=o[0],l=o[1],p=o[2],h=o[3],n=Math.sqrt(s*s+l*l+p*p+h*h);return n>1e-5?(t[0]=s/n,t[1]=l/n,t[2]=p/n,t[3]=h/n):(t[0]=0,t[1]=0,t[2]=0,t[3]=1),t}function An(o,d){return Math.abs(o[0]-d[0])<U&&Math.abs(o[1]-d[1])<U&&Math.abs(o[2]-d[2])<U&&Math.abs(o[3]-d[3])<U}function _n(o,d){return o[0]===d[0]&&o[1]===d[1]&&o[2]===d[2]&&o[3]===d[3]}function mn(o){const d=o??new f(4);return d[0]=0,d[1]=0,d[2]=0,d[3]=1,d}const N=_.create(),Dn=_.create(),fn=_.create();function dn(o,d,t){const s=t??new f(4),l=_.dot(o,d);return l<-.999999?(_.cross(Dn,o,N),_.len(N)<1e-6&&_.cross(fn,o,N),_.normalize(N,N),G(N,Math.PI,s),s):l>.999999?(s[0]=0,s[1]=0,s[2]=0,s[3]=1,s):(_.cross(o,d,N),s[0]=N[0],s[1]=N[1],s[2]=N[2],s[3]=1+l,gn(s,s))}const pn=new f(4),Sn=new f(4);function Mn(o,d,t,s,l,p){const h=p??new f(4);return wn(o,s,l,pn),wn(d,t,l,Sn),wn(pn,Sn,2*l*(1-l),h),h}return{create:I,fromValues:S,set:M,fromAxisAngle:G,toAxisAngle:A,angle:q,multiply:k,mul:Z,rotateX:nn,rotateY:an,rotateZ:xn,slerp:wn,inverse:zn,conjugate:bn,fromMat:Tn,fromEuler:On,copy:F,clone:un,add:Un,subtract:yn,sub:vn,mulScalar:Bn,scale:ln,divScalar:Gn,dot:$,lerp:In,length:B,len:L,lengthSq:O,lenSq:en,normalize:gn,equalsApproximately:An,equals:_n,identity:mn,rotationTo:dn,sqlerp:Mn}}const Ie=new Map;function pt(f){let _=Ie.get(f);return _||(_=dt(f),Ie.set(f,_)),_}function wt(f){function _(t,s,l,p){const h=new f(4);return t!==void 0&&(h[0]=t,s!==void 0&&(h[1]=s,l!==void 0&&(h[2]=l,p!==void 0&&(h[3]=p)))),h}const I=_;function S(t,s,l,p,h){const n=h??new f(4);return n[0]=t,n[1]=s,n[2]=l,n[3]=p,n}function M(t,s){const l=s??new f(4);return l[0]=Math.ceil(t[0]),l[1]=Math.ceil(t[1]),l[2]=Math.ceil(t[2]),l[3]=Math.ceil(t[3]),l}function G(t,s){const l=s??new f(4);return l[0]=Math.floor(t[0]),l[1]=Math.floor(t[1]),l[2]=Math.floor(t[2]),l[3]=Math.floor(t[3]),l}function A(t,s){const l=s??new f(4);return l[0]=Math.round(t[0]),l[1]=Math.round(t[1]),l[2]=Math.round(t[2]),l[3]=Math.round(t[3]),l}function q(t,s=0,l=1,p){const h=p??new f(4);return h[0]=Math.min(l,Math.max(s,t[0])),h[1]=Math.min(l,Math.max(s,t[1])),h[2]=Math.min(l,Math.max(s,t[2])),h[3]=Math.min(l,Math.max(s,t[3])),h}function k(t,s,l){const p=l??new f(4);return p[0]=t[0]+s[0],p[1]=t[1]+s[1],p[2]=t[2]+s[2],p[3]=t[3]+s[3],p}function Z(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+s[0]*l,h[1]=t[1]+s[1]*l,h[2]=t[2]+s[2]*l,h[3]=t[3]+s[3]*l,h}function nn(t,s,l){const p=l??new f(4);return p[0]=t[0]-s[0],p[1]=t[1]-s[1],p[2]=t[2]-s[2],p[3]=t[3]-s[3],p}const an=nn;function xn(t,s){return Math.abs(t[0]-s[0])<U&&Math.abs(t[1]-s[1])<U&&Math.abs(t[2]-s[2])<U&&Math.abs(t[3]-s[3])<U}function wn(t,s){return t[0]===s[0]&&t[1]===s[1]&&t[2]===s[2]&&t[3]===s[3]}function zn(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+l*(s[0]-t[0]),h[1]=t[1]+l*(s[1]-t[1]),h[2]=t[2]+l*(s[2]-t[2]),h[3]=t[3]+l*(s[3]-t[3]),h}function bn(t,s,l,p){const h=p??new f(4);return h[0]=t[0]+l[0]*(s[0]-t[0]),h[1]=t[1]+l[1]*(s[1]-t[1]),h[2]=t[2]+l[2]*(s[2]-t[2]),h[3]=t[3]+l[3]*(s[3]-t[3]),h}function Tn(t,s,l){const p=l??new f(4);return p[0]=Math.max(t[0],s[0]),p[1]=Math.max(t[1],s[1]),p[2]=Math.max(t[2],s[2]),p[3]=Math.max(t[3],s[3]),p}function On(t,s,l){const p=l??new f(4);return p[0]=Math.min(t[0],s[0]),p[1]=Math.min(t[1],s[1]),p[2]=Math.min(t[2],s[2]),p[3]=Math.min(t[3],s[3]),p}function F(t,s,l){const p=l??new f(4);return p[0]=t[0]*s,p[1]=t[1]*s,p[2]=t[2]*s,p[3]=t[3]*s,p}const un=F;function Un(t,s,l){const p=l??new f(4);return p[0]=t[0]/s,p[1]=t[1]/s,p[2]=t[2]/s,p[3]=t[3]/s,p}function yn(t,s){const l=s??new f(4);return l[0]=1/t[0],l[1]=1/t[1],l[2]=1/t[2],l[3]=1/t[3],l}const vn=yn;function Bn(t,s){return t[0]*s[0]+t[1]*s[1]+t[2]*s[2]+t[3]*s[3]}function ln(t){const s=t[0],l=t[1],p=t[2],h=t[3];return Math.sqrt(s*s+l*l+p*p+h*h)}const Gn=ln;function $(t){const s=t[0],l=t[1],p=t[2],h=t[3];return s*s+l*l+p*p+h*h}const In=$;function B(t,s){const l=t[0]-s[0],p=t[1]-s[1],h=t[2]-s[2],n=t[3]-s[3];return Math.sqrt(l*l+p*p+h*h+n*n)}const L=B;function O(t,s){const l=t[0]-s[0],p=t[1]-s[1],h=t[2]-s[2],n=t[3]-s[3];return l*l+p*p+h*h+n*n}const en=O;function gn(t,s){const l=s??new f(4),p=t[0],h=t[1],n=t[2],i=t[3],e=Math.sqrt(p*p+h*h+n*n+i*i);return e>1e-5?(l[0]=p/e,l[1]=h/e,l[2]=n/e,l[3]=i/e):(l[0]=0,l[1]=0,l[2]=0,l[3]=0),l}function An(t,s){const l=s??new f(4);return l[0]=-t[0],l[1]=-t[1],l[2]=-t[2],l[3]=-t[3],l}function _n(t,s){const l=s??new f(4);return l[0]=t[0],l[1]=t[1],l[2]=t[2],l[3]=t[3],l}const mn=_n;function N(t,s,l){const p=l??new f(4);return p[0]=t[0]*s[0],p[1]=t[1]*s[1],p[2]=t[2]*s[2],p[3]=t[3]*s[3],p}const Dn=N;function fn(t,s,l){const p=l??new f(4);return p[0]=t[0]/s[0],p[1]=t[1]/s[1],p[2]=t[2]/s[2],p[3]=t[3]/s[3],p}const dn=fn;function pn(t){const s=t??new f(4);return s[0]=0,s[1]=0,s[2]=0,s[3]=0,s}function Sn(t,s,l){const p=l??new f(4),h=t[0],n=t[1],i=t[2],e=t[3];return p[0]=s[0]*h+s[4]*n+s[8]*i+s[12]*e,p[1]=s[1]*h+s[5]*n+s[9]*i+s[13]*e,p[2]=s[2]*h+s[6]*n+s[10]*i+s[14]*e,p[3]=s[3]*h+s[7]*n+s[11]*i+s[15]*e,p}function Mn(t,s,l){const p=l??new f(4);return gn(t,p),F(p,s,p)}function o(t,s,l){const p=l??new f(4);return ln(t)>s?Mn(t,s,p):_n(t,p)}function d(t,s,l){const p=l??new f(4);return zn(t,s,.5,p)}return{create:_,fromValues:I,set:S,ceil:M,floor:G,round:A,clamp:q,add:k,addScaled:Z,subtract:nn,sub:an,equalsApproximately:xn,equals:wn,lerp:zn,lerpV:bn,max:Tn,min:On,mulScalar:F,scale:un,divScalar:Un,inverse:yn,invert:vn,dot:Bn,length:ln,len:Gn,lengthSq:$,lenSq:In,distance:B,dist:L,distanceSq:O,distSq:en,normalize:gn,negate:An,copy:_n,clone:mn,multiply:N,mul:Dn,divide:fn,div:dn,zero:pn,transformMat4:Sn,setLength:Mn,truncate:o,midpoint:d}}const Se=new Map;function vt(f){let _=Se.get(f);return _||(_=wt(f),Se.set(f,_)),_}function ve(f,_,I,S,M,G){return{mat3:ut(f),mat4:ft(_),quat:pt(I),vec2:Re(S),vec3:pe(M),vec4:vt(G)}}const{mat3:zt,mat4:hn,quat:bt,vec2:Gt,vec3:It,vec4:St}=ve(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);ve(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array);ve(st,Array,Array,Array,Array,Array);const Wn=.07,Xn=.7,jn=2,Yn=.7;function gt(f){let _=new ArrayBuffer(64*f);var I=0;const S=.5;for(var M=-jn;I<f;M+=S*Wn)for(var G=-.95*Xn;G<.95*Xn&&I<f;G+=S*Wn)for(var A=-.95*Yn;A<0&&I<f;A+=S*Wn){let q=1e-4*Math.random();const k=64*I;({position:new Float32Array(_,k,3),velocity:new Float32Array(_,k+16,3),force:new Float32Array(_,k+32,3),density:new Float32Array(_,k+44,1),nearDensity:new Float32Array(_,k+48,1)}).position.set([G+q,M,A]),I++}return _}async function _t(){const f=document.querySelector("canvas"),_=await navigator.gpu.requestAdapter();if(!_)throw new Error;const I=await _.requestDevice(),S=f.getContext("webgpu");if(!S)throw new Error;const{devicePixelRatio:M}=window;f.width=M*f.clientWidth,f.height=M*f.clientHeight;const G=navigator.gpu.getPreferredCanvasFormat();return S.configure({device:I,format:G}),{canvas:f,device:I,presentationFormat:G,context:S}}function ht(f,_,I,S){var M=[f[0]*2-1,1-f[1]*2,0,1];M[2]=-I[4*2+2]+I[4*3+2]/_;var G=hn.multiply(S,M),A=G[3];return[G[0]/A,G[1]/A,G[2]/A]}function xt(f,_){const I=f.clientWidth/f.clientHeight,S=hn.perspective(_,I,.1,50),M=hn.lookAt([0,0,5.5],[0,0,0],[0,1,0]),G=new Float32Array(4);G[0]=1.5,G[1]=1,G[2]=2,G[3]=1;const A=hn.multiply(M,G),q=Math.abs(A[2]),k=hn.multiply(S,A),Z=[k[0]/k[3],k[1]/k[3],k[2]/k[3]];console.log("camera: ",A),console.log("ndc: ",Z);const nn=[(Z[0]+1)/2,(1-Z[1])/2],an=hn.inverse(S),xn=ht(nn,q,S,an);return console.log(xn),{projection:S,view:M}}function yt(f,_,I,S){var M=hn.identity();hn.translate(M,S,M),hn.rotateY(M,I,M),hn.rotateX(M,_,M),hn.translate(M,[0,0,f],M);var G=hn.multiply(M,[0,0,0,1]);return hn.lookAt([G[0],G[1],G[2]],S,[0,1,0])}const Te=.025,Ee=2*Te;async function mt(){const{canvas:f,device:_,presentationFormat:I,context:S}=await _t();S.configure({device:_,format:I,alphaMode:"premultiplied"});const M=_.createShaderModule({code:Ye}),G=_.createShaderModule({code:He}),A=_.createShaderModule({code:qe}),q=_.createShaderModule({code:Ke}),k=_.createShaderModule({code:je}),Z=_.createShaderModule({code:$e}),nn=_.createShaderModule({code:Xe}),an=_.createShaderModule({code:Ze}),xn=_.createShaderModule({code:Ne}),wn=_.createShaderModule({code:Ve}),zn=_.createShaderModule({code:We}),bn=_.createShaderModule({code:Je}),Tn=_.createShaderModule({code:Qe}),On=_.createShaderModule({code:Ce}),F={kernelRadius:Wn,kernelRadiusPow2:Math.pow(Wn,2),kernelRadiusPow4:Math.pow(Wn,4),kernelRadiusPow5:Math.pow(Wn,5),kernelRadiusPow6:Math.pow(Wn,6),kernelRadiusPow9:Math.pow(Wn,9),stiffness:15,nearStiffness:1.5,mass:1,restDensity:35e3,viscosity:3e5,dt:.008,xHalf:Xn,yHalf:jn,zHalf:Yn},un=_.createRenderPipeline({label:"circles pipeline",layout:"auto",vertex:{module:G},fragment:{module:G,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),Un=_.createRenderPipeline({label:"ball pipeline",layout:"auto",vertex:{module:Z},fragment:{module:Z,targets:[{format:I}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),yn=90*Math.PI/180,{projection:vn,view:Bn}=xt(f,yn),ln={screenHeight:f.height,screenWidth:f.width},Gn={depth_threshold:Te*10,max_filter_size:100,projected_particle_constant:10*Ee*.05*(f.height/2)/Math.tan(yn/2)},$=_.createRenderPipeline({label:"filter pipeline",layout:"auto",vertex:{module:M,constants:ln},fragment:{module:q,constants:Gn,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"}}),In=_.createRenderPipeline({label:"fluid rendering pipeline",layout:"auto",vertex:{module:M,constants:ln},fragment:{module:k,targets:[{format:I}]},primitive:{topology:"triangle-list",cullMode:"none"}}),B=_.createRenderPipeline({label:"thickness pipeline",layout:"auto",vertex:{module:nn},fragment:{module:nn,targets:[{format:"r16float",writeMask:GPUColorWrite.RED,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}}),L=_.createRenderPipeline({label:"thickness filter pipeline",layout:"auto",vertex:{module:M,constants:ln},fragment:{module:an,targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),O=_.createRenderPipeline({label:"show pipeline",layout:"auto",vertex:{module:M,constants:ln},fragment:{module:A,targets:[{format:I}]},primitive:{topology:"triangle-list",cullMode:"none"}}),en=1*Wn,gn=2*Xn,An=2*jn,_n=2*Yn,mn=4*en,N=Math.ceil((gn+mn)/en),Dn=Math.ceil((An+mn)/en),fn=Math.ceil((_n+mn)/en),dn=N*Dn*fn,pn=mn/2,Sn={xHalf:Xn,yHalf:jn,zHalf:Yn,gridCount:dn,xGrids:N,yGrids:Dn,cellSize:en,offset:pn},Mn=_.createComputePipeline({label:"grid clear pipeline",layout:"auto",compute:{module:Tn}}),o=_.createComputePipeline({label:"grid build pipeline",layout:"auto",compute:{module:bn,constants:Sn}}),d=_.createComputePipeline({label:"reorder pipeline",layout:"auto",compute:{module:On,constants:Sn}}),t=_.createComputePipeline({label:"density pipeline",layout:"auto",compute:{module:xn,constants:{kernelRadius:F.kernelRadius,kernelRadiusPow2:F.kernelRadiusPow2,kernelRadiusPow5:F.kernelRadiusPow5,kernelRadiusPow6:F.kernelRadiusPow6,mass:F.mass,xHalf:Xn,yHalf:jn,zHalf:Yn,xGrids:N,yGrids:Dn,zGrids:fn,gridCount:dn,cellSize:en,offset:pn}}}),s=_.createComputePipeline({label:"force pipeline",layout:"auto",compute:{module:wn,constants:{kernelRadius:F.kernelRadius,kernelRadiusPow2:F.kernelRadiusPow2,kernelRadiusPow5:F.kernelRadiusPow5,kernelRadiusPow6:F.kernelRadiusPow6,kernelRadiusPow9:F.kernelRadiusPow9,mass:F.mass,stiffness:F.stiffness,nearStiffness:F.nearStiffness,viscosity:F.viscosity,restDensity:F.restDensity,xHalf:Xn,yHalf:jn,zHalf:Yn,xGrids:N,yGrids:Dn,zGrids:fn,gridCount:dn,cellSize:en,offset:pn}}}),l=_.createComputePipeline({label:"integrate pipeline",layout:"auto",compute:{module:zn,constants:{dt:F.dt}}}),h=_.createTexture({label:"depth map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),i=_.createTexture({label:"temporary texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),r=_.createTexture({label:"thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),g=_.createTexture({label:"temporary thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),y=_.createTexture({size:[f.width,f.height,1],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});y.createView();let c;{const qn=["cubemap/posx.jpg","cubemap/negx.jpg","cubemap/posy.jpg","cubemap/negy.jpg","cubemap/posz.jpg","cubemap/negz.jpg"].map(async Nn=>{const Vn=await fetch(Nn);return createImageBitmap(await Vn.blob())}),Fn=await Promise.all(qn);console.log(Fn[0].width,Fn[0].height),c=_.createTexture({dimension:"2d",size:[Fn[0].width,Fn[0].height,6],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let Nn=0;Nn<Fn.length;Nn++){const Vn=Fn[Nn];_.queue.copyExternalImageToTexture({source:Vn},{texture:c,origin:[0,0,Nn]},[Vn.width,Vn.height])}}const u=c.createView({dimension:"cube"}),w=_.createSampler({magFilter:"linear",minFilter:"linear"}),v=2e4,x=gt(v),m=new ArrayBuffer(144),D={size:new Float32Array(m,0,1),view_matrix:new Float32Array(m,16,16),projection_matrix:new Float32Array(m,80,16)},P=new ArrayBuffer(8),E=new ArrayBuffer(8),z={blur_dir:new Float32Array(P)},b={blur_dir:new Float32Array(E)};z.blur_dir.set([1,0]),b.blur_dir.set([0,1]);const R=new ArrayBuffer(272),T={texel_size:new Float32Array(R,0,2),inv_projection_matrix:new Float32Array(R,16,16),projection_matrix:new Float32Array(R,80,16),view_matrix:new Float32Array(R,144,16),inv_view_matrix:new Float32Array(R,208,16)};T.texel_size.set([1/f.width,1/f.height]),T.projection_matrix.set(vn);const V=hn.identity(),Y=hn.identity();hn.inverse(vn,V),hn.inverse(Bn,Y),T.inv_projection_matrix.set(V),T.inv_view_matrix.set(Y);const W=new ArrayBuffer(12),X={xHalf:new Float32Array(W,0,1),yHalf:new Float32Array(W,4,1),zHalf:new Float32Array(W,8,1)};X.xHalf.set([Xn]),X.yHalf.set([jn]),X.zHalf.set([Yn]);const H=_.createBuffer({label:"particles buffer",size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),K=_.createBuffer({label:"cell particle count buffer",size:4*(dn+1),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),J=_.createBuffer({label:"target particles buffer",size:x.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),tn=_.createBuffer({label:"particle cell offset buffer",size:4*v,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});_.queue.writeBuffer(H,0,x);const Q=_.createBuffer({label:"uniform buffer",size:m.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),rn=_.createBuffer({label:"filter uniform buffer",size:P.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),on=_.createBuffer({label:"filter uniform buffer",size:E.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),sn=_.createBuffer({label:"filter uniform buffer",size:R.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),cn=_.createBuffer({label:"real box size buffer",size:W.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});_.queue.writeBuffer(rn,0,P),_.queue.writeBuffer(on,0,E),_.queue.writeBuffer(sn,0,R),_.queue.writeBuffer(cn,0,W);const En=_.createBindGroup({layout:Mn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}}]}),Rn=_.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}},{binding:1,resource:{buffer:tn}},{binding:2,resource:{buffer:H}}]}),Pn=_.createBindGroup({layout:d.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}},{binding:3,resource:{buffer:tn}}]}),$n=_.createBindGroup({layout:t.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Qn=_.createBindGroup({layout:s.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Jn=_.createBindGroup({layout:l.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:cn}}]}),Cn=_.createBindGroup({label:"ball bind group",layout:Un.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:Q}}]}),ne=_.createBindGroup({label:"circle bind group",layout:un.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:Q}}]});_.createBindGroup({label:"show bind group",layout:O.getBindGroupLayout(0),entries:[{binding:1,resource:r}]});const ee=[_.createBindGroup({label:"filterX bind group",layout:$.getBindGroupLayout(0),entries:[{binding:1,resource:h},{binding:2,resource:{buffer:rn}}]}),_.createBindGroup({label:"filterY bind group",layout:$.getBindGroupLayout(0),entries:[{binding:1,resource:i},{binding:2,resource:{buffer:on}}]})],te=_.createBindGroup({label:"fluid bind group",layout:In.getBindGroupLayout(0),entries:[{binding:0,resource:w},{binding:1,resource:h},{binding:2,resource:{buffer:sn}},{binding:3,resource:r},{binding:4,resource:u}]}),ae=_.createBindGroup({label:"thickness bind group",layout:B.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:Q}}]}),ue=[_.createBindGroup({label:"thickness filterX bind group",layout:L.getBindGroupLayout(0),entries:[{binding:1,resource:r},{binding:2,resource:{buffer:rn}}]}),_.createBindGroup({label:"thickness filterY bind group",layout:L.getBindGroupLayout(0),entries:[{binding:1,resource:g},{binding:2,resource:{buffer:on}}]})];let Zn=!1,re=0,C=0,ge=Math.PI/4,oe=-Math.PI/12;const _e=.005,he=-.99*Math.PI/2,xe=0;let we=1;const ie=[{MIN_DISTANCE:1.7,MAX_DISTANCE:3,INIT_DISTANCE:2.2}][0];let se=ie.INIT_DISTANCE;const ye=document.getElementById("fluidCanvas");ye.addEventListener("mousedown",Ln=>{Zn=!0,re=Ln.clientX,C=Ln.clientY}),ye.addEventListener("wheel",Ln=>{Ln.preventDefault();var qn=Ln.deltaY;se+=(qn>0?1:-1)*.05,se<ie.MIN_DISTANCE&&(se=ie.MIN_DISTANCE),se>ie.MAX_DISTANCE&&(se=ie.MAX_DISTANCE)}),document.addEventListener("mousemove",Ln=>{if(Zn){const qn=Ln.clientX,Fn=Ln.clientY,Nn=re-qn,Vn=C-Fn;ge+=_e*Nn,oe+=_e*Vn,oe>xe&&(oe=xe),oe<he&&(oe=he),re=qn,C=Fn,console.log("Dragging... Delta:",Nn,Vn)}}),document.addEventListener("mouseup",()=>{Zn&&(Zn=!1,console.log("Drag ended"))});async function me(){const Ln=performance.now(),qn={colorAttachments:[{view:h,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},Fn={colorAttachments:[{view:S.getCurrentTexture().createView(),clearValue:{r:.7,g:.7,b:.7,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},Nn=[{colorAttachments:[{view:i,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:h,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];S.getCurrentTexture().createView();const Vn={colorAttachments:[{view:S.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Oe={colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Ue=[{colorAttachments:[{view:g,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];D.size.set([Ee]),D.projection_matrix.set(vn);const De=yt(se,oe,ge,[0,-jn,0]);D.view_matrix.set(De),T.view_matrix.set(De),_.queue.writeBuffer(Q,0,m),_.queue.writeBuffer(sn,0,R);const Be=document.getElementById("slider"),Ae=document.getElementById("slider-value"),ke=document.getElementById("particle");let Me=parseInt(Be.value)/200+.5;const Le=Math.max(Me-we,-.02);we+=Le,Ae.textContent=Me.toFixed(2),X.zHalf.set([Yn*we]),_.queue.writeBuffer(cn,0,W);const Kn=_.createCommandEncoder(),j=Kn.beginComputePass();for(let kn=0;kn<1;kn++)j.setBindGroup(0,En),j.setPipeline(Mn),j.dispatchWorkgroups(Math.ceil((dn+1)/64)),j.setBindGroup(0,Rn),j.setPipeline(o),j.dispatchWorkgroups(Math.ceil(v/64)),new rt({device:_,data:K,count:dn+1}).dispatch(j),j.setBindGroup(0,Pn),j.setPipeline(d),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,$n),j.setPipeline(t),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Pn),j.setPipeline(d),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Qn),j.setPipeline(s),j.dispatchWorkgroups(Math.ceil(v/64)),j.setBindGroup(0,Jn),j.setPipeline(l),j.dispatchWorkgroups(Math.ceil(v/64));if(j.end(),ke.checked){const kn=Kn.beginRenderPass(Fn);kn.setBindGroup(0,Cn),kn.setPipeline(Un),kn.draw(6,v),kn.end()}else{const kn=Kn.beginRenderPass(qn);kn.setBindGroup(0,ne),kn.setPipeline(un),kn.draw(6,v),kn.end();for(var Hn=0;Hn<6;Hn++){const fe=Kn.beginRenderPass(Nn[Hn%2]);fe.setBindGroup(0,ee[Hn%2]),fe.setPipeline($),fe.draw(6),fe.end()}const ce=Kn.beginRenderPass(Oe);ce.setBindGroup(0,ae),ce.setPipeline(B),ce.draw(6,v),ce.end();for(var Hn=0;Hn<4;Hn++){const de=Kn.beginRenderPass(Ue[Hn%2]);de.setBindGroup(0,ue[Hn%2]),de.setPipeline(L),de.draw(6),de.end()}const le=Kn.beginRenderPass(Vn);le.setBindGroup(0,te),le.setPipeline(In),le.draw(6),le.end()}_.queue.submit([Kn.finish()]);const Fe=performance.now();console.log(`js: ${(Fe-Ln).toFixed(1)}ms`),requestAnimationFrame(me)}requestAnimationFrame(me)}mt();
