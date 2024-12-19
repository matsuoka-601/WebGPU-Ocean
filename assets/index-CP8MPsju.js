(function(){const g=document.createElement("link").relList;if(g&&g.supports&&g.supports("modulepreload"))return;for(const D of document.querySelectorAll('link[rel="modulepreload"]'))G(D);new MutationObserver(D=>{for(const I of D)if(I.type==="childList")for(const B of I.addedNodes)B.tagName==="LINK"&&B.rel==="modulepreload"&&G(B)}).observe(document,{childList:!0,subtree:!0});function S(D){const I={};return D.integrity&&(I.integrity=D.integrity),D.referrerPolicy&&(I.referrerPolicy=D.referrerPolicy),D.crossOrigin==="use-credentials"?I.credentials="include":D.crossOrigin==="anonymous"?I.credentials="omit":I.credentials="same-origin",I}function G(D){if(D.ep)return;D.ep=!0;const I=S(D);fetch(D.href,I)}})();var Xe=`struct Particle {
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
    let scale = 315.0 / (64. * 3.1415926535 * kernelRadiusPow2 * kernelRadiusPow2 * kernelRadiusPow5);
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
fn computeDensity(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x < arrayLength(&particles)) {
        particles[id.x].density = 0.0;
        particles[id.x].nearDensity = 0.0;
        let n = arrayLength(&particles);
        let pos_i = particles[id.x].position;

        let v = cellPosition(pos_i);
        if (v.x < xGrids && 0 <= v.x && 
            v.y < yGrids && 0 <= v.y && 
            v.z < zGrids && 0 <= v.z) 
        {
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

        
        
        
        
        
        
        
        
        
    }
}`,Ze=`struct Particle {
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
    let scale: f32 = 45.0 / (3.1415926535 * kernelRadiusPow6); 
    let d = kernelRadius - r;
    return scale * d * d;
}

fn nearDensityKernelGradient(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * kernelRadiusPow5); 
    let a = kernelRadiusPow9;
    let d = kernelRadius - r;
    return scale * d * d;
}

fn viscosityKernelLaplacian(r: f32) -> f32 {
    let scale: f32 = 45.0 / (3.1415926535 * kernelRadiusPow6);
    
    let d = kernelRadius - r;
    return scale * d;
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
        if (v.x < xGrids && 0 <= v.x && 
            v.y < yGrids && 0 <= v.y && 
            v.z < zGrids && 0 <= v.z) 
        {
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
        }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        fVisc *= viscosity;
        let fGrv: vec3f = density_i * vec3f(0.0, -9.8, 0.0);
        particles[id.x].force = fPress + fVisc + fGrv;
    }
}`,$e=`struct Particle {
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
}`,Qe=`struct VertexOutput {
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

    let speed = sqrt(dot(particles[instance_index].velocity, particles[instance_index].velocity));
    let sz = max(0., uniforms.size - 0.01 * speed);
    let corner = vec3(corner_positions[vertex_index] * sz, 0.0);
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
}`,Je=`@group(0) @binding(1) var texture: texture_2d<f32>;

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var r = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    
    return vec4(0, r, r, 1.0);
}`,Ce=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,nt=`@group(0) @binding(0) var texture_sampler: sampler;
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
    var lightDir = normalize((uniforms.view_matrix * vec4f(1, 1, 1, 0.)).xyz);
    var H: vec3f        = normalize(lightDir - rayDir);
    var specular: f32   = pow(max(0.0, dot(H, normal)), 250.);
    var diffuse: f32  = max(0.0, dot(lightDir, normal)) * 1.0;

    var density = 1.0; 
    
    var thickness = textureLoad(thickness_texture, vec2u(input.iuv), 0).r;
    if (thickness < 0.0) {
        return vec4f(0.7, 0.7, 0.7, 1.);
    }
    var diffuseColor = vec3f(0.085, 0.6375, 0.9);
    var transmittance: vec3f = exp(-density * thickness * (1.0 - diffuseColor)); 
    var refractionColor: vec3f = vec3f(0.7, 0.7, 0.7) * transmittance;

    let F0 = 0.02;
    var fresnel: f32 = clamp(F0 + (1.0 - F0) * pow(1.0 - dot(normal, -rayDir), 5.0), 0., 0.3);

    var reflectionDir: vec3f = reflect(rayDir, normal);
    
    var reflectionDirWorld: vec3f = (uniforms.inv_view_matrix * vec4f(reflectionDir, 0.0)).xyz;
    var reflectionColor: vec3f = textureSampleLevel(envmap_texture, texture_sampler, reflectionDirWorld, 0.).rgb; 
    
    var finalColor = 0.2 * specular + mix(refractionColor, reflectionColor, fresnel);

    return vec4f(finalColor, 1.0);

    

    
    
    
    
    
    
    
    
    
    
    
}`,et=`struct VertexOutput {
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
}`,tt=`struct Uniforms {
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

    let speed = sqrt(dot(particles[instance_index].velocity, particles[instance_index].velocity));
    let sz = max(0., uniforms.size - 0.01 * speed);
    let corner = vec3(corner_positions[vertex_index] * sz, 0.0);
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
}`,rt=`@group(0) @binding(1) var texture: texture_2d<f32>;
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
}`,st=`struct VertexOutput {
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
}`,ot=`@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
    if (id.x < arrayLength(&cellParticleCount)) {
        cellParticleCount[id.x] = 0u;
    }
    let a = f32(id.x) / 0.;
}`,it=`struct Particle {
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
}`,ct=`struct Particle {
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
}`;const at=`

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
}`,ut=`

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
}`;function lt(f,g){const S={x:g,y:1};if(g>f.limits.maxComputeWorkgroupsPerDimension){const G=Math.floor(Math.sqrt(g)),D=Math.ceil(g/G);S.x=G,S.y=D}return S}class ft{constructor({device:g,data:S,count:G,workgroup_size:D={x:16,y:16},avoid_bank_conflicts:I=!1}){if(this.device=g,this.workgroup_size=D,this.threads_per_workgroup=D.x*D.y,this.items_per_workgroup=2*this.threads_per_workgroup,Math.log2(this.threads_per_workgroup)%1!==0)throw new Error(`workgroup_size.x * workgroup_size.y must be a power of two. (current: ${this.threads_per_workgroup})`);this.pipelines=[],this.shaderModule=this.device.createShaderModule({label:"prefix-sum",code:I?ut:at}),this.create_pass_recursive(S,G)}create_pass_recursive(g,S){const G=Math.ceil(S/this.items_per_workgroup),D=lt(this.device,G),I=this.device.createBuffer({label:"prefix-sum-block-sum",size:G*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),B=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]}),W=this.device.createBindGroup({label:"prefix-sum-bind-group",layout:B,entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:I}}]}),V=this.device.createPipelineLayout({bindGroupLayouts:[B]}),q=this.device.createComputePipeline({label:"prefix-sum-scan-pipeline",layout:V,compute:{module:this.shaderModule,entryPoint:"reduce_downsweep",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ITEMS_PER_WORKGROUP:this.items_per_workgroup,ELEMENT_COUNT:S}}});if(this.pipelines.push({pipeline:q,bindGroup:W,dispatchSize:D}),G>1){this.create_pass_recursive(I,G);const Q=this.device.createComputePipeline({label:"prefix-sum-add-block-pipeline",layout:V,compute:{module:this.shaderModule,entryPoint:"add_block_sums",constants:{WORKGROUP_SIZE_X:this.workgroup_size.x,WORKGROUP_SIZE_Y:this.workgroup_size.y,THREADS_PER_WORKGROUP:this.threads_per_workgroup,ELEMENT_COUNT:S}}});this.pipelines.push({pipeline:Q,bindGroup:W,dispatchSize:D})}}get_dispatch_chain(){return this.pipelines.flatMap(g=>[g.dispatchSize.x,g.dispatchSize.y,1])}dispatch(g,S,G=0){for(let D=0;D<this.pipelines.length;D++){const{pipeline:I,bindGroup:B,dispatchSize:W}=this.pipelines[D];g.setPipeline(I),g.setBindGroup(0,B),S==null?g.dispatchWorkgroups(W.x,W.y,1):g.dispatchWorkgroupsIndirect(S,G+D*3*4)}}}function dt(f,g){return class extends f{constructor(...S){super(...S),g(this)}}}const pt=dt(Array,f=>f.fill(0));let U=1e-6;function wt(f){function g(n=0,i=0){const e=new f(2);return n!==void 0&&(e[0]=n,i!==void 0&&(e[1]=i)),e}const S=g;function G(n,i,e){const r=e??new f(2);return r[0]=n,r[1]=i,r}function D(n,i){const e=i??new f(2);return e[0]=Math.ceil(n[0]),e[1]=Math.ceil(n[1]),e}function I(n,i){const e=i??new f(2);return e[0]=Math.floor(n[0]),e[1]=Math.floor(n[1]),e}function B(n,i){const e=i??new f(2);return e[0]=Math.round(n[0]),e[1]=Math.round(n[1]),e}function W(n,i=0,e=1,r){const a=r??new f(2);return a[0]=Math.min(e,Math.max(i,n[0])),a[1]=Math.min(e,Math.max(i,n[1])),a}function V(n,i,e){const r=e??new f(2);return r[0]=n[0]+i[0],r[1]=n[1]+i[1],r}function q(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+i[0]*e,a[1]=n[1]+i[1]*e,a}function Q(n,i){const e=n[0],r=n[1],a=i[0],v=i[1],y=Math.sqrt(e*e+r*r),c=Math.sqrt(a*a+v*v),u=y*c,w=u&&En(n,i)/u;return Math.acos(w)}function cn(n,i,e){const r=e??new f(2);return r[0]=n[0]-i[0],r[1]=n[1]-i[1],r}const yn=cn;function pn(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U}function Sn(n,i){return n[0]===i[0]&&n[1]===i[1]}function Gn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e*(i[0]-n[0]),a[1]=n[1]+e*(i[1]-n[1]),a}function Bn(n,i,e,r){const a=r??new f(2);return a[0]=n[0]+e[0]*(i[0]-n[0]),a[1]=n[1]+e[1]*(i[1]-n[1]),a}function kn(n,i,e){const r=e??new f(2);return r[0]=Math.max(n[0],i[0]),r[1]=Math.max(n[1],i[1]),r}function L(n,i,e){const r=e??new f(2);return r[0]=Math.min(n[0],i[0]),r[1]=Math.min(n[1],i[1]),r}function an(n,i,e){const r=e??new f(2);return r[0]=n[0]*i,r[1]=n[1]*i,r}const Ln=an;function mn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i,r[1]=n[1]/i,r}function wn(n,i){const e=i??new f(2);return e[0]=1/n[0],e[1]=1/n[1],e}const Nn=wn;function un(n,i,e){const r=e??new f(3),a=n[0]*i[1]-n[1]*i[0];return r[0]=0,r[1]=0,r[2]=a,r}function En(n,i){return n[0]*i[0]+n[1]*i[1]}function Z(n){const i=n[0],e=n[1];return Math.sqrt(i*i+e*e)}const Tn=Z;function A(n){const i=n[0],e=n[1];return i*i+e*e}const k=A;function O(n,i){const e=n[0]-i[0],r=n[1]-i[1];return Math.sqrt(e*e+r*r)}const tn=O;function vn(n,i){const e=n[0]-i[0],r=n[1]-i[1];return e*e+r*r}const Hn=vn;function _n(n,i){const e=i??new f(2),r=n[0],a=n[1],v=Math.sqrt(r*r+a*a);return v>1e-5?(e[0]=r/v,e[1]=a/v):(e[0]=0,e[1]=0),e}function Mn(n,i){const e=i??new f(2);return e[0]=-n[0],e[1]=-n[1],e}function N(n,i){const e=i??new f(2);return e[0]=n[0],e[1]=n[1],e}const Dn=N;function ln(n,i,e){const r=e??new f(2);return r[0]=n[0]*i[0],r[1]=n[1]*i[1],r}const Pn=ln;function fn(n,i,e){const r=e??new f(2);return r[0]=n[0]/i[0],r[1]=n[1]/i[1],r}const Rn=fn;function zn(n=1,i){const e=i??new f(2),r=Math.random()*2*Math.PI;return e[0]=Math.cos(r)*n,e[1]=Math.sin(r)*n,e}function s(n){const i=n??new f(2);return i[0]=0,i[1]=0,i}function d(n,i,e){const r=e??new f(2),a=n[0],v=n[1];return r[0]=a*i[0]+v*i[4]+i[12],r[1]=a*i[1]+v*i[5]+i[13],r}function t(n,i,e){const r=e??new f(2),a=n[0],v=n[1];return r[0]=i[0]*a+i[4]*v+i[8],r[1]=i[1]*a+i[5]*v+i[9],r}function o(n,i,e,r){const a=r??new f(2),v=n[0]-i[0],y=n[1]-i[1],c=Math.sin(e),u=Math.cos(e);return a[0]=v*u-y*c+i[0],a[1]=v*c+y*u+i[1],a}function l(n,i,e){const r=e??new f(2);return _n(n,r),an(r,i,r)}function p(n,i,e){const r=e??new f(2);return Z(n)>i?l(n,i,r):N(n,r)}function x(n,i,e){const r=e??new f(2);return Gn(n,i,.5,r)}return{create:g,fromValues:S,set:G,ceil:D,floor:I,round:B,clamp:W,add:V,addScaled:q,angle:Q,subtract:cn,sub:yn,equalsApproximately:pn,equals:Sn,lerp:Gn,lerpV:Bn,max:kn,min:L,mulScalar:an,scale:Ln,divScalar:mn,inverse:wn,invert:Nn,cross:un,dot:En,length:Z,len:Tn,lengthSq:A,lenSq:k,distance:O,dist:tn,distanceSq:vn,distSq:Hn,normalize:_n,negate:Mn,copy:N,clone:Dn,multiply:ln,mul:Pn,divide:fn,div:Rn,random:zn,zero:s,transformMat4:d,transformMat3:t,rotate:o,setLength:l,truncate:p,midpoint:x}}const Te=new Map;function Ne(f){let g=Te.get(f);return g||(g=wt(f),Te.set(f,g)),g}function vt(f){function g(c,u,w){const _=new f(3);return c!==void 0&&(_[0]=c,u!==void 0&&(_[1]=u,w!==void 0&&(_[2]=w))),_}const S=g;function G(c,u,w,_){const h=_??new f(3);return h[0]=c,h[1]=u,h[2]=w,h}function D(c,u){const w=u??new f(3);return w[0]=Math.ceil(c[0]),w[1]=Math.ceil(c[1]),w[2]=Math.ceil(c[2]),w}function I(c,u){const w=u??new f(3);return w[0]=Math.floor(c[0]),w[1]=Math.floor(c[1]),w[2]=Math.floor(c[2]),w}function B(c,u){const w=u??new f(3);return w[0]=Math.round(c[0]),w[1]=Math.round(c[1]),w[2]=Math.round(c[2]),w}function W(c,u=0,w=1,_){const h=_??new f(3);return h[0]=Math.min(w,Math.max(u,c[0])),h[1]=Math.min(w,Math.max(u,c[1])),h[2]=Math.min(w,Math.max(u,c[2])),h}function V(c,u,w){const _=w??new f(3);return _[0]=c[0]+u[0],_[1]=c[1]+u[1],_[2]=c[2]+u[2],_}function q(c,u,w,_){const h=_??new f(3);return h[0]=c[0]+u[0]*w,h[1]=c[1]+u[1]*w,h[2]=c[2]+u[2]*w,h}function Q(c,u){const w=c[0],_=c[1],h=c[2],m=u[0],M=u[1],b=u[2],E=Math.sqrt(w*w+_*_+h*h),P=Math.sqrt(m*m+M*M+b*b),z=E*P,T=z&&En(c,u)/z;return Math.acos(T)}function cn(c,u,w){const _=w??new f(3);return _[0]=c[0]-u[0],_[1]=c[1]-u[1],_[2]=c[2]-u[2],_}const yn=cn;function pn(c,u){return Math.abs(c[0]-u[0])<U&&Math.abs(c[1]-u[1])<U&&Math.abs(c[2]-u[2])<U}function Sn(c,u){return c[0]===u[0]&&c[1]===u[1]&&c[2]===u[2]}function Gn(c,u,w,_){const h=_??new f(3);return h[0]=c[0]+w*(u[0]-c[0]),h[1]=c[1]+w*(u[1]-c[1]),h[2]=c[2]+w*(u[2]-c[2]),h}function Bn(c,u,w,_){const h=_??new f(3);return h[0]=c[0]+w[0]*(u[0]-c[0]),h[1]=c[1]+w[1]*(u[1]-c[1]),h[2]=c[2]+w[2]*(u[2]-c[2]),h}function kn(c,u,w){const _=w??new f(3);return _[0]=Math.max(c[0],u[0]),_[1]=Math.max(c[1],u[1]),_[2]=Math.max(c[2],u[2]),_}function L(c,u,w){const _=w??new f(3);return _[0]=Math.min(c[0],u[0]),_[1]=Math.min(c[1],u[1]),_[2]=Math.min(c[2],u[2]),_}function an(c,u,w){const _=w??new f(3);return _[0]=c[0]*u,_[1]=c[1]*u,_[2]=c[2]*u,_}const Ln=an;function mn(c,u,w){const _=w??new f(3);return _[0]=c[0]/u,_[1]=c[1]/u,_[2]=c[2]/u,_}function wn(c,u){const w=u??new f(3);return w[0]=1/c[0],w[1]=1/c[1],w[2]=1/c[2],w}const Nn=wn;function un(c,u,w){const _=w??new f(3),h=c[2]*u[0]-c[0]*u[2],m=c[0]*u[1]-c[1]*u[0];return _[0]=c[1]*u[2]-c[2]*u[1],_[1]=h,_[2]=m,_}function En(c,u){return c[0]*u[0]+c[1]*u[1]+c[2]*u[2]}function Z(c){const u=c[0],w=c[1],_=c[2];return Math.sqrt(u*u+w*w+_*_)}const Tn=Z;function A(c){const u=c[0],w=c[1],_=c[2];return u*u+w*w+_*_}const k=A;function O(c,u){const w=c[0]-u[0],_=c[1]-u[1],h=c[2]-u[2];return Math.sqrt(w*w+_*_+h*h)}const tn=O;function vn(c,u){const w=c[0]-u[0],_=c[1]-u[1],h=c[2]-u[2];return w*w+_*_+h*h}const Hn=vn;function _n(c,u){const w=u??new f(3),_=c[0],h=c[1],m=c[2],M=Math.sqrt(_*_+h*h+m*m);return M>1e-5?(w[0]=_/M,w[1]=h/M,w[2]=m/M):(w[0]=0,w[1]=0,w[2]=0),w}function Mn(c,u){const w=u??new f(3);return w[0]=-c[0],w[1]=-c[1],w[2]=-c[2],w}function N(c,u){const w=u??new f(3);return w[0]=c[0],w[1]=c[1],w[2]=c[2],w}const Dn=N;function ln(c,u,w){const _=w??new f(3);return _[0]=c[0]*u[0],_[1]=c[1]*u[1],_[2]=c[2]*u[2],_}const Pn=ln;function fn(c,u,w){const _=w??new f(3);return _[0]=c[0]/u[0],_[1]=c[1]/u[1],_[2]=c[2]/u[2],_}const Rn=fn;function zn(c=1,u){const w=u??new f(3),_=Math.random()*2*Math.PI,h=Math.random()*2-1,m=Math.sqrt(1-h*h)*c;return w[0]=Math.cos(_)*m,w[1]=Math.sin(_)*m,w[2]=h*c,w}function s(c){const u=c??new f(3);return u[0]=0,u[1]=0,u[2]=0,u}function d(c,u,w){const _=w??new f(3),h=c[0],m=c[1],M=c[2],b=u[3]*h+u[7]*m+u[11]*M+u[15]||1;return _[0]=(u[0]*h+u[4]*m+u[8]*M+u[12])/b,_[1]=(u[1]*h+u[5]*m+u[9]*M+u[13])/b,_[2]=(u[2]*h+u[6]*m+u[10]*M+u[14])/b,_}function t(c,u,w){const _=w??new f(3),h=c[0],m=c[1],M=c[2];return _[0]=h*u[0*4+0]+m*u[1*4+0]+M*u[2*4+0],_[1]=h*u[0*4+1]+m*u[1*4+1]+M*u[2*4+1],_[2]=h*u[0*4+2]+m*u[1*4+2]+M*u[2*4+2],_}function o(c,u,w){const _=w??new f(3),h=c[0],m=c[1],M=c[2];return _[0]=h*u[0]+m*u[4]+M*u[8],_[1]=h*u[1]+m*u[5]+M*u[9],_[2]=h*u[2]+m*u[6]+M*u[10],_}function l(c,u,w){const _=w??new f(3),h=u[0],m=u[1],M=u[2],b=u[3]*2,E=c[0],P=c[1],z=c[2],T=m*z-M*P,R=M*E-h*z,F=h*P-m*E;return _[0]=E+T*b+(m*F-M*R)*2,_[1]=P+R*b+(M*T-h*F)*2,_[2]=z+F*b+(h*R-m*T)*2,_}function p(c,u){const w=u??new f(3);return w[0]=c[12],w[1]=c[13],w[2]=c[14],w}function x(c,u,w){const _=w??new f(3),h=u*4;return _[0]=c[h+0],_[1]=c[h+1],_[2]=c[h+2],_}function n(c,u){const w=u??new f(3),_=c[0],h=c[1],m=c[2],M=c[4],b=c[5],E=c[6],P=c[8],z=c[9],T=c[10];return w[0]=Math.sqrt(_*_+h*h+m*m),w[1]=Math.sqrt(M*M+b*b+E*E),w[2]=Math.sqrt(P*P+z*z+T*T),w}function i(c,u,w,_){const h=_??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0],M[1]=m[1]*Math.cos(w)-m[2]*Math.sin(w),M[2]=m[1]*Math.sin(w)+m[2]*Math.cos(w),h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function e(c,u,w,_){const h=_??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[2]*Math.sin(w)+m[0]*Math.cos(w),M[1]=m[1],M[2]=m[2]*Math.cos(w)-m[0]*Math.sin(w),h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function r(c,u,w,_){const h=_??new f(3),m=[],M=[];return m[0]=c[0]-u[0],m[1]=c[1]-u[1],m[2]=c[2]-u[2],M[0]=m[0]*Math.cos(w)-m[1]*Math.sin(w),M[1]=m[0]*Math.sin(w)+m[1]*Math.cos(w),M[2]=m[2],h[0]=M[0]+u[0],h[1]=M[1]+u[1],h[2]=M[2]+u[2],h}function a(c,u,w){const _=w??new f(3);return _n(c,_),an(_,u,_)}function v(c,u,w){const _=w??new f(3);return Z(c)>u?a(c,u,_):N(c,_)}function y(c,u,w){const _=w??new f(3);return Gn(c,u,.5,_)}return{create:g,fromValues:S,set:G,ceil:D,floor:I,round:B,clamp:W,add:V,addScaled:q,angle:Q,subtract:cn,sub:yn,equalsApproximately:pn,equals:Sn,lerp:Gn,lerpV:Bn,max:kn,min:L,mulScalar:an,scale:Ln,divScalar:mn,inverse:wn,invert:Nn,cross:un,dot:En,length:Z,len:Tn,lengthSq:A,lenSq:k,distance:O,dist:tn,distanceSq:vn,distSq:Hn,normalize:_n,negate:Mn,copy:N,clone:Dn,multiply:ln,mul:Pn,divide:fn,div:Rn,random:zn,zero:s,transformMat4:d,transformMat4Upper3x3:t,transformMat3:o,transformQuat:l,getTranslation:p,getAxis:x,getScaling:n,rotateX:i,rotateY:e,rotateZ:r,setLength:a,truncate:v,midpoint:y}}const Re=new Map;function ge(f){let g=Re.get(f);return g||(g=vt(f),Re.set(f,g)),g}function _t(f){const g=Ne(f),S=ge(f);function G(s,d,t,o,l,p,x,n,i){const e=new f(12);return e[3]=0,e[7]=0,e[11]=0,s!==void 0&&(e[0]=s,d!==void 0&&(e[1]=d,t!==void 0&&(e[2]=t,o!==void 0&&(e[4]=o,l!==void 0&&(e[5]=l,p!==void 0&&(e[6]=p,x!==void 0&&(e[8]=x,n!==void 0&&(e[9]=n,i!==void 0&&(e[10]=i))))))))),e}function D(s,d,t,o,l,p,x,n,i,e){const r=e??new f(12);return r[0]=s,r[1]=d,r[2]=t,r[3]=0,r[4]=o,r[5]=l,r[6]=p,r[7]=0,r[8]=x,r[9]=n,r[10]=i,r[11]=0,r}function I(s,d){const t=d??new f(12);return t[0]=s[0],t[1]=s[1],t[2]=s[2],t[3]=0,t[4]=s[4],t[5]=s[5],t[6]=s[6],t[7]=0,t[8]=s[8],t[9]=s[9],t[10]=s[10],t[11]=0,t}function B(s,d){const t=d??new f(12),o=s[0],l=s[1],p=s[2],x=s[3],n=o+o,i=l+l,e=p+p,r=o*n,a=l*n,v=l*i,y=p*n,c=p*i,u=p*e,w=x*n,_=x*i,h=x*e;return t[0]=1-v-u,t[1]=a+h,t[2]=y-_,t[3]=0,t[4]=a-h,t[5]=1-r-u,t[6]=c+w,t[7]=0,t[8]=y+_,t[9]=c-w,t[10]=1-r-v,t[11]=0,t}function W(s,d){const t=d??new f(12);return t[0]=-s[0],t[1]=-s[1],t[2]=-s[2],t[4]=-s[4],t[5]=-s[5],t[6]=-s[6],t[8]=-s[8],t[9]=-s[9],t[10]=-s[10],t}function V(s,d){const t=d??new f(12);return t[0]=s[0],t[1]=s[1],t[2]=s[2],t[4]=s[4],t[5]=s[5],t[6]=s[6],t[8]=s[8],t[9]=s[9],t[10]=s[10],t}const q=V;function Q(s,d){return Math.abs(s[0]-d[0])<U&&Math.abs(s[1]-d[1])<U&&Math.abs(s[2]-d[2])<U&&Math.abs(s[4]-d[4])<U&&Math.abs(s[5]-d[5])<U&&Math.abs(s[6]-d[6])<U&&Math.abs(s[8]-d[8])<U&&Math.abs(s[9]-d[9])<U&&Math.abs(s[10]-d[10])<U}function cn(s,d){return s[0]===d[0]&&s[1]===d[1]&&s[2]===d[2]&&s[4]===d[4]&&s[5]===d[5]&&s[6]===d[6]&&s[8]===d[8]&&s[9]===d[9]&&s[10]===d[10]}function yn(s){const d=s??new f(12);return d[0]=1,d[1]=0,d[2]=0,d[4]=0,d[5]=1,d[6]=0,d[8]=0,d[9]=0,d[10]=1,d}function pn(s,d){const t=d??new f(12);if(t===s){let v;return v=s[1],s[1]=s[4],s[4]=v,v=s[2],s[2]=s[8],s[8]=v,v=s[6],s[6]=s[9],s[9]=v,t}const o=s[0*4+0],l=s[0*4+1],p=s[0*4+2],x=s[1*4+0],n=s[1*4+1],i=s[1*4+2],e=s[2*4+0],r=s[2*4+1],a=s[2*4+2];return t[0]=o,t[1]=x,t[2]=e,t[4]=l,t[5]=n,t[6]=r,t[8]=p,t[9]=i,t[10]=a,t}function Sn(s,d){const t=d??new f(12),o=s[0*4+0],l=s[0*4+1],p=s[0*4+2],x=s[1*4+0],n=s[1*4+1],i=s[1*4+2],e=s[2*4+0],r=s[2*4+1],a=s[2*4+2],v=a*n-i*r,y=-a*x+i*e,c=r*x-n*e,u=1/(o*v+l*y+p*c);return t[0]=v*u,t[1]=(-a*l+p*r)*u,t[2]=(i*l-p*n)*u,t[4]=y*u,t[5]=(a*o-p*e)*u,t[6]=(-i*o+p*x)*u,t[8]=c*u,t[9]=(-r*o+l*e)*u,t[10]=(n*o-l*x)*u,t}function Gn(s){const d=s[0],t=s[0*4+1],o=s[0*4+2],l=s[1*4+0],p=s[1*4+1],x=s[1*4+2],n=s[2*4+0],i=s[2*4+1],e=s[2*4+2];return d*(p*e-i*x)-l*(t*e-i*o)+n*(t*x-p*o)}const Bn=Sn;function kn(s,d,t){const o=t??new f(12),l=s[0],p=s[1],x=s[2],n=s[4],i=s[5],e=s[6],r=s[8],a=s[9],v=s[10],y=d[0],c=d[1],u=d[2],w=d[4],_=d[5],h=d[6],m=d[8],M=d[9],b=d[10];return o[0]=l*y+n*c+r*u,o[1]=p*y+i*c+a*u,o[2]=x*y+e*c+v*u,o[4]=l*w+n*_+r*h,o[5]=p*w+i*_+a*h,o[6]=x*w+e*_+v*h,o[8]=l*m+n*M+r*b,o[9]=p*m+i*M+a*b,o[10]=x*m+e*M+v*b,o}const L=kn;function an(s,d,t){const o=t??yn();return s!==o&&(o[0]=s[0],o[1]=s[1],o[2]=s[2],o[4]=s[4],o[5]=s[5],o[6]=s[6]),o[8]=d[0],o[9]=d[1],o[10]=1,o}function Ln(s,d){const t=d??g.create();return t[0]=s[8],t[1]=s[9],t}function mn(s,d,t){const o=t??g.create(),l=d*4;return o[0]=s[l+0],o[1]=s[l+1],o}function wn(s,d,t,o){const l=o===s?s:V(s,o),p=t*4;return l[p+0]=d[0],l[p+1]=d[1],l}function Nn(s,d){const t=d??g.create(),o=s[0],l=s[1],p=s[4],x=s[5];return t[0]=Math.sqrt(o*o+l*l),t[1]=Math.sqrt(p*p+x*x),t}function un(s,d){const t=d??S.create(),o=s[0],l=s[1],p=s[2],x=s[4],n=s[5],i=s[6],e=s[8],r=s[9],a=s[10];return t[0]=Math.sqrt(o*o+l*l+p*p),t[1]=Math.sqrt(x*x+n*n+i*i),t[2]=Math.sqrt(e*e+r*r+a*a),t}function En(s,d){const t=d??new f(12);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=1,t[6]=0,t[8]=s[0],t[9]=s[1],t[10]=1,t}function Z(s,d,t){const o=t??new f(12),l=d[0],p=d[1],x=s[0],n=s[1],i=s[2],e=s[1*4+0],r=s[1*4+1],a=s[1*4+2],v=s[2*4+0],y=s[2*4+1],c=s[2*4+2];return s!==o&&(o[0]=x,o[1]=n,o[2]=i,o[4]=e,o[5]=r,o[6]=a),o[8]=x*l+e*p+v,o[9]=n*l+r*p+y,o[10]=i*l+a*p+c,o}function Tn(s,d){const t=d??new f(12),o=Math.cos(s),l=Math.sin(s);return t[0]=o,t[1]=l,t[2]=0,t[4]=-l,t[5]=o,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function A(s,d,t){const o=t??new f(12),l=s[0*4+0],p=s[0*4+1],x=s[0*4+2],n=s[1*4+0],i=s[1*4+1],e=s[1*4+2],r=Math.cos(d),a=Math.sin(d);return o[0]=r*l+a*n,o[1]=r*p+a*i,o[2]=r*x+a*e,o[4]=r*n-a*l,o[5]=r*i-a*p,o[6]=r*e-a*x,s!==o&&(o[8]=s[8],o[9]=s[9],o[10]=s[10]),o}function k(s,d){const t=d??new f(12),o=Math.cos(s),l=Math.sin(s);return t[0]=1,t[1]=0,t[2]=0,t[4]=0,t[5]=o,t[6]=l,t[8]=0,t[9]=-l,t[10]=o,t}function O(s,d,t){const o=t??new f(12),l=s[4],p=s[5],x=s[6],n=s[8],i=s[9],e=s[10],r=Math.cos(d),a=Math.sin(d);return o[4]=r*l+a*n,o[5]=r*p+a*i,o[6]=r*x+a*e,o[8]=r*n-a*l,o[9]=r*i-a*p,o[10]=r*e-a*x,s!==o&&(o[0]=s[0],o[1]=s[1],o[2]=s[2]),o}function tn(s,d){const t=d??new f(12),o=Math.cos(s),l=Math.sin(s);return t[0]=o,t[1]=0,t[2]=-l,t[4]=0,t[5]=1,t[6]=0,t[8]=l,t[9]=0,t[10]=o,t}function vn(s,d,t){const o=t??new f(12),l=s[0*4+0],p=s[0*4+1],x=s[0*4+2],n=s[2*4+0],i=s[2*4+1],e=s[2*4+2],r=Math.cos(d),a=Math.sin(d);return o[0]=r*l-a*n,o[1]=r*p-a*i,o[2]=r*x-a*e,o[8]=r*n+a*l,o[9]=r*i+a*p,o[10]=r*e+a*x,s!==o&&(o[4]=s[4],o[5]=s[5],o[6]=s[6]),o}const Hn=Tn,_n=A;function Mn(s,d){const t=d??new f(12);return t[0]=s[0],t[1]=0,t[2]=0,t[4]=0,t[5]=s[1],t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function N(s,d,t){const o=t??new f(12),l=d[0],p=d[1];return o[0]=l*s[0*4+0],o[1]=l*s[0*4+1],o[2]=l*s[0*4+2],o[4]=p*s[1*4+0],o[5]=p*s[1*4+1],o[6]=p*s[1*4+2],s!==o&&(o[8]=s[8],o[9]=s[9],o[10]=s[10]),o}function Dn(s,d){const t=d??new f(12);return t[0]=s[0],t[1]=0,t[2]=0,t[4]=0,t[5]=s[1],t[6]=0,t[8]=0,t[9]=0,t[10]=s[2],t}function ln(s,d,t){const o=t??new f(12),l=d[0],p=d[1],x=d[2];return o[0]=l*s[0*4+0],o[1]=l*s[0*4+1],o[2]=l*s[0*4+2],o[4]=p*s[1*4+0],o[5]=p*s[1*4+1],o[6]=p*s[1*4+2],o[8]=x*s[2*4+0],o[9]=x*s[2*4+1],o[10]=x*s[2*4+2],o}function Pn(s,d){const t=d??new f(12);return t[0]=s,t[1]=0,t[2]=0,t[4]=0,t[5]=s,t[6]=0,t[8]=0,t[9]=0,t[10]=1,t}function fn(s,d,t){const o=t??new f(12);return o[0]=d*s[0*4+0],o[1]=d*s[0*4+1],o[2]=d*s[0*4+2],o[4]=d*s[1*4+0],o[5]=d*s[1*4+1],o[6]=d*s[1*4+2],s!==o&&(o[8]=s[8],o[9]=s[9],o[10]=s[10]),o}function Rn(s,d){const t=d??new f(12);return t[0]=s,t[1]=0,t[2]=0,t[4]=0,t[5]=s,t[6]=0,t[8]=0,t[9]=0,t[10]=s,t}function zn(s,d,t){const o=t??new f(12);return o[0]=d*s[0*4+0],o[1]=d*s[0*4+1],o[2]=d*s[0*4+2],o[4]=d*s[1*4+0],o[5]=d*s[1*4+1],o[6]=d*s[1*4+2],o[8]=d*s[2*4+0],o[9]=d*s[2*4+1],o[10]=d*s[2*4+2],o}return{clone:q,create:G,set:D,fromMat4:I,fromQuat:B,negate:W,copy:V,equalsApproximately:Q,equals:cn,identity:yn,transpose:pn,inverse:Sn,invert:Bn,determinant:Gn,mul:L,multiply:kn,setTranslation:an,getTranslation:Ln,getAxis:mn,setAxis:wn,getScaling:Nn,get3DScaling:un,translation:En,translate:Z,rotation:Tn,rotate:A,rotationX:k,rotateX:O,rotationY:tn,rotateY:vn,rotationZ:Hn,rotateZ:_n,scaling:Mn,scale:N,uniformScaling:Pn,uniformScale:fn,scaling3D:Dn,scale3D:ln,uniformScaling3D:Rn,uniformScale3D:zn}}const Oe=new Map;function gt(f){let g=Oe.get(f);return g||(g=_t(f),Oe.set(f,g)),g}function xt(f){const g=ge(f);function S(n,i,e,r,a,v,y,c,u,w,_,h,m,M,b,E){const P=new f(16);return n!==void 0&&(P[0]=n,i!==void 0&&(P[1]=i,e!==void 0&&(P[2]=e,r!==void 0&&(P[3]=r,a!==void 0&&(P[4]=a,v!==void 0&&(P[5]=v,y!==void 0&&(P[6]=y,c!==void 0&&(P[7]=c,u!==void 0&&(P[8]=u,w!==void 0&&(P[9]=w,_!==void 0&&(P[10]=_,h!==void 0&&(P[11]=h,m!==void 0&&(P[12]=m,M!==void 0&&(P[13]=M,b!==void 0&&(P[14]=b,E!==void 0&&(P[15]=E)))))))))))))))),P}function G(n,i,e,r,a,v,y,c,u,w,_,h,m,M,b,E,P){const z=P??new f(16);return z[0]=n,z[1]=i,z[2]=e,z[3]=r,z[4]=a,z[5]=v,z[6]=y,z[7]=c,z[8]=u,z[9]=w,z[10]=_,z[11]=h,z[12]=m,z[13]=M,z[14]=b,z[15]=E,z}function D(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=0,e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=0,e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function I(n,i){const e=i??new f(16),r=n[0],a=n[1],v=n[2],y=n[3],c=r+r,u=a+a,w=v+v,_=r*c,h=a*c,m=a*u,M=v*c,b=v*u,E=v*w,P=y*c,z=y*u,T=y*w;return e[0]=1-m-E,e[1]=h+T,e[2]=M-z,e[3]=0,e[4]=h-T,e[5]=1-_-E,e[6]=b+P,e[7]=0,e[8]=M+z,e[9]=b-P,e[10]=1-_-m,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function B(n,i){const e=i??new f(16);return e[0]=-n[0],e[1]=-n[1],e[2]=-n[2],e[3]=-n[3],e[4]=-n[4],e[5]=-n[5],e[6]=-n[6],e[7]=-n[7],e[8]=-n[8],e[9]=-n[9],e[10]=-n[10],e[11]=-n[11],e[12]=-n[12],e[13]=-n[13],e[14]=-n[14],e[15]=-n[15],e}function W(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],e}const V=W;function q(n,i){return Math.abs(n[0]-i[0])<U&&Math.abs(n[1]-i[1])<U&&Math.abs(n[2]-i[2])<U&&Math.abs(n[3]-i[3])<U&&Math.abs(n[4]-i[4])<U&&Math.abs(n[5]-i[5])<U&&Math.abs(n[6]-i[6])<U&&Math.abs(n[7]-i[7])<U&&Math.abs(n[8]-i[8])<U&&Math.abs(n[9]-i[9])<U&&Math.abs(n[10]-i[10])<U&&Math.abs(n[11]-i[11])<U&&Math.abs(n[12]-i[12])<U&&Math.abs(n[13]-i[13])<U&&Math.abs(n[14]-i[14])<U&&Math.abs(n[15]-i[15])<U}function Q(n,i){return n[0]===i[0]&&n[1]===i[1]&&n[2]===i[2]&&n[3]===i[3]&&n[4]===i[4]&&n[5]===i[5]&&n[6]===i[6]&&n[7]===i[7]&&n[8]===i[8]&&n[9]===i[9]&&n[10]===i[10]&&n[11]===i[11]&&n[12]===i[12]&&n[13]===i[13]&&n[14]===i[14]&&n[15]===i[15]}function cn(n){const i=n??new f(16);return i[0]=1,i[1]=0,i[2]=0,i[3]=0,i[4]=0,i[5]=1,i[6]=0,i[7]=0,i[8]=0,i[9]=0,i[10]=1,i[11]=0,i[12]=0,i[13]=0,i[14]=0,i[15]=1,i}function yn(n,i){const e=i??new f(16);if(e===n){let R;return R=n[1],n[1]=n[4],n[4]=R,R=n[2],n[2]=n[8],n[8]=R,R=n[3],n[3]=n[12],n[12]=R,R=n[6],n[6]=n[9],n[9]=R,R=n[7],n[7]=n[13],n[13]=R,R=n[11],n[11]=n[14],n[14]=R,e}const r=n[0*4+0],a=n[0*4+1],v=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],_=n[1*4+3],h=n[2*4+0],m=n[2*4+1],M=n[2*4+2],b=n[2*4+3],E=n[3*4+0],P=n[3*4+1],z=n[3*4+2],T=n[3*4+3];return e[0]=r,e[1]=c,e[2]=h,e[3]=E,e[4]=a,e[5]=u,e[6]=m,e[7]=P,e[8]=v,e[9]=w,e[10]=M,e[11]=z,e[12]=y,e[13]=_,e[14]=b,e[15]=T,e}function pn(n,i){const e=i??new f(16),r=n[0*4+0],a=n[0*4+1],v=n[0*4+2],y=n[0*4+3],c=n[1*4+0],u=n[1*4+1],w=n[1*4+2],_=n[1*4+3],h=n[2*4+0],m=n[2*4+1],M=n[2*4+2],b=n[2*4+3],E=n[3*4+0],P=n[3*4+1],z=n[3*4+2],T=n[3*4+3],R=M*T,F=z*b,Y=w*T,H=z*_,K=w*b,J=M*_,rn=v*T,$=z*y,C=v*b,nn=M*y,sn=v*_,on=w*y,gn=h*P,xn=E*m,bn=c*P,On=E*u,Un=c*m,Jn=h*u,Cn=r*P,ne=E*a,Yn=r*m,ee=h*a,te=r*u,Xn=c*a,Zn=R*u+H*m+K*P-(F*u+Y*m+J*P),re=F*a+rn*m+nn*P-(R*a+$*m+C*P),se=Y*a+$*u+sn*P-(H*a+rn*u+on*P),oe=J*a+C*u+on*m-(K*a+nn*u+sn*m),j=1/(r*Zn+c*re+h*se+E*oe);return e[0]=j*Zn,e[1]=j*re,e[2]=j*se,e[3]=j*oe,e[4]=j*(F*c+Y*h+J*E-(R*c+H*h+K*E)),e[5]=j*(R*r+$*h+C*E-(F*r+rn*h+nn*E)),e[6]=j*(H*r+rn*c+on*E-(Y*r+$*c+sn*E)),e[7]=j*(K*r+nn*c+sn*h-(J*r+C*c+on*h)),e[8]=j*(gn*_+On*b+Un*T-(xn*_+bn*b+Jn*T)),e[9]=j*(xn*y+Cn*b+ee*T-(gn*y+ne*b+Yn*T)),e[10]=j*(bn*y+ne*_+te*T-(On*y+Cn*_+Xn*T)),e[11]=j*(Jn*y+Yn*_+Xn*b-(Un*y+ee*_+te*b)),e[12]=j*(bn*M+Jn*z+xn*w-(Un*z+gn*w+On*M)),e[13]=j*(Yn*z+gn*v+ne*M-(Cn*M+ee*z+xn*v)),e[14]=j*(Cn*w+Xn*z+On*v-(te*z+bn*v+ne*w)),e[15]=j*(te*M+Un*v+ee*w-(Yn*w+Xn*M+Jn*v)),e}function Sn(n){const i=n[0],e=n[0*4+1],r=n[0*4+2],a=n[0*4+3],v=n[1*4+0],y=n[1*4+1],c=n[1*4+2],u=n[1*4+3],w=n[2*4+0],_=n[2*4+1],h=n[2*4+2],m=n[2*4+3],M=n[3*4+0],b=n[3*4+1],E=n[3*4+2],P=n[3*4+3],z=h*P,T=E*m,R=c*P,F=E*u,Y=c*m,H=h*u,K=r*P,J=E*a,rn=r*m,$=h*a,C=r*u,nn=c*a,sn=z*y+F*_+Y*b-(T*y+R*_+H*b),on=T*e+K*_+$*b-(z*e+J*_+rn*b),gn=R*e+J*y+C*b-(F*e+K*y+nn*b),xn=H*e+rn*y+nn*_-(Y*e+$*y+C*_);return i*sn+v*on+w*gn+M*xn}const Gn=pn;function Bn(n,i,e){const r=e??new f(16),a=n[0],v=n[1],y=n[2],c=n[3],u=n[4],w=n[5],_=n[6],h=n[7],m=n[8],M=n[9],b=n[10],E=n[11],P=n[12],z=n[13],T=n[14],R=n[15],F=i[0],Y=i[1],H=i[2],K=i[3],J=i[4],rn=i[5],$=i[6],C=i[7],nn=i[8],sn=i[9],on=i[10],gn=i[11],xn=i[12],bn=i[13],On=i[14],Un=i[15];return r[0]=a*F+u*Y+m*H+P*K,r[1]=v*F+w*Y+M*H+z*K,r[2]=y*F+_*Y+b*H+T*K,r[3]=c*F+h*Y+E*H+R*K,r[4]=a*J+u*rn+m*$+P*C,r[5]=v*J+w*rn+M*$+z*C,r[6]=y*J+_*rn+b*$+T*C,r[7]=c*J+h*rn+E*$+R*C,r[8]=a*nn+u*sn+m*on+P*gn,r[9]=v*nn+w*sn+M*on+z*gn,r[10]=y*nn+_*sn+b*on+T*gn,r[11]=c*nn+h*sn+E*on+R*gn,r[12]=a*xn+u*bn+m*On+P*Un,r[13]=v*xn+w*bn+M*On+z*Un,r[14]=y*xn+_*bn+b*On+T*Un,r[15]=c*xn+h*bn+E*On+R*Un,r}const kn=Bn;function L(n,i,e){const r=e??cn();return n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11]),r[12]=i[0],r[13]=i[1],r[14]=i[2],r[15]=1,r}function an(n,i){const e=i??g.create();return e[0]=n[12],e[1]=n[13],e[2]=n[14],e}function Ln(n,i,e){const r=e??g.create(),a=i*4;return r[0]=n[a+0],r[1]=n[a+1],r[2]=n[a+2],r}function mn(n,i,e,r){const a=r===n?r:W(n,r),v=e*4;return a[v+0]=i[0],a[v+1]=i[1],a[v+2]=i[2],a}function wn(n,i){const e=i??g.create(),r=n[0],a=n[1],v=n[2],y=n[4],c=n[5],u=n[6],w=n[8],_=n[9],h=n[10];return e[0]=Math.sqrt(r*r+a*a+v*v),e[1]=Math.sqrt(y*y+c*c+u*u),e[2]=Math.sqrt(w*w+_*_+h*h),e}function Nn(n,i,e,r,a){const v=a??new f(16),y=Math.tan(Math.PI*.5-.5*n);if(v[0]=y/i,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=y,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[11]=-1,v[12]=0,v[13]=0,v[15]=0,Number.isFinite(r)){const c=1/(e-r);v[10]=r*c,v[14]=r*e*c}else v[10]=-1,v[14]=-e;return v}function un(n,i,e,r=1/0,a){const v=a??new f(16),y=1/Math.tan(n*.5);if(v[0]=y/i,v[1]=0,v[2]=0,v[3]=0,v[4]=0,v[5]=y,v[6]=0,v[7]=0,v[8]=0,v[9]=0,v[11]=-1,v[12]=0,v[13]=0,v[15]=0,r===1/0)v[10]=0,v[14]=e;else{const c=1/(r-e);v[10]=e*c,v[14]=r*e*c}return v}function En(n,i,e,r,a,v,y){const c=y??new f(16);return c[0]=2/(i-n),c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2/(r-e),c[6]=0,c[7]=0,c[8]=0,c[9]=0,c[10]=1/(a-v),c[11]=0,c[12]=(i+n)/(n-i),c[13]=(r+e)/(e-r),c[14]=a/(a-v),c[15]=1,c}function Z(n,i,e,r,a,v,y){const c=y??new f(16),u=i-n,w=r-e,_=a-v;return c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[10]=v/_,c[11]=-1,c[12]=0,c[13]=0,c[14]=a*v/_,c[15]=0,c}function Tn(n,i,e,r,a,v=1/0,y){const c=y??new f(16),u=i-n,w=r-e;if(c[0]=2*a/u,c[1]=0,c[2]=0,c[3]=0,c[4]=0,c[5]=2*a/w,c[6]=0,c[7]=0,c[8]=(n+i)/u,c[9]=(r+e)/w,c[11]=-1,c[12]=0,c[13]=0,c[15]=0,v===1/0)c[10]=0,c[14]=a;else{const _=1/(v-a);c[10]=a*_,c[14]=v*a*_}return c}const A=g.create(),k=g.create(),O=g.create();function tn(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(i,n,O),O),g.normalize(g.cross(e,O,A),A),g.normalize(g.cross(O,A,k),k),a[0]=A[0],a[1]=A[1],a[2]=A[2],a[3]=0,a[4]=k[0],a[5]=k[1],a[6]=k[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function vn(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(n,i,O),O),g.normalize(g.cross(e,O,A),A),g.normalize(g.cross(O,A,k),k),a[0]=A[0],a[1]=A[1],a[2]=A[2],a[3]=0,a[4]=k[0],a[5]=k[1],a[6]=k[2],a[7]=0,a[8]=O[0],a[9]=O[1],a[10]=O[2],a[11]=0,a[12]=n[0],a[13]=n[1],a[14]=n[2],a[15]=1,a}function Hn(n,i,e,r){const a=r??new f(16);return g.normalize(g.subtract(n,i,O),O),g.normalize(g.cross(e,O,A),A),g.normalize(g.cross(O,A,k),k),a[0]=A[0],a[1]=k[0],a[2]=O[0],a[3]=0,a[4]=A[1],a[5]=k[1],a[6]=O[1],a[7]=0,a[8]=A[2],a[9]=k[2],a[10]=O[2],a[11]=0,a[12]=-(A[0]*n[0]+A[1]*n[1]+A[2]*n[2]),a[13]=-(k[0]*n[0]+k[1]*n[1]+k[2]*n[2]),a[14]=-(O[0]*n[0]+O[1]*n[1]+O[2]*n[2]),a[15]=1,a}function _n(n,i){const e=i??new f(16);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=n[0],e[13]=n[1],e[14]=n[2],e[15]=1,e}function Mn(n,i,e){const r=e??new f(16),a=i[0],v=i[1],y=i[2],c=n[0],u=n[1],w=n[2],_=n[3],h=n[1*4+0],m=n[1*4+1],M=n[1*4+2],b=n[1*4+3],E=n[2*4+0],P=n[2*4+1],z=n[2*4+2],T=n[2*4+3],R=n[3*4+0],F=n[3*4+1],Y=n[3*4+2],H=n[3*4+3];return n!==r&&(r[0]=c,r[1]=u,r[2]=w,r[3]=_,r[4]=h,r[5]=m,r[6]=M,r[7]=b,r[8]=E,r[9]=P,r[10]=z,r[11]=T),r[12]=c*a+h*v+E*y+R,r[13]=u*a+m*v+P*y+F,r[14]=w*a+M*v+z*y+Y,r[15]=_*a+b*v+T*y+H,r}function N(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=r,e[6]=a,e[7]=0,e[8]=0,e[9]=-a,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Dn(n,i,e){const r=e??new f(16),a=n[4],v=n[5],y=n[6],c=n[7],u=n[8],w=n[9],_=n[10],h=n[11],m=Math.cos(i),M=Math.sin(i);return r[4]=m*a+M*u,r[5]=m*v+M*w,r[6]=m*y+M*_,r[7]=m*c+M*h,r[8]=m*u-M*a,r[9]=m*w-M*v,r[10]=m*_-M*y,r[11]=m*h-M*c,n!==r&&(r[0]=n[0],r[1]=n[1],r[2]=n[2],r[3]=n[3],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function ln(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=0,e[2]=-a,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=a,e[9]=0,e[10]=r,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Pn(n,i,e){const r=e??new f(16),a=n[0*4+0],v=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[2*4+0],w=n[2*4+1],_=n[2*4+2],h=n[2*4+3],m=Math.cos(i),M=Math.sin(i);return r[0]=m*a-M*u,r[1]=m*v-M*w,r[2]=m*y-M*_,r[3]=m*c-M*h,r[8]=m*u+M*a,r[9]=m*w+M*v,r[10]=m*_+M*y,r[11]=m*h+M*c,n!==r&&(r[4]=n[4],r[5]=n[5],r[6]=n[6],r[7]=n[7],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function fn(n,i){const e=i??new f(16),r=Math.cos(n),a=Math.sin(n);return e[0]=r,e[1]=a,e[2]=0,e[3]=0,e[4]=-a,e[5]=r,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function Rn(n,i,e){const r=e??new f(16),a=n[0*4+0],v=n[0*4+1],y=n[0*4+2],c=n[0*4+3],u=n[1*4+0],w=n[1*4+1],_=n[1*4+2],h=n[1*4+3],m=Math.cos(i),M=Math.sin(i);return r[0]=m*a+M*u,r[1]=m*v+M*w,r[2]=m*y+M*_,r[3]=m*c+M*h,r[4]=m*u-M*a,r[5]=m*w-M*v,r[6]=m*_-M*y,r[7]=m*h-M*c,n!==r&&(r[8]=n[8],r[9]=n[9],r[10]=n[10],r[11]=n[11],r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function zn(n,i,e){const r=e??new f(16);let a=n[0],v=n[1],y=n[2];const c=Math.sqrt(a*a+v*v+y*y);a/=c,v/=c,y/=c;const u=a*a,w=v*v,_=y*y,h=Math.cos(i),m=Math.sin(i),M=1-h;return r[0]=u+(1-u)*h,r[1]=a*v*M+y*m,r[2]=a*y*M-v*m,r[3]=0,r[4]=a*v*M-y*m,r[5]=w+(1-w)*h,r[6]=v*y*M+a*m,r[7]=0,r[8]=a*y*M+v*m,r[9]=v*y*M-a*m,r[10]=_+(1-_)*h,r[11]=0,r[12]=0,r[13]=0,r[14]=0,r[15]=1,r}const s=zn;function d(n,i,e,r){const a=r??new f(16);let v=i[0],y=i[1],c=i[2];const u=Math.sqrt(v*v+y*y+c*c);v/=u,y/=u,c/=u;const w=v*v,_=y*y,h=c*c,m=Math.cos(e),M=Math.sin(e),b=1-m,E=w+(1-w)*m,P=v*y*b+c*M,z=v*c*b-y*M,T=v*y*b-c*M,R=_+(1-_)*m,F=y*c*b+v*M,Y=v*c*b+y*M,H=y*c*b-v*M,K=h+(1-h)*m,J=n[0],rn=n[1],$=n[2],C=n[3],nn=n[4],sn=n[5],on=n[6],gn=n[7],xn=n[8],bn=n[9],On=n[10],Un=n[11];return a[0]=E*J+P*nn+z*xn,a[1]=E*rn+P*sn+z*bn,a[2]=E*$+P*on+z*On,a[3]=E*C+P*gn+z*Un,a[4]=T*J+R*nn+F*xn,a[5]=T*rn+R*sn+F*bn,a[6]=T*$+R*on+F*On,a[7]=T*C+R*gn+F*Un,a[8]=Y*J+H*nn+K*xn,a[9]=Y*rn+H*sn+K*bn,a[10]=Y*$+H*on+K*On,a[11]=Y*C+H*gn+K*Un,n!==a&&(a[12]=n[12],a[13]=n[13],a[14]=n[14],a[15]=n[15]),a}const t=d;function o(n,i){const e=i??new f(16);return e[0]=n[0],e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n[1],e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n[2],e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function l(n,i,e){const r=e??new f(16),a=i[0],v=i[1],y=i[2];return r[0]=a*n[0*4+0],r[1]=a*n[0*4+1],r[2]=a*n[0*4+2],r[3]=a*n[0*4+3],r[4]=v*n[1*4+0],r[5]=v*n[1*4+1],r[6]=v*n[1*4+2],r[7]=v*n[1*4+3],r[8]=y*n[2*4+0],r[9]=y*n[2*4+1],r[10]=y*n[2*4+2],r[11]=y*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}function p(n,i){const e=i??new f(16);return e[0]=n,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=n,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=n,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function x(n,i,e){const r=e??new f(16);return r[0]=i*n[0*4+0],r[1]=i*n[0*4+1],r[2]=i*n[0*4+2],r[3]=i*n[0*4+3],r[4]=i*n[1*4+0],r[5]=i*n[1*4+1],r[6]=i*n[1*4+2],r[7]=i*n[1*4+3],r[8]=i*n[2*4+0],r[9]=i*n[2*4+1],r[10]=i*n[2*4+2],r[11]=i*n[2*4+3],n!==r&&(r[12]=n[12],r[13]=n[13],r[14]=n[14],r[15]=n[15]),r}return{create:S,set:G,fromMat3:D,fromQuat:I,negate:B,copy:W,clone:V,equalsApproximately:q,equals:Q,identity:cn,transpose:yn,inverse:pn,determinant:Sn,invert:Gn,multiply:Bn,mul:kn,setTranslation:L,getTranslation:an,getAxis:Ln,setAxis:mn,getScaling:wn,perspective:Nn,perspectiveReverseZ:un,ortho:En,frustum:Z,frustumReverseZ:Tn,aim:tn,cameraAim:vn,lookAt:Hn,translation:_n,translate:Mn,rotationX:N,rotateX:Dn,rotationY:ln,rotateY:Pn,rotationZ:fn,rotateZ:Rn,axisRotation:zn,rotation:s,axisRotate:d,rotate:t,scaling:o,scale:l,uniformScaling:p,uniformScale:x}}const Ue=new Map;function ht(f){let g=Ue.get(f);return g||(g=xt(f),Ue.set(f,g)),g}function yt(f){const g=ge(f);function S(s,d,t,o){const l=new f(4);return s!==void 0&&(l[0]=s,d!==void 0&&(l[1]=d,t!==void 0&&(l[2]=t,o!==void 0&&(l[3]=o)))),l}const G=S;function D(s,d,t,o,l){const p=l??new f(4);return p[0]=s,p[1]=d,p[2]=t,p[3]=o,p}function I(s,d,t){const o=t??new f(4),l=d*.5,p=Math.sin(l);return o[0]=p*s[0],o[1]=p*s[1],o[2]=p*s[2],o[3]=Math.cos(l),o}function B(s,d){const t=d??g.create(3),o=Math.acos(s[3])*2,l=Math.sin(o*.5);return l>U?(t[0]=s[0]/l,t[1]=s[1]/l,t[2]=s[2]/l):(t[0]=1,t[1]=0,t[2]=0),{angle:o,axis:t}}function W(s,d){const t=Z(s,d);return Math.acos(2*t*t-1)}function V(s,d,t){const o=t??new f(4),l=s[0],p=s[1],x=s[2],n=s[3],i=d[0],e=d[1],r=d[2],a=d[3];return o[0]=l*a+n*i+p*r-x*e,o[1]=p*a+n*e+x*i-l*r,o[2]=x*a+n*r+l*e-p*i,o[3]=n*a-l*i-p*e-x*r,o}const q=V;function Q(s,d,t){const o=t??new f(4),l=d*.5,p=s[0],x=s[1],n=s[2],i=s[3],e=Math.sin(l),r=Math.cos(l);return o[0]=p*r+i*e,o[1]=x*r+n*e,o[2]=n*r-x*e,o[3]=i*r-p*e,o}function cn(s,d,t){const o=t??new f(4),l=d*.5,p=s[0],x=s[1],n=s[2],i=s[3],e=Math.sin(l),r=Math.cos(l);return o[0]=p*r-n*e,o[1]=x*r+i*e,o[2]=n*r+p*e,o[3]=i*r-x*e,o}function yn(s,d,t){const o=t??new f(4),l=d*.5,p=s[0],x=s[1],n=s[2],i=s[3],e=Math.sin(l),r=Math.cos(l);return o[0]=p*r+x*e,o[1]=x*r-p*e,o[2]=n*r+i*e,o[3]=i*r-n*e,o}function pn(s,d,t,o){const l=o??new f(4),p=s[0],x=s[1],n=s[2],i=s[3];let e=d[0],r=d[1],a=d[2],v=d[3],y=p*e+x*r+n*a+i*v;y<0&&(y=-y,e=-e,r=-r,a=-a,v=-v);let c,u;if(1-y>U){const w=Math.acos(y),_=Math.sin(w);c=Math.sin((1-t)*w)/_,u=Math.sin(t*w)/_}else c=1-t,u=t;return l[0]=c*p+u*e,l[1]=c*x+u*r,l[2]=c*n+u*a,l[3]=c*i+u*v,l}function Sn(s,d){const t=d??new f(4),o=s[0],l=s[1],p=s[2],x=s[3],n=o*o+l*l+p*p+x*x,i=n?1/n:0;return t[0]=-o*i,t[1]=-l*i,t[2]=-p*i,t[3]=x*i,t}function Gn(s,d){const t=d??new f(4);return t[0]=-s[0],t[1]=-s[1],t[2]=-s[2],t[3]=s[3],t}function Bn(s,d){const t=d??new f(4),o=s[0]+s[5]+s[10];if(o>0){const l=Math.sqrt(o+1);t[3]=.5*l;const p=.5/l;t[0]=(s[6]-s[9])*p,t[1]=(s[8]-s[2])*p,t[2]=(s[1]-s[4])*p}else{let l=0;s[5]>s[0]&&(l=1),s[10]>s[l*4+l]&&(l=2);const p=(l+1)%3,x=(l+2)%3,n=Math.sqrt(s[l*4+l]-s[p*4+p]-s[x*4+x]+1);t[l]=.5*n;const i=.5/n;t[3]=(s[p*4+x]-s[x*4+p])*i,t[p]=(s[p*4+l]+s[l*4+p])*i,t[x]=(s[x*4+l]+s[l*4+x])*i}return t}function kn(s,d,t,o,l){const p=l??new f(4),x=s*.5,n=d*.5,i=t*.5,e=Math.sin(x),r=Math.cos(x),a=Math.sin(n),v=Math.cos(n),y=Math.sin(i),c=Math.cos(i);switch(o){case"xyz":p[0]=e*v*c+r*a*y,p[1]=r*a*c-e*v*y,p[2]=r*v*y+e*a*c,p[3]=r*v*c-e*a*y;break;case"xzy":p[0]=e*v*c-r*a*y,p[1]=r*a*c-e*v*y,p[2]=r*v*y+e*a*c,p[3]=r*v*c+e*a*y;break;case"yxz":p[0]=e*v*c+r*a*y,p[1]=r*a*c-e*v*y,p[2]=r*v*y-e*a*c,p[3]=r*v*c+e*a*y;break;case"yzx":p[0]=e*v*c+r*a*y,p[1]=r*a*c+e*v*y,p[2]=r*v*y-e*a*c,p[3]=r*v*c-e*a*y;break;case"zxy":p[0]=e*v*c-r*a*y,p[1]=r*a*c+e*v*y,p[2]=r*v*y+e*a*c,p[3]=r*v*c-e*a*y;break;case"zyx":p[0]=e*v*c-r*a*y,p[1]=r*a*c+e*v*y,p[2]=r*v*y-e*a*c,p[3]=r*v*c+e*a*y;break;default:throw new Error(`Unknown rotation order: ${o}`)}return p}function L(s,d){const t=d??new f(4);return t[0]=s[0],t[1]=s[1],t[2]=s[2],t[3]=s[3],t}const an=L;function Ln(s,d,t){const o=t??new f(4);return o[0]=s[0]+d[0],o[1]=s[1]+d[1],o[2]=s[2]+d[2],o[3]=s[3]+d[3],o}function mn(s,d,t){const o=t??new f(4);return o[0]=s[0]-d[0],o[1]=s[1]-d[1],o[2]=s[2]-d[2],o[3]=s[3]-d[3],o}const wn=mn;function Nn(s,d,t){const o=t??new f(4);return o[0]=s[0]*d,o[1]=s[1]*d,o[2]=s[2]*d,o[3]=s[3]*d,o}const un=Nn;function En(s,d,t){const o=t??new f(4);return o[0]=s[0]/d,o[1]=s[1]/d,o[2]=s[2]/d,o[3]=s[3]/d,o}function Z(s,d){return s[0]*d[0]+s[1]*d[1]+s[2]*d[2]+s[3]*d[3]}function Tn(s,d,t,o){const l=o??new f(4);return l[0]=s[0]+t*(d[0]-s[0]),l[1]=s[1]+t*(d[1]-s[1]),l[2]=s[2]+t*(d[2]-s[2]),l[3]=s[3]+t*(d[3]-s[3]),l}function A(s){const d=s[0],t=s[1],o=s[2],l=s[3];return Math.sqrt(d*d+t*t+o*o+l*l)}const k=A;function O(s){const d=s[0],t=s[1],o=s[2],l=s[3];return d*d+t*t+o*o+l*l}const tn=O;function vn(s,d){const t=d??new f(4),o=s[0],l=s[1],p=s[2],x=s[3],n=Math.sqrt(o*o+l*l+p*p+x*x);return n>1e-5?(t[0]=o/n,t[1]=l/n,t[2]=p/n,t[3]=x/n):(t[0]=0,t[1]=0,t[2]=0,t[3]=1),t}function Hn(s,d){return Math.abs(s[0]-d[0])<U&&Math.abs(s[1]-d[1])<U&&Math.abs(s[2]-d[2])<U&&Math.abs(s[3]-d[3])<U}function _n(s,d){return s[0]===d[0]&&s[1]===d[1]&&s[2]===d[2]&&s[3]===d[3]}function Mn(s){const d=s??new f(4);return d[0]=0,d[1]=0,d[2]=0,d[3]=1,d}const N=g.create(),Dn=g.create(),ln=g.create();function Pn(s,d,t){const o=t??new f(4),l=g.dot(s,d);return l<-.999999?(g.cross(Dn,s,N),g.len(N)<1e-6&&g.cross(ln,s,N),g.normalize(N,N),I(N,Math.PI,o),o):l>.999999?(o[0]=0,o[1]=0,o[2]=0,o[3]=1,o):(g.cross(s,d,N),o[0]=N[0],o[1]=N[1],o[2]=N[2],o[3]=1+l,vn(o,o))}const fn=new f(4),Rn=new f(4);function zn(s,d,t,o,l,p){const x=p??new f(4);return pn(s,o,l,fn),pn(d,t,l,Rn),pn(fn,Rn,2*l*(1-l),x),x}return{create:S,fromValues:G,set:D,fromAxisAngle:I,toAxisAngle:B,angle:W,multiply:V,mul:q,rotateX:Q,rotateY:cn,rotateZ:yn,slerp:pn,inverse:Sn,conjugate:Gn,fromMat:Bn,fromEuler:kn,copy:L,clone:an,add:Ln,subtract:mn,sub:wn,mulScalar:Nn,scale:un,divScalar:En,dot:Z,lerp:Tn,length:A,len:k,lengthSq:O,lenSq:tn,normalize:vn,equalsApproximately:Hn,equals:_n,identity:Mn,rotationTo:Pn,sqlerp:zn}}const Ae=new Map;function mt(f){let g=Ae.get(f);return g||(g=yt(f),Ae.set(f,g)),g}function Mt(f){function g(t,o,l,p){const x=new f(4);return t!==void 0&&(x[0]=t,o!==void 0&&(x[1]=o,l!==void 0&&(x[2]=l,p!==void 0&&(x[3]=p)))),x}const S=g;function G(t,o,l,p,x){const n=x??new f(4);return n[0]=t,n[1]=o,n[2]=l,n[3]=p,n}function D(t,o){const l=o??new f(4);return l[0]=Math.ceil(t[0]),l[1]=Math.ceil(t[1]),l[2]=Math.ceil(t[2]),l[3]=Math.ceil(t[3]),l}function I(t,o){const l=o??new f(4);return l[0]=Math.floor(t[0]),l[1]=Math.floor(t[1]),l[2]=Math.floor(t[2]),l[3]=Math.floor(t[3]),l}function B(t,o){const l=o??new f(4);return l[0]=Math.round(t[0]),l[1]=Math.round(t[1]),l[2]=Math.round(t[2]),l[3]=Math.round(t[3]),l}function W(t,o=0,l=1,p){const x=p??new f(4);return x[0]=Math.min(l,Math.max(o,t[0])),x[1]=Math.min(l,Math.max(o,t[1])),x[2]=Math.min(l,Math.max(o,t[2])),x[3]=Math.min(l,Math.max(o,t[3])),x}function V(t,o,l){const p=l??new f(4);return p[0]=t[0]+o[0],p[1]=t[1]+o[1],p[2]=t[2]+o[2],p[3]=t[3]+o[3],p}function q(t,o,l,p){const x=p??new f(4);return x[0]=t[0]+o[0]*l,x[1]=t[1]+o[1]*l,x[2]=t[2]+o[2]*l,x[3]=t[3]+o[3]*l,x}function Q(t,o,l){const p=l??new f(4);return p[0]=t[0]-o[0],p[1]=t[1]-o[1],p[2]=t[2]-o[2],p[3]=t[3]-o[3],p}const cn=Q;function yn(t,o){return Math.abs(t[0]-o[0])<U&&Math.abs(t[1]-o[1])<U&&Math.abs(t[2]-o[2])<U&&Math.abs(t[3]-o[3])<U}function pn(t,o){return t[0]===o[0]&&t[1]===o[1]&&t[2]===o[2]&&t[3]===o[3]}function Sn(t,o,l,p){const x=p??new f(4);return x[0]=t[0]+l*(o[0]-t[0]),x[1]=t[1]+l*(o[1]-t[1]),x[2]=t[2]+l*(o[2]-t[2]),x[3]=t[3]+l*(o[3]-t[3]),x}function Gn(t,o,l,p){const x=p??new f(4);return x[0]=t[0]+l[0]*(o[0]-t[0]),x[1]=t[1]+l[1]*(o[1]-t[1]),x[2]=t[2]+l[2]*(o[2]-t[2]),x[3]=t[3]+l[3]*(o[3]-t[3]),x}function Bn(t,o,l){const p=l??new f(4);return p[0]=Math.max(t[0],o[0]),p[1]=Math.max(t[1],o[1]),p[2]=Math.max(t[2],o[2]),p[3]=Math.max(t[3],o[3]),p}function kn(t,o,l){const p=l??new f(4);return p[0]=Math.min(t[0],o[0]),p[1]=Math.min(t[1],o[1]),p[2]=Math.min(t[2],o[2]),p[3]=Math.min(t[3],o[3]),p}function L(t,o,l){const p=l??new f(4);return p[0]=t[0]*o,p[1]=t[1]*o,p[2]=t[2]*o,p[3]=t[3]*o,p}const an=L;function Ln(t,o,l){const p=l??new f(4);return p[0]=t[0]/o,p[1]=t[1]/o,p[2]=t[2]/o,p[3]=t[3]/o,p}function mn(t,o){const l=o??new f(4);return l[0]=1/t[0],l[1]=1/t[1],l[2]=1/t[2],l[3]=1/t[3],l}const wn=mn;function Nn(t,o){return t[0]*o[0]+t[1]*o[1]+t[2]*o[2]+t[3]*o[3]}function un(t){const o=t[0],l=t[1],p=t[2],x=t[3];return Math.sqrt(o*o+l*l+p*p+x*x)}const En=un;function Z(t){const o=t[0],l=t[1],p=t[2],x=t[3];return o*o+l*l+p*p+x*x}const Tn=Z;function A(t,o){const l=t[0]-o[0],p=t[1]-o[1],x=t[2]-o[2],n=t[3]-o[3];return Math.sqrt(l*l+p*p+x*x+n*n)}const k=A;function O(t,o){const l=t[0]-o[0],p=t[1]-o[1],x=t[2]-o[2],n=t[3]-o[3];return l*l+p*p+x*x+n*n}const tn=O;function vn(t,o){const l=o??new f(4),p=t[0],x=t[1],n=t[2],i=t[3],e=Math.sqrt(p*p+x*x+n*n+i*i);return e>1e-5?(l[0]=p/e,l[1]=x/e,l[2]=n/e,l[3]=i/e):(l[0]=0,l[1]=0,l[2]=0,l[3]=0),l}function Hn(t,o){const l=o??new f(4);return l[0]=-t[0],l[1]=-t[1],l[2]=-t[2],l[3]=-t[3],l}function _n(t,o){const l=o??new f(4);return l[0]=t[0],l[1]=t[1],l[2]=t[2],l[3]=t[3],l}const Mn=_n;function N(t,o,l){const p=l??new f(4);return p[0]=t[0]*o[0],p[1]=t[1]*o[1],p[2]=t[2]*o[2],p[3]=t[3]*o[3],p}const Dn=N;function ln(t,o,l){const p=l??new f(4);return p[0]=t[0]/o[0],p[1]=t[1]/o[1],p[2]=t[2]/o[2],p[3]=t[3]/o[3],p}const Pn=ln;function fn(t){const o=t??new f(4);return o[0]=0,o[1]=0,o[2]=0,o[3]=0,o}function Rn(t,o,l){const p=l??new f(4),x=t[0],n=t[1],i=t[2],e=t[3];return p[0]=o[0]*x+o[4]*n+o[8]*i+o[12]*e,p[1]=o[1]*x+o[5]*n+o[9]*i+o[13]*e,p[2]=o[2]*x+o[6]*n+o[10]*i+o[14]*e,p[3]=o[3]*x+o[7]*n+o[11]*i+o[15]*e,p}function zn(t,o,l){const p=l??new f(4);return vn(t,p),L(p,o,p)}function s(t,o,l){const p=l??new f(4);return un(t)>o?zn(t,o,p):_n(t,p)}function d(t,o,l){const p=l??new f(4);return Sn(t,o,.5,p)}return{create:g,fromValues:S,set:G,ceil:D,floor:I,round:B,clamp:W,add:V,addScaled:q,subtract:Q,sub:cn,equalsApproximately:yn,equals:pn,lerp:Sn,lerpV:Gn,max:Bn,min:kn,mulScalar:L,scale:an,divScalar:Ln,inverse:mn,invert:wn,dot:Nn,length:un,len:En,lengthSq:Z,lenSq:Tn,distance:A,dist:k,distanceSq:O,distSq:tn,normalize:vn,negate:Hn,copy:_n,clone:Mn,multiply:N,mul:Dn,divide:ln,div:Pn,zero:fn,transformMat4:Rn,setLength:zn,truncate:s,midpoint:d}}const Be=new Map;function Dt(f){let g=Be.get(f);return g||(g=Mt(f),Be.set(f,g)),g}function De(f,g,S,G,D,I){return{mat3:gt(f),mat4:ht(g),quat:mt(S),vec2:Ne(G),vec3:ge(D),vec4:Dt(I)}}const{mat3:Et,mat4:dn,quat:Tt,vec2:Rt,vec3:Ot,vec4:Ut}=De(Float32Array,Float32Array,Float32Array,Float32Array,Float32Array,Float32Array);De(Float64Array,Float64Array,Float64Array,Float64Array,Float64Array,Float64Array);De(pt,Array,Array,Array,Array,Array);const qn=.07,ue=2,le=2,fe=2,me=4e4,Me=64;function ke(f,g){let S=new ArrayBuffer(Me*f);var G=0;const D=.6;for(var I=-g.yHalf*.95;G<f;I+=D*qn)for(var B=-.95*g.xHalf;B<.95*g.xHalf&&G<f;B+=D*qn)for(var W=-.95*g.zHalf;W<0*g.zHalf&&G<f;W+=D*qn){let V=1e-4*Math.random();const q=64*G;({position:new Float32Array(S,q,3),velocity:new Float32Array(S,q+16,3),force:new Float32Array(S,q+32,3),density:new Float32Array(S,q+44,1),nearDensity:new Float32Array(S,q+48,1)}).position.set([B+V,I,W]),G++}return S}async function Pt(){const f=document.querySelector("canvas"),g=await navigator.gpu.requestAdapter();if(!g)throw new Error;const S=await g.requestDevice(),G=f.getContext("webgpu");if(!G)throw new Error;let D=1;f.width=D*f.clientWidth,f.height=D*f.clientHeight;const I=navigator.gpu.getPreferredCanvasFormat();return G.configure({device:S,format:I}),{canvas:f,device:S,presentationFormat:I,context:G}}function zt(f,g,S,G){var D=[f[0]*2-1,1-f[1]*2,0,1];D[2]=-S[4*2+2]+S[4*3+2]/g;var I=dn.multiply(G,D),B=I[3];return[I[0]/B,I[1]/B,I[2]/B]}function bt(f,g){const S=f.clientWidth/f.clientHeight,G=dn.perspective(g,S,.1,50),D=dn.lookAt([0,0,5.5],[0,0,0],[0,1,0]),I=new Float32Array(4);I[0]=1.5,I[1]=1,I[2]=2,I[3]=1;const B=dn.multiply(D,I),W=Math.abs(B[2]),V=dn.multiply(G,B),q=[V[0]/V[3],V[1]/V[3],V[2]/V[3]];console.log("camera: ",B),console.log("ndc: ",q);const Q=[(q[0]+1)/2,(1-q[1])/2],cn=dn.inverse(G),yn=zt(Q,W,G,cn);return console.log(yn),{projection:G,view:D}}function It(f,g,S,G){var D=dn.identity();dn.translate(D,G,D),dn.rotateY(D,S,D),dn.rotateX(D,g,D),dn.translate(D,[0,0,f],D);var I=dn.multiply(D,[0,0,0,1]);return dn.lookAt([I[0],I[1],I[2]],G,[0,1,0])}const Fe=.04,Le=2*Fe;async function St(){const{canvas:f,device:g,presentationFormat:S,context:G}=await Pt();G.configure({device:g,format:S,alphaMode:"premultiplied"});const D=g.createShaderModule({code:et}),I=g.createShaderModule({code:Qe}),B=g.createShaderModule({code:Je}),W=g.createShaderModule({code:Ce}),V=g.createShaderModule({code:nt}),q=g.createShaderModule({code:st}),Q=g.createShaderModule({code:tt}),cn=g.createShaderModule({code:rt}),yn=g.createShaderModule({code:Xe}),pn=g.createShaderModule({code:Ze}),Sn=g.createShaderModule({code:$e}),Gn=g.createShaderModule({code:it}),Bn=g.createShaderModule({code:ot}),kn=g.createShaderModule({code:ct}),L={kernelRadius:qn,kernelRadiusPow2:Math.pow(qn,2),kernelRadiusPow4:Math.pow(qn,4),kernelRadiusPow5:Math.pow(qn,5),kernelRadiusPow6:Math.pow(qn,6),kernelRadiusPow9:Math.pow(qn,9),stiffness:20,nearStiffness:1,mass:1,restDensity:15e3,viscosity:100,dt:.006,xHalf:ue,yHalf:le,zHalf:fe},an=g.createRenderPipeline({label:"circles pipeline",layout:"auto",vertex:{module:I},fragment:{module:I,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),Ln=g.createRenderPipeline({label:"ball pipeline",layout:"auto",vertex:{module:q},fragment:{module:q,targets:[{format:S}]},primitive:{topology:"triangle-list"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth32float"}}),mn=90*Math.PI/180,{projection:wn,view:Nn}=bt(f,mn),un={screenHeight:f.height,screenWidth:f.width},En={depth_threshold:Fe*10,max_filter_size:100,projected_particle_constant:10*Le*.05*(f.height/2)/Math.tan(mn/2)},Z=g.createRenderPipeline({label:"filter pipeline",layout:"auto",vertex:{module:D,constants:un},fragment:{module:W,constants:En,targets:[{format:"r32float"}]},primitive:{topology:"triangle-list"}}),Tn=g.createRenderPipeline({label:"fluid rendering pipeline",layout:"auto",vertex:{module:D,constants:un},fragment:{module:V,targets:[{format:S}]},primitive:{topology:"triangle-list",cullMode:"none"}}),A=g.createRenderPipeline({label:"thickness pipeline",layout:"auto",vertex:{module:Q},fragment:{module:Q,targets:[{format:"r16float",writeMask:GPUColorWrite.RED,blend:{color:{operation:"add",srcFactor:"one",dstFactor:"one"},alpha:{operation:"add",srcFactor:"one",dstFactor:"one"}}}]},primitive:{topology:"triangle-list",cullMode:"none"}}),k=g.createRenderPipeline({label:"thickness filter pipeline",layout:"auto",vertex:{module:D,constants:un},fragment:{module:cn,targets:[{format:"r16float"}]},primitive:{topology:"triangle-list"}}),O=g.createRenderPipeline({label:"show pipeline",layout:"auto",vertex:{module:D,constants:un},fragment:{module:B,targets:[{format:S}]},primitive:{topology:"triangle-list",cullMode:"none"}}),tn=1*qn,vn=2*ue,Hn=2*le,_n=2*fe,Mn=4*tn,N=Math.ceil((vn+Mn)/tn),Dn=Math.ceil((Hn+Mn)/tn),ln=Math.ceil((_n+Mn)/tn),Pn=N*Dn*ln,fn=Mn/2,Rn={xHalf:ue,yHalf:le,zHalf:fe,gridCount:Pn,xGrids:N,yGrids:Dn,cellSize:tn,offset:fn},zn=g.createComputePipeline({label:"grid clear pipeline",layout:"auto",compute:{module:Bn}}),s=g.createComputePipeline({label:"grid build pipeline",layout:"auto",compute:{module:Gn,constants:Rn}}),d=g.createComputePipeline({label:"reorder pipeline",layout:"auto",compute:{module:kn,constants:Rn}}),t=g.createComputePipeline({label:"density pipeline",layout:"auto",compute:{module:yn,constants:{kernelRadius:L.kernelRadius,kernelRadiusPow2:L.kernelRadiusPow2,kernelRadiusPow5:L.kernelRadiusPow5,kernelRadiusPow6:L.kernelRadiusPow6,mass:L.mass,xHalf:ue,yHalf:le,zHalf:fe,xGrids:N,yGrids:Dn,zGrids:ln,cellSize:tn,offset:fn}}}),o=g.createComputePipeline({label:"force pipeline",layout:"auto",compute:{module:pn,constants:{kernelRadius:L.kernelRadius,kernelRadiusPow2:L.kernelRadiusPow2,kernelRadiusPow5:L.kernelRadiusPow5,kernelRadiusPow6:L.kernelRadiusPow6,kernelRadiusPow9:L.kernelRadiusPow9,mass:L.mass,stiffness:L.stiffness,nearStiffness:L.nearStiffness,viscosity:L.viscosity,restDensity:L.restDensity,xHalf:ue,yHalf:le,zHalf:fe,xGrids:N,yGrids:Dn,zGrids:ln,cellSize:tn,offset:fn}}}),l=g.createComputePipeline({label:"integrate pipeline",layout:"auto",compute:{module:Sn,constants:{dt:L.dt}}}),x=g.createTexture({label:"depth map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),i=g.createTexture({label:"temporary texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r32float"}).createView(),r=g.createTexture({label:"thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),v=g.createTexture({label:"temporary thickness map texture",size:[f.width,f.height,1],usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING,format:"r16float"}).createView(),y=g.createTexture({size:[f.width,f.height,1],format:"depth32float",usage:GPUTextureUsage.RENDER_ATTACHMENT});y.createView();let c;{const An=["cubemap/posx.png","cubemap/negx.png","cubemap/posy.png","cubemap/negy.png","cubemap/posz.png","cubemap/negz.png"].map(async Vn=>{const Kn=await fetch(Vn);return createImageBitmap(await Kn.blob())}),hn=await Promise.all(An);console.log(hn[0].width,hn[0].height),c=g.createTexture({dimension:"2d",size:[hn[0].width,hn[0].height,6],format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT});for(let Vn=0;Vn<hn.length;Vn++){const Kn=hn[Vn];g.queue.copyExternalImageToTexture({source:Kn},{texture:c,origin:[0,0,Vn]},[Kn.width,Kn.height])}}const u=c.createView({dimension:"cube"}),w=g.createSampler({magFilter:"linear",minFilter:"linear"}),_=new ArrayBuffer(144),h={size:new Float32Array(_,0,1),view_matrix:new Float32Array(_,16,16),projection_matrix:new Float32Array(_,80,16)},m=new ArrayBuffer(8),M=new ArrayBuffer(8),b={blur_dir:new Float32Array(m)},E={blur_dir:new Float32Array(M)};b.blur_dir.set([1,0]),E.blur_dir.set([0,1]);const P=new ArrayBuffer(272),z={texel_size:new Float32Array(P,0,2),inv_projection_matrix:new Float32Array(P,16,16),projection_matrix:new Float32Array(P,80,16),view_matrix:new Float32Array(P,144,16),inv_view_matrix:new Float32Array(P,208,16)};z.texel_size.set([1/f.width,1/f.height]),z.projection_matrix.set(wn);const T=dn.identity(),R=dn.identity();dn.inverse(wn,T),dn.inverse(Nn,R),z.inv_projection_matrix.set(T),z.inv_view_matrix.set(R);const F=new ArrayBuffer(12),Y={xHalf:new Float32Array(F,0,1),yHalf:new Float32Array(F,4,1),zHalf:new Float32Array(F,8,1)},H=g.createBuffer({label:"particles buffer",size:Me*me,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),K=g.createBuffer({label:"cell particle count buffer",size:4*(Pn+1),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),J=g.createBuffer({label:"target particles buffer",size:Me*me,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),rn=g.createBuffer({label:"particle cell offset buffer",size:4*me,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),$=g.createBuffer({label:"uniform buffer",size:_.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),C=g.createBuffer({label:"filter uniform buffer",size:m.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),nn=g.createBuffer({label:"filter uniform buffer",size:M.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),sn=g.createBuffer({label:"filter uniform buffer",size:P.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),on=g.createBuffer({label:"real box size buffer",size:F.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});g.queue.writeBuffer(C,0,m),g.queue.writeBuffer(nn,0,M),g.queue.writeBuffer(sn,0,P),g.queue.writeBuffer(on,0,F);const gn=g.createBindGroup({layout:zn.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}}]}),xn=g.createBindGroup({layout:s.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:K}},{binding:1,resource:{buffer:rn}},{binding:2,resource:{buffer:H}}]}),bn=g.createBindGroup({layout:d.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}},{binding:3,resource:{buffer:rn}}]}),On=g.createBindGroup({layout:t.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Un=g.createBindGroup({layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:J}},{binding:2,resource:{buffer:K}}]}),Jn=g.createBindGroup({layout:l.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:on}}]}),Cn=g.createBindGroup({label:"ball bind group",layout:Ln.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:$}}]}),ne=g.createBindGroup({label:"circle bind group",layout:an.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:$}}]});g.createBindGroup({label:"show bind group",layout:O.getBindGroupLayout(0),entries:[{binding:1,resource:r}]});const Yn=[g.createBindGroup({label:"filterX bind group",layout:Z.getBindGroupLayout(0),entries:[{binding:1,resource:x},{binding:2,resource:{buffer:C}}]}),g.createBindGroup({label:"filterY bind group",layout:Z.getBindGroupLayout(0),entries:[{binding:1,resource:i},{binding:2,resource:{buffer:nn}}]})],ee=g.createBindGroup({label:"fluid bind group",layout:Tn.getBindGroupLayout(0),entries:[{binding:0,resource:w},{binding:1,resource:x},{binding:2,resource:{buffer:sn}},{binding:3,resource:r},{binding:4,resource:u}]}),te=g.createBindGroup({label:"thickness bind group",layout:A.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:H}},{binding:1,resource:{buffer:$}}]}),Xn=[g.createBindGroup({label:"thickness filterX bind group",layout:k.getBindGroupLayout(0),entries:[{binding:1,resource:r},{binding:2,resource:{buffer:C}}]}),g.createBindGroup({label:"thickness filterY bind group",layout:k.getBindGroupLayout(0),entries:[{binding:1,resource:v},{binding:2,resource:{buffer:nn}}]})];let Zn=!1,re=0,se=0,oe=Math.PI/4,j=-Math.PI/12;const Pe=.005,ze=-.99*Math.PI/2,be=0;let xe=1,ce=1;const de=[{MIN_DISTANCE:1.3,MAX_DISTANCE:3,INIT_DISTANCE:1.6},{MIN_DISTANCE:1.8,MAX_DISTANCE:3,INIT_DISTANCE:2.1},{MIN_DISTANCE:2,MAX_DISTANCE:3,INIT_DISTANCE:2.3},{MIN_DISTANCE:2.3,MAX_DISTANCE:3,INIT_DISTANCE:2.7}];let $n=de[ce].INIT_DISTANCE;const Ie=document.getElementById("fluidCanvas");Ie.addEventListener("mousedown",In=>{Zn=!0,re=In.clientX,se=In.clientY}),Ie.addEventListener("wheel",In=>{In.preventDefault();var An=In.deltaY;$n+=(An>0?1:-1)*.05;const hn=de[ce];$n<hn.MIN_DISTANCE&&($n=hn.MIN_DISTANCE),$n>hn.MAX_DISTANCE&&($n=hn.MAX_DISTANCE)}),document.addEventListener("mousemove",In=>{if(Zn){const An=In.clientX,hn=In.clientY,Vn=re-An,Kn=se-hn;oe+=Pe*Vn,j+=Pe*Kn,j>be&&(j=be),j<ze&&(j=ze),re=An,se=hn}}),document.addEventListener("mouseup",()=>{Zn&&(Zn=!1)});let He=document.getElementById("error-reason");g.lost.then(In=>{const An=In.reason?`reason: ${In.reason}`:"unknown reason";He.textContent=An});let Ve=document.getElementById("number-button"),he=!1,pe="";Ve.addEventListener("change",function(In){const An=In.target;(An==null?void 0:An.name)==="options"&&(he=!0,pe=An.value)});let ie=new Map;ie.set("10000",{xHalf:.7,yHalf:2,zHalf:.7}),ie.set("20000",{xHalf:1,yHalf:2,zHalf:1}),ie.set("30000",{xHalf:1.2,yHalf:2,zHalf:1.2}),ie.set("40000",{xHalf:1.4,yHalf:2,zHalf:1.4});let en={boxSize:ie.get("20000")??{xHalf:1,yHalf:2,zHalf:1},numParticles:2e4};const We=ke(en.numParticles,en.boxSize);g.queue.writeBuffer(H,0,We);async function Se(){if(performance.now(),he){console.log(pe),en.boxSize=ie.get(pe)??{xHalf:.8,yHalf:2,zHalf:.8},en.numParticles=parseInt(pe),oe=Math.PI/4,j=-Math.PI/12;const Fn=ke(en.numParticles,en.boxSize);g.queue.writeBuffer(H,0,Fn),ce=en.numParticles/1e4-1,$n=de[ce].INIT_DISTANCE;let Qn=document.getElementById("slider");Qn.value="100",console.log(de[ce]),he=!1}const In={colorAttachments:[{view:x,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},An={colorAttachments:[{view:G.getCurrentTexture().createView(),clearValue:{r:.7,g:.7,b:.7,a:1},loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:y.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}},hn=[{colorAttachments:[{view:i,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:x,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];G.getCurrentTexture().createView();const Vn={colorAttachments:[{view:G.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Kn={colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},Ge=[{colorAttachments:[{view:v,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]},{colorAttachments:[{view:r,clearValue:{r:0,g:0,b:0,a:1},loadOp:"clear",storeOp:"store"}]}];h.size.set([Le]),h.projection_matrix.set(wn);const ye=It($n,j,oe,[0,-en.boxSize.yHalf,0]);h.view_matrix.set(ye),z.view_matrix.set(ye),dn.inverse(ye,R),z.inv_view_matrix.set(R),g.queue.writeBuffer($,0,_),g.queue.writeBuffer(sn,0,P);const qe=document.getElementById("slider"),Ke=document.getElementById("slider-value"),je=document.getElementById("particle");let Ee=parseInt(qe.value)/200+.5;const Ye=Math.max(Ee-xe,-.01);xe+=Ye,Ke.textContent=Ee.toFixed(2),Y.xHalf.set([en.boxSize.xHalf]),Y.yHalf.set([en.boxSize.yHalf]),Y.zHalf.set([en.boxSize.zHalf*xe]),g.queue.writeBuffer(on,0,F);const Wn=g.createCommandEncoder(),X=Wn.beginComputePass();for(let Fn=0;Fn<2;Fn++)X.setBindGroup(0,gn),X.setPipeline(zn),X.dispatchWorkgroups(Math.ceil((Pn+1)/64)),X.setBindGroup(0,xn),X.setPipeline(s),X.dispatchWorkgroups(Math.ceil(en.numParticles/64)),new ft({device:g,data:K,count:Pn+1}).dispatch(X),X.setBindGroup(0,bn),X.setPipeline(d),X.dispatchWorkgroups(Math.ceil(en.numParticles/64)),X.setBindGroup(0,On),X.setPipeline(t),X.dispatchWorkgroups(Math.ceil(en.numParticles/64)),X.setBindGroup(0,bn),X.setPipeline(d),X.dispatchWorkgroups(Math.ceil(en.numParticles/64)),X.setBindGroup(0,Un),X.setPipeline(o),X.dispatchWorkgroups(Math.ceil(en.numParticles/64)),X.setBindGroup(0,Jn),X.setPipeline(l),X.dispatchWorkgroups(Math.ceil(en.numParticles/64));if(X.end(),je.checked){const Fn=Wn.beginRenderPass(An);Fn.setBindGroup(0,Cn),Fn.setPipeline(Ln),Fn.draw(6,en.numParticles),Fn.end()}else{const Fn=Wn.beginRenderPass(In);Fn.setBindGroup(0,ne),Fn.setPipeline(an),Fn.draw(6,en.numParticles),Fn.end();for(var ae=0;ae<4;ae++){const ve=Wn.beginRenderPass(hn[0]);ve.setBindGroup(0,Yn[0]),ve.setPipeline(Z),ve.draw(6),ve.end();const jn=Wn.beginRenderPass(hn[1]);jn.setBindGroup(0,Yn[1]),jn.setPipeline(Z),jn.draw(6),jn.end()}const Qn=Wn.beginRenderPass(Kn);Qn.setBindGroup(0,te),Qn.setPipeline(A),Qn.draw(6,en.numParticles),Qn.end();for(var ae=0;ae<1;ae++){const jn=Wn.beginRenderPass(Ge[0]);jn.setBindGroup(0,Xn[0]),jn.setPipeline(k),jn.draw(6),jn.end();const _e=Wn.beginRenderPass(Ge[1]);_e.setBindGroup(0,Xn[1]),_e.setPipeline(k),_e.draw(6),_e.end()}const we=Wn.beginRenderPass(Vn);we.setBindGroup(0,ee),we.setPipeline(Tn),we.draw(6),we.end()}g.queue.submit([Wn.finish()]),performance.now(),requestAnimationFrame(Se)}requestAnimationFrame(Se)}St();
