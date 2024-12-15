struct VertexOutput {
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

// HSVをRGBに変換する関数
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let i = floor(h * 6.0); // i は整数部
    let f = h * 6.0 - i; // f は小数部
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);

    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;

    // 色相に基づく条件分岐
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

    return vec3f(r, g, b); // RGBA形式で返す
}

// speed に基づいて色を計算する関数
fn get_color_by_speed(speed: f32, max_speed: f32) -> vec3f {
    // speed を -max_speed から max_speed の範囲で正規化
    let normalized_speed = clamp(abs(speed) / max_speed, 0.0, 1.0);
    // 色相を計算
    let hue = (1.0 - normalized_speed) * 0.7;
    let saturation = 1.0;
    let value = 1.0;

    return hsv_to_rgb(hue, saturation, value);
}

fn value_to_color(value: f32, min: f32, max: f32) -> vec3<f32> {
    // 入力値を0～1に正規化
    let normalized = (value - min) / (max - min);
    
    // 色を虹色に変化させる
    let r = normalized;
    let g = (normalized - 0.5);
    let b = (1.0 - normalized);
    
    return vec3<f32>(r, g, b); // RGB色を返す
}

fn value_to_color2(value: f32) -> vec3<f32> {
    // let col0 = vec3f(29, 71, 158) / 256;
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

    // return mix(col1, col2, value);
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
    // var color: vec3f = get_color_by_speed(input.speed, 2.);
    // var color: vec3f = mix(vec3f(0, 0.2, 0.8), vec3f(0, 0.8, 1.0), input.speed / 2);
    var color: vec3f = value_to_color2(input.speed / 1.5);
    // var color: vec3f = vec3f(0.0, input.speed / 1.5, 1.);
    // var color: vec3f = vec3f(0.0, input.speed / 1.5, 1.);
    // var color: vec3f = value_to_color(input.speed, 0.0, 2.0);

    // ここ，負だな．やっぱり右手系なのか？
    // out.frag_color = vec4(real_view_pos.z, 0., 0., 1.);
    // out.frag_color = vec4(0.5 + 0.5 * normal, 1.);
    out.frag_color = vec4(color * diffuse, 1.);
    return out;
}