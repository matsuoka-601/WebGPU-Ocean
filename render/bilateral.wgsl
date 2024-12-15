// @group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

struct FragmentInput {
    @location(0) uv: vec2f,  
    @location(1) iuv: vec2f
}

override depth_threshold: f32;  // これは何？
override projected_particle_constant: f32; // これは Babylon.js で計算していたやつか．
override max_filter_size: f32;
struct FilterUniforms {
    blur_dir: vec2f, // 解像度で割る
}


@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    // 正かどうかを確かめる
    var depth: f32 = abs(textureLoad(texture, vec2u(input.iuv), 0).r);

    // ここが有効になるためには，背景の depth を適切に設定しなきゃいけないな．
    if (depth >= 1e4 || depth <= 0.) {
        return vec4f(vec3f(depth), 1.);
    }

    // depth は正か？
    var filter_size: i32 = min(i32(max_filter_size), i32(ceil(projected_particle_constant / depth)));

    // ここのパラメータ設定がよくわからない
    var sigma: f32 = f32(filter_size) / 3.0; 
    var two_sigma: f32 = 2.0 * sigma * sigma;
    var sigma_depth: f32 = depth_threshold / 3.0;
    var two_sigma_depth: f32 = 2.0 * sigma_depth * sigma_depth;

    var sum: f32 = 0.0;
    var wsum: f32 = 0.0;
    for (var x: i32 = -filter_size; x <= filter_size; x++) {
        var coords: vec2f = vec2f(f32(x));
        var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * uniforms.blur_dir), 0).r);
        // sampled_depth = (depth + sampled_depth) / 2;

        var rr: f32 = dot(coords, coords);
        var w: f32 = exp(-rr / two_sigma);

        var r_depth: f32 = sampled_depth - depth;
        var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);
        sum += sampled_depth * w * wd;
        wsum += w * wd;
    }

    // for (var x: i32 = -filter_size; x <= filter_size; x++) {
    //     for (var y: i32 = -filter_size; y <= filter_size; y++) {
    //         var coords: vec2f = vec2f(f32(x), f32(y));
    //         var sampled_depth: f32 = abs(textureLoad(texture, vec2u(input.iuv + coords * vec2f(1.0, 1.0)), 0).r);

    //         var rr: f32 = dot(coords, coords);
    //         var w: f32 = exp(-rr / two_sigma);

    //         var r_depth: f32 = sampled_depth - depth;
    //         var wd: f32 = exp(-r_depth * r_depth / two_sigma_depth);

    //         sum += sampled_depth * w * wd;
    //         wsum += w * wd;
    //     }
    // }

    sum /= wsum;
    // if (wsum > 0.) {
    //  sum /= wsum;
    // }

    return vec4f(sum, 0., 0., 1.);
}
