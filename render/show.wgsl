// @group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;

struct FragmentInput {
    @location(0) uv: vec2f, 
    @location(1) iuv: vec2f, 
}

@fragment
fn fs(input: FragmentInput) -> @location(0) vec4f {
    var r = abs(textureLoad(texture, vec2u(input.iuv), 0).r);
    // r *= 0.15;
    return vec4(0, r, r, 1.0);
}