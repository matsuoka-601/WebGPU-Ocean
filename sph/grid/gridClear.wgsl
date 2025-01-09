@group(0) @binding(0) var<storage, read_write> cellParticleCount : array<u32>;

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
    // TODO : ここは変えなくていいか？
    if (id.x < arrayLength(&cellParticleCount)) {
        cellParticleCount[id.x] = 0u;
    }
}

