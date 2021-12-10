#ifndef RLPBR_EDITOR_COVER_SAMPLER_GLSL_INCLUDED
#define RLPBR_EDITOR_COVER_SAMPLER_GLSL_INCLUDED

uint32_t samplerSeedHash(uint32_t v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;

    return v;
}

struct Sampler {
    uint32_t v;
};

Sampler makeSampler(uint32_t idx,
                    uint32_t base_frame_idx)
{
    Sampler rng;

    rng.v = idx;

    uint32_t v1 = samplerSeedHash(base_frame_idx);
    uint32_t s0 = 0;

    for (int n = 0; n < 4; n++) {
        s0 += 0x9e3779b9;
        rng.v += ((v1 << 4) + 0xa341316c) ^
            (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((rng.v << 4) + 0xad90777d) ^
            (rng.v + s0) ^ ((rng.v >> 5) + 0x7e95761e);
    }

    return rng;
}

float samplerGet1D(inout Sampler rng)
{
    const uint32_t LCG_A = 1664525u;
    const uint32_t LCG_C = 1013904223u;
    rng.v = (LCG_A * rng.v + LCG_C);
    uint32_t next = rng.v & 0x00FFFFFF;

    return float(next) / float(0x01000000);
}

#endif
