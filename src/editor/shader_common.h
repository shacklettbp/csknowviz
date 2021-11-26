#ifndef EDITOR_SHADER_COMMON_H_INCLUDED
#define EDITOR_SHADER_COMMON_H_INCLUDED

struct DrawPushConst {
    mat4 proj;
    mat4 view;
};

struct NavmeshPushConst {
    DrawPushConst base;
    vec4 color;
};

struct CoverPushConst {
    uint32_t idxOffset;
    uint32_t numGroundSamples;
    uint32_t sqrtSphereSamples;
    float agentHeight;
    float originJitter;
    float cornerEpsilon;
};

struct CandidatePair {
    vec3 origin;
    vec3 candidate;
    vec2 pad;
};

#endif
