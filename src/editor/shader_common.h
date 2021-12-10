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
    uint32_t originIdx;
    uint32_t voxelStartIdx;
    uint32_t voxelsPerDispatch;
    uint32_t pointsPerVoxel;
    uint32_t numPoints;
    uint32_t numGroundSamples;
    uint32_t sqrtSearchSamples;
    uint32_t sqrtSphereSamples;
    float agentHeight;
    float searchRadius;
    float cornerEpsilon;
};

struct CandidatePair {
    vec3 origin;
    float pad1;
    vec3 candidate;
    float pad2;
};

struct GPUAABB {
    float pMinX;
    float pMinY;
    float pMinZ;
    float pMaxX;
    float pMaxY;
    float pMaxZ;
};

#endif
