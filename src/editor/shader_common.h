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
    float agentHeight;
    float eyeHeight;
    float torsoHeight;
    uint32_t sqrtOffsetSamples;
    float offsetRadius;
    int numVoxelTests;
    int numVoxels;
};

struct CandidatePair {
    vec3 origin;
    uint32_t voxelID;
    vec3 hitPos;
    float pad2;
};

struct GPUAABB {
    float pMinX;
    float pMinY;
    float pMinZ;
    float pMaxX;
    float pMaxY;
    float pMaxZ;
    float aabbMinY;
    float aabbMaxY;
};

#endif
