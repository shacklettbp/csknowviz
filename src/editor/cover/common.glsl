#ifndef EDITOR_COVER_COMMON_H_INCLUDED
#define EDITOR_COVER_COMMON_H_INCLUDED

#define SHADER_CONST const
#include "rlpbr_core/device.h"
#undef SHADER_CONST

#include "shader_common.h"
#include "vulkan/shaders/utils.glsl"

#define ISTHREAD0 (gl_GlobalInvocationID.x == 0 && \
                   gl_GlobalInvocationID.y == 0 && \
                   gl_GlobalInvocationID.z == 0)

struct PackedVertex {
    vec4 data[2];
};

struct PackedMeshInfo {
    u32vec4 data;
};

layout (set = 1, binding = 0) readonly buffer Vertices {
    PackedVertex vertices[];
};

layout (set = 1, binding = 1, scalar) readonly buffer Indices {
    uint32_t indices[];
};

layout (set = 1, binding = 2) readonly buffer MeshInfos {
    PackedMeshInfo meshInfos[];
};

layout (set = 2, binding = 0)
    uniform accelerationStructureEXT tlas;

struct Vertex {
    vec3 position;
    vec2 uv;
};

struct Triangle {
    Vertex a;
    Vertex b;
    Vertex c;
};

struct MeshInfo {
    uint32_t indexOffset;
};

Vertex unpackVertex(uint32_t idx)
{
    PackedVertex packed = vertices[nonuniformEXT(idx)];

    vec4 a = packed.data[0];
    vec4 b = packed.data[1];

    Vertex vert;
    vert.position = vec3(a.x, a.y, a.z);
    vert.uv = vec2(b.z, b.w);

    return vert;
}

MeshInfo unpackMeshInfo(uint32_t mesh_idx)
{
    MeshInfo mesh_info;
    mesh_info.indexOffset = meshInfos[nonuniformEXT(mesh_idx)].data.x;

    return mesh_info;
}


u32vec3 fetchTriangleIndices(uint32_t index_offset)
{
    // FIXME: maybe change all this to triangle offset
    return u32vec3(
        indices[nonuniformEXT(index_offset)],
        indices[nonuniformEXT(index_offset + 1)],
        indices[nonuniformEXT(index_offset + 2)]);
}

Triangle fetchTriangle(uint32_t index_offset)
{
    u32vec3 indices = fetchTriangleIndices(index_offset);

    return Triangle(
        unpackVertex(indices.x),
        unpackVertex(indices.y),
        unpackVertex(indices.z));
}

bool traceRay(in rayQueryEXT ray_query,
              in vec3 ray_origin,
              in vec3 ray_dir,
              in uint32_t mask,
              in float dist)
{
    rayQueryInitializeEXT(ray_query, tlas,
                          gl_RayFlagsTerminateOnFirstHitEXT, mask,
                          ray_origin, 0.f, ray_dir, dist);

    while (rayQueryProceedEXT(ray_query)) {
        if (rayQueryGetIntersectionTypeEXT(ray_query, false) ==
            gl_RayQueryCandidateIntersectionTriangleEXT) {
            rayQueryConfirmIntersectionEXT(ray_query);
        }
    }

    subgroupBarrier();

    return rayQueryGetIntersectionTypeEXT(ray_query, true) !=
        gl_RayQueryCommittedIntersectionNoneEXT;
}

vec3 sphereDir(vec2 uv)
{
    // Compute radius r (branchless).
    uv = 2.f * uv - 1.f;
    float d = 1.f - (abs(uv.x) + abs(uv.y));
    float r = 1.f - abs(d);
    
    // Compute phi in the first quadrant (branchless, except for the
    // division-by-zero test), using sign(u) to map the result to the
    // correct quadrant below.
    float phi = (r == 0) ? 0 : (M_PI / 4.f) * 
        ((abs(uv.y) - abs(uv.x)) / r + 1);

    float f = r * sqrt(2.f - r*r);

    float x = f * sign(uv.x) * cos(phi);
    float y = f * sign(uv.y) * sin(phi);
    float z = sign(d) * (1.f - r*r);

    return vec3(x, y, z);
}

#define INTERPOLATE_ATTR(a, b, c, barys) \
    (a + barys.x * (b - a) + \
     barys.y * (c - a))

vec3 interpolatePosition(vec3 a, vec3 b, vec3 c, vec2 barys)
{
    return INTERPOLATE_ATTR(a, b, c, barys);
}

#undef INTERPOLATE_ATTR

#define MAKE_HIT_PARAMS(suffix, committed)                                    \
    void getHitParams##suffix(in rayQueryEXT ray_query,                       \
                              out vec2 barys,                                 \
                              out uint32_t tri_idx,                           \
                              out uint32_t geo_idx,                           \
                              out uint32_t mesh_offset,                       \
                              out mat4x3 o2w)                                 \
{                                                                             \
    barys = rayQueryGetIntersectionBarycentricsEXT(ray_query, committed);     \
                                                                              \
    tri_idx = uint32_t(rayQueryGetIntersectionPrimitiveIndexEXT(              \
            ray_query, committed));                                           \
                                                                              \
    geo_idx = uint32_t(rayQueryGetIntersectionGeometryIndexEXT(               \
            ray_query, committed));                                           \
                                                                              \
    mesh_offset = uint32_t(                                                   \
        rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(     \
            ray_query, committed));                                           \
                                                                              \
    o2w = rayQueryGetIntersectionObjectToWorldEXT(ray_query, committed);      \
}

MAKE_HIT_PARAMS(Committed, true)
MAKE_HIT_PARAMS(Uncommitted, false)
#undef MAKE_HIT_PARAMS

#define MAKE_WORLD_SPACE_HIT(suffix) \
    vec3 getWorldSpaceHit##suffix(in rayQueryEXT ray_query)                   \
    {                                                                         \
        vec2 barys;                                                           \
        uint32_t tri_idx, geo_idx, mesh_offset;                               \
        mat4x3 o2w;                                                           \
        getHitParams##suffix(ray_query, barys, tri_idx, geo_idx,              \
                             mesh_offset, o2w);                               \
                                                                              \
        MeshInfo mesh_info = unpackMeshInfo(mesh_offset + geo_idx);           \
                                                                              \
        uint32_t index_offset = mesh_info.indexOffset + tri_idx * 3;          \
        Triangle hit_tri = fetchTriangle(index_offset);                       \
                                                                              \
        vec3 obj_position = interpolatePosition(hit_tri.a.position,           \
            hit_tri.b.position, hit_tri.c.position, barys);                   \
                                                                              \
        return transformPosition(o2w, obj_position);                          \
    }

MAKE_WORLD_SPACE_HIT(Committed)
MAKE_WORLD_SPACE_HIT(Uncommitted)
#undef MAKE_WORLD_SPACE_HIT

void getWorldSpaceHitParams(in rayQueryEXT ray_query,
                            out vec3 world_pos,
                            out vec3 geo_normal)
{
    vec2 barys;
    uint32_t tri_idx, geo_idx, mesh_offset;
    mat4x3 o2w;
    getHitParamsCommitted(ray_query, barys, tri_idx, geo_idx,
                          mesh_offset, o2w);

    MeshInfo mesh_info = unpackMeshInfo(mesh_offset + geo_idx);

    uint32_t index_offset = mesh_info.indexOffset + tri_idx * 3;
    Triangle hit_tri = fetchTriangle(index_offset);

    vec3 obj_position = interpolatePosition(hit_tri.a.position,
        hit_tri.b.position, hit_tri.c.position, barys);

    world_pos = transformPosition(o2w, obj_position);

    vec3 obj_geo_normal = cross(
        hit_tri.c.position - hit_tri.a.position,
        hit_tri.b.position - hit_tri.a.position);

    mat4x3 w2o = rayQueryGetIntersectionWorldToObjectEXT(ray_query, true);
    geo_normal = transformNormal(w2o, obj_geo_normal);
}


#endif
