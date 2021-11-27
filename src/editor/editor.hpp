#pragma once

#include <set>
#include "renderer.hpp"
#include "utils.hpp"
#include "json.hpp"

namespace RLpbr {
namespace editor {

struct AreaLight {
    std::array<glm::vec3, 4> vertices;
    glm::vec3 translate;
};

struct NavmeshData {
    std::vector<AABB> aabbs;
    std::vector<OverlayVertex> overlayVerts;
    std::vector<uint32_t> overlayIdxs;
};


struct compareVec
{
    bool operator() (const glm::vec3& lhs, const glm::vec3& rhs) const
    {
        return (lhs.x < rhs.x) || 
            (lhs.x == rhs.x && lhs.y < rhs.y) ||
            (lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
    }
};

struct compareAABB
{
    compareVec c;
    bool operator() (const AABB& lhs, const AABB& rhs) const
    {
        if (glm::all(glm::equal(lhs.pMin, rhs.pMin))) {
            return c(lhs.pMax, rhs.pMax);
        }
        else {
            return c(lhs.pMin, rhs.pMin);
        }
    }
};

struct CoverResults {
    std::set<AABB, compareAABB> aabbs;
    std::vector<OverlayVertex> overlayVerts;
    std::vector<uint32_t> overlayIdxs;
};

struct CoverData {
    std::optional<NavmeshData> navmesh;
    bool showNavmesh = false;
    std::map<glm::vec3, CoverResults, compareVec> results;
    bool showCover = false;
    float sampleSpacing = 1.f;
    float agentHeight = 72.f;
    int sqrtSphereSamples = 48;
    float originJitterDiv = 2.f;
    float cornerEpsilon = 0.5f;

    glm::vec3 nearestCamPoint = glm::vec3(0.f);
};

struct EditorScene {
    std::string scenePath;
    EditorVkScene hdl;

    std::vector<char> cpuData;
    PackedVertex *verts;
    uint32_t *indices;
    AABB bbox;
    uint32_t totalTriangles;

    EditorCam cam;
    Renderer::OverlayConfig overlayCfg;

    CoverData cover;
};

class Editor {
public:
    Editor(uint32_t gpu_id, uint32_t img_width, uint32_t img_height);

    void loadScene(const char *scene_name);

    void loop();

private:
    void startFrame();
    void render(EditorScene &scene, float frame_duration);

    Renderer renderer_;
    uint32_t cur_scene_idx_;
    std::vector<EditorScene> scenes_;
};

}
}
