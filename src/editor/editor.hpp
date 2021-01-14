#pragma once

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

struct CoverData {
    std::optional<NavmeshData> navmesh;
    bool showNavmesh = false;
    std::vector<AABB> coverAABBs;
    bool showCover = false;
};

struct EditorScene {
    std::string scenePath;
    std::shared_ptr<Scene> hdl;
    vk::TLAS tlas;

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
