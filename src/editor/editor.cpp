#include "editor.hpp"
#include "file_select.hpp"

#include <rlpbr/environment.hpp>
#include "rlpbr_core/utils.hpp"
#include "rlpbr_core/scene.hpp"
#include "vulkan/utils.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <vulkan/vulkan_core.h>
#include "imgui_extensions.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <queue>
#include <omp.h>
#include <chrono>
#include <optional>
#include <filesystem>
#include <cmath>
#include "contiguous.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/ext/vector_common.hpp>

#include "shader.hpp"

#define NUM_ANGLES 361
#define CLUSTER_RADIUS 5
#define INVALID_INDEX std::numeric_limits<uint64_t>::max()
#define INVALID_CLUSTER std::numeric_limits<uint32_t>::max()

using namespace std;

namespace RLpbr {
namespace editor {

namespace InternalConfig {
inline constexpr float cameraMoveSpeed = 5.f * 100.f;
inline constexpr float mouseSpeed = 2e-4f;

inline constexpr auto nsPerFrame = chrono::nanoseconds(8333333);
inline constexpr auto nsPerFrameLongWait =
    chrono::nanoseconds(7000000);
inline constexpr float secondsPerFrame =
    chrono::duration<float>(nsPerFrame).count();
}

struct SceneProperties {
    AABB aabb;
    uint32_t totalTriangles; // Post transform
};

static SceneProperties computeSceneProperties(
    const PackedVertex *vertices,
    const uint32_t *indices,
    const std::vector<ObjectInfo> &objects,
    const std::vector<MeshInfo> &meshes,
    const std::vector<ObjectInstance> &instances,
    const std::vector<InstanceTransform> &transforms)
{
    AABB bounds {
        glm::vec3(INFINITY, INFINITY, INFINITY),
        glm::vec3(-INFINITY, -INFINITY, -INFINITY),
    };

    auto updateBounds = [&bounds](const glm::vec3 &point) {
        bounds.pMin = glm::min(bounds.pMin, point);
        bounds.pMax = glm::max(bounds.pMax, point);
    };

    uint32_t total_triangles = 0;
    for (int inst_idx = 0; inst_idx < (int)instances.size(); inst_idx++) {
        const ObjectInstance &inst = instances[inst_idx];
        const InstanceTransform &txfm = transforms[inst_idx];

        const ObjectInfo &obj = objects[inst.objectIndex];
        for (int mesh_offset = 0; mesh_offset < (int)obj.numMeshes;
             mesh_offset++) {
            uint32_t mesh_idx = obj.meshIndex + mesh_offset;
            const MeshInfo &mesh = meshes[mesh_idx];

            for (int tri_idx = 0; tri_idx < (int)mesh.numTriangles; tri_idx++) {
                uint32_t base_idx = tri_idx * 3 + mesh.indexOffset;

                glm::u32vec3 tri_indices(indices[base_idx],
                                         indices[base_idx + 1],
                                         indices[base_idx + 2]);

                auto a = vertices[tri_indices.x].position;
                auto b = vertices[tri_indices.y].position;
                auto c = vertices[tri_indices.z].position;

                a = txfm.mat * glm::vec4(a, 1.f);
                b = txfm.mat * glm::vec4(b, 1.f);
                c = txfm.mat * glm::vec4(c, 1.f);

                updateBounds(a);
                updateBounds(b);
                updateBounds(c);

                total_triangles++;
            }
        }
    }

    return {
        bounds,
        total_triangles,
    };
}

void Editor::loadScene(const char *scene_name, std::filesystem::path output_path)
{
    SceneLoadData load_data = SceneLoadData::loadFromDisk(scene_name, true);
    const vector<char> &loaded_gpu_data = *get_if<vector<char>>(&load_data.data);
    vector<char> cpu_data(loaded_gpu_data);

    PackedVertex *verts = (PackedVertex *)cpu_data.data();
    assert((uintptr_t)verts % std::alignment_of_v<PackedVertex> == 0);
    uint32_t *indices =
        (uint32_t *)(cpu_data.data() + load_data.hdr.indexOffset);

    EditorVkScene render_data = 
        renderer_.loadScene(move(load_data));

    auto [scene_aabb, total_triangles] = computeSceneProperties(verts, indices,
        render_data.scene->objectInfo, render_data.scene->meshInfo,
        render_data.scene->envInit.defaultInstances,
        render_data.scene->envInit.defaultTransforms);

    EditorCam default_cam;
    default_cam.position = glm::vec3(0.f, 10.f, 0.f);
    default_cam.view = glm::vec3(0.f, -1.f, 0.f);
    default_cam.up = glm::vec3(0.f, 0.f, 1.f);
    default_cam.right = glm::cross(default_cam.view, default_cam.up);

    scenes_.emplace_back(EditorScene {
        string(scene_name),
        output_path,
        move(render_data),
        move(cpu_data),
        verts,
        indices,
        scene_aabb,
        total_triangles,
        default_cam,
        Renderer::OverlayConfig(),
        {},
    });
}

#if 0
static void updateNavmeshSettings(NavmeshConfig *cfg,
                                  const AABB &full_bbox,
                                  bool build_inprogress,
                                  bool has_navmesh,
                                  bool *build_navmesh,
                                  bool *save_navmesh,
                                  bool *load_navmesh)
{
    ImGui::Begin("Navmesh Build", nullptr, ImGuiWindowFlags_None);
    ImGui::PushItemWidth(ImGui::GetFontSize() * 10.f);

    ImGui::TextUnformatted("Agent Settings:");
    ImGui::InputFloat("Agent Height", &cfg->agentHeight);
    ImGui::InputFloat("Agent Radius", &cfg->agentRadius);
    ImGui::InputFloat("Max Slope", &cfg->maxSlope);
    ImGui::InputFloat("Max Climb", &cfg->agentMaxClimb);

    ImGui::TextUnformatted("Voxelization Settings:");
    ImGui::InputFloat("Cell Size", &cfg->cellSize);
    ImGui::InputFloat("Cell Height", &cfg->cellHeight);

    ImGui::TextUnformatted("Meshification Settings:");
    ImGui::InputFloat("Max Edge Length", &cfg->maxEdgeLen);
    ImGui::InputFloat("Max Edge Error", &cfg->maxError);
    ImGui::InputFloat("Minimum Region Size", &cfg->regionMinSize);
    ImGui::InputFloat("Region Merge Size", &cfg->regionMergeSize);
    ImGui::InputFloat("Detail Sampling Distance", &cfg->detailSampleDist);
    ImGui::InputFloat("Detail Sampling Max Error", &cfg->detailSampleMaxError);
    ImGui::PopItemWidth();

    ImGui::PushItemWidth(ImGui::GetFontSize() * 15.f);
    glm::vec3 speed(0.05f);
    ImGuiEXT::DragFloat3SeparateRange("Bounding box min",
        glm::value_ptr(cfg->bbox.pMin),
        glm::value_ptr(speed),
        glm::value_ptr(full_bbox.pMin),
        glm::value_ptr(full_bbox.pMax),
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);

    ImGuiEXT::DragFloat3SeparateRange("Bounding box max",
        glm::value_ptr(cfg->bbox.pMax),
        glm::value_ptr(speed),
        glm::value_ptr(full_bbox.pMin),
        glm::value_ptr(full_bbox.pMax),
        "%.2f",
        ImGuiSliderFlags_AlwaysClamp);
    ImGui::PopItemWidth();

    bool reset_settings = false;
    if (!build_inprogress) {
        *build_navmesh = ImGui::Button("Build Navmesh");
        ImGui::SameLine();
        reset_settings = ImGui::Button("Reset to Defaults");

        ImGuiEXT::PushDisabled(!has_navmesh);
        *save_navmesh = ImGui::Button("Save Navmesh");
        ImGuiEXT::PopDisabled();
        ImGui::SameLine();
        *load_navmesh = ImGui::Button("Load Navmesh");
    } else {
        ImGui::TextUnformatted("Navmesh building...");
        *build_navmesh = false;
        *save_navmesh = false;
        *load_navmesh = false;
    }

    if (reset_settings) {
        *cfg = NavmeshConfig {
            full_bbox,
        };
    }

    ImGui::End();
}
#endif

static void handleCamera(GLFWwindow *window, EditorScene &scene)
{
    auto keyPressed = [&](uint32_t key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    glm::vec3 translate(0.f);

    auto cursorPosition = [window]() {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        return glm::vec2(mouse_x, -mouse_y);
    };


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        glm::vec2 mouse_cur = cursorPosition();
        glm::vec2 mouse_delta = mouse_cur - scene.cam.mousePrev;

        auto around_right = glm::angleAxis(
            mouse_delta.y * InternalConfig::mouseSpeed, scene.cam.right);

        auto around_up = glm::angleAxis(
            -mouse_delta.x * InternalConfig::mouseSpeed, glm::vec3(0, 1, 0));

        auto rotation = around_up * around_right;

        scene.cam.up = rotation * scene.cam.up;
        scene.cam.view = rotation * scene.cam.view;
        scene.cam.right = rotation * scene.cam.right;

        if (keyPressed(GLFW_KEY_W)) {
            translate += scene.cam.view;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= scene.cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= scene.cam.view;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += scene.cam.right;
        }

        scene.cam.mousePrev = mouse_cur;
    } else {
        if (keyPressed(GLFW_KEY_W)) {
            translate += scene.cam.up;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= scene.cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= scene.cam.up;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += scene.cam.right;
        }

        scene.cam.mousePrev = cursorPosition();
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }

    scene.cam.position += translate * InternalConfig::cameraMoveSpeed *
        InternalConfig::secondsPerFrame;
}

static glm::vec3 randomEpisodeColor(uint32_t idx) {
    auto rand = [](glm::vec2 co) {
        const float a  = 12.9898f;
        const float b  = 78.233f;
        const float c  = 43758.5453f;
        float dt = glm::dot(co, glm::vec2(a, b));
        float sn = fmodf(dt, 3.14);
        return glm::fract(glm::sin(sn) * c);
    };

    return glm::vec3(rand(glm::vec2(idx, idx)),
                rand(glm::vec2(idx + 1, idx)),
                rand(glm::vec2(idx, idx + 1)));
}

static optional<vector<AABB>> loadNavmeshCSV(const char *filename, 
        std::unordered_map<uint64_t, uint64_t> &navmesh_index_to_aabb_index,
        std::unordered_map<uint64_t, uint64_t> &aabb_index_to_navmesh_index,
        std::unordered_map<uint64_t, std::vector<uint64_t>> &aabb_neighbors) {
    ifstream csv(filename);

    if (!csv.is_open()) {
        cerr << "Failed to open CSV: " << filename << endl;
        return optional<vector<AABB>>();
    }

    vector<AABB> bboxes;

    string line;
    getline(csv, line);

    getline(csv, line);
    while (!csv.eof()) {
        auto getColumn = [&]() {
            auto pos = line.find(",");

            auto col = line.substr(0, pos);

            line = line.substr(pos + 1);

            return col;
        };

        bool finished_last_column = false;
        auto getColumnSemicolon = [&]() {
            auto pos = line.find(";");

            if (pos == std::string::npos) {
                finished_last_column = true;
                line.pop_back();
                return line;
            }
            else {
                auto col = line.substr(0, pos);

                line = line.substr(pos + 1);

                return col;
            }
        };

        getColumn();
        getColumn();
        uint64_t cur_navmesh_index = stoul(getColumn());
        navmesh_index_to_aabb_index[cur_navmesh_index] = bboxes.size();
        aabb_index_to_navmesh_index[bboxes.size()] = cur_navmesh_index;

        glm::vec3 pmin;
        pmin.x = stof(getColumn());
        pmin.y = stof(getColumn());
        pmin.z = stof(getColumn());

        glm::vec3 pmax;
        pmax.x = stof(getColumn());
        pmax.y = stof(getColumn());
        pmax.z = stof(getColumn());

        // Transform into renderer Y-up orientation
        pmin = glm::vec3(-pmin.x, pmin.z, pmin.y);
        pmax = glm::vec3(-pmax.x, pmax.z, pmax.y);

        glm::vec3 new_pmin = glm::min(pmin, pmax);
        glm::vec3 new_pmax = glm::max(pmin, pmax);

        bboxes.push_back({
            new_pmin,
            new_pmax,
        });

        while (!finished_last_column) {
            auto last_col_str = getColumnSemicolon();
            if (last_col_str.size() != 0) {
                uint64_t neighbor_index = stoul(last_col_str);
                aabb_neighbors[cur_navmesh_index].push_back(neighbor_index);
            }
        }

        getline(csv, line);
    }

    return optional<vector<AABB>> {
        move(bboxes),
    };
}

template <typename IterT>
pair<vector<OverlayVertex>, vector<uint32_t>> generateAABBVerts(
    IterT begin, IterT end, int red = 0, int green = 255, int blue = 0)
{
    vector<OverlayVertex> overlay_verts;
    vector<uint32_t> overlay_idxs;

    auto addVertex = [&](glm::vec3 pos) {
        overlay_verts.push_back({
            pos,
            glm::u8vec4(red, green, blue, 255),
        });
    };

    for (IterT iter = begin; iter != end; iter++) {
        const AABB &aabb = *iter;
        uint32_t start_idx = overlay_verts.size();

        addVertex(aabb.pMin);
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMin.z});
        addVertex({aabb.pMax.x, aabb.pMax.y, aabb.pMin.z});
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMin.z});

        addVertex({aabb.pMin.x, aabb.pMin.y, aabb.pMax.z});
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMax.z});
        addVertex(aabb.pMax);
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMax.z});

        auto addLine = [&](uint32_t a, uint32_t b) {
            overlay_idxs.push_back(start_idx + a);
            overlay_idxs.push_back(start_idx + b);
        };

        addLine(0, 1);
        addLine(1, 2);
        addLine(2, 3);
        addLine(3, 0);

        addLine(4, 5);
        addLine(5, 6);
        addLine(6, 7);
        addLine(7, 4);

        addLine(0, 4);
        addLine(1, 5);
        addLine(2, 6);
        addLine(3, 7);
    }

    return {
        move(overlay_verts),
        move(overlay_idxs),
    };
}

template <typename IterT>
void appendAABBVerts(
    vector<OverlayVertex> &overlay_verts, vector<uint32_t> &overlay_idxs,
    IterT begin, IterT end,
    int red = 255, int green = 0, int blue = 0)
{
    auto addVertex = [&](glm::vec3 pos) {
        overlay_verts.push_back({
            pos,
            glm::u8vec4(red, green, blue, 255),
        });
    };

    for (IterT iter = begin; iter != end; iter++) {
        const AABB &aabb = *iter;
        uint32_t start_idx = overlay_verts.size();

        addVertex(aabb.pMin);
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMin.z});
        addVertex({aabb.pMax.x, aabb.pMax.y, aabb.pMin.z});
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMin.z});

        addVertex({aabb.pMin.x, aabb.pMin.y, aabb.pMax.z});
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMax.z});
        addVertex(aabb.pMax);
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMax.z});

        auto addLine = [&](uint32_t a, uint32_t b) {
            overlay_idxs.push_back(start_idx + a);
            overlay_idxs.push_back(start_idx + b);
        };

        addLine(0, 1);
        addLine(1, 2);
        addLine(2, 3);
        addLine(3, 0);

        addLine(4, 5);
        addLine(5, 6);
        addLine(6, 7);
        addLine(7, 4);

        addLine(0, 4);
        addLine(1, 5);
        addLine(2, 6);
        addLine(3, 7);
    }
}

static vector<glm::ivec3> cluster_colors{
        {255, 0, 0},
        {0, 0, 255},
        {255, 255, 0},
        {0, 255, 255},
        {255, 255, 255},
        {155, 0, 0},
        {0, 0, 155},
        {155, 155, 0},
        {0, 155, 155},
        {155, 155, 155},
        {55, 0, 0},
        {0, 0, 55},
        {55, 55, 0},
        {0, 55, 55},
        {55, 55, 55},
        {0, 0, 0},
        };

void appendClusteredAABBVerts(
    vector<OverlayVertex> &overlay_verts, vector<uint32_t> &overlay_idxs,
    vector<AABB> &aabbs,
    vector<uint32_t> &cluster_indices)
{
    auto addVertex = [&](glm::vec3 pos, uint32_t cluster_index) {
        glm::ivec3 color = cluster_colors[cluster_index % cluster_colors.size()];
        int alpha_decay = cluster_index / cluster_colors.size();
        overlay_verts.push_back({
            pos,
            glm::u8vec4(color.r, color.g, color.b, 
                    255 / (int) std::round(std::pow(2, alpha_decay)))
        });
    };

    for (uint64_t i = 0; i < aabbs.size(); i++) {
        const AABB &aabb = aabbs[i];
        uint32_t cluster_index = cluster_indices[i];
        uint32_t start_idx = overlay_verts.size();

        addVertex(aabb.pMin, cluster_index);
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMin.z}, cluster_index);
        addVertex({aabb.pMax.x, aabb.pMax.y, aabb.pMin.z}, cluster_index);
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMin.z}, cluster_index);

        addVertex({aabb.pMin.x, aabb.pMin.y, aabb.pMax.z}, cluster_index);
        addVertex({aabb.pMax.x, aabb.pMin.y, aabb.pMax.z}, cluster_index);
        addVertex(aabb.pMax, cluster_index);
        addVertex({aabb.pMin.x, aabb.pMax.y, aabb.pMax.z}, cluster_index);

        auto addLine = [&](uint32_t a, uint32_t b) {
            overlay_idxs.push_back(start_idx + a);
            overlay_idxs.push_back(start_idx + b);
        };

        addLine(0, 1);
        addLine(1, 2);
        addLine(2, 3);
        addLine(3, 0);

        addLine(4, 5);
        addLine(5, 6);
        addLine(6, 7);
        addLine(7, 4);

        addLine(0, 4);
        addLine(1, 5);
        addLine(2, 6);
        addLine(3, 7);
    }
}

optional<NavmeshData> loadNavmesh()
{
    const char *filename = fileDialog();

    std::unordered_map<uint64_t, uint64_t> navmesh_index_to_aabb_index;
    std::unordered_map<uint64_t, uint64_t> aabb_index_to_navmesh_index;
    std::unordered_map<uint64_t, std::vector<uint64_t>> aabb_neighbors;
    auto aabbs_opt = loadNavmeshCSV(filename, navmesh_index_to_aabb_index, 
            aabb_index_to_navmesh_index, aabb_neighbors);
    if (!aabbs_opt.has_value()) {
        return optional<NavmeshData>();
    } 

    auto aabbs = move(*aabbs_opt);

    auto [overlay_verts, overlay_idxs] =
        generateAABBVerts(aabbs.begin(), aabbs.end());

    return NavmeshData {
        move(aabbs),
        move(overlay_verts),
        move(overlay_idxs),
        move(navmesh_index_to_aabb_index),
        move(aabb_index_to_navmesh_index),
        move(aabb_neighbors)
    };
}

static inline int atan2Positive(float opposite, float adjacent) {
    int unshifted = (int) (std::atan2(opposite, adjacent) * 180.f / M_PI);
    if (unshifted < 0) {
        return NUM_ANGLES + unshifted;
    }
    else {
        return unshifted;
    }
}

static inline bool sameCluster(
        std::array<std::array<uint64_t, NUM_ANGLES>, NUM_ANGLES> &nearest_per_angle,
        std::array<std::array<float, NUM_ANGLES>, NUM_ANGLES> &distance_per_angle,
        uint32_t cur_theta, uint32_t cur_phi,
        uint32_t other_theta, uint32_t other_phi) {
    return (nearest_per_angle[other_theta][other_phi] != INVALID_INDEX) &&
        (std::abs(distance_per_angle[cur_theta][cur_phi] - distance_per_angle[other_theta][other_phi]) <= 100);
}

static void detectCover(EditorScene &scene,
                        const ComputeContext<Renderer::numCoverShaders> &ctx)
{
    using namespace vk;
    const DeviceState &dev = ctx.dev;
    MemoryAllocator &alloc = ctx.alloc;
    CoverData &cover_data = scene.cover;
    cover_data.results.clear();

    vector<glm::vec4> launch_points;

    for (const AABB &aabb : cover_data.navmesh->aabbs) {
        glm::vec2 min2d = glm::vec2(aabb.pMin.x, aabb.pMin.z);
        glm::vec2 max2d = glm::vec2(aabb.pMax.x, aabb.pMax.z);

        glm::vec2 diff = max2d - min2d;

        glm::i32vec2 num_samples(diff / cover_data.sampleSpacing);

        glm::vec2 sample_extent = glm::vec2(num_samples) *
            cover_data.sampleSpacing;

        glm::vec2 extra = diff - sample_extent;
        assert(extra.x >= 0 && extra.y >= 0);

        glm::vec2 start = min2d + extra / 2.f +
            cover_data.sampleSpacing / 2.f;

        for (int i = 0; i < (int)num_samples.x; i++) {
            for (int j = 0; j < (int)num_samples.y; j++) {
                glm::vec2 pos2d = start + glm::vec2(i, j) *
                    cover_data.sampleSpacing;

                glm::vec3 pos(pos2d.x, aabb.pMin.y, pos2d.y);

                /*
                if (pos2d.x >= -400.f && pos2d.x <= -170.f &&
                        pos2d.y >= -350.f && pos2d.y <= 120.f) {
                    launch_points.emplace_back(pos, aabb.pMax.y);
                }
                */
                //launch_points.emplace_back(pos, aabb.pMax.y);
                if (cover_data.definedLaunchRegion) {
                    if (pos2d.x >= cover_data.launchRegion.pMin.x &&
                            pos2d.x <= cover_data.launchRegion.pMax.x &&
                            pos2d.y >= cover_data.launchRegion.pMin.z &&
                            pos2d.y <= cover_data.launchRegion.pMax.z) {
                        launch_points.emplace_back(pos, aabb.pMax.y);
                    }
                }
                else {
                    launch_points.emplace_back(pos, aabb.pMax.y);
                }
            }
        }
    }

    optional<HostBuffer> voxel_staging;
    optional<LocalBuffer> voxel_buffer;
    uint32_t num_voxels = 0;
    uint64_t num_voxel_bytes = 0;
    vector<GPUAABB> voxels_tmp;
    {
        glm::vec3 voxel_size = {cover_data.voxelSizeXZ, 
            cover_data.voxelSizeY - cover_data.torsoHeight, cover_data.voxelSizeXZ};
        glm::vec3 voxel_stride = {cover_data.voxelStrideXZ, 
            cover_data.voxelSizeY, cover_data.voxelStrideXZ};
        for (uint64_t aabb_index = 0; aabb_index < cover_data.navmesh->aabbs.size();
                aabb_index++) {
            const AABB &aabb = cover_data.navmesh->aabbs[aabb_index];
            glm::vec3 pmin = aabb.pMin;
            pmin.y += cover_data.torsoHeight;
            glm::vec3 pmax = aabb.pMax;
            pmax.y += cover_data.agentHeight;

            glm::vec3 diff = pmax - pmin;

            glm::i32vec3 num_fullsize(diff / voxel_stride);

            /*
            glm::vec3 sample_extent = glm::vec3(num_fullsize) *
                voxel_stride;

            glm::vec3 extra = diff - sample_extent;
            assert(extra.x >= 0 && extra.y >= 0 && extra.z >= 0);

            if (pmin.x <= 1715.8f && pmax.x >= 1715.8f &&
                    pmin.z <= 1011.8 && pmax.z >= 1011.8) {
                diff.x += 1.f;
            }
            uint64_t navmesh_index = cover_data.navmesh->aabbIndexToNavmeshIndex[aabb_index];
            const std::vector<uint64_t> &neighbor_navmesh_indices = cover_data.navmesh->aabbNeighbors[navmesh_index];
            std::vector<AABB> neighbors;
            for (const auto &neighbor_navmesh_index : neighbor_navmesh_indices) {
                uint64_t neighbor_aabb_index = cover_data.navmesh->navmeshIndexToAABBIndex[neighbor_navmesh_index];
                neighbors.push_back(cover_data.navmesh->aabbs[neighbor_aabb_index]);
            }
            */

            for (int i = 0; i <= num_fullsize.x; i++) {
                for (int k = 0; k <= num_fullsize.z; k++) {
                    glm::vec3 cur_pmin = pmin + glm::vec3(i, 0, k) *
                        voxel_stride;

                    cur_pmin.x -= voxel_size.x / 2.f;
                    cur_pmin.z -= voxel_size.z / 2.f;

                    glm::vec3 cur_pmax = cur_pmin + voxel_size; 
                    //glm::min(cur_pmin + cur_size, pmax);
                    voxels_tmp.push_back(GPUAABB {
                        cur_pmin.x,
                        cur_pmin.y,
                        cur_pmin.z,
                        cur_pmax.x,
                        cur_pmax.y,
                        cur_pmax.z,
                        pmin.y,
                        pmax.y,
                    });

                }
            }
        }

        num_voxels = voxels_tmp.size();
        num_voxel_bytes = num_voxels * sizeof(GPUAABB);

        voxel_staging.emplace(alloc.makeStagingBuffer(num_voxel_bytes));
        memcpy(voxel_staging->ptr, voxels_tmp.data(), num_voxel_bytes);

        voxel_staging->flush(dev);

        voxel_buffer = alloc.makeLocalBuffer(num_voxel_bytes, true);
    }

    cout << launch_points.size() << " launch points. " <<
        num_voxels << " voxels." << endl;

    uint64_t num_launch_bytes = launch_points.size() * sizeof(glm::vec4);
    HostBuffer ground_points =
        alloc.makeHostBuffer(num_launch_bytes, true);

    memcpy(ground_points.ptr, launch_points.data(), num_launch_bytes);

    ground_points.flush(dev);

    uint32_t max_candidates = 125952 * 1000;
    uint64_t num_candidate_bytes = max_candidates * sizeof(CandidatePair);
    uint64_t extra_candidate_bytes =
        alloc.alignStorageBufferOffset(sizeof(uint32_t));
    assert(extra_candidate_bytes >= 8);

    uint64_t total_candidate_buffer_bytes =
        num_candidate_bytes + extra_candidate_bytes;

    optional<LocalBuffer> candidate_buffer_opt = alloc.makeLocalBuffer(
        total_candidate_buffer_bytes, true);
    if (!candidate_buffer_opt.has_value()) {
        cerr << "Out of memory while allocating intermediate buffer" << endl;
        abort();
    }
    LocalBuffer candidate_buffer_gpu = move(*candidate_buffer_opt);

    HostBuffer candidate_buffer_cpu = alloc.makeHostBuffer(
        total_candidate_buffer_bytes, true);

    DescriptorUpdates desc_updates(5);
    VkDescriptorBufferInfo ground_info;
    ground_info.buffer = ground_points.buffer;
    ground_info.offset = 0;
    ground_info.range = num_launch_bytes;

    desc_updates.storage(ctx.descSets[0], &ground_info, 0);
    desc_updates.storage(ctx.descSets[1], &ground_info, 0);

    VkDescriptorBufferInfo num_candidates_info;
    num_candidates_info.buffer = candidate_buffer_gpu.buffer;
    num_candidates_info.offset = 0;
    num_candidates_info.range = sizeof(uint32_t);;
    desc_updates.storage(ctx.descSets[1], &num_candidates_info, 1);

    VkDescriptorBufferInfo candidates_info;
    candidates_info.buffer = candidate_buffer_gpu.buffer;
    candidates_info.offset = extra_candidate_bytes;
    candidates_info.range = num_candidate_bytes;

    desc_updates.storage(ctx.descSets[1], &candidates_info, 2);

    VkDescriptorBufferInfo voxel_info;
    voxel_info.buffer = voxel_buffer->buffer;
    voxel_info.offset = 0;
    voxel_info.range = num_voxel_bytes;

    desc_updates.storage(ctx.descSets[1], &voxel_info, 3);

    desc_updates.update(dev);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, ctx.cmdPool, 0));

    VkCommandBuffer cmd = ctx.cmdBuffer;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    {
        uint32_t zero = 0;
        dev.dt.cmdUpdateBuffer(cmd, candidate_buffer_gpu.buffer,
                               4, sizeof(uint32_t), &zero);
    }

    dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx.pipelines[0].hdls[0]);

    CoverPushConst push_const;
    push_const.idxOffset = 0;
    push_const.numGroundSamples = launch_points.size();
    push_const.agentHeight = cover_data.agentHeight;
    push_const.eyeHeight = cover_data.eyeHeight;
    push_const.torsoHeight = cover_data.torsoHeight;
    push_const.sqrtOffsetSamples = cover_data.sqrtOffsetSamples;
    push_const.offsetRadius = cover_data.offsetRadius;
    push_const.numVoxelTests = cover_data.numVoxelTests;
    push_const.numVoxels = num_voxels;

    dev.dt.cmdPushConstants(cmd, ctx.pipelines[0].layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(CoverPushConst), 
                            &push_const);

    array<VkDescriptorSet, 3> bind_sets;
    bind_sets[0] = ctx.descSets[0];
    bind_sets[1] = scene.hdl.computeDescSet.hdl;
    bind_sets[2] = cover_data.tlasWithAABBs->tlasDesc.hdl;

    dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 ctx.pipelines[0].layout,
                                 0, bind_sets.size(), bind_sets.data(),
                                 0, nullptr);

    dev.dt.cmdDispatch(cmd, divideRoundUp(uint32_t(launch_points.size()), 32u),
                       1, 1);

    VkMemoryBarrier ground_barrier;
    ground_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    ground_barrier.pNext = nullptr;
    ground_barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    ground_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    dev.dt.cmdPipelineBarrier(cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1, &ground_barrier,
                              0, nullptr,
                              0, nullptr);

    VkBufferCopy voxel_copy_info;
    voxel_copy_info.srcOffset = 0;
    voxel_copy_info.dstOffset = 0;
    voxel_copy_info.size = num_voxel_bytes;

    dev.dt.cmdCopyBuffer(cmd, voxel_staging->buffer,
                         voxel_buffer->buffer,
                         1, &voxel_copy_info);

    VkMemoryBarrier voxel_barrier;
    voxel_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    voxel_barrier.pNext = nullptr;
    voxel_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    voxel_barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    dev.dt.cmdPipelineBarrier(cmd,
                              VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1, &voxel_barrier,
                              0, nullptr,
                              0, nullptr);

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &ctx.fence));

    VkSubmitInfo submit {
        VK_STRUCTURE_TYPE_SUBMIT_INFO,
        nullptr,
        0, nullptr, nullptr,
        1, &cmd,
        0, nullptr,
    };

    REQ_VK(dev.dt.queueSubmit(ctx.computeQueue, 1, &submit,
                              ctx.fence));

    waitForFenceInfinitely(dev, ctx.fence);

    voxel_staging.reset();

    uint32_t potential_candidates = min(num_voxels, max_candidates); // arbitrary
    assert(potential_candidates <= max_candidates);

    uint32_t points_per_dispatch = max_candidates / potential_candidates;

    int num_iters = launch_points.size() / points_per_dispatch;

    if (num_iters * points_per_dispatch < launch_points.size()) {
        num_iters++;
    }

    /*
    const int regionSize = 800;
    auto candidateRegions = new ArrayLookup[regionSize][regionSize][regionSize];
    */

    std::array<std::array<uint64_t, NUM_ANGLES>, NUM_ANGLES> nearest_per_angle;
    std::array<std::array<uint32_t, NUM_ANGLES>, NUM_ANGLES> cluster_per_angle;
    std::array<std::array<float, NUM_ANGLES>, NUM_ANGLES> distance_per_angle;
    auto &cover_results = cover_data.results;
    for (int i = 0; i < num_iters; i++) {
        REQ_VK(dev.dt.resetCommandPool(dev.hdl, ctx.cmdPool, 0));
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

        dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx.pipelines[1].hdls[0]);

        bind_sets[0] = ctx.descSets[1];

        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     ctx.pipelines[1].layout,
                                     0, bind_sets.size(), bind_sets.data(),
                                     0, nullptr);

        VkBufferCopy zero_copy;
        zero_copy.dstOffset = 0;
        zero_copy.srcOffset = sizeof(uint32_t);
        zero_copy.size = sizeof(uint32_t);

        dev.dt.cmdCopyBuffer(cmd, candidate_buffer_gpu.buffer,
                             candidate_buffer_gpu.buffer,
                             1, &zero_copy);

        VkMemoryBarrier zero_barrier;
        zero_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        zero_barrier.pNext = nullptr;
        zero_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        zero_barrier.dstAccessMask =
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

        dev.dt.cmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &zero_barrier, 0, nullptr, 0, nullptr);

        push_const.idxOffset = i * points_per_dispatch;

        dev.dt.cmdPushConstants(cmd, ctx.pipelines[1].layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(CoverPushConst), 
                                &push_const);


        uint32_t dispatch_points = min(
            uint32_t(launch_points.size() - push_const.idxOffset),
                     points_per_dispatch);
        std::cout << "dispatch_points " << dispatch_points << std::endl;
        for (uint32_t dispatch_point_idx = push_const.idxOffset;
                dispatch_point_idx < push_const.idxOffset + dispatch_points;
                dispatch_point_idx++) {
            std::cout << "launch point " << dispatch_point_idx << ": " <<
                glm::to_string(launch_points[dispatch_point_idx]) << std::endl;
        }

        dev.dt.cmdDispatch(cmd, divideRoundUp(num_voxels, 32u),
                           dispatch_points, 1);

        VkMemoryBarrier dispatch_barrier;
        dispatch_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        dispatch_barrier.pNext = nullptr;
        dispatch_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        dispatch_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        dev.dt.cmdPipelineBarrier(cmd,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  0,
                                  1, &dispatch_barrier,
                                  0, nullptr,
                                  0, nullptr);

        VkBufferCopy result_copy_info;
        result_copy_info.srcOffset = 0;
        result_copy_info.dstOffset = 0;
        result_copy_info.size = total_candidate_buffer_bytes;

        dev.dt.cmdCopyBuffer(cmd, candidate_buffer_gpu.buffer,
                             candidate_buffer_cpu.buffer,
                             1, &result_copy_info);

        REQ_VK(dev.dt.endCommandBuffer(cmd));

        REQ_VK(dev.dt.resetFences(dev.hdl, 1, &ctx.fence));

        REQ_VK(dev.dt.queueSubmit(ctx.computeQueue, 1, &submit,
                                  ctx.fence));

        waitForFenceInfinitely(dev, ctx.fence);

        uint32_t num_candidates;
        memcpy(&num_candidates, candidate_buffer_cpu.ptr, sizeof(uint32_t));
        cout << "Iter " << i << " / " << num_iters << ": Found " << num_candidates << " candidate corner points" << endl;

        assert(num_candidates < max_candidates);

        CandidatePair *candidate_data =
            (CandidatePair *)((char *)candidate_buffer_cpu.ptr + extra_candidate_bytes);


        //std::unordered_map<glm::vec3, std::vector<glm::vec3>> origins_to_pmins;
        //std::unordered_map<glm::vec3, std::vector<glm::vec3>> origins_to_pmaxs;
        std::unordered_map<glm::vec3, std::vector<AABB>> origins_to_aabbs;
        for (uint64_t candidate_idx = 0; candidate_idx < num_candidates; candidate_idx++) {
            const auto &candidate = candidate_data[candidate_idx];
            GPUAABB gpuaabb = voxels_tmp[candidate.voxelID];
            gpuaabb.pMinY = candidate.voxelMinY;
            gpuaabb.pMaxY = candidate.voxelMinY + cover_data.voxelSizeY - cover_data.torsoHeight;
            glm::vec3 region_p_min = {gpuaabb.pMinX, gpuaabb.pMinY, gpuaabb.pMinZ};
            glm::vec3 region_p_max = {gpuaabb.pMaxX, gpuaabb.pMaxY, gpuaabb.pMaxZ};
            origins_to_aabbs[candidate.origin].push_back({region_p_min - 2.f, region_p_max + 2.f});
            /*
            //candidates.push_back(region_min_p);
            origins_to_pmins[candidate.origin].push_back(region_p_min);
            origins_to_pmaxs[candidate.origin].push_back(region_p_max);
            */
            if (cover_data.showAllCoverEdges) {
                cover_results[candidate.origin].allEdges.push_back({region_p_min - 2.f, region_p_max + 2.f});
            }
            //cover_results[candidate.origin].cover_regions.insert({region_p_min - 1.f, region_p_max + 1.f});
        }
        
//        std::vector<glm::vec3> origins;
        for (const auto &[origin, aabbs] : origins_to_aabbs) {
            //origins.push_back(origin_and_cover_result.first);
            for (int theta = 0; theta < NUM_ANGLES; theta++) {
                for (int phi = 0; phi < NUM_ANGLES; phi++) {
                   nearest_per_angle[theta][phi] = INVALID_INDEX;
                   cluster_per_angle[theta][phi] = INVALID_CLUSTER;
                }
            }

            for (uint64_t aabb_index = 0; aabb_index < aabbs.size(); aabb_index++) {
                const AABB &aabb = aabbs[aabb_index];
                float minX = aabb.pMin.x - origin.x;
                float maxX = aabb.pMax.x - origin.x;
                float minY = aabb.pMin.y - origin.y;
                float maxY = aabb.pMax.y - origin.y;
                float minZ = aabb.pMin.z - origin.z;
                float maxZ = aabb.pMax.z - origin.z;
                int theta_0 = atan2Positive(minZ, minX);
                int phi_0 = atan2Positive(std::hypot(minX, minZ), minY);
                int theta_1 = atan2Positive(maxZ, maxX);
                int phi_1 = atan2Positive(std::hypot(maxX, maxZ), maxY); 
                int min_theta = std::min(theta_0, theta_1);
                int max_theta = std::max(theta_0, theta_1);
                int min_phi = std::min(phi_0, phi_1);
                int max_phi = std::max(phi_0, phi_1);
                float cur_distance = glm::length(aabb.pMin - origin);
                auto nearest_phi_per_theta = [&] (int theta) {
                    for (int phi = min_phi; phi <= max_phi; phi++) {
                        if (nearest_per_angle[theta][phi] == INVALID_INDEX || 
                               cur_distance < distance_per_angle[theta][phi]) {
                            nearest_per_angle[theta][phi] = aabb_index;
                            distance_per_angle[theta][phi] = cur_distance;
                        }
                    }
                };

                // handle wrap arounda
                if (max_theta - min_theta <= 200) {
                    for (int theta = min_theta; theta <= max_theta; theta++) {
                        nearest_phi_per_theta(theta);
                    }
                }
                else {
                    for (int theta = max_theta; theta < NUM_ANGLES; theta++) {
                        nearest_phi_per_theta(theta);
                    }

                    for (int theta = 0; theta <= min_theta; theta++) {
                        nearest_phi_per_theta(theta);
                    }
                }
            }
 
            // cluster all aabbs that are within CLUSTER_RADIUS of eachother
            uint32_t next_cluster_index = 0;
            for (int theta = 0; theta < NUM_ANGLES; theta++) {
                for (int phi = 0; phi < NUM_ANGLES; phi++) {
                    if (nearest_per_angle[theta][phi] == INVALID_INDEX) {
                        continue;
                    }

                    else if (cluster_per_angle[theta][phi] == INVALID_CLUSTER) {
                        cluster_per_angle[theta][phi] = next_cluster_index++;
                    }

                    uint32_t cur_cluster_index = cluster_per_angle[theta][phi];

                    for (int cluster_theta = theta - CLUSTER_RADIUS; 
                            cluster_theta <= theta + CLUSTER_RADIUS;
                            cluster_theta++) {
                        for (int cluster_phi = phi - CLUSTER_RADIUS;
                                cluster_phi <= phi + CLUSTER_RADIUS;
                                cluster_phi++) {
                            // indexes adjust for wrap around range 0 to NUM_ANGLES-1
                            int cluster_theta_index = 
                                ((cluster_theta % NUM_ANGLES) + NUM_ANGLES) % NUM_ANGLES;
                            int cluster_phi_index = 
                                ((cluster_phi % NUM_ANGLES) + NUM_ANGLES) % NUM_ANGLES;
                            if (nearest_per_angle[cluster_theta_index][cluster_phi_index] != INVALID_INDEX) {
                                cluster_per_angle[cluster_theta_index][cluster_phi_index] = 
                                    cur_cluster_index;
                            }
                        }
                    }
                }
            }

            // merge the clusters
            // first build a cluster adjacency matrix
            bool * unmerged_cluster_matrix = new bool[next_cluster_index * next_cluster_index];
            for (uint32_t unmerged_index = 0; unmerged_index < next_cluster_index * next_cluster_index; 
                    unmerged_index++) {
                unmerged_cluster_matrix[unmerged_index] = false;
            }

            for (int theta = 0; theta < NUM_ANGLES; theta++) {
                for (int phi = 0; phi < NUM_ANGLES; phi++) {
                    if (nearest_per_angle[theta][phi] == INVALID_INDEX) {
                        continue;
                    }

                    uint32_t cur_cluster_index = cluster_per_angle[theta][phi];

                    for (int cluster_theta = theta - CLUSTER_RADIUS; 
                            cluster_theta <= theta + CLUSTER_RADIUS;
                            cluster_theta++) {
                        for (int cluster_phi = phi - CLUSTER_RADIUS;
                                cluster_phi <= phi + CLUSTER_RADIUS;
                                cluster_phi++) {
                            // indexes adjust for wrap around range 0 to NUM_ANGLES-1
                            int cluster_theta_index = 
                                ((cluster_theta % NUM_ANGLES) + NUM_ANGLES) % NUM_ANGLES;
                            int cluster_phi_index = 
                                ((cluster_phi % NUM_ANGLES) + NUM_ANGLES) % NUM_ANGLES;
                            if (nearest_per_angle[cluster_theta_index][cluster_phi_index] != INVALID_INDEX) {
                                uint32_t other_cluster_index = 
                                    cluster_per_angle[cluster_theta_index][cluster_phi_index];
                                unmerged_cluster_matrix[
                                    cur_cluster_index * next_cluster_index + other_cluster_index] = true;
                                unmerged_cluster_matrix[
                                    other_cluster_index * next_cluster_index + cur_cluster_index] = true;
                            }
                        }
                    }
                }
            }

            // then merge adjacency matrix with Warshall's Algorithm
            bool changed_value = true;
            while (changed_value) {
                changed_value = false;
                for (uint32_t unmerged_i = 0; unmerged_i < next_cluster_index; unmerged_i++) {
                    for (uint32_t unmerged_j = 0; unmerged_j < next_cluster_index; unmerged_j++) {
                        bool old_value = unmerged_cluster_matrix[unmerged_i * next_cluster_index + 
                            unmerged_j];
                        for (uint32_t unmerged_k = 0; unmerged_k < next_cluster_index; unmerged_k++) {
                            unmerged_cluster_matrix[unmerged_i * next_cluster_index + unmerged_j] |= 
                                unmerged_cluster_matrix[unmerged_i * next_cluster_index + unmerged_k] &&
                                unmerged_cluster_matrix[unmerged_k * next_cluster_index + unmerged_j];
                        }
                        if (unmerged_cluster_matrix[unmerged_i * next_cluster_index + unmerged_j] != old_value) {
                            changed_value = true;
                        }
                    }
                }
            }

            // get the min cluster number for each cluster
            uint32_t * min_connected_cluster = new uint32_t[next_cluster_index];
            for (uint32_t unmerged_i = 0; unmerged_i < next_cluster_index; unmerged_i++) {
                for (uint32_t unmerged_j = 0; unmerged_j < next_cluster_index; unmerged_j++) {
                    if (unmerged_cluster_matrix[unmerged_i * next_cluster_index + unmerged_j]) {
                        min_connected_cluster[unmerged_i] = unmerged_j;
                        break;
                    }
                }
            }

            cover_results[origin].num_clusters = next_cluster_index;
            std::unordered_set<uint64_t> added_aabbs;
            for (int theta = 0; theta < NUM_ANGLES; theta++) {
                for (int phi = 0; phi < NUM_ANGLES; phi++) {
                    uint64_t aabb_index = nearest_per_angle[theta][phi];
                    if (aabb_index != INVALID_INDEX && added_aabbs.find(aabb_index) == added_aabbs.end()) {
                        cover_results[origin].aabbs.push_back(aabbs[aabb_index]);
                        cover_results[origin].edgeClusterIndices.push_back(
                                min_connected_cluster[cluster_per_angle[theta][phi]]);
                        added_aabbs.insert(aabb_index);
                    }
                }
            }

            delete[] unmerged_cluster_matrix;
            delete[] min_connected_cluster;
        }

        /*
        //std::chrono::steady_clock::time_point begin_cluster = std::chrono::steady_clock::now();
        #pragma omp parallel for
        for (int origin_idx = 0; origin_idx < (int) origins.size(); origin_idx++) {
            glm::vec3 origin = origins[origin_idx];
            const std::vector<glm::vec3> &p_mins = origins_to_pmins[origin];
            ContiguousClusters clusters(p_mins, cover_data.voxelSizeXZ / 1.8);
            const std::vector<std::vector<int64_t>> &pmin_indices_per_cluster = clusters.getIndicesPerCluster();
            const auto& clusters_aabbs = clusters.getClusters();
            for (uint64_t idx = 0; idx < clusters_aabbs.size(); idx++) {
                const auto& taabb = clusters_aabbs[idx];
                std::cout << "cluster " << idx << ": {" << glm::to_string(taabb.pMin) << ", " << glm::to_string(taabb.pMax) << "}" << std::endl;
            }
            for (const auto &pmin_indices : pmin_indices_per_cluster) {
                std::vector<std::pair<int64_t, float>> indices_and_distances;
                for (const auto &pmin_index : pmin_indices) {
                    indices_and_distances.push_back({pmin_index, glm::length(p_mins[pmin_index] - origin)});
                }
                sort(indices_and_distances.begin(), indices_and_distances.end(), 
                        [](const std::pair<int64_t, float>& lhs, const std::pair<int64_t, float>& rhs) {
                            return lhs.second < rhs.second;
                        });
                for (uint64_t index_index = 0; index_index < indices_and_distances.size() && 
                        indices_and_distances[index_index].second <= indices_and_distances[0].second + 48;
                        index_index++) {
                    int64_t cur_index = indices_and_distances[index_index].first;
                    cover_results[origin].aabbs.push_back({p_mins[cur_index], origins_to_pmaxs[origin][cur_index]});
                }
            }
        }
        */
        //std::chrono::steady_clock::time_point end_cluster = std::chrono::steady_clock::now();
        //std::cout << origins.size() << " cluster time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cluster - begin_cluster).count() << "[ms]" << std::endl;

    }

    cout << "Unique origin points: " << cover_results.size() << endl;

    for (auto &[_, result] : cover_results) {
        vector<OverlayVertex> overlay_verts;
        vector<uint32_t> overlay_idxs;

        if (cover_data.showAllCoverEdges) {
            appendAABBVerts(overlay_verts, overlay_idxs, 
                    result.aabbs.begin(), 
                    result.aabbs.end(),
                    0, 0, 255);

            appendAABBVerts(overlay_verts, overlay_idxs, 
                    result.allEdges.begin(), 
                    result.allEdges.end(),
                    255, 0, 0);
        }
        else {
            appendClusteredAABBVerts(overlay_verts, overlay_idxs, result.aabbs,
                    result.edgeClusterIndices);
        }

        result.overlayVerts = move(overlay_verts);
        result.overlayIdxs = move(overlay_idxs);
    }

    cover_data.showCover = true;
}

static vector<AABB> transformNavmeshAABBS(const vector<AABB> &orig_aabbs)
{
    vector<AABB> new_aabbs;
    new_aabbs.reserve(orig_aabbs.size());

    for (const AABB &aabb : orig_aabbs) {
        glm::vec3 new_pmin = aabb.pMin;
        new_pmin.y += 55;

        glm::vec3 new_pmax = aabb.pMax;
        new_pmax.y += 72;

        new_aabbs.push_back({
            new_pmin,
            new_pmax,
        });
    }

    return new_aabbs;
}

static glm::vec3  transformOrigin(const glm::vec3 &origin) {
    glm::vec3 result;
    result.x = (origin.x + 16.f) * -1;
    result.y = origin.z + 16.f;
    result.z = origin.y - 50.f;
    return result;
}

static void handleCover(EditorScene &scene,
                        Renderer &renderer)
{
    const ComputeContext<Renderer::numCoverShaders> &ctx =
        renderer.getCoverContext();
    CoverData &cover = scene.cover;

    ImGui::Begin("Cover Detection");

    if (ImGui::Button("Load Navmesh")) {
        cover.navmesh = loadNavmesh();
        vector<AABB> tlas_aabbs = transformNavmeshAABBS(cover.navmesh->aabbs);
        tie(cover.navmeshAABBGPU, cover.tlasWithAABBs) = 
            renderer.buildTLASWithAABBS(*scene.hdl.scene, tlas_aabbs.data(),
                                        tlas_aabbs.size());
                                                        
        cover.showNavmesh = true;
    }
    ImGuiEXT::PushDisabled(!cover.navmesh.has_value());

    ImGui::SameLine();

    if (ImGui::Button("Detect Cover")) {
        detectCover(scene, ctx);
    }

    {
        glm::vec3 cur_pos = scene.cam.position;

        if (!cover.fixOrigin) {
            float min_dist = INFINITY;
            for (const auto &[key_pos, _] : cover.results) {
                float dist = glm::distance(key_pos, cur_pos);

                if (dist < min_dist) {
                    min_dist = dist;
                    cover.nearestCamPoint = key_pos;
                }
            }
        }
    }

    ImGui::Separator();

    //bool oldCoverEdges = cover.showAllCoverEdges;
    ImGui::Checkbox("Show Navmesh", &cover.showNavmesh);
    ImGui::Checkbox("Show Cover", &cover.showCover);
    ImGui::Checkbox("Show All Edges", &cover.showAllCoverEdges);
    ImGui::Checkbox("Fix Origin", &cover.fixOrigin);

    /*
    if (oldCoverEdges != cover.showAllCoverEdges) {
        for (auto &[_, result] : cover.results) {
            auto [overlay_verts, overlay_idxs] =
                generateAABBVerts(result.aabbs.begin(), result.aabbs.end(), 0, 0, 255);

            if (cover.showAllCoverEdges) {
                appendAABBVerts(overlay_verts, overlay_idxs, 
                        result.allEdges.begin(), 
                        result.allEdges.end(),
                        255, 0, 0);
            }
            result.overlayVerts = move(overlay_verts);
            result.overlayIdxs = move(overlay_idxs);
        }
    }
    */

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::PushItemWidth(digit_width * 6);
    ImGui::DragFloat("Sample Spacing", &cover.sampleSpacing, 0.1f, 0.1f, 100.f, "%.1f");
    ImGui::DragFloat("Voxel Size XZ", &cover.voxelSizeXZ,
                     0.1f, 0.1f, 100.f, "%.1f");

    ImGui::DragFloat("Voxel Size Y", &cover.voxelSizeY,
                     0.1f, 0.1f, 100.f, "%.1f");

    ImGui::DragFloat("Agent Height", &cover.agentHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragFloat("Eye Height", &cover.eyeHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragFloat("Torso Height", &cover.torsoHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragInt("# Offset Samples (sqrt)", &cover.sqrtOffsetSamples, 1, 1, 1000);
    ImGui::DragFloat("Offset Radius", &cover.offsetRadius, 0.01f, 0.f, 100.f,
                     "%.2f");
    ImGui::DragInt("# Voxel Tests", &cover.numVoxelTests, 1, 1, 1000);
    ImGui::DragInt("# AABBs Clustered", &cover.numAABBs, 1, 1, 1000);

    if (!cover.triedLoadingLaunchRegion) {
        cover.triedLoadingLaunchRegion = true;        
        if (std::filesystem::exists(scene.outputPath / "region.csv")) {
            std::fstream region_csv(scene.outputPath / "region.csv");
            string tmp_str;
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMin.x = std::stof(tmp_str);
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMin.y = std::stof(tmp_str);
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMin.z = std::stof(tmp_str);
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMax.x = std::stof(tmp_str);
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMax.y = std::stof(tmp_str);
            getline(region_csv, tmp_str, ',');
            cover.launchRegion.pMax.z = std::stof(tmp_str);
            cover.definedLaunchRegion = true;
            region_csv.close();
            std::cout << "loaded region: {" << glm::to_string(cover.launchRegion.pMin) << "} , {"
                << glm::to_string(cover.launchRegion.pMax) << "}" << std::endl;

        }
    }

    if (cover.definedLaunchRegion || !cover.definingLaunchRegion) {
        if (ImGui::Button("Define Launch Region")) {
            cover.definedLaunchRegion = false;
            cover.definingLaunchRegion = true;
            cover.launchRegion.pMin = scene.cam.position;
        }
    }

    else if (cover.definingLaunchRegion) {
        if (ImGui::Button("Finish Launch Region")) {
            cover.definedLaunchRegion = true;
            cover.definingLaunchRegion = false;
            glm::vec3 tmp_pos = cover.launchRegion.pMin;
            cover.launchRegion.pMin = glm::min(tmp_pos, scene.cam.position) - 1.f;
            cover.launchRegion.pMax = glm::max(tmp_pos, scene.cam.position) + 1.f;
        }
    }

    if (ImGui::Button("Clear Launch Region")) {
        cover.definedLaunchRegion = false;
        cover.definingLaunchRegion = false;
    }
    
    if (cover.definedLaunchRegion) {
        if (ImGui::Button("Save Launch Region")) {
            std::fstream region_csv(scene.outputPath / "region.csv", std::fstream::out | std::fstream::trunc);
            region_csv << cover.launchRegion.pMin.x << ","
                << cover.launchRegion.pMin.y << ","
                << cover.launchRegion.pMin.z << ","
                << cover.launchRegion.pMax.x << ","
                << cover.launchRegion.pMax.y << ","
                << cover.launchRegion.pMax.z;
            region_csv.close();
        }
    }

    if (cover.results.size() > 0) {
        if (ImGui::Button("Create CSGO Scripts")) {
            std::filesystem::remove_all(scene.outputPath / "scripts");
            std::filesystem::create_directory(scene.outputPath / "scripts");
            uint64_t origin_idx = 0;
            for (const auto &[origin, cover_result] : cover.results) {
                std::fstream cover_csv(scene.outputPath / "scripts" / (std::to_string(origin_idx) + ".cfg"), 
                        std::fstream::out | std::fstream::trunc);
                glm::vec3 transformed_origin = transformOrigin(origin);
                cover_csv << "sv_cheats 1" << std::endl;
                cover_csv << "setpos " << transformed_origin.x << " " 
                    << transformed_origin.y << " " << transformed_origin.z << std::endl;

                for (const auto &aabb : cover_result.aabbs) {
                    cover_csv << "box "
                        << aabb.pMin.x * -1 << " "
                        << aabb.pMin.z << " "
                        << aabb.pMin.y << " "
                        << aabb.pMax.x * -1 << " "
                        << aabb.pMax.z << " "
                        << aabb.pMax.y << std::endl;;
                }

                cover_csv.close();
                origin_idx++;
            }
        }


        if (ImGui::Button("Create CSV Output")) {
            uint64_t origin_idx = 0;
            std::fstream cover_csv(scene.outputPath / "dimension_table_cover_edges.csv", 
                    std::fstream::out | std::fstream::trunc);
            std::fstream origins_csv(scene.outputPath / "dimension_table_cover_origins.csv", 
                    std::fstream::out | std::fstream::trunc);
            std::fstream unconverted_cover_csv(scene.outputPath / "unconverted_cover_edges.csv", 
                    std::fstream::out | std::fstream::trunc);
            std::fstream unconverted_origins_csv(scene.outputPath / "unconverted_origins.csv", 
                    std::fstream::out | std::fstream::trunc);

            origins_csv << "id,x,y,z\n";
            cover_csv << "id,origin_id,cluster_id,min_x,min_y,min_z,max_x,max_y,max_z\n";
            for (const auto &[origin, cover_result] : cover.results) {
                unconverted_origins_csv << origin_idx << "," 
                    << origin.x << ","
                    << origin.y << ","
                    << origin.z << "\n";
                origins_csv << origin_idx << "," 
                    << origin.x * -1 << ","
                    << origin.z << ","
                    << origin.y << "\n";

                for (uint64_t aabb_index = 0; aabb_index < cover_result.aabbs.size(); aabb_index++) {
                    const auto &aabb = cover_result.aabbs[aabb_index];
                    cover_csv << aabb_index << ","
                        << origin_idx << ","
                        << cover_result.edgeClusterIndices[aabb_index] << ","
                        << aabb.pMin.x * -1 << ","
                        << aabb.pMin.z << ","
                        << aabb.pMin.y << ","
                        << aabb.pMax.x * -1 << ","
                        << aabb.pMax.z << ","
                        << aabb.pMax.y << "\n";
                }

                for (uint64_t aabb_index = 0; aabb_index < cover_result.aabbs.size(); aabb_index++) {
                    const auto &aabb = cover_result.aabbs[aabb_index];
                    unconverted_cover_csv << aabb_index << ","
                        << origin_idx << ","
                        << cover_result.edgeClusterIndices[aabb_index] << ","
                        << aabb.pMin.x << ","
                        << aabb.pMin.y << ","
                        << aabb.pMin.z << ","
                        << aabb.pMax.x << ","
                        << aabb.pMax.y << ","
                        << aabb.pMax.z << "\n";
                }
                origin_idx++;
            }

            cover_csv.close();
            origins_csv.close();
            unconverted_cover_csv.close();
            unconverted_origins_csv.close();
        }
    }

    if (!cover.showAllEdges && std::filesystem::exists(scene.outputPath / "unconverted_origins.csv")) {
        if (ImGui::Button("Load Origins and Edges")) {
            std::fstream origins_csv(scene.outputPath / "unconverted_origins.csv");
            std::fstream edges_csv(scene.outputPath / "unconverted_cover_edges.csv");
            string tmp_str;
            cover.results.clear();
            std::vector<glm:vec3> origins;
            // first fetch all the origins
            while (getline(origins_csv, tmp_str, ',')) {
                // skip the index, no need when loading into unordered map
                // will keep sorting order by putting in vector first
                glm::vec3 origin;
                getline(origins_csv, tmp_str, ',');
                origin.x = std::stof(tmp_str);
                getline(origins_csv, tmp_str, ',');
                origin.y = std::stof(tmp_str);
                getline(origins_csv, tmp_str);
                origin.z = std::stof(tmp_str);
                origins.push_back(origin);
            }

            while (getline(edges_csv, tmp_str, ',')) {
                // TODO: ADD ADJUSTMENT FOR INDEX WHEN REGENERATE EDGES WITH INDEX
                uint32_t origin_idx = (uint32_t) std::stoul(tmp_str);
                glm::vec3 origin = origins[origin_idx];
                getline(edges_csv, tmp_str, ',');
                uint32_t cluster_idx = (uint32_t) std::stoul(tmp_str);
                cover.results[origin].edgeClusterIndices.push_back(cluster_idx);

                AABB edge;
                getline(edges_csv, tmp_str, ',');
                edge.pMin.x = std::stof(tmp_str);
                getline(edges_csv, tmp_str, ',');
                edge.pMin.y = std::stof(tmp_str);
                getline(edges_csv, tmp_str, ',');
                edge.pMin.z = std::stof(tmp_str);
                getline(edges_csv, tmp_str, ',');
                edge.pMax.x = std::stof(tmp_str);
                getline(edges_csv, tmp_str, ',');
                edge.pMax.y = std::stof(tmp_str);
                getline(edges_csv, tmp_str);
                edge.pMax.z = std::stof(tmp_str);
                cover.results[origin].aabbs.push_back(edge);
            }

            origins_csv.close();
            edges_csv.close();

            for (auto &[_, result] : cover.results) {
                vector<OverlayVertex> overlay_verts;
                vector<uint32_t> overlay_idxs;

                appendClusteredAABBVerts(overlay_verts, overlay_idxs, result.aabbs,
                        result.edgeClusterIndices);

                result.overlayVerts = move(overlay_verts);
                result.overlayIdxs = move(overlay_idxs);
            }

            std::cout << "loaded " << cover.results.size() 
                << " origins and their edges " << std::endl;

            unconverted_cover_csv.close();
            unconverted_origins_csv.close();
        }
    }

    ImGuiEXT::PopDisabled();

    ImGuiEXT::PushDisabled(scene.cover.results.empty());

    if (ImGui::Button("Snap to Sample")) {
        scene.cam.position = cover.nearestCamPoint;
    }

    ImGuiEXT::PopDisabled();

    ImGui::PopItemWidth();
    ImGui::End();
}

#if 0
static void handleLights(EditorScene &scene)
{
    ImGui::Begin("Lights");
    ImGui::Separator();
    if (ImGui::Button("Add Light")) {
        const float area_offset = 0.25f;
        AreaLight new_light {
            {
                glm::vec3(-area_offset, 0, -area_offset),
                glm::vec3(-area_offset, 0, area_offset),
                glm::vec3(area_offset, 0, -area_offset),
                glm::vec3(area_offset, 0, area_offset),
            },
            glm::vec3(0),
        };

        if (scene.selectedLight == -1) {
            new_light.translate = scene.cam.position + 0.5f * scene.cam.view;
        } else {
            new_light.translate = scene.lights[scene.selectedLight].translate +
                glm::vec3(2.f * area_offset, 0, 2.f * area_offset);
        }

        scene.lights.push_back(new_light);
        scene.selectedLight = scene.lights.size() - 1;
    }

    ImGuiEXT::PushDisabled(scene.lights.size() <= 1);
    ImGui::PushItemWidth(4.f * ImGui::GetFontSize());

    ImGui::DragInt("Selected Light", &scene.selectedLight, 0.05f,
                   0, scene.lights.size() - 1,
                   "%d", ImGuiSliderFlags_AlwaysClamp);

    ImGui::PopItemWidth();
    ImGuiEXT::PopDisabled();

    ImGuiEXT::PushDisabled(scene.selectedLight == -1);

    AreaLight *cur_light = nullptr;
    if (scene.selectedLight != -1) {
        cur_light = &scene.lights[scene.selectedLight];
    }

    glm::vec3 fake(0);
    auto pos_ptr = glm::value_ptr(fake);
    if (cur_light) {
        pos_ptr = glm::value_ptr(cur_light->translate);
    }

    ImGui::PushItemWidth(ImGui::GetFontSize() * 15.f);
    glm::vec3 speed(0.01f);
    ImGuiEXT::DragFloat3SeparateRange("Position",
        pos_ptr,
        glm::value_ptr(speed),
        glm::value_ptr(scene.bbox.pMin),
        glm::value_ptr(scene.bbox.pMax),
        "%.3f",
        ImGuiSliderFlags_AlwaysClamp);

    ImGuiEXT::PopDisabled();


    bool save = ImGui::Button("Save Lights");
    ImGui::SameLine();
    bool load = ImGui::Button("Load Lights");

    if (save || load) {
        filesystem::path lights_path =
            filesystem::path(scene.scenePath).replace_extension("lights");
        if (save) {
            ofstream lights_out(lights_path, ios::binary);
            uint32_t num_lights = scene.lights.size();
            lights_out.write((char *)&num_lights, sizeof(uint32_t));
            for (const AreaLight &light : scene.lights) {
                lights_out.write((char *)&light, sizeof(AreaLight));
            }
        }

        if (load) {
            ifstream lights_in(lights_path, ios::binary);
            uint32_t num_lights;
            lights_in.read((char *)&num_lights, sizeof(uint32_t));
            scene.lights.clear();
            scene.lights.reserve(num_lights);
            for (int i = 0; i < (int)num_lights; i++) {
                AreaLight light;
                lights_in.read((char *)&light, sizeof(AreaLight));
                scene.lights.push_back(light);
            }
        }
    }

    ImGui::End();
}
#endif

static float throttleFPS(chrono::time_point<chrono::steady_clock> start) {
    using namespace chrono;
    using namespace chrono_literals;
    
    auto end = steady_clock::now();
    while (end - start <
           InternalConfig::nsPerFrameLongWait) {
        this_thread::sleep_for(1ms);
    
        end = steady_clock::now();
    }
    
    while (end - start < InternalConfig::nsPerFrame) {
        this_thread::yield();
    
        end = steady_clock::now();
    }

    return duration<float>(end - start).count();
}

void Editor::loop()
{
    auto window = renderer_.getWindow();

    float frame_duration = InternalConfig::secondsPerFrame;
    while (!glfwWindowShouldClose(window)) {
        EditorScene &scene = scenes_[cur_scene_idx_];

        auto start_time = chrono::steady_clock::now();

        startFrame();

        handleCamera(window, scene);
        handleCover(scene, renderer_);
        render(scene, frame_duration);

        frame_duration = throttleFPS(start_time);
    }

    renderer_.waitForIdle();
}

void Editor::startFrame()
{
    renderer_.waitUntilFrameReady();

    glfwPollEvents();

    renderer_.startFrame();
    ImGui::NewFrame();
}

static void renderCFGUI(Renderer::OverlayConfig &cfg,
                        EditorCam &cam)
{
    ImGui::Begin("Render Settings");

    ImGui::TextUnformatted("Camera");
    ImGui::Separator();

    auto side_size = ImGui::CalcTextSize(" Bottom " );
    side_size.y *= 1.4f;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                        ImVec2(0.5f, 0.f));

    if (ImGui::Button("Top", side_size)) {
        cam.position = glm::vec3(0.f, 10.f, 0.f);
        cam.view = glm::vec3(0, -1, 0.f);
        cam.up = glm::vec3(0, 0, 1.f);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Left", side_size)) {
        cam.position = glm::vec3(-10.f, 0, 0);
        cam.view = glm::vec3(1, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Right", side_size)) {
        cam.position = glm::vec3(10.f, 0, 0);
        cam.view = glm::vec3(-1, 0, 0);
        cam.up = glm::vec3(0, 1, 0);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Bottom", side_size)) {
        cam.position = glm::vec3(0, -10, 0);
        cam.view = glm::vec3(0, 1, 0);
        cam.up = glm::vec3(0, 0, 1);
        cam.right = glm::cross(cam.view, cam.up);
    }

    ImGui::PopStyleVar();

    auto ortho_size = ImGui::CalcTextSize(" Orthographic ");
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                        ImVec2(0.5f, 0.f));
    if (ImGui::Selectable("Perspective", cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = true;
    }
    ImGui::SameLine();

    if (ImGui::Selectable("Orthographic", !cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = false;
    }

    ImGui::SameLine();

    ImGui::PopStyleVar();

    ImGui::TextUnformatted("Projection");

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::SetNextItemWidth(digit_width * 6);
    if (cam.perspective) {
        ImGui::DragFloat("FOV", &cam.fov, 1.f, 1.f, 179.f, "%.0f");
    } else {
        ImGui::DragFloat("View Size", &cam.orthoHeight,
                          0.5f, 0.f, 100.f, "%0.1f");
    }

    ImGui::NewLine();
    ImGui::TextUnformatted("Overlay Rendering");

    ImGui::Checkbox("Enable Overlay", &cfg.showOverlay);
    ImGui::SetNextItemWidth(digit_width * 6);
    ImGui::DragFloat("Line Width", &cfg.lineWidth, 0.1f, 1.f, 10.f,
                     "%0.1f");
    ImGui::Checkbox("Edges always visible", &cfg.linesNoDepthTest);

    ImGui::End();
}

static void statusUI(EditorScene &scene, float frame_duration)
{
    auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkSize.x, 0.f),
                            0, ImVec2(1.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
    ImGui::Begin("FPS Counter", nullptr,
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    ImGui::Text("%.3f ms per frame (%.1f FPS)",
                1000.f * frame_duration, 1.f / frame_duration);

    ImGui::Text("Position: (%.1f, %.1f, %.1f)",
                scene.cam.position.x,
                scene.cam.position.y,
                scene.cam.position.z);
    ImGui::End();
}

void Editor::render(EditorScene &scene, float frame_duration)
{
    renderCFGUI(scene.overlayCfg, scene.cam);
    statusUI(scene, frame_duration);

    ImGui::Render();

    uint32_t total_vertices = 0;
    uint32_t total_tri_indices = 0;
    uint32_t total_line_indices = 0;

    CoverResults *cover_results = nullptr;
    if (scene.cover.showCover && scene.cover.results.size() > 0) {
        cover_results = &scene.cover.results.find(scene.cover.nearestCamPoint)->second;
    }

    if (scene.overlayCfg.showOverlay) {
        if (scene.cover.showNavmesh) {
            total_vertices += scene.cover.navmesh->overlayVerts.size();
            total_line_indices += scene.cover.navmesh->overlayIdxs.size();
        }

        if (scene.cover.showCover && cover_results != nullptr) {

            total_vertices += cover_results->overlayVerts.size();
            total_line_indices += cover_results->overlayIdxs.size();
        }
    }

    vector<OverlayVertex> tmp_verts(total_vertices);
    vector<uint32_t> tmp_indices(total_tri_indices + total_line_indices);

    if (scene.overlayCfg.showOverlay) {
        OverlayVertex *vert_ptr = tmp_verts.data();
        uint32_t *idx_ptr = tmp_indices.data();

        if (scene.cover.showNavmesh) {
            memcpy(vert_ptr, scene.cover.navmesh->overlayVerts.data(),
                   sizeof(OverlayVertex) * scene.cover.navmesh->overlayVerts.size());

            int vert_offset = vert_ptr - tmp_verts.data();
            vert_ptr += scene.cover.navmesh->overlayVerts.size();

            for (uint32_t idx : scene.cover.navmesh->overlayIdxs) {
                *idx_ptr++ = idx + vert_offset;
            }
        }

        if (scene.cover.showCover && cover_results != nullptr) {
            memcpy(vert_ptr, cover_results->overlayVerts.data(),
                   sizeof(OverlayVertex) * cover_results->overlayVerts.size());

            int vert_offset = vert_ptr - tmp_verts.data();
            vert_ptr += cover_results->overlayVerts.size();

            for (uint32_t idx : cover_results->overlayIdxs) {
                *idx_ptr++ = idx + vert_offset;
            }
        }
    }

    renderer_.render(scene.hdl, scene.cam, scene.overlayCfg,
                     tmp_verts.data(), tmp_indices.data(),
                     tmp_verts.size(), 
                     total_tri_indices, total_line_indices);
}

Editor::Editor(uint32_t gpu_id, uint32_t img_width, uint32_t img_height)
    : renderer_(gpu_id, img_width, img_height),
      cur_scene_idx_(0),
      scenes_()
{}

}
}

using namespace RLpbr;
using namespace RLpbr::editor;

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "%s width height scene.bps\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    std::filesystem::path p(argv[3]);
    Editor editor(0, stoul(argv[1]), stoul(argv[2]));
    editor.loadScene(argv[3], p.parent_path());

    editor.loop();
}
