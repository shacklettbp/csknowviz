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
#include "contiguous.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/ext/vector_common.hpp>

#include "shader.hpp"

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

void Editor::loadScene(const char *scene_name)
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

static optional<vector<AABB>> loadNavmeshCSV(const char *filename)
{
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

        getColumn();
        getColumn();
        getColumn();

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

        getline(csv, line);
    }

    return optional<vector<AABB>> {
        move(bboxes),
    };
}

template <typename IterT>
pair<vector<OverlayVertex>, vector<uint32_t>> generateAABBVerts(
    IterT begin, IterT end)
{
    vector<OverlayVertex> overlay_verts;
    vector<uint32_t> overlay_idxs;

    auto addVertex = [&](glm::vec3 pos) {
        overlay_verts.push_back({
            pos,
            glm::u8vec4(0, 255, 0, 255),
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

optional<NavmeshData> loadNavmesh()
{
    const char *filename = fileDialog();

    auto aabbs_opt = loadNavmeshCSV(filename);
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
    };
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

                if (pos2d.x >= 1430.f && pos2d.x <= 1440.f &&
                        pos2d.y >= 1930.f && pos2d.y <= 1940.f) {
                    launch_points.emplace_back(pos, aabb.pMax.y);
                }
            }
        }
    }

    optional<HostBuffer> voxel_staging;
    optional<LocalBuffer> voxel_buffer;
    uint32_t num_voxels = 0;
    uint64_t num_voxel_bytes = 0;
    {
        vector<GPUAABB> voxels_tmp;
        glm::vec3 voxel_size = {cover_data.voxelSizeXZ, 
            cover_data.voxelSizeY, cover_data.voxelSizeXZ};
        for (const AABB &aabb : cover_data.navmesh->aabbs) {
            glm::vec3 pmin = aabb.pMin;
            pmin.y += cover_data.torsoHeight;
            //pmin.y += 50;
            glm::vec3 pmax = aabb.pMax;
            pmax.y += cover_data.agentHeight;

            glm::vec3 diff = pmax - pmin;

            glm::i32vec3 num_fullsize(diff / voxel_size);

            glm::vec3 sample_extent = glm::vec3(num_fullsize) *
                voxel_size;

            glm::vec3 extra = diff - sample_extent;
            assert(extra.x >= 0 && extra.y >= 0 && extra.z >= 0);

            for (int i = 0; i <= num_fullsize.x; i++) {
                for (int j = 0; j <= num_fullsize.y; j++) {
                    for (int k = 0; k <= num_fullsize.z; k++) {
                        glm::vec3 cur_size(voxel_size);
                        if (i == num_fullsize.x) {
                            cur_size.x = extra.x;
                        }

                        if (j == num_fullsize.y) {
                            cur_size.y = extra.y;
                        }

                        if (k == num_fullsize.z) {
                            cur_size.z = extra.z;
                        }

                        float cur_volume = cur_size.x * cur_size.y * cur_size.z;
                        if (cur_volume == 0) {
                            continue;
                        }

                        glm::vec3 cur_pmin = pmin + glm::vec3(i, j, k) *
                            voxel_size;

                        glm::vec3 cur_pmax = cur_pmin + cur_size;

                        voxels_tmp.push_back(GPUAABB {
                            cur_pmin.x,
                            cur_pmin.y,
                            cur_pmin.z,
                            cur_pmax.x,
                            cur_pmax.y,
                            cur_pmax.z,
                        });
                    }
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


        std::unordered_map<glm::vec3, std::vector<glm::vec3>> originsToCandidates;
        std::vector<glm::vec3> candidates;
        for (uint64_t candidate_idx = 0; candidate_idx < num_candidates; candidate_idx++) {
            const auto &candidate = candidate_data[candidate_idx];
            candidates.push_back(candidate.hitPos);
            originsToCandidates[candidate.origin].push_back(candidate.hitPos);
            //cover_results[candidate.origin].aabbs.push_back({candidate.candidate - 1.0f, candidate.candidate + 1.0f});
            // inserting default values so can update them in parallel loop below
            glm::vec3 small(0.5, 0.5, 0.5);
            cover_results[candidate.origin].aabbs.push_back({candidate.hitPos - small, candidate.hitPos + small});
        }
        
        /*
        std::vector<glm::vec3> origins;
        for (const auto &originAndCandidates : originsToCandidates) {
            origins.push_back(originAndCandidates.first);
        }

        //std::chrono::steady_clock::time_point begin_cluster = std::chrono::steady_clock::now();
        #pragma omp parallel for
        for (int origin_idx = 0; origin_idx < (int) origins.size(); origin_idx++) {
            glm::vec3 origin = origins[origin_idx];
            ContiguousClusters clusters(originsToCandidates[origin]);
            cover_results[origins[origin_idx]].aabbs = clusters.getClusters();
        }
        */
        //std::chrono::steady_clock::time_point end_cluster = std::chrono::steady_clock::now();
        //std::cout << origins.size() << " cluster time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cluster - begin_cluster).count() << "[ms]" << std::endl;

    }

    cout << "Unique origin points: " << cover_results.size() << endl;

    for (auto &[_, result] : cover_results) {
        auto [overlay_verts, overlay_idxs] =
            generateAABBVerts(result.aabbs.begin(), result.aabbs.end());
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

        float min_dist = INFINITY;
        for (const auto &[key_pos, _] : cover.results) {
            float dist = glm::distance(key_pos, cur_pos);

            if (dist < min_dist) {
                min_dist = dist;
                cover.nearestCamPoint = key_pos;
            }
        }
    }

    ImGui::Separator();

    ImGui::Checkbox("Show Navmesh", &cover.showNavmesh);
    ImGui::Checkbox("Show Cover", &cover.showCover);

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::PushItemWidth(digit_width * 6);
    ImGui::DragFloat("Sample Spacing", &cover.sampleSpacing, 0.1f, 0.1f, 100.f, "%.1f");
    ImGui::DragFloat("Voxel Size XZ", &cover.voxelSizeXZ,
                     0.1f, 0.1f, 100.f, "%.1f");

    ImGui::DragFloat("Voxel Size Y", &cover.voxelSizeY,
                     0.1f, 0.1f, 100.f, "%.1f");

    ImGui::DragFloat("Agent Height", &cover.agentHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragFloat("Torso Height", &cover.torsoHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragInt("# Offset Samples (sqrt)", &cover.sqrtOffsetSamples, 1, 1, 1000);
    ImGui::DragFloat("Offset Radius", &cover.offsetRadius, 0.01f, 0.f, 100.f,
                     "%.2f");
    ImGui::DragInt("# Voxel Tests", &cover.numVoxelTests, 1, 1, 1000);

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

    Editor editor(0, stoul(argv[1]), stoul(argv[2]));
    editor.loadScene(argv[3]);

    editor.loop();
}
