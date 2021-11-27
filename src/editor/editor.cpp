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

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

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

                launch_points.emplace_back(pos, 0.f);
            }
        }
    }

    cout << "Finding ground for " << launch_points.size() << " points." <<
        endl;

    uint64_t num_launch_bytes = launch_points.size() * sizeof(glm::vec4);
    HostBuffer ground_points =
        alloc.makeHostBuffer(num_launch_bytes, true);

    memcpy(ground_points.ptr, launch_points.data(), num_launch_bytes);

    ground_points.flush(dev);

    uint32_t max_candidates = 125952 * 2;
    uint64_t num_candidate_bytes = max_candidates * sizeof(CandidatePair);
    uint64_t extra_candidate_bytes =
        alloc.alignStorageBufferOffset(sizeof(uint32_t));

#if 0
    optional<LocalBuffer> candidate_buffer_opt = alloc.makeLocalBuffer(
        num_candidate_bytes + extra_candidate_bytes, true);
    if (!candidate_buffer_opt.has_value()) {
        cerr << "Out of memory while allocating intermediate buffer" << endl;
        abort();
    }
    LocalBuffer candidate_buffer = move(*candidate_buffer_opt);
#endif
    HostBuffer candidate_buffer = alloc.makeHostBuffer(
        num_candidate_bytes + extra_candidate_bytes, true);
    candidate_buffer.flush(dev);

    DescriptorUpdates desc_updates(3);
    VkDescriptorBufferInfo ground_info;
    ground_info.buffer = ground_points.buffer;
    ground_info.offset = 0;
    ground_info.range = num_launch_bytes;

    desc_updates.storage(ctx.descSets[0], &ground_info, 0);
    desc_updates.storage(ctx.descSets[1], &ground_info, 0);

    VkDescriptorBufferInfo filter_count_info;
    filter_count_info.buffer = candidate_buffer.buffer;
    filter_count_info.offset = 0;
    filter_count_info.range = sizeof(uint32_t);;
    desc_updates.storage(ctx.descSets[1], &filter_count_info, 1);

    VkDescriptorBufferInfo filter_candidates_info;
    filter_candidates_info.buffer = candidate_buffer.buffer;
    filter_candidates_info.offset = extra_candidate_bytes;
    filter_candidates_info.range = num_candidate_bytes;

    desc_updates.storage(ctx.descSets[1], &filter_candidates_info, 2);

    desc_updates.update(dev);

    REQ_VK(dev.dt.resetCommandPool(dev.hdl, ctx.cmdPool, 0));

    VkCommandBuffer cmd = ctx.cmdBuffer;

    VkCommandBufferBeginInfo begin_info {};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

    dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                           ctx.pipelines[0].hdls[0]);

    CoverPushConst push_const;
    push_const.idxOffset = 0;
    push_const.numGroundSamples = launch_points.size();
    push_const.sqrtSphereSamples = cover_data.sqrtSphereSamples;
    push_const.agentHeight = cover_data.agentHeight;
    push_const.cornerEpsilon = cover_data.cornerEpsilon;
    push_const.originJitter =
        cover_data.sampleSpacing / cover_data.originJitterDiv;

    dev.dt.cmdPushConstants(cmd, ctx.pipelines[0].layout,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(CoverPushConst), 
                            &push_const);

    array<VkDescriptorSet, 2> bind_sets;
    bind_sets[0] = ctx.descSets[0];
    bind_sets[1] = scene.hdl.computeDescSet.hdl;

    dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 ctx.pipelines[0].layout,
                                 0, bind_sets.size(), bind_sets.data(),
                                 0, nullptr);

    dev.dt.cmdDispatch(cmd, divideRoundUp(uint32_t(launch_points.size()), 32u), 1, 1);

    VkMemoryBarrier barrier;
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.pNext = nullptr;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    dev.dt.cmdPipelineBarrier(cmd,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              0,
                              1, &barrier,
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

    uint32_t points_per_dispatch = max_candidates / 
        (cover_data.sqrtSphereSamples * cover_data.sqrtSphereSamples);

    int num_iters = launch_points.size() / points_per_dispatch;

    if (num_iters * points_per_dispatch < launch_points.size()) {
        num_iters++;
    }

    auto &cover_results = cover_data.results;
    for (int i = 0; i < num_iters; i++) {
        memset(candidate_buffer.ptr, 0, sizeof(uint32_t));

        REQ_VK(dev.dt.resetCommandPool(dev.hdl, ctx.cmdPool, 0));
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &begin_info));

        dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                               ctx.pipelines[1].hdls[0]);

        bind_sets[0] = ctx.descSets[1];

        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                     ctx.pipelines[1].layout,
                                     0, bind_sets.size(), bind_sets.data(),
                                     0, nullptr);

        push_const.idxOffset = i * points_per_dispatch;

        dev.dt.cmdPushConstants(cmd, ctx.pipelines[1].layout,
                                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                sizeof(CoverPushConst), 
                                &push_const);

        uint32_t dispatch_points = min(uint32_t(launch_points.size() - push_const.idxOffset),
                                       points_per_dispatch);

        dev.dt.cmdDispatch(cmd, divideRoundUp(cover_data.sqrtSphereSamples, 8),
                           divideRoundUp(cover_data.sqrtSphereSamples, 4),
                           dispatch_points);

        REQ_VK(dev.dt.endCommandBuffer(cmd));

        REQ_VK(dev.dt.resetFences(dev.hdl, 1, &ctx.fence));

        REQ_VK(dev.dt.queueSubmit(ctx.computeQueue, 1, &submit,
                                  ctx.fence));

        waitForFenceInfinitely(dev, ctx.fence);

        uint32_t num_candidates;
        memcpy(&num_candidates, candidate_buffer.ptr, sizeof(uint32_t));
        cout << "Iter " << i << ": Found " << num_candidates << " candidate corner points" << endl;

        assert(num_candidates < max_candidates);

        CandidatePair *candidate_data =
            (CandidatePair *)((char *)candidate_buffer.ptr + extra_candidate_bytes);


        for (int candidate_idx = 0; candidate_idx < (int)num_candidates; candidate_idx++) {
            const auto &candidate = candidate_data[candidate_idx];
            //cout << glm::to_string(candidate.origin) << " " <<
            //    glm::to_string(candidate.candidate) << "\n";
            glm::vec3 pMin = candidate.candidate;
            pMin.x = std::floor(pMin.x / 0.1f) * 0.1f;
            pMin.y = std::floor(pMin.y / 0.1f) * 0.1f;
            pMin.z = std::floor(pMin.z / 0.1f) * 0.1f;
            glm::vec3 pMax = candidate.candidate;
            pMax.x = std::ceil(pMax.x / 0.1f) * 0.1f;
            pMax.y = std::ceil(pMax.y / 0.1f) * 0.1f;
            pMax.z = std::ceil(pMax.z / 0.1f) * 0.1f;
            std::cout << "saving origin " << glm::to_string(candidate.origin) << 
                "and candidate pmin " << glm::to_string(pMin) << ", pmax " << glm::to_string(pMax) << std::endl;
            bool hit = false;
            if (cover_results.find(candidate.origin) != cover_results.end()) {
                hit = true;
                AABB f = *(cover_results.find(candidate.origin)->second.aabbs.begin());
                auto compare = compareAABB{};
                std::cout << "a lt b: " << compare(f, {pMin, pMax}) << " and b lt a " << compare({pMin, pMax}, f) << std::endl;
                if (compare(f, {pMin, pMax}) == compare({pMin, pMax}, f)) {
                    std::cout << "hi" << std::endl;
                }
            }
            cover_results[candidate.origin].aabbs.insert({pMin, pMax});
            if (hit) {
                std::cout << "new set size " << cover_results[candidate.origin].aabbs.size() << std::endl; 
            }
        }

        for (auto &originAndAABBs : cover_results) {
            float boxSize = 0.2f;
            std::set<AABB, compareAABB> resultAABBs = originAndAABBs.second.aabbs;
            std::set<AABB, compareAABB> largerAABBs; 
            int initAABBSize = resultAABBs.size();
            while (true) {
                for (const auto origAABB : resultAABBs) {
                    AABB largerAABB;
                    largerAABB.pMin.x = std::floor(origAABB.pMin.x / boxSize) * boxSize;
                    largerAABB.pMin.y = std::floor(origAABB.pMin.y / boxSize) * boxSize;
                    largerAABB.pMin.z = std::floor(origAABB.pMin.z / boxSize) * boxSize;
                    largerAABB.pMax.x = std::ceil(origAABB.pMax.x / boxSize) * boxSize;
                    largerAABB.pMax.y = std::ceil(origAABB.pMax.y / boxSize) * boxSize;
                    largerAABB.pMax.z = std::ceil(origAABB.pMax.z / boxSize) * boxSize;
                    largerAABBs.insert(largerAABB);
                }
                if (largerAABBs.size() != resultAABBs.size()) {
                    resultAABBs = largerAABBs;
                    boxSize *= 2;
                }
                else {
                    break;
                }
            }
            if (resultAABBs.size() != originAndAABBs.second.aabbs.size()) {
                originAndAABBs.second.aabbs = resultAABBs;
            }
            std::cout << "for origin " << glm::to_string(originAndAABBs.first)
                << " decreased " << initAABBSize << " to " << resultAABBs.size() << std::endl;
        }
        //cout << endl;
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

static void handleCover(EditorScene &scene,
                        const ComputeContext<Renderer::numCoverShaders> &ctx)
{
    CoverData &cover = scene.cover;

    ImGui::Begin("Cover Detection");

    if (ImGui::Button("Load Navmesh")) {
        cover.navmesh = loadNavmesh();
        cover.showNavmesh = true;
    }
    ImGuiEXT::PushDisabled(!cover.navmesh.has_value());

    ImGui::SameLine();

    if (ImGui::Button("Detect Cover")) {
        detectCover(scene, ctx);
    }

    ImGui::Separator();

    ImGui::Checkbox("Show Navmesh", &cover.showNavmesh);
    ImGui::Checkbox("Show Cover", &cover.showCover);

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::PushItemWidth(digit_width * 6);
    ImGui::DragFloat("Sample Spacing", &cover.sampleSpacing, 0.1f, 0.1f, 10.f, "%.1f");
    ImGui::DragFloat("Agent Height", &cover.agentHeight, 1.f, 1.f, 200.f, "%.0f");
    ImGui::DragInt("# Sphere Samples (sqrt)", &cover.sqrtSphereSamples, 1, 1, 1000);
    ImGui::DragFloat("Jitter Divisor", &cover.originJitterDiv, 0.1f, 1.f, 100.f, "%.1f");
    ImGui::DragFloat("Corner Epsilon", &cover.cornerEpsilon, 0.01f, 0.01f, 100.f, "%.2f");

    ImGui::PopItemWidth();

    ImGuiEXT::PopDisabled();

    ImGui::End();

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
        handleCover(scene, renderer_.getCoverContext());
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

static void fpsCounterUI(float frame_duration)
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

    ImGui::End();
}

void Editor::render(EditorScene &scene, float frame_duration)
{
    renderCFGUI(scene.overlayCfg, scene.cam);
    fpsCounterUI(frame_duration);

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
