#pragma once

#include <glm/vector_relational.hpp>
#include "vulkan/render.hpp"
#include "vulkan/scene.hpp"

namespace RLpbr {
namespace editor {

template <size_t N>
struct Pipeline {
    vk::ShaderPipeline shader;
    VkPipelineLayout layout;
    std::array<VkPipeline, N> hdls;
    vk::FixedDescriptorPool descPool;
};

struct Framebuffer {
    vk::LocalImage colorAttachment;
    vk::LocalImage depthAttachment;
    VkImageView colorView;
    VkImageView depthView;

    VkFramebuffer hdl;
};

struct OverlayVertex {
    glm::vec3 position;
    glm::u8vec4 color;
};

struct HostRenderInput {
    vk::HostBuffer buffer;
    InstanceTransform *transformPtr;
    uint32_t *matIndices;
    OverlayVertex *overlayVertices;
    uint32_t *overlayIndices;
    uint32_t overlayIndexOffset;
    uint32_t maxTransforms;
    uint32_t maxMatIndices;
    uint32_t maxOverlayVertices;
    uint32_t maxOverlayIndices;
};

struct Frame {
    Framebuffer fb;
    VkCommandPool cmdPool;
    VkCommandBuffer drawCmd;
    VkFence cpuFinished;
    VkSemaphore renderFinished;
    VkSemaphore swapchainReady;
    VkDescriptorSet defaultShaderSet;
    VkDescriptorSet overlayShaderSet;
    HostRenderInput renderInput;
};

struct EditorCam {
    glm::vec3 position;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;

    bool perspective = true;
    float fov = 90.f;
    float orthoHeight = 5.f;
    glm::vec2 mousePrev {0.f, 0.f};
};

template <int N>
struct ComputeContext {
    const vk::DeviceState &dev;
    vk::MemoryAllocator &alloc;
    VkQueue computeQueue;
    VkFence fence;
    VkCommandPool cmdPool;
    VkCommandBuffer cmdBuffer;

    std::array<Pipeline<1>, N> pipelines;
    std::array<VkDescriptorSet, N> descSets;
};

class EditorVkScene {
public:
    std::shared_ptr<Scene> scene;
    vk::TLAS tlas;
    vk::DescriptorSet renderDescSet;
    vk::DescriptorSet computeDescSet;
};

class Renderer {
public:
    struct OverlayConfig {
        bool showOverlay = true;
        float lineWidth = 2.f;
        bool linesNoDepthTest = false;
    };

    Renderer(uint32_t gpu_id, uint32_t img_width,
             uint32_t img_height);
    Renderer(const Renderer &) = delete;
    GLFWwindow *getWindow();

    EditorVkScene loadScene(SceneLoadData &&load_data);

    void waitUntilFrameReady();
    void startFrame();
    void render(EditorVkScene &editor_scene, const EditorCam &cam,
                const OverlayConfig &cfg,
                const OverlayVertex *extra_vertices,
                const uint32_t *extra_indices,
                uint32_t num_extra_vertices,
                uint32_t num_overlay_tri_indices,
                uint32_t num_overlay_line_indices);

    static constexpr int numCoverShaders = 3;
    ComputeContext<numCoverShaders> & getCoverContext();

    void waitForIdle();

    vk::InstanceState inst;
    vk::DeviceState dev;
    vk::MemoryAllocator alloc;

private:
    VkQueue render_queue_;
    VkQueue transfer_queue_;
    VkQueue compute_queue_;
    VkQueue render_transfer_queue_;
    VkQueue compute_transfer_queue_;

    // Fixme remove
    vk::QueueState transfer_wrapper_;
    vk::QueueState render_transfer_wrapper_;
    vk::QueueState present_wrapper_;

    glm::u32vec2 fb_dims_;
    std::array<VkClearValue, 2> fb_clear_;
    vk::PresentationState present_;
    VkPipelineCache pipeline_cache_;
    VkSampler repeat_sampler_;
    VkSampler clamp_sampler_;
    VkRenderPass render_pass_;
    VkRenderPass gui_render_pass_;
    Pipeline<1> default_pipeline_;
    Pipeline<3> overlay_pipeline_;

    ComputeContext<numCoverShaders> cover_context_;

    vk::DescriptorManager scene_render_pool_;
    vk::DescriptorManager scene_compute_pool_;

    uint32_t cur_frame_;
    DynArray<Frame> frames_;


    vk::VulkanLoader loader_;
};

}
}
