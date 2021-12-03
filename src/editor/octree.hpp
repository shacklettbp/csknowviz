#include <optional>
#include <vector>
#include <algorithm>
#include <glm/glm.hpp>
#include "utils.hpp"

namespace RLpbr {
namespace editor {

class Octree {
public:
    int getMaxDepth() const {
        if (m_subtrees.empty()) {
            return 1;
        }
        else {
            int max_depth = 0;
            for (const auto &subtree : m_subtrees) {
                max_depth = std::max(max_depth, subtree.getMaxDepth());
            }
            return max_depth + 1;
        }
    }
    void getConnectedComponent(float radius, AABB &region, 
            std::vector<glm::vec3> &result_vecs, std::vector<uint64_t> &result_indices);
    void getPointsInAABB(AABB region, std::vector<glm::vec3> &result_vecs,
            std::vector<uint64_t> &result_indices, std::optional<AABB> ignore_region = {});
    bool removePointsInAABB(AABB region);

    Octree(std::vector<glm::vec3> points, std::vector<uint64_t> indices, int cur_depth = 1);
    Octree() {};
    
private:
    std::vector<Octree> m_subtrees;
    std::vector<glm::vec3> m_elements;
    std::vector<uint64_t> m_indices;
    AABB m_region;
};

const int MIN_SIZE = 100;
const int MAX_DEPTH = 20;
const int X_INDEX = 4;
const int Y_INDEX = 2;
const int Z_INDEX = 1;
const int NUM_SUBTREES = 8;

inline bool aabbContains(AABB outer, AABB inner) {
    return 
        (outer.pMin.x <= inner.pMin.x && outer.pMax.x >= inner.pMax.x) &&
        (outer.pMin.y <= inner.pMin.y && outer.pMax.y >= inner.pMax.y) &&
        (outer.pMin.z <= inner.pMin.z && outer.pMax.z >= inner.pMax.z);
}

inline bool aabbOverlap(AABB b0, AABB b1) {
    return
        (b0.pMin.x <= b1.pMax.x && b1.pMin.x <= b0.pMax.x) &&
        (b0.pMin.y <= b1.pMax.y && b1.pMin.y <= b0.pMax.y) &&
        (b0.pMin.z <= b1.pMax.z && b1.pMin.z <= b0.pMax.z);
}

inline bool aabbContains(AABB region, glm::vec3 point) {
    return 
        (region.pMin.x <= point.x && region.pMax.x >= point.x) &&
        (region.pMin.y <= point.y && region.pMax.y >= point.y) &&
        (region.pMin.z <= point.z && region.pMax.z >= point.z);
}

void Octree::getConnectedComponent(float radius, AABB &region, std::vector<glm::vec3> &result_vecs, 
        std::vector<uint64_t> &result_indices) {
    AABB old_region = {{0,0,0},{0,0,0}};
    std::optional<AABB> ignore_region = {};
    while (glm::any(glm::greaterThan(old_region.pMin, region.pMin)) ||
            glm::any(glm::lessThan(old_region.pMax, region.pMax))) {
        getPointsInAABB(region, result_vecs, result_indices, ignore_region);
        old_region = region;
        ignore_region = old_region;
        for (const auto &result_vec: result_vecs) {
            region.pMin = glm::min(region.pMin, result_vec - radius);
            region.pMax = glm::max(region.pMax, result_vec + radius);
        }
    }
}

void Octree::getPointsInAABB(AABB region, std::vector<glm::vec3> &result_vecs, 
        std::vector<uint64_t> &result_indices, std::optional<AABB> ignore_region) {
    if (ignore_region.has_value() && aabbContains(ignore_region.value(), m_region)) {
        return;
    }
    if (aabbOverlap(region, m_region)) {
        if (m_subtrees.empty()) {
            for (uint64_t i = 0; i < m_elements.size(); i++) {
                const glm::vec3 &element = m_elements[i];
                uint64_t index = m_indices[i];
                if (aabbContains(region, element) && 
                        !(ignore_region.has_value() && aabbContains(ignore_region.value(), element))) {
                    result_vecs.push_back(element);
                    result_indices.push_back(index);
                }
            }
        }
        else {
            for (auto &subtree : m_subtrees) {
                subtree.getPointsInAABB(region, result_vecs, result_indices, ignore_region);
            }
        }
    }
}

bool Octree::removePointsInAABB(AABB region) {
    if (aabbContains(region, m_region)) {
        m_region = {{0, 0, 0}, {0, 0, 0}};
        m_subtrees.clear();
        m_elements.clear();
        return true;
    }
    else if (aabbOverlap(region, m_region)) {
        bool all_empty = true;
        for (auto &subtree : m_subtrees) {
            all_empty &= subtree.removePointsInAABB(region);
        }
        if (all_empty) {
            m_region = {{0, 0, 0}, {0, 0, 0}};
            m_subtrees.clear();
            return true;
        }
    }
    return false;
}

Octree::Octree(std::vector<glm::vec3> points, std::vector<uint64_t> indices, int cur_depth) {
    if (points.empty()) {
        m_region = {{0, 0, 0}, {0, 0, 0}};
        return;
    }
    m_region = {points[0], points[0]};

    for (const auto &point: points) {
        m_region.pMin = glm::min(point, m_region.pMin);
        m_region.pMax = glm::max(point, m_region.pMax);
    }
    
    if (points.size() < MIN_SIZE || glm::all(glm::equal(m_region.pMin, m_region.pMax)) ||
             cur_depth >= MAX_DEPTH) {
        m_elements = points;
        m_indices = indices;
    }
    else {
        std::vector<std::vector<glm::vec3>> elements_for_subtrees(NUM_SUBTREES);
        std::vector<std::vector<uint64_t>> indices_for_subtrees(NUM_SUBTREES);
        glm::vec3 m_avg = (m_region.pMin + m_region.pMax) / 2.0f;
        for (uint64_t i = 0; i < points.size(); i++) {
            const glm::vec3 &point = points[i];
            uint64_t index = indices[i];
            int subtree_index = (point.x <= m_avg.x ? 0 : X_INDEX) + 
                (point.y <= m_avg.y ? 0 : Y_INDEX) + 
                (point.z <= m_avg.z ? 0 : Z_INDEX);
            elements_for_subtrees[subtree_index].push_back(point);
            indices_for_subtrees[subtree_index].push_back(index);
        }
        for (int subtree_index = 0; subtree_index < NUM_SUBTREES; subtree_index++) {
            m_subtrees.push_back(Octree(elements_for_subtrees[subtree_index], indices_for_subtrees[subtree_index], cur_depth + 1));
        }
    }
}

}
}
