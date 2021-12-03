#include <optional>
#include <vector>
#include <algorithm>
#include <queue>
#include <glm/glm.hpp>
#include <cmath>
#include "utils.hpp"
#define NUM_SUBTREES 8

namespace RLpbr {
namespace editor {


const int MIN_SIZE = 100;
const int MAX_ELEMENTS = 100;
const int MAX_DEPTH = 20;
const int X_INDEX = 4;
const int Y_INDEX = 2;
const int Z_INDEX = 1;

class Octree {
public:
    void getConnectedComponent(float radius, AABB &region, 
            std::vector<glm::vec3> &result_vecs, std::vector<uint64_t> &result_indices);
    void getPointsInAABB(AABB region, std::vector<glm::vec3> &result_vecs,
            std::vector<uint64_t> &result_indices, std::optional<AABB> ignore_region = {});
    bool removePointsInAABB(AABB region);

    void build(std::vector<glm::vec3> &points, std::vector<uint64_t> &indices);
    
private:
    void resize_indices(uint64_t num_indices);
    void resize_nodes(uint64_t num_nodes);

    // need one of everything except m_indices and m_points per node
    // number of indices is dependent on points, not number of notes in octree
    std::vector<glm::vec3> m_points;
    std::vector<uint64_t> m_indices;
    // the indices point to nodes, this is the indices of those indices
    std::vector<uint64_t> m_indices_start, m_indices_length;
    std::vector<std::array<uint64_t, NUM_SUBTREES>> m_subtrees;
    std::vector<AABB> m_regions;
    std::vector<bool> m_leafs;
    std::vector<uint64_t> m_depths;
};


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
    std::queue<uint64_t> frontier;
    frontier.push(0);

    for (; frontier.empty(); frontier.pop()) {
        uint64_t cur_node = frontier.front();
        if (ignore_region.has_value() && aabbContains(ignore_region.value(), m_regions[cur_node])) {
            continue;
        }
        if (aabbOverlap(region, m_regions[cur_node])) {
            if (m_leafs[cur_node]) {
                // the indices point to nodes, this is the index of and index
                for (uint64_t index_for_index = m_indices_start[cur_node]; 
                        index_for_index < m_indices_start[cur_node] + m_indices_length[cur_node]; 
                        index_for_index++) {
                    const glm::vec3 &point = m_points[index_for_index];
                    uint64_t index = m_indices[index_for_index];
                    if (aabbContains(region, point) && 
                            !(ignore_region.has_value() && aabbContains(ignore_region.value(), point))) {
                        result_vecs.push_back(point);
                        result_indices.push_back(index);
                    }
                }
            }
            else {
                for (auto &subtree : m_subtrees[cur_node]) {
                    frontier.push(subtree);
                }
            }
        }
    }
}

bool Octree::removePointsInAABB(AABB region) {
    /*
    if (aabbContains(region, m_regions)) {
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
    */
    return false;
}

void Octree::build(std::vector<glm::vec3> &points, std::vector<uint64_t> &indices) {
    // handle empty case
    if (points.empty()) {
        resize_indices(1);
        resize_nodes(1);
        m_regions[0] = {{0, 0, 0}, {0, 0, 0}};
        return;
    }

    // initialize points and size indices, which can be done
    // without knowledge of tree size
    m_points = points;
    resize_indices(points.size());

    uint64_t cur_node = 0, cur_point = 0;
    std::queue<std::vector<uint64_t>> indices_for_subtrees;
    std::queue<uint64_t> parent_node_for_subtrees;
    std::queue<uint64_t> index_in_parent_node_for_subtrees;
    indices_for_subtrees.push(indices);
    // init these with harmless garbage for first node
    parent_node_for_subtrees.push(0);
    index_in_parent_node_for_subtrees.push(0);
    while (!indices_for_subtrees.empty()) {
        // ensure vectors are large enough
        resize_nodes(cur_node + 1);

        // get indices relevant for current node
        const std::vector<uint64_t> &cur_indices = indices_for_subtrees.front();
        indices_for_subtrees.pop();
        const uint64_t &parent_node = parent_node_for_subtrees.front();
        parent_node_for_subtrees.pop();
        const uint64_t &index_in_parent_node = index_in_parent_node_for_subtrees.front();
        index_in_parent_node_for_subtrees.pop();
        
        // create AABB for all points in this region of octree
        AABB region = {points[0], points[0]};
        for (const auto &index : cur_indices) {
            region.pMin = glm::min(points[index], region.pMin);
            region.pMax = glm::max(points[index], region.pMax);
        }
        m_regions[cur_node] = region;

        // update parent node with pointer to child node
        m_subtrees[parent_node][index_in_parent_node] = cur_node;

        // set depth
        if (cur_node == 0) {
            m_depths[cur_node] = 0;
        }
        else {
            m_depths[cur_node] = m_depths[parent_node] + 1;
        }

        // if points fit in this node, no need to add anything else to frontier
        if (cur_indices.size() <= MAX_ELEMENTS || glm::all(glm::equal(region.pMin, region.pMax)) ||
                 m_depths[cur_node] >= MAX_DEPTH) {
            m_leafs[cur_node] = true;
            m_indices_start[cur_node] = cur_point;
            m_indices_length[cur_node] = cur_indices.size();
            std::copy(cur_indices.begin(), cur_indices.end(), m_indices.begin() + cur_point);
            cur_point += cur_indices.size();
        }
        // otherwise, split cur nodes points up and assign to sub nodes
        else {
            m_leafs[cur_node] = false;
            std::array<std::vector<uint64_t>, NUM_SUBTREES> cur_indices_for_subtrees;
            glm::vec3 middle = (region.pMin + region.pMax) / 2.0f;
            for (uint64_t i = 0; i < points.size(); i++) {
                const glm::vec3 &point = points[i];
                uint64_t index = indices[i];
                int subtree_index = (point.x <= middle.x ? 0 : X_INDEX) + 
                    (point.y <= middle.y ? 0 : Y_INDEX) + 
                    (point.z <= middle.z ? 0 : Z_INDEX);
                cur_indices_for_subtrees[subtree_index].push_back(index);
            }
            for (uint64_t i = 0; i < cur_indices_for_subtrees.size(); i++) {
                indices_for_subtrees.push(cur_indices_for_subtrees[i]);
                parent_node_for_subtrees.push(cur_node);
                index_in_parent_node_for_subtrees.push(i);
            }
        }
        cur_node++;
    }
    
}

void Octree::resize_indices(uint64_t required_indices) {
    if (m_indices.size() < required_indices) {
        uint64_t times_double_indices = std::ceil(std::log2(required_indices * 1.0f / MIN_SIZE));
        uint64_t num_indices = MIN_SIZE * std::round(std::pow(2, times_double_indices));
        m_indices.resize(num_indices);
    }
}

void Octree::resize_nodes(uint64_t required_nodes) {
    if (m_indices_start.size() < required_nodes) {
        uint64_t times_double_nodes = std::ceil(std::log2(required_nodes * 1.0f / MIN_SIZE));
        uint64_t num_nodes = MIN_SIZE * std::round(std::pow(2, times_double_nodes));
        m_indices_start.resize(num_nodes);
        m_indices_length.resize(num_nodes);
        m_subtrees.resize(num_nodes);
        m_regions.resize(num_nodes);
        m_leafs.resize(num_nodes);
        m_depths.resize(num_nodes);
    }
}

}
}
