#include <optional>
#include <vector>
#include <algorithm>
#include <queue>
#include <stack>
#include <glm/glm.hpp>
#include <cmath>
#include <omp.h>
#include <random>
#include "utils.hpp"
#define NUM_AXIS 3

namespace RLpbr {
namespace editor {

class ContiguousClusters {
public:
    std::vector<AABB> getClusters();
    ContiguousClusters(std::vector<glm::vec3> &all_points, float bin_width = 5.0f, uint64_t histogram_samples = 200);
    
private:
    std::vector<AABB> m_clusters;
    uint64_t getBinIndex(glm::vec3 point, AABB outer_region, int axis);
    std::vector<bool> getBinaryHistogram(std::vector<glm::vec3> &points, AABB region, int axis);
    inline AABB binIndexToAABB(uint64_t start_bin_index, uint64_t end_bin_index, AABB region, int arg_axis);

    const float BIN_WIDTH;
    const uint64_t HISTOGRAM_SAMPLES;
    uint64_t num_points_considered;
};


inline bool 
aabbContains(AABB outer, AABB inner) {
    return 
        (outer.pMin.x <= inner.pMin.x && outer.pMax.x >= inner.pMax.x) &&
        (outer.pMin.y <= inner.pMin.y && outer.pMax.y >= inner.pMax.y) &&
        (outer.pMin.z <= inner.pMin.z && outer.pMax.z >= inner.pMax.z);
}


inline bool 
aabbOverlap(AABB b0, AABB b1) {
    return
        (b0.pMin.x <= b1.pMax.x && b1.pMin.x <= b0.pMax.x) &&
        (b0.pMin.y <= b1.pMax.y && b1.pMin.y <= b0.pMax.y) &&
        (b0.pMin.z <= b1.pMax.z && b1.pMin.z <= b0.pMax.z);
}


inline bool 
aabbContains(AABB region, glm::vec3 point) {
    return 
        (region.pMin.x <= point.x && region.pMax.x >= point.x) &&
        (region.pMin.y <= point.y && region.pMax.y >= point.y) &&
        (region.pMin.z <= point.z && region.pMax.z >= point.z);
}


std::vector<AABB> 
ContiguousClusters::getClusters() {
    return m_clusters;
}


inline uint64_t 
ContiguousClusters::getBinIndex(glm::vec3 point, AABB region, int axis) {
    return std::floor(point[axis] / BIN_WIDTH) - std::floor(region.pMin[axis] / BIN_WIDTH);
}


inline AABB 
ContiguousClusters::binIndexToAABB(uint64_t start_bin_index, uint64_t end_bin_index, AABB region, int axis) {
    AABB result = region;
    result.pMin[axis] = region.pMin[axis] + BIN_WIDTH * start_bin_index;
    // plus 1 as include all points up to next bin
    result.pMax[axis] = region.pMin[axis] + BIN_WIDTH * (end_bin_index + 1);
    return result;
}


std::vector<bool> 
ContiguousClusters::getBinaryHistogram(std::vector<glm::vec3> &points, AABB region, int axis) {
    std::vector<bool> result(getBinIndex(region.pMax, region, axis) + 1);

    for (uint64_t bin_num = 0; bin_num < result.size(); bin_num++) {
        for (const auto &point : points) {
            num_points_considered++;
            if (aabbContains(region, point) && getBinIndex(point, region, axis) == bin_num) {
                result[bin_num] = true;
                break;
            }
        }
    }

    return result;
}


ContiguousClusters::ContiguousClusters(std::vector<glm::vec3> &all_points, float bin_width,
        uint64_t histogram_samples) : BIN_WIDTH(bin_width), HISTOGRAM_SAMPLES(histogram_samples) {

    std::chrono::steady_clock::time_point begin_clock = std::chrono::steady_clock::now();
    num_points_considered = 0;
    // handle empty case
    if (all_points.empty()) {
        m_clusters.push_back({{0, 0, 0}, {0, 0, 0}});
        return;
    }

    auto rng = std::default_random_engine {};
    std::vector<glm::vec3> points; 
    std::sample(all_points.begin(), all_points.end(), std::back_inserter(points), HISTOGRAM_SAMPLES, rng);

    AABB outer_region{points[0], points[0]};
    for (const auto &point : points) {
        outer_region.pMin = glm::min(point, outer_region.pMin);
        outer_region.pMax = glm::max(point, outer_region.pMax);
    }

    // before looking at each axis, this stores the contigous regions that will be 
    // split up by looking at the region
    // need 1 extra at end as will take last entry as final result
    std::vector<AABB> contiguous_regions_before_axis[NUM_AXIS+1];
    contiguous_regions_before_axis[0].push_back(outer_region);
    for (int axis = 0; axis < NUM_AXIS; axis++) {
        std::vector<std::vector<AABB>> tmp_regions(omp_get_max_threads());
//        #pragma omp parallel for
        for (uint64_t region_index = 0; 
                region_index < contiguous_regions_before_axis[axis].size(); 
                region_index++) {
            int thread_num = omp_get_thread_num();
            std::vector<bool> binary_histogram = 
                getBinaryHistogram(points, contiguous_regions_before_axis[axis][region_index], axis);

            // for loop through histogram, collecting contiguous regions
            bool in_region = false;
            uint64_t region_start_bin_index;
            for (uint64_t binIndex = 0; binIndex < binary_histogram.size(); binIndex++) {
                // case 1: looking for a region and found a filled bin
                if (!in_region && binary_histogram[binIndex]) {
                    in_region = true;
                    region_start_bin_index = binIndex;
                }
                // case 2: in a region and found an empty bin
                else if (in_region && !binary_histogram[binIndex]) {
                    in_region = false;
                    contiguous_regions_before_axis[axis+1]
                        .push_back(binIndexToAABB(region_start_bin_index, binIndex - 1, 
                                    contiguous_regions_before_axis[axis][region_index], axis));
                }
            }
            // if end histogram while active, write result
            if (in_region) {
                contiguous_regions_before_axis[axis+1]
                    .push_back(binIndexToAABB(region_start_bin_index, binary_histogram.size() - 1, 
                                contiguous_regions_before_axis[axis][region_index], axis));
            }
        }
        std::cout << "at axis " << axis << " found num regions " << 
                    contiguous_regions_before_axis[axis+1].size() << std::endl;
    }
    m_clusters = contiguous_regions_before_axis[NUM_AXIS];    

    std::chrono::steady_clock::time_point end_clock = std::chrono::steady_clock::now();
    std::cout << "contiguous time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - begin_clock).count() << "[ms]" << std::endl;

    std::cout << "processed " << num_points_considered << " points" << std::endl;
}

}
}
