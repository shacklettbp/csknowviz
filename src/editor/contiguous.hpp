#include <optional>
#include <vector>
#include <algorithm>
#include <queue>
#include <stack>
#include <glm/glm.hpp>
#include <cmath>
#include <omp.h>
#include <random>
#include <chrono>
#include "utils.hpp"
#define NUM_AXIS 3

namespace RLpbr {
namespace editor {

class ContiguousClusters {
public:
    std::vector<AABB> getClusters();
    ContiguousClusters(std::vector<glm::vec3> &points, float bin_width = 20.0f);
    
private:
    struct Bin {
        int64_t index_start;
        AABB aabb;
    };

    std::vector<AABB> m_clusters;
    uint64_t getHistogramIndex(glm::vec3 point, AABB outer_region, int axis);
    inline AABB histIndexToAABB(uint64_t start_hist_index, uint64_t end_hist_index, const AABB &region, int arg_axis);
    void getBins(const std::vector<glm::vec3> &points, const std::vector<uint64_t> &input_indices, 
            uint64_t num_indices, const AABB &region, int axis, 
            std::vector<Bin> &output_bins, std::vector<uint64_t> &output_indices);
    void getBins(const std::vector<glm::vec3> &points, const std::vector<uint64_t> &input_indices, 
        uint64_t input_indices_start, uint64_t input_indices_length, const AABB &region, int axis, 
        std::vector<Bin> &output_bins, std::vector<uint64_t> &output_indices, uint64_t output_indices_start);

    const float HISTOGRAM_WIDTH;
};


std::vector<AABB> 
ContiguousClusters::getClusters() {
    return m_clusters;
}


inline uint64_t 
ContiguousClusters::getHistogramIndex(glm::vec3 point, AABB region, int axis) {
    return std::floor(point[axis] / HISTOGRAM_WIDTH) - std::floor(region.pMin[axis] / HISTOGRAM_WIDTH);
}


inline AABB 
ContiguousClusters::histIndexToAABB(uint64_t start_hist_index, uint64_t end_hist_index, const AABB &region, int axis) {
    AABB result = region;
    result.pMin[axis] = region.pMin[axis] + HISTOGRAM_WIDTH * start_hist_index;
    // plus 1 as include all points up to next hist
    result.pMax[axis] = region.pMin[axis] + HISTOGRAM_WIDTH * (end_hist_index + 1);
    return result;
}


// create a histogram, then group together adjacent parts of histogram into bins
// return bins by adding them to output_bins, put indices in sorted order for bins with output_indices
void
ContiguousClusters::getBins(const std::vector<glm::vec3> &points, const std::vector<uint64_t> &input_indices, 
        uint64_t input_indices_start, uint64_t input_indices_length, const AABB &region, int axis, 
        std::vector<Bin> &output_bins, std::vector<uint64_t> &output_indices, uint64_t output_indices_start) {
    // compute histogram
    std::vector<uint64_t> histogram(getHistogramIndex(region.pMax, region, axis) + 1);
    for (uint64_t index_index = input_indices_start; 
            index_index < input_indices_start + input_indices_length; 
            index_index++) {
        uint64_t point_index = input_indices[index_index];
        histogram[getHistogramIndex(points[point_index], region, axis)]++;
    }

    // for loop through histogram, getting bin starts and map from histogram to bin
    std::vector<Bin> bins;
    std::vector<uint64_t> bin_points_added_to_output;
    std::vector<uint64_t> histogram_to_bin(getHistogramIndex(region.pMax, region, axis) + 1);
    bool in_region = false;
    uint64_t cur_bin_length = 0;
    for (uint64_t hist_index = 0; hist_index < histogram.size(); hist_index++) {
        // mark start of bin 
        if (!in_region && histogram[hist_index] != 0) {
            in_region = true;
            if (bins.size() == 0) {
                bins.push_back({});
                bins.back().index_start = output_indices_start;
            }
            else {
                uint64_t last_bin_start = bins.back().index_start;
                bins.push_back({});
                bins.back().index_start = last_bin_start + cur_bin_length;
            }
            bin_points_added_to_output.push_back(0);
            cur_bin_length = 0;
        }
        else if (in_region && histogram[hist_index] == 0) {
            in_region = false;
        }

        // record data for histogram to bin and bin length while in bin
        if (in_region) {
            histogram_to_bin[hist_index] = bins.size() - 1;
            cur_bin_length += histogram[hist_index];
        }
    }

    // send points to bins via histogram
    for (uint64_t index_index = input_indices_start; 
            index_index < input_indices_start + input_indices_length; 
            index_index++) {
        uint64_t point_index = input_indices[index_index];
        const glm::vec3 &point = points[point_index];
        uint64_t hist_index = getHistogramIndex(point, region, axis);
        uint64_t bin_index = histogram_to_bin[hist_index];
        if (bin_points_added_to_output[bin_index] == 0) {
            bins[bin_index].aabb = {point, point};
        }
        else {
            bins[bin_index].aabb.pMin = glm::min(point, bins[bin_index].aabb.pMin);
            bins[bin_index].aabb.pMax = glm::max(point, bins[bin_index].aabb.pMax);
        }

        output_indices[bins[bin_index].index_start + bin_points_added_to_output[bin_index]++] = point_index;
    }

    // add bins to output_bins
    output_bins.insert(output_bins.end(), bins.begin(), bins.end());
}


ContiguousClusters::ContiguousClusters(std::vector<glm::vec3> &points, float bin_width) : HISTOGRAM_WIDTH(bin_width) {
    //std::chrono::steady_clock::time_point begin_clock = std::chrono::steady_clock::now();
    // handle empty case
    if (points.empty()) {
        m_clusters.push_back({{0, 0, 0}, {0, 0, 0}});
        return;
    }

    int read_index = 0, write_index;
    // double buffer bins, all points are in first bin at start
    std::vector<Bin> bins[2];
    bins[read_index].resize(1);
    bins[read_index][0].aabb = {points[0], points[0]};
    for (const auto &point : points) {
        bins[read_index][0].aabb.pMin = glm::min(point, bins[read_index][0].aabb.pMin);
        bins[read_index][0].aabb.pMax = glm::max(point, bins[read_index][0].aabb.pMax);
    }

    // double buffer the point indices so read and write to different locations on each axis
    // at start, all indices in are in order for first bin
    std::vector<uint64_t> indices[2];
    indices[0].resize(points.size());
    indices[1].resize(points.size());
    for (uint64_t point_index = 0; point_index < points.size(); point_index++) {
        indices[read_index][point_index] = point_index;
    }

    for (int axis = 0; axis < NUM_AXIS; axis++) {
        read_index = axis % 2;
        write_index = (axis + 1) % 2;
        bins[write_index].clear();

        // for all bins in prior axis (those that are being read), compute new bins
        // make sure to flatten all new bins into write bins array
        uint64_t prior_read_indices_length;
        for (uint64_t read_bin_index = 0; 
                read_bin_index < bins[read_index].size(); 
                read_bin_index++) {
            // get subset of indieces for current bin as input and output locations
            int64_t read_indices_end = read_bin_index == bins[read_index].size() - 1 ?
                points.size() : bins[read_index][read_bin_index + 1].index_start;
            int64_t read_indices_start = bins[read_index][read_bin_index].index_start;
            int64_t read_indices_length = read_indices_end - read_indices_start;

            int64_t write_indices_start = read_bin_index == 0 ? 0 : 
                bins[read_index][read_bin_index - 1].index_start + prior_read_indices_length;

            getBins(points, indices[read_index], read_indices_start, read_indices_length, 
                    bins[read_index][read_bin_index].aabb, axis, 
                    bins[write_index], indices[write_index], write_indices_start);
            prior_read_indices_length = read_indices_length;
        }
    }

    for (const auto &bin : bins[write_index]) {
        m_clusters.push_back(bin.aabb);
    }

    //std::chrono::steady_clock::time_point end_clock = std::chrono::steady_clock::now();
    //std::cout << "contiguous time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - begin_clock).count() << "[ms]" << std::endl;
}

}
}
