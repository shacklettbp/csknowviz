#pragma once

#include <unordered_set>
#include <unordered_map>
#include <utility>
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

struct pairHash 
{
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);
 
        return h1 ^ h2;
    }
};

// https://gist.github.com/etcwilde/0eb1870fbce202184499
class Points
{
public:
	Points() : m_verts() { }
	Points(const std::vector<glm::vec3>& verts) : m_verts(verts) { }
	inline void addVertex(const glm::vec3& v) { m_verts.push_back(v); }
	const glm::vec3& operator[](unsigned int i) const { return m_verts[i]; }
	unsigned int size() const { return m_verts.size(); }
private:
	std::vector<glm::vec3> m_verts;
};

class PointsAdaptor
{
public:
	PointsAdaptor(const Points& pts) : m_pts(pts) { }

	inline unsigned int kdtree_get_point_count() const
	{ return m_pts.size(); }

	inline float kdtree_distance(const float* p1, const unsigned int index_2,
			unsigned int) const
	{
		return glm::length(glm::vec3(p1[0], p1[1], p1[2]) -
				m_pts[index_2]);
	}

	inline float kdtree_get_pt(const unsigned int i, int dim) const
	{
		if (dim == 0) return m_pts[i].x;
		else if (dim == 1) return m_pts[i].y;
		else return m_pts[i].z;
	}


	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const { return false; }

private:
	const Points& m_pts;
};

struct ArrayLookup {
    size_t start;
    size_t length;
};

class Vec3IndexLessThan {
public:
    Vec3IndexLessThan(const std::vector<glm::vec3>& points) : m_points(points) { }

    inline glm::ivec3 getGridCoordinates(const int idx) {
        glm::vec3 point = m_points[idx];
        return {std::floor(point.x) / gridSize + halfArrayDim, 
            std::floor(point.y) / gridSize + halfArrayDim, 
            std::floor(point.z) / gridSize + halfArrayDim};
    }

    inline bool operator() (const int& idx0, const int& idx1) {
        glm::ivec3 c0 = getGridCoordinates(idx0);
        glm::ivec3 c1 = getGridCoordinates(idx1);
        return (c0.x < c1.x) || (c0.x == c1.x && c0.y < c1.y) ||
            (c0.x == c1.x && c0.y == c1.y && c0.z < c1.z);
    }

    const int gridSize = 10;
    const int halfArrayDim = 400;

private:
    const std::vector<glm::vec3>& m_points;
};

struct CoverResults {
    std::unordered_set<AABB, AABB::HashFunction> aabbs;
    std::vector<OverlayVertex> overlayVerts;
    std::vector<uint32_t> overlayIdxs;
};

struct CoverData {
    std::optional<NavmeshData> navmesh {};
    std::optional<vk::LocalBuffer> navmeshAABBGPU {};
    std::optional<ExpandedTLAS> tlasWithAABBs {};
    bool showNavmesh = false;
    std::unordered_map<glm::vec3, CoverResults> results {};
    bool showCover = false;
    float sampleSpacing = 1.f;
    float agentHeight = 72.f;
    int sqrtSphereSamples = 48;
    int sqrtSearchSamples = 5;
    float searchRadius = 0.05f;
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
