/**
 * @file ground_detection_types.hpp
 * @brief Core data structures and configuration types for grid-based ground segmentation.
 *
 * Defines spatial grid representation, terrain labels, primitive classification,
 * and configuration parameters used by the grid-based segmentation algorithm.
 */

#pragma once

#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <map>
#include <queue>

namespace ground_segmentation
{

/**
 * @struct Point
 * @brief Simple 3D point representation (double precision).
 *
 * Used for lightweight internal representations where full PCL point types
 * are not required.
 */

struct Point
{
  double x;
  double y;
  double z;
};

/**
 * @enum TerrainType
 * @brief Semantic classification label assigned to a grid cell.
 *
 * UNDEFINED  - Not processed yet
 * UNKNOWN    - Insufficient evidence
 * GROUND     - Traversable ground surface
 * OBSTACLE   - Non-traversable object or structure
 */

enum TerrainType
{
  UNDEFINED,
  UNKNOWN,
  GROUND,
  OBSTACLE
};

/**
 * @enum PrimitiveType
 * @brief Geometric primitive classification based on local PCA.
 *
 * LINE   - Strong linear structure (e.g., edge, curb)
 * PLANE  - Planar surface (candidate ground)
 * NOISE  - Irregular or scattered structure
 */

enum PrimitiveType
{
  LINE,
  PLANE,
  NOISE
};

/**
 * @struct GridCell
 * @brief Represents one discretized 3D voxel in the spatial grid.
 *
 * Each cell stores:
 *  - Raw points assigned to the voxel
 *  - PCA eigenvalues/eigenvectors
 *  - Estimated surface normal
 *  - Plane fitting inliers
 *  - Terrain classification
 *
 * Cells are classified using geometric consistency, slope,
 * and region growing.
 *
 * @tparam PointT PCL point type
 */

template<typename PointT>
struct GridCell
{
  int x;
  int y;
  int z;
  bool expanded;
  bool in_queue;
  bool explored;
  TerrainType terrain_type;
  PrimitiveType primitive_type;
  Eigen::Vector4d centroid;
  pcl::PointIndices::Ptr inliers;
  Eigen::Matrix3d eigenvectors;
  Eigen::Vector3d eigenvalues;
  /** The points in the Grid Cell */
  typename pcl::PointCloud<PointT>::Ptr points;

  /** slope of the plane */
  double slope;

  /** Normal of the plane*/
  Eigen::Vector3d normal;

  GridCell()
  : points(new pcl::PointCloud<PointT>),
    inliers(new pcl::PointIndices)
  {
    x = 0;
    y = 0;
    z = 0;
    in_queue = false;
    expanded = false;
    explored = false;
    slope = std::numeric_limits<double>::quiet_NaN();
  }
};

/**
 * @struct Index3D
 * @brief Integer 3D grid index used as key in hash map.
 *
 * Provides:
 *  - Hash function for unordered_map
 *  - Equality operator
 *  - Offset addition operator
 *
 * Designed for efficient sparse 3D grid storage.
 */

struct Index3D
{
  Index3D(int x, int y, int z)
  : x(x), y(y), z(z) {}
  Index3D()
  {
    x = std::numeric_limits<int>::min();
    y = std::numeric_limits<int>::min();
    z = std::numeric_limits<int>::min();
  }
  int x, y, z;
  struct HashFunction
  {
    size_t operator()(Index3D const & ind) const
    {
      size_t xx = ind.x, yy = ind.y, zz = ind.z;
      // distribute bits equally over 64bits
      return (xx) ^ (yy << 21) ^ ((zz << 42) | (zz >> 22));
    }
  };
  bool operator==(const Index3D & oth) const
  {
    return (x == oth.x) & (y == oth.y) & (z == oth.z); // use non-lazy `&` to avoid branching
  }

  Index3D operator+(Index3D const & obj) const
  {
    return Index3D(x + obj.x, y + obj.y, z + obj.z);
  }
};

/**
 * @struct GridConfig
 * @brief Configuration parameters controlling segmentation behavior.
 *
 * Contains grid resolution, geometric thresholds,
 * slope limits, and processing phase control.
 *
 * Important relationships:
 *   groundInlierThreshold < cellSizeZ
 *   slopeThresholdDegrees determines traversability angle limit
 */

struct GridConfig
{
  double cellSizeX;   // meters
  double cellSizeY;   // meters
  double cellSizeZ;   // meters

  double slopeThresholdDegrees;   //degrees
  double groundInlierThreshold;    // meters
  double centroidSearchRadius; // meters
  double distToGround; // meters
  double maxGroundHeightDeviation; // meters

  uint16_t processing_phase;

  GridConfig()
  {
    cellSizeX = 2;
    cellSizeY = 2;
    cellSizeZ = 10;
    slopeThresholdDegrees = 30;
    groundInlierThreshold = 0.1;
    centroidSearchRadius = 5.0;
    maxGroundHeightDeviation = 0.3;
    distToGround = 0.0;
    processing_phase = 1;
  }
};
} //namespace ground_segmentation
