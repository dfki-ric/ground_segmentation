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

namespace pointcloud_obstacle_detection{

struct Point {
    double x;
    double y;
    double z;
};

enum TerrainType {
    UNDEFINED,
    UNKNOWN,
    GROUND,
    OBSTACLE
};

enum GridType {
    SQUARE,
    POLAR,
    HEXAGONAL
};

enum PrimitiveType {
    LINE,
    PLANE,
    NOISE
};

enum Confidence {
    HIGH,
    MEDIUM,
    LOW
};

template<typename PointT> 
struct GridCell {
    int x;
    int y;
    int z;
    bool expanded;
    bool explored;
    TerrainType terrain_type;
    PrimitiveType primitive_type;
    Confidence confidence;
    Eigen::Vector4d centroid;
    pcl::PointIndices::Ptr inliers;
    Eigen::Matrix3d eigenvectors;
    Eigen::Vector3d eigenvalues;
    /** The points in the Grid Cell */
    typename pcl::PointCloud<PointT>::Ptr points;
    typename pcl::PointCloud<PointT>::Ptr ground_points;

    /** The plane that has been fitted to the mls at the location of this node */
    Eigen::Hyperplane<double, 3> plane;

    /** slope of the plane */
    double slope;

    /** Normal of the plane*/
    Eigen::Vector3d normal;

    /** normalized direction of the slope. Only valid if slope > 0 */
    Eigen::Vector3d slopeDirection;

    /** The atan2(slopeDirection.y(), slopeDirection.x()), i.e. angle of slopeDirection projected on the xy plane.
     * Precomputed for performance reasons */
    double slopeDirectionAtan2;

    GridCell() : points(new pcl::PointCloud<PointT>), 
                 ground_points(new pcl::PointCloud<PointT>), 
                 inliers(new pcl::PointIndices){
        x = 0;
        y = 0;
        z = 0;
        expanded = false;
        explored = false;
        confidence = Confidence::LOW;
    }
};

struct Index3D {
    Index3D(int x, int y, int z): x(x), y(y), z(z) {}
    Index3D(){
        x = std::numeric_limits<int>::min(); 
        y = std::numeric_limits<int>::min(); 
        z = std::numeric_limits<int>::min();}
    int x, y, z;
    struct HashFunction
    {
        size_t operator()(Index3D const& ind) const {
            size_t xx=ind.x, yy=ind.y, zz=ind.z;
            // distribute bits equally over 64bits
            return (xx) ^ (yy << 21) ^ ((zz<<42) | (zz>>22));
        }
    };
    bool operator==(const Index3D& oth) const {
        return (x==oth.x) & (y==oth.y) & (z==oth.z); // use non-lazy `&` to avoid branching
    }
 
    Index3D operator+(Index3D const& obj)
    {
        Index3D res;
        res.x = x + obj.x;
        res.y = y + obj.y;
        res.z = z + obj.z;
        return res;
    }
};

struct GridConfig{
    double radialCellSize; // meters
    double angularCellSize; // meters

    double cellSizeX; // meters
    double cellSizeY; // meters
    double cellSizeZ; // meters

    double maxX;
    double maxY;
    double maxZ;

    double startCellDistanceThreshold; // meters
    double slopeThresholdDegrees; //degrees
    double groundInlierThreshold;
    bool returnGroundPoints;
    uint16_t neighborsIndexThreshold;
    uint16_t minPoints;
    uint16_t ransac_iterations;
    GridType grid_type;

    uint16_t processing_phase;

    GridConfig(){
        radialCellSize = 2;
        angularCellSize = 0.349066;

        cellSizeX = 2;
        cellSizeY = 2;
        cellSizeZ = 10;

        maxX = 20;
        maxY = 20;
        maxZ = 20;

        startCellDistanceThreshold = 20; // meters
        slopeThresholdDegrees = 30; //degrees
        groundInlierThreshold = 0.1; // meters
        neighborsIndexThreshold = 1;
        returnGroundPoints = true;
        minPoints = 5;
        ransac_iterations = 50;
        grid_type = GridType::SQUARE;

        processing_phase = 1;
    }
};
} //namespace pointcloud_obstacle_detection
