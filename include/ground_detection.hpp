#pragma once

#include "process_pointcloud.hpp"
#include "ground_detection_statistics.hpp"

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

namespace pointcloud_obstacle_detection{

struct Point {
    double x;
    double y;
    double z;
};

enum TerrainType {
    UNDEFINED,
    GROUND,
    OBSTACLE
};

struct GridCell {
    int row;
    int col;
    int height;
    double z_height;
    bool expanded;
    bool explored;
    TerrainType terrain_type;
    std::vector<GridCell> neighbors;
    Eigen::Vector4d centroid;
    pcl::PointIndices::Ptr inliers;

    /** The points in the Grid Cell */
    pcl::PointCloud<pcl::PointXYZ>::Ptr points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr inlier_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr outlier_pts;

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

    GridCell() : points(new pcl::PointCloud<pcl::PointXYZ>), 
                 inlier_pts(new pcl::PointCloud<pcl::PointXYZ>),
                 outlier_pts(new pcl::PointCloud<pcl::PointXYZ>),
                 inliers(new pcl::PointIndices){
        row = 0;
        col = 0;
        height = 0;
        z_height = 0.0;
        expanded = false;
        explored = false;
    }
};

struct Index3D {
    Index3D(int x, int y, int z): x(x), y(y), z(z) {}
    Index3D(){x = NAN; y = NAN; z= NAN;}
    int x, y, z;
};

struct GridConfig{
    double radialCellSize; // meters
    double angularCellSize; // meters
    double cellSizeZ; // meters
    double startCellDistanceThreshold; // meters
    double slopeThresholdDegrees; //degrees
    double groundInlierThreshold;
    bool returnGroundPoints;
    int neighborsRadius;
    int minPoints;
    int ransac_iterations;

    GridConfig(){
        radialCellSize = 2;
        angularCellSize = 0.785398;
        cellSizeZ = 1;
        startCellDistanceThreshold = 5; // meters
        slopeThresholdDegrees = 30; //degrees
        groundInlierThreshold = 0.2; // meters
        neighborsRadius = 1;
        returnGroundPoints = true;
        minPoints = 5;
        ransac_iterations = 50;
    }
};

class PointCloudGrid {

public:
    PointCloudGrid(const GridConfig& config);
    void clear();
    GroundDetectionStatistics& getStatistics();
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input, const Eigen::Quaterniond& R_body2World);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentPoints();
    std::map<int, std::map<int, std::map<int, GridCell>>>& getGridCells();

private:

    void addPoint(const pcl::PointXYZ& point);
    std::vector<Index3D> getGroundCells();
    std::vector<Index3D> getNeighbors(const GridCell& cell, const TerrainType& type, const std::vector<Index3D>& indices);
    double computeGridDistance(const GridCell& cell1, const GridCell& cell2);
    double computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2);
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    double computeSlope(const Eigen::Vector3d& normal);
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    int computeMeanHeight(const std::vector<Index3D> ids);
    double computeMeanPointsHeight(const std::vector<Index3D> ids);
    Index3D cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height);
    bool fitGroundPlane(GridCell& cell, const double& inlier_threshold);
    void selectStartCell(GridCell& cell);
    std::pair<size_t,pcl::PointXYZ>  findLowestPoint(const GridCell& cell);
    std::vector<Index3D> expandGrid(std::queue<Index3D> q);
    std::vector<Index3D> explore(std::queue<Index3D> q);

    std::vector<Index3D> indices;
    std::map<int, std::map<int, std::map<int, GridCell>>> gridCells;
    GridConfig grid_config;
    std::vector<Index3D> ground_cells;
    std::vector<Index3D> non_ground_cells;
    std::vector<Index3D> undefined_cells;
    std::vector<Index3D> selected_cells_first_quadrant;
    std::vector<Index3D> selected_cells_second_quadrant;
    std::vector<Index3D> selected_cells_third_quadrant;
    std::vector<Index3D> selected_cells_fourth_quadrant;
    Eigen::Quaterniond orientation;
    GridCell robot_cell;
    ProcessPointCloud processor;
    unsigned int total_ground_cells;
    GroundDetectionStatistics statistics;
};

} //namespace pointcloud_obstacle_detection
