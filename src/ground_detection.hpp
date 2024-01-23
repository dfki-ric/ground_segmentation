#pragma once

#include "processPointClouds.hpp"

#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <vector>
#include <cmath>
#include <map>

#include <Eigen/Dense>

using namespace pointcloud_obstacle_detection;

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
    int x, y, z;
};

struct GridConfig{
    double cellSizeX; // meters
    double cellSizeY; // meters
    double cellSizeZ; // meters

    double gridSizeX; // meters
    double gridSizeY; // meters
    double gridSizeZ; // meters

    double startCellDistanceThreshold; // meters
    double slopeThresholdDegrees; //degrees
    double groundInlierThreshold;
    
    bool returnGroundPoints;


    GridConfig(){
        cellSizeX = 1;
        cellSizeY = 1;
        cellSizeZ = 2;

        gridSizeX = 100;
        gridSizeY = 100;
        gridSizeZ = 100;

        startCellDistanceThreshold = 4; // meters
        slopeThresholdDegrees = 30; //degrees
        groundInlierThreshold = 0.1; // meters
    }

};

class PointCloudGrid {

public:
    PointCloudGrid();
    PointCloudGrid(const GridConfig& config);
    void clear();
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input, const Eigen::Quaterniond& R_body2World);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentPoints();

private:

    void addPoint(const pcl::PointXYZ& point);
    std::vector<Index3D> getGroundCells();
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    double calculateDistance(const GridCell& cell1, const GridCell& cell2);
    int calculateMeanHeight(const std::vector<Index3D> ids);
    std::vector<Index3D> getNeighbors(const GridCell& cell, const TerrainType& type);
    int countGroundNeighbors(const GridCell& cell);
    Index3D cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height);
    bool fitGroundPlane(GridCell& cell, const double& inlier_threshold, const double& inlier_percentage);
    void selectStartCell(GridCell& cell);
    double computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2);
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
    ProcessPointClouds processor;
    unsigned int total_ground_cells;
};
