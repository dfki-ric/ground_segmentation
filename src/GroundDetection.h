#pragma once

#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

#include <vector>
#include <cmath>
#include <map>

#include <Eigen/Dense>

struct Point {
    double x;
    double y;
    double z;
};

struct GridCell {
    int row;
    int col;
    double height;
    bool isGround;
    bool isFrontier;
    std::vector<GridCell> neighbors;
    pcl::PointIndices::Ptr source_indices;

    /** The points in the Grid Cell */
    pcl::PointCloud<pcl::PointXYZI>::Ptr points;

    /** The plane that has been fitted to the mls at the location of this node */
    Eigen::Hyperplane<double, 3> plane;
    
    /** slope of the plane */
    double slope;
    
    /** normalized direction of the slope. Only valid if slope > 0 */
    Eigen::Vector3d slopeDirection;

    /** The atan2(slopeDirection.y(), slopeDirection.x()), i.e. angle of slopeDirection projected on the xy plane.
     * Precomputed for performance reasons */
    double slopeDirectionAtan2; 

    GridCell() : isGround(false), points(new pcl::PointCloud<pcl::PointXYZI>), source_indices(new pcl::PointIndices){}
};

class PointCloudGrid {
public:
    PointCloudGrid(double cellSizeX, double cellSizeY, double cellSizeZ);
    void clear();
    void addPoint(const pcl::PointXYZI& point, const unsigned int index);
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    std::vector<GridCell> getGroundCells();
    pcl::PointCloud<pcl::PointXYZI>::Ptr getGroundPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr getGroundPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr source);
    pcl::PointCloud<pcl::PointXYZI>::Ptr removeGroundPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr source);

private:
    double cellSizeX;
    double cellSizeY;
    double cellSizeZ;

    int gridWidth;
    int gridDepth;
    int gridHeight;

    std::map<int, std::map<int, std::map<int, GridCell>>> gridCells;
};
