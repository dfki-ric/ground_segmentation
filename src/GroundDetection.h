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
    bool expanded;
    std::vector<GridCell> neighbors;
    Eigen::Vector4d centroid;
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

    GridCell() : isGround(false), points(new pcl::PointCloud<pcl::PointXYZI>), source_indices(new pcl::PointIndices){
        row = -1;
        col = -1;
        height = -1;
        expanded = false;        
    }
};

struct Index3D {
    int x, y, z;
};


class PointCloudGrid {

public:
    PointCloudGrid(double cellSizeX, double cellSizeY, double cellSizeZ);
    void clear();
    void addPoint(const pcl::PointXYZI& point, const unsigned int index);
    pcl::PointCloud<pcl::PointXYZI>::Ptr getGroundPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr removeGroundPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr source);

private:
    std::vector<GridCell> getGroundCells();
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    double calculateDistance(const GridCell& cell1, const GridCell& cell2);
    int calculateMeanHeight(const std::vector<GridCell> cells);
    int countGroundNeighbors(const GridCell& cell);
    GridCell cellClosestToMeanHeight(const std::vector<GridCell>& cells, const int mean_height);
    bool PointCloudGrid::fitPlane(GridCell& cell);
    void selectStartCell(GridCell& cell);

    double cellSizeX;
    double cellSizeY;
    double cellSizeZ;
    int gridWidth;
    int gridDepth;
    int gridHeight;

    std::vector<Index3D> indices;    
    std::map<int, std::map<int, std::map<int, GridCell>>> gridCells;
    std::vector<GridCell> ground_cells;
    std::vector<GridCell> selected_cells_first_quadrant;
    std::vector<GridCell> selected_cells_second_quadrant;
    std::vector<GridCell> selected_cells_third_quadrant;
    std::vector<GridCell> selected_cells_fourth_quadrant;

    GridCell robot_cell;

};
