#pragma once

#include "processPointClouds.hpp"
#include "GroundDetectionStatistics.hpp"

#include <pcl/common/transforms.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Polygon_mesh_processing/intersection.h>

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <map>

typedef CGAL::Exact_predicates_inexact_constructions_kernel  Kernel;
typedef CGAL::Polyhedron_3<Kernel>                           Polyhedron_3;
typedef Kernel::Point_3                                      Point_3;
typedef CGAL::Surface_mesh<Point_3>                          Surface_mesh;
typedef Polyhedron_3::Vertex_const_iterator                  Vertex_const_iterator; 
typedef std::pair<Kernel::Point_3, Kernel::Vector_3>         Point_with_normal;
typedef std::vector<Point_with_normal>                       Pwn_vector;
typedef CGAL::First_of_pair_property_map<Point_with_normal>  Point_map;
typedef CGAL::Second_of_pair_property_map<Point_with_normal> Normal_map;

// Efficient RANSAC types
#include <CGAL/property_map.h>
#include <CGAL/Shape_detection/Efficient_RANSAC.h>
#include <CGAL/structure_point_set.h>
#include <CGAL/jet_estimate_normals.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/grid_simplify_point_set.h>

typedef Kernel::FT                                           FT;
typedef Kernel::Vector_3                                     Vector;
typedef CGAL::Point_set_3<Point_3>                           Point_set;
typedef CGAL::Shape_detection::Efficient_RANSAC_traits
<Kernel, Point_set, Point_set::Point_map, Point_set::Vector_map> Traits;
typedef CGAL::Shape_detection::Efficient_RANSAC<Traits>      Efficient_ransac;
typedef CGAL::Shape_detection::Plane<Traits>                 Plane;

//Classification
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/bounding_box.h>
#include <CGAL/tags.h>
#include <CGAL/Real_timer.h>

namespace Classification = CGAL::Classification;

typedef CGAL::Simple_cartesian<double> Kernel_c;
typedef Kernel_c::Point_3 Point_c;
typedef CGAL::Point_set_3<Point_c> Point_set_c;
typedef Kernel_c::Iso_cuboid_3 Iso_cuboid_3;
typedef Point_set_c::Point_map Pmap;
typedef Point_set_c::Property_map<int> Imap;
typedef Classification::Sum_of_weighted_features_classifier                         Classifier;
typedef Classification::Point_set_feature_generator<Kernel_c, Point_set_c, Pmap>    Feature_generator;

#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif

typedef std::vector<Point_c> Point_range;
typedef CGAL::Identity_property_map<Point_c> IPmap;
typedef Classification::Planimetric_grid<Kernel_c, Point_range, IPmap>             Planimetric_grid;
typedef Classification::Point_set_neighborhood<Kernel_c, Point_range, IPmap>       Neighborhood;
typedef Classification::Local_eigen_analysis                                    Local_eigen_analysis;
typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set                                               Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Feature::Distance_to_plane<Point_range, IPmap>           Distance_to_plane;
typedef Classification::Feature::Elevation<Kernel_c, Point_range, IPmap>           Elevation;
typedef Classification::Feature::Vertical_dispersion<Kernel_c, Point_range, IPmap> Dispersion;

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

    // define polyhedron to hold convex hull
    Polyhedron_3 poly;

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
    GridConfig(){
        radialCellSize = 1;
        angularCellSize = 0.1;
        cellSizeZ = 0.2;
        startCellDistanceThreshold = 4; // meters
        slopeThresholdDegrees = 30; //degrees
        groundInlierThreshold = 0.1; // meters
        neighborsRadius = 3;
        returnGroundPoints = true;
        minPoints = 5;
    }
};

class PointCloudGrid {

public:
    PointCloudGrid(const GridConfig& config);
    void clear();
    GroundDetectionStatistics& getStatistics();
    void setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input, const Eigen::Quaterniond& R_body2World);
    std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,pcl::PointCloud<pcl::PointXYZ>::Ptr> segmentPoints();

private:

    void addPoint(const pcl::PointXYZ& point);
    std::vector<Index3D> getGroundCells();
    std::vector<Index3D> getNeighbors(const GridCell& cell, const TerrainType& type, const std::vector<Index3D>& indices);
    double computeGridDistance(const GridCell& cell1, const GridCell& cell2);
    double computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2);
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    int computeMeanHeight(const std::vector<Index3D> ids);
    double computeMeanPointsHeight(const std::vector<Index3D> ids);
    Index3D cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height);
    bool fitGroundPlane(GridCell& cell, const double& inlier_threshold);
    std::vector<GridCell> fitPlanes(GridCell& cell, const double& threshold);
    void selectStartCell(GridCell& cell);
    std::pair<size_t,pcl::PointXYZ>  findLowestPoint(const GridCell& cell);
    std::vector<Index3D> expandGrid(std::queue<Index3D> q);
    std::vector<Index3D> explore(std::queue<Index3D> q);
    void train(Point_set_c& pts);
    void classify(std::vector<Kernel_c::Point_3>& pts);

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
    GroundDetectionStatistics statistics;
};

} //namespace pointcloud_obstacle_detection
