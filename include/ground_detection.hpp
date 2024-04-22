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
    int row;
    int col;
    int height;
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
                 inliers(new pcl::PointIndices){
        row = 0;
        col = 0;
        height = 0;
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

template<typename PointT> 
class PointCloudGrid {

public:
    PointCloudGrid(const GridConfig& config);
    void clear();
    GroundDetectionStatistics& getStatistics();
    void setInputCloud(typename pcl::PointCloud<PointT>::Ptr input, const Eigen::Quaterniond& R_body2World);
    std::pair<typename pcl::PointCloud<PointT>::Ptr,typename pcl::PointCloud<PointT>::Ptr> segmentPoints();
    std::map<int, std::map<int, std::map<int, GridCell<PointT>>>>& getGridCells();

private:

    std::vector<Index3D> generateIndices(const uint16_t& radius);
    void cleanUp();
    void addPoint(const PointT& point);
    std::vector<Index3D> getGroundCells();
    std::vector<Index3D> getNeighbors(const GridCell<PointT>& cell, const TerrainType& type, const std::vector<Index3D>& indices, const double& radius);
    double computeGridDistance(const GridCell<PointT>& cell1, const GridCell<PointT>& cell2);
    double computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2);
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    double computeSlope(const Eigen::Vector3d& normal);
    bool neighborCheck(const GridCell<PointT>& cell, GridCell<PointT>& neighbor);
    Eigen::Vector3d computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const;
    int computeMeanHeight(const std::vector<Index3D> ids);
    double computeMeanPointsHeight(const std::vector<Index3D> ids);
    Index3D cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height);
    bool fitGroundPlane(GridCell<PointT>& cell, const double& inlier_threshold);
    void selectStartCell(GridCell<PointT>& cell);
    std::pair<size_t,PointT>  findLowestPoint(const GridCell<PointT>& cell);
    std::vector<Index3D> expandGrid(std::queue<Index3D> q);
    std::vector<Index3D> explore(std::queue<Index3D> q);

    std::vector<Index3D> indices;
    std::vector<Index3D> obs_indices;

    //TODO
    std::map<int, std::map<int, std::map<int, GridCell<PointT>>>> gridCells;
    GridConfig grid_config;
    std::vector<Index3D> ground_cells;
    std::vector<Index3D> non_ground_cells;
    std::vector<Index3D> undefined_cells;
    std::vector<Index3D> unknown_cells;
    std::vector<Index3D> selected_cells_first_quadrant;
    std::vector<Index3D> selected_cells_second_quadrant;
    std::vector<Index3D> selected_cells_third_quadrant;
    std::vector<Index3D> selected_cells_fourth_quadrant;
    Eigen::Quaterniond orientation;
    GridCell<PointT> robot_cell;
    ProcessPointCloud<PointT> processor;
    uint32_t total_ground_cells;
    GroundDetectionStatistics statistics;
};
    
template<typename PointT>
PointCloudGrid<PointT>::PointCloudGrid(const GridConfig& config){
    grid_config = config;
    robot_cell.row = 0;
    robot_cell.col = 0;
    robot_cell.height = 0;
    total_ground_cells = 0;
    
    indices = generateIndices(grid_config.neighborsIndexThreshold);

    for (int dx = -1; dx <= 1; ++dx) {
       for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 0; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0){
                    continue;
                }
                Index3D idx;
                idx.x = dx;
                idx.y = dy;
                idx.z = dz;
                obs_indices.push_back(idx);
            }
       }
    }
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::generateIndices(const uint16_t& radius){
    std::vector<Index3D> idxs;

    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0){
                    continue;
                }
                Index3D idx;
                idx.x = dx;
                idx.y = dy;
                idx.z = dz;
                idxs.push_back(idx);
            }
        }
    }
    return idxs;    
}

template<typename PointT>
void PointCloudGrid<PointT>::clear(){
    gridCells.clear();
    total_ground_cells = 0;
}

template<typename PointT>
GroundDetectionStatistics& PointCloudGrid<PointT>::getStatistics(){
    return statistics;
}

template<typename PointT>
void PointCloudGrid<PointT>::cleanUp(){
    ground_cells.clear();
    non_ground_cells.clear();
    undefined_cells.clear();
    unknown_cells.clear();
    selected_cells_first_quadrant.clear();
    selected_cells_second_quadrant.clear();
    selected_cells_third_quadrant.clear();
    selected_cells_fourth_quadrant.clear();
    total_ground_cells = 0;
}

template<typename PointT>
void PointCloudGrid<PointT>::addPoint(const PointT& point){
    double cell_x = 0;
    double cell_y = 0;
    double cell_z = 0;

    if(grid_config.grid_type == GridType::POLAR){
        Eigen::Vector3d radial_vector(point.x,point.y,point.z);
        double radial_angle = std::atan2(point.y, point.x);

        cell_x = radial_vector.norm() / grid_config.radialCellSize;
        cell_y = radial_angle / grid_config.angularCellSize;
        cell_z = point.z / grid_config.cellSizeZ;
    }
    else 
    if(grid_config.grid_type == GridType::SQUARE){
        cell_x = point.x / grid_config.cellSizeX;
        cell_y = point.y / grid_config.cellSizeY;
        cell_z = point.z / grid_config.cellSizeZ;
    }

    if((cell_x <= -grid_config.maxX || cell_x > grid_config.maxX) ||
       (cell_y <= -grid_config.maxY || cell_y > grid_config.maxY) ||
       (cell_z <= -grid_config.maxZ || cell_z > grid_config.maxZ)){
        return;
    }

    int row = static_cast<int>(std::floor(cell_x));
    int col = static_cast<int>(std::floor(cell_y));
    int height = static_cast<int>(std::floor(cell_z));

    gridCells[row][col][height].row = row;
    gridCells[row][col][height].col = col;
    gridCells[row][col][height].height = height;
    gridCells[row][col][height].points->push_back(point);
}

// Function to calculate the slope from the normal vector
template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Vector3d& normal)
{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = normal;
    planeNormal = orientation * planeNormal;
    planeNormal.normalize(); //just in case
    return acos(planeNormal.dot(zNormal));
}

template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = plane.normal();
    planeNormal = orientation * planeNormal;
    planeNormal.normalize(); //just in case
    return acos(planeNormal.dot(zNormal));
}

template<typename PointT>
bool PointCloudGrid<PointT>::neighborCheck(const GridCell<PointT>& cell, GridCell<PointT>& neighbor){
    pcl::KdTreeFLANN<PointT> kdtree;
    size_t nearest_index{0};
    std::vector<int> point_indices(1);
    std::vector<float> point_distances(1);
    Eigen::Vector3d ground_normal;
    if (cell.normal.isApprox(Eigen::Vector3d::Zero())){
        ground_normal = cell.eigenvectors.col(0);
    }
    else{
        ground_normal = cell.normal;
    }
   
    uint16_t count = 0;
    kdtree.setInputCloud(cell.points);
    for (typename pcl::PointCloud<PointT>::iterator it = neighbor.points->begin(); it != neighbor.points->end(); ++it){
        Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
        PointT search_point;
        search_point.x = it->x;
        search_point.y = it->y;
        search_point.z = it->z;
        if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
            nearest_index = point_indices[0];
        }
        Eigen::Vector3d ground_point(cell.points->points.at(nearest_index).x,
                                        cell.points->points.at(nearest_index).y,
                                        cell.points->points.at(nearest_index).z);

        Eigen::Vector3d diff = obstacle_point - ground_point;
        
        ground_normal.normalize();

        double distance = std::abs(diff.dot(ground_normal)); 
        if (distance < grid_config.groundInlierThreshold){
            count++;
        }
        if ((count / neighbor.points->size()) > 0.9){   
            neighbor.terrain_type = TerrainType::GROUND;
            return true;
        }
    }
    return false;
}

template<typename PointT>
Eigen::Vector3d PointCloudGrid<PointT>::computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const{
    /** The vector of maximum slope on a plane is the projection of (0,0,1) onto the plane.
     *  (0,0,1) is the steepest vector possible in the global frame, thus by projecting it onto
     *  the plane we get the steepest vector possible on that plane.
     */
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d planeNormal(plane.normal().normalized());
    const Eigen::Vector3d projection = zNormal - zNormal.dot(planeNormal) * planeNormal;
    return projection;
}

template<typename PointT>
double PointCloudGrid<PointT>::computeGridDistance(const GridCell<PointT>& cell1, const GridCell<PointT>& cell2){

    double dx = cell1.row - cell2.row;
    double dy = cell1.col - cell2.col;
    double dz = cell1.height - cell2.height;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

template<typename PointT>
int PointCloudGrid<PointT>::computeMeanHeight(const std::vector<Index3D> ids){

    // Calculate the mean height of selected cells
    double total_height = 0.0;
    for (const Index3D& id : ids) {
        total_height += gridCells[id.x][id.y][id.z].height;
    }
    // Find the cell closest to the mean height
    int mean_height = std::floor(total_height / ids.size());
    return mean_height;
}

template<typename PointT>
double PointCloudGrid<PointT>::computeMeanPointsHeight(const std::vector<Index3D> ids){
    // Calculate the mean height of selected cells
    int total_points = 0;
    double mean_height = 0.0;
    for (const Index3D& id : ids) {
        GridCell<PointT>& cell = gridCells[id.x][id.y][id.z];

        // Compute the transformation matrix
        Eigen::Affine3f transform = pcl::getTransFromUnitVectorsZY(Eigen::Vector3f::UnitZ(), cell.normal.template cast<float>());

        // Apply the transformation to the point clou
        typename pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);
        pcl::transformPointCloud(*(cell.points), *transformed_cloud, transform.cast<double>());

        for (typename pcl::PointCloud<PointT>::iterator it = transformed_cloud->begin(); it != transformed_cloud->end(); ++it)
        {
            mean_height += (*it).z;
            total_points++;
        }
    }
    if (total_points != 0){
        mean_height /= total_points;
    }
    return mean_height;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getNeighbors(const GridCell<PointT>& cell, const TerrainType& type, const std::vector<Index3D>& idx, const double& radius){

    std::vector<Index3D> neighbors;
    for (int i = 0; i < idx.size(); ++i){
        int neighborX = cell.row + idx[i].x;
        int neighborY = cell.col + idx[i].y;
        int neighborZ = cell.height + idx[i].z;

        GridCell<PointT>& neighbor = gridCells[neighborX][neighborY][neighborZ];
        if (neighbor.points->size() > 0 && computeDistance(cell.centroid,neighbor.centroid) < radius && neighbor.terrain_type == type){
            Index3D id;
            id.x = neighbor.row;
            id.y = neighbor.col;
            id.z = neighbor.height;
            neighbors.push_back(id);
        }
    }
    return neighbors;
}

template<typename PointT>
Index3D PointCloudGrid<PointT>::cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height){

    int min_height_difference = std::numeric_limits<int>::max();
    int max_ground_neighbors = std::numeric_limits<int>::min();
    Index3D closest_to_mean_height;

    for (const Index3D& id : ids) {
        const GridCell<PointT>& cell = gridCells[id.x][id.y][id.z];

        double height_difference = std::abs(cell.height - mean_height);
        int neighbor_count = getNeighbors(cell, TerrainType::GROUND, indices, 3).size();

        if (neighbor_count == 0){
            continue;
        }

        if (height_difference <= min_height_difference) {
            if (neighbor_count >= max_ground_neighbors){
                closest_to_mean_height = id;
                min_height_difference = height_difference;
                max_ground_neighbors = neighbor_count;
            }
        }
    }
    return closest_to_mean_height;
}

template<typename PointT>
bool PointCloudGrid<PointT>::fitGroundPlane(GridCell<PointT>& cell, const double& threshold){

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_MSAC);
    seg.setMaxIterations(grid_config.ransac_iterations);
    seg.setInputCloud(cell.points);
    seg.setDistanceThreshold(threshold); // Adjust this threshold based on your needs
    seg.segment(*inliers, *coefficients);
    cell.inliers = inliers;
    if (cell.inliers->indices.size() == 0){
        return false;
    }

    if (cell.inliers->indices.size() / cell.points->size() > 0.95){
        cell.confidence = Confidence::HIGH;
    }
    Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    double distToOrigin = coefficients->values[3];
    cell.plane = Eigen::Hyperplane<double, 3>(plane_normal, distToOrigin);
    //const Eigen::Vector3d slopeDir = computeSlopeDirection(cell.plane);
    cell.slope = computeSlope(cell.plane);
    //cell.slopeDirection = slopeDir;
    //cell.slopeDirectionAtan2 = std::atan2(slopeDir.y(), slopeDir.x());
    return true;
}

template<typename PointT>
void PointCloudGrid<PointT>::selectStartCell(GridCell<PointT>& cell){

    Index3D id;
    id.x = cell.row;
    id.y = cell.col;
    id.z = cell.height;

    double distance = computeGridDistance(robot_cell, cell);
    if (distance <= grid_config.startCellDistanceThreshold) {
        // This grid cell is within the specified radius around the robot

        if (cell.row >= 0 && cell.col > 0){
            selected_cells_first_quadrant.push_back(id);
        }
        else if (cell.row <= 0 && cell.col > 0){
            selected_cells_second_quadrant.push_back(id);
        }
        else if (cell.row <= 0 && cell.col < 0){
            selected_cells_third_quadrant.push_back(id);
        }
        else if (cell.row >= 0 && cell.col < 0) {
            selected_cells_fourth_quadrant.push_back(id);
        }
    }
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getGroundCells(){

    if (gridCells.empty()){
        return ground_cells;
    }

    //clear internal variables
    this->cleanUp();

    for (auto& rowPair : gridCells){
        for (auto& colPair : rowPair.second){
            for (auto& heightPair : colPair.second){
                GridCell<PointT>& cell = heightPair.second;

                if ((cell.points->size() == 0)){continue;}

                Index3D id;
                id.x = cell.row;
                id.y = cell.col;
                id.z = cell.height;

                pcl::compute3DCentroid(*(cell.points), cell.centroid);
                // Compute the covariance matrix
                Eigen::Matrix3d covariance_matrix;
                pcl::computeCovarianceMatrixNormalized(*cell.points, cell.centroid, covariance_matrix);

                // Compute eigenvectors and eigenvalues
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
                cell.eigenvectors = eigen_solver.eigenvectors();
                cell.eigenvalues = eigen_solver.eigenvalues();
                // Compute the ratio of eigenvalues to determine point distribution
                double ratio = cell.eigenvalues[2] / cell.eigenvalues.sum();
                if (ratio > 0.950){ 
                    //The points form a line  
                    cell.primitive_type = PrimitiveType::LINE; 
                    
                    Eigen::Vector3d v = cell.eigenvectors.col(2);
                    // Ensure all normals point upward
                    if (v(2) < 0) {
                        v *= -1; // flip the normal direction
                    }
                    v = orientation * v;
                    v.normalize(); //just in case

                    double angle_rad = acos(v.dot(Eigen::Vector3d::UnitZ()));

                    if (angle_rad > ((90-grid_config.slopeThresholdDegrees) * (M_PI / 180))){
                        cell.terrain_type = TerrainType::UNKNOWN;  
                        unknown_cells.push_back(id);
                    }
                    else{
                        cell.terrain_type = TerrainType::OBSTACLE;
                        non_ground_cells.push_back(id);  
                    }
                    continue;
                } 
                else 
                if (ratio > 0.4){
                    //The points form a plane
                    cell.primitive_type = PrimitiveType::PLANE; 
                } 
                else{
                    //The points are not a plane
                    //Assumping it is an obstacle? Like plants etc.
                    cell.terrain_type = TerrainType::UNKNOWN;
                    cell.primitive_type = PrimitiveType::NOISE; 
                    unknown_cells.push_back(id);
                    continue;
                }

                // Normal is the eigenvector corresponding to the smallest eigenvalue
                Eigen::Vector3d normal = cell.eigenvectors.col(0);

                // Ensure all normals point upward
                if (normal(2) < 0) {
                    normal *= -1; // flip the normal direction
                }

                normal.normalize();
                cell.normal = normal;

                if (cell.points->size() <= grid_config.minPoints){
                    Eigen::Vector4f squared_diff_sum(0, 0, 0, 0);

                    for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                        Eigen::Vector4f diff = (*it).getVector4fMap() - cell.centroid.template cast<float>();
                        squared_diff_sum += diff.array().square().matrix();                     
                    }

                    Eigen::Vector4f variance = squared_diff_sum / cell.points->size();

                    if (variance[0] < variance[2] && variance[1] < variance[2]){
                        cell.terrain_type = TerrainType::OBSTACLE;
                        non_ground_cells.push_back(id);                    
                    }
                    else{
                        if (cell.terrain_type != TerrainType::UNKNOWN){
                            cell.terrain_type = TerrainType::UNKNOWN;
                            unknown_cells.push_back(id);
                        }                        
                    }
                    continue;
                }

                if (!fitGroundPlane(cell, grid_config.groundInlierThreshold)){
                    cell.terrain_type = TerrainType::OBSTACLE;
                    non_ground_cells.push_back(id);
                    continue;
                }

                if (cell.slope < (grid_config.slopeThresholdDegrees * (M_PI / 180)) ){
                    cell.terrain_type = TerrainType::GROUND;
                    total_ground_cells += 1;
                    selectStartCell(cell);
                }
                else{
                    cell.terrain_type = TerrainType::OBSTACLE;
                    non_ground_cells.push_back(id);
                }
            }
        }
    }

    std::queue<Index3D> q;
    std::vector<Index3D> selected_cells;
    selected_cells.insert(selected_cells.end(), selected_cells_first_quadrant.begin(),  selected_cells_first_quadrant.end());
    selected_cells.insert(selected_cells.end(), selected_cells_second_quadrant.begin(), selected_cells_second_quadrant.end());
    selected_cells.insert(selected_cells.end(), selected_cells_third_quadrant.begin(),  selected_cells_third_quadrant.end());
    selected_cells.insert(selected_cells.end(), selected_cells_fourth_quadrant.begin(), selected_cells_fourth_quadrant.end());

    int cells_mean_height = computeMeanHeight(selected_cells);

    Index3D closest_to_mean_height_q1 = cellClosestToMeanHeight(selected_cells_first_quadrant, cells_mean_height);
    Index3D closest_to_mean_height_q2 = cellClosestToMeanHeight(selected_cells_second_quadrant, cells_mean_height);
    Index3D closest_to_mean_height_q3 = cellClosestToMeanHeight(selected_cells_third_quadrant, cells_mean_height);
    Index3D closest_to_mean_height_q4 = cellClosestToMeanHeight(selected_cells_fourth_quadrant, cells_mean_height);
    
    q.push(closest_to_mean_height_q1);    
    q.push(closest_to_mean_height_q2);    
    q.push(closest_to_mean_height_q3);    
    q.push(closest_to_mean_height_q4);       

    //TODO
/*    
    std::queue<Index3D> q;

    if (selected_cells_first_quadrant.size() > 0){
        int cells_q1_mean_height = computeMeanHeight(selected_cells_first_quadrant);
        Index3D closest_to_mean_height_q1 = cellClosestToMeanHeight(selected_cells_first_quadrant, cells_q1_mean_height);
        q.push(closest_to_mean_height_q1);
    }

    if (selected_cells_second_quadrant.size() > 0){
        int cells_q2_mean_height = computeMeanHeight(selected_cells_second_quadrant);
        Index3D closest_to_mean_height_q2 = cellClosestToMeanHeight(selected_cells_second_quadrant, cells_q2_mean_height);
        q.push(closest_to_mean_height_q2);
    }

    if (selected_cells_third_quadrant.size() > 0){
        int cells_q3_mean_height = computeMeanHeight(selected_cells_third_quadrant);
        Index3D closest_to_mean_height_q3 = cellClosestToMeanHeight(selected_cells_third_quadrant, cells_q3_mean_height);
        q.push(closest_to_mean_height_q3);
    }

    if (selected_cells_fourth_quadrant.size() > 0){
        int cells_q4_mean_height = computeMeanHeight(selected_cells_fourth_quadrant);
        Index3D closest_to_mean_height_q4 = cellClosestToMeanHeight(selected_cells_fourth_quadrant, cells_q4_mean_height);
        q.push(closest_to_mean_height_q4);
    }
*/
    ground_cells = expandGrid(q);
    return ground_cells;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::expandGrid(std::queue<Index3D> q){
    std::vector<Index3D> result;
    int count{0};
    while (!q.empty()){
        Index3D idx = q.front();
        q.pop();
        GridCell<PointT>& current_cell = gridCells[idx.x][idx.y][idx.z];
        if (current_cell.expanded == true || current_cell.points->size() == 0){
            continue;
        }
        current_cell.expanded = true;
        for (int i = 0; i < indices.size(); ++i){
            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height + indices[i].z; 
            GridCell<PointT>& neighbor = gridCells[neighborX][neighborY][neighborZ];
            if(neighbor.points->size() == 0 || neighbor.expanded || neighbor.terrain_type == TerrainType::OBSTACLE){
                continue;
            }

            Index3D id;
            id.x = neighbor.row;
            id.y = neighbor.col;
            id.z = neighbor.height;

            if (indices[i].z !=0 && grid_config.processing_phase == 2){
                if (!neighborCheck(current_cell,neighbor)){
                    continue;
                }
            }
            if (neighbor.terrain_type == TerrainType::UNKNOWN && (neighbor.primitive_type == PrimitiveType::LINE || 
                                                                  neighbor.primitive_type == PrimitiveType::PLANE)){
                if (!neighborCheck(current_cell,neighbor)){
                    continue;
                }
            }
            if (neighbor.terrain_type == TerrainType::GROUND ){
                q.push(id);
            }
        }
        result.emplace_back(idx);
    }
    return result;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::explore(std::queue<Index3D> q){
    std::vector<Index3D> result;
    std::map<int, std::map<int, std::map<int, GridCell<PointT>>>> copy = gridCells;

    while (!q.empty()){

        Index3D& idx = q.front();
        q.pop();

        GridCell<PointT>& current_cell = copy[idx.x][idx.y][idx.z];

        if (current_cell.explored == true){
            continue;
        }
        current_cell.explored = true;
      
        for (int i = 0; i < indices.size(); ++i){

            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height + indices[i].z;

            GridCell<PointT>& neighbor = copy[neighborX][neighborY][neighborZ];

            if(neighbor.points->size() == 0 || neighbor.explored){
                continue;
            }

            if (neighbor.terrain_type == TerrainType::GROUND){
                Index3D n;
                n.x = neighbor.row;
                n.y = neighbor.col;
                n.z = neighbor.height;
                q.push(n);
            }        
        }
        result.emplace_back(idx);
    }
    return result;
}

template<typename PointT>
double PointCloudGrid<PointT>::computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2) {
    Eigen::Vector3d diff = centroid1.head<3>() - centroid2.head<3>();
    return diff.norm();
}

template<typename PointT>
void PointCloudGrid<PointT>::setInputCloud(typename pcl::PointCloud<PointT>::Ptr input, const Eigen::Quaterniond& R_body2World){

    this->clear();
    orientation = R_body2World;
    unsigned int index = 0;
    for (typename pcl::PointCloud<PointT>::iterator it = input->begin(); it != input->end(); ++it)
    {
        this->addPoint(*it);
        index++;
    }
    ground_cells = getGroundCells();
}

template<typename PointT>
std::pair<size_t,PointT> PointCloudGrid<PointT>::findLowestPoint(const GridCell<PointT>& cell){
    double min_height = std::numeric_limits<float>::max();
    PointT min_point;
    size_t min_point_index;

    for (size_t i = 0; i < cell.points->size(); ++i) {
        double height = cell.points->points[i].z;

        if (height < min_height) {
            min_height = height;
            min_point = cell.points->points[i];
            min_point_index = i;
        }
    }
    return std::make_pair(min_point_index, min_point);
}

template<typename PointT>
std::pair<typename pcl::PointCloud<PointT>::Ptr,typename pcl::PointCloud<PointT>::Ptr> PointCloudGrid<PointT>::segmentPoints(){

    typename pcl::PointCloud<PointT>::Ptr ground_points(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr ground_inliers(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr non_ground_points(new pcl::PointCloud<PointT>());
    typename pcl::PointCloud<PointT>::Ptr non_ground_inliers(new pcl::PointCloud<PointT>());

    pcl::ExtractIndices<PointT> extract_ground;
    const TerrainType type_ground = TerrainType::GROUND;
    const TerrainType type_obstacle = TerrainType::OBSTACLE;
    pcl::KdTreeFLANN<PointT> kdtree;

    if (grid_config.returnGroundPoints){
        for (auto& id : ground_cells){
            GridCell<PointT>& cell = gridCells[id.x][id.y][id.z];
            if (cell.points->size() < grid_config.minPoints || cell.primitive_type == PrimitiveType::LINE){
                for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin(); it != cell.points->end(); ++it)
                {
                    ground_points->points.push_back(*it);
                }
                continue;
            }

            extract_ground.setInputCloud(cell.points);
            extract_ground.setIndices(cell.inliers);

            extract_ground.setNegative(false);
            extract_ground.filter(*ground_inliers);

            extract_ground.setNegative(true);
            extract_ground.filter(*non_ground_inliers);

            if (ground_inliers->size() == 0) continue;

            size_t nearest_index{0};
            std::vector<int> point_indices(1);
            std::vector<float> point_distances(1);

            kdtree.setInputCloud(ground_inliers);

            Eigen::Vector3d ground_normal;
            if (cell.normal.isApprox(Eigen::Vector3d::Zero())){
                ground_normal = cell.eigenvectors.col(0);
            }
            else{
                ground_normal = cell.normal;
            }

            ground_normal.normalize();

            for (typename pcl::PointCloud<PointT>::iterator it = non_ground_inliers->begin(); it != non_ground_inliers->end(); ++it){
                Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
                PointT search_point;
                search_point.x = it->x;
                search_point.y = it->y;
                search_point.z = it->z;

                if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
                    nearest_index = point_indices[0];
                }

                Eigen::Vector3d ground_point(ground_inliers->points.at(nearest_index).x,
                                            ground_inliers->points.at(nearest_index).y,
                                            ground_inliers->points.at(nearest_index).z);

                Eigen::Vector3d diff = obstacle_point - ground_point;

                double distance = std::abs(diff.dot(ground_normal)); 

                if (distance > grid_config.groundInlierThreshold){
                    non_ground_points->points.push_back(*it);
                }
                else{
                    ground_points->points.push_back(*it);
                }
            }

            for (typename pcl::PointCloud<PointT>::iterator it = ground_inliers->begin(); it != ground_inliers->end(); ++it){
                ground_points->points.push_back(*it);
            }
        }
    }
   
    if (grid_config.processing_phase == 1 && ground_points->points.size() > 0){

        double grid_cell_radius = std::sqrt(grid_config.cellSizeX*grid_config.cellSizeX +   
                                            grid_config.cellSizeY*grid_config.cellSizeY);

        pcl::NormalEstimation<PointT, pcl::Normal> ne;
        ne.setInputCloud(ground_points);
        typename pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
        ne.setSearchMethod(tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch(0.5);
        ne.compute(*cloud_normals);
        
        kdtree.setInputCloud(ground_points);
        for (const auto& id : unknown_cells){
            GridCell<PointT>& cell = gridCells[id.x][id.y][id.z];
            if (cell.terrain_type == TerrainType::UNKNOWN){
                size_t nearest_index{0};
                std::vector<int> point_indices(1);
                std::vector<float> point_distances(1);
                for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                    Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
                    PointT search_point;
                    search_point.x = it->x;
                    search_point.y = it->y;
                    search_point.z = it->z;
                    if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0){
                        nearest_index = point_indices[0];
                    }
                    Eigen::Vector3d ground_point(ground_points->points.at(nearest_index).x,
                                                 ground_points->points.at(nearest_index).y,
                                                 ground_points->points.at(nearest_index).z);

                    Eigen::Vector3d diff = obstacle_point - ground_point;
        
                    if (diff.norm() > 0.5){continue;}

                    Eigen::Vector3d ground_normal(cloud_normals->points[nearest_index].normal_x,
                                                  cloud_normals->points[nearest_index].normal_y,
                                                  cloud_normals->points[nearest_index].normal_z);
                    // Ensure all normals point upward
                    if (ground_normal(2) < 0) {
                        ground_normal *= -1; // flip the normal direction
                    }
                    ground_normal.normalize();

                    if (!ground_normal.allFinite()){continue;}

                    double distance = std::abs(diff.dot(ground_normal)); 
                    if (distance <= grid_config.groundInlierThreshold){
                        ground_points->points.push_back(*it);
                    }
                    else{
                        non_ground_points->points.push_back(*it);
                    }
                }
            }
        }
    }
    
    for (const auto& id : non_ground_cells){
        GridCell<PointT>& cell = gridCells[id.x][id.y][id.z];

        typename pcl::PointCloud<PointT>::Ptr close_ground_points(new pcl::PointCloud<PointT>());

        std::vector<Index3D> test = getNeighbors(cell, type_ground, obs_indices, 3);
        if (test.size() == 0){
            for (typename pcl::PointCloud<PointT>::iterator it = cell.points->points.begin(); it != cell.points->points.end(); ++it){
                non_ground_points->points.push_back(*it);
            }
            continue;    
        }
        else{
            for (const auto& gp : test){
                GridCell<PointT>& ground_cell = gridCells[gp.x][gp.y][gp.z];

                if (ground_cell.confidence != Confidence::HIGH){
                    continue;
                }

                extract_ground.setInputCloud(ground_cell.points);
                extract_ground.setIndices(ground_cell.inliers);

                extract_ground.setNegative(false);
                extract_ground.filter(*ground_inliers);

                for (typename pcl::PointCloud<PointT>::iterator it = ground_inliers->points.begin(); it != ground_inliers->points.end(); ++it){
                    close_ground_points->points.push_back(*it);
                }
            }
        }

        std::vector<Index3D> potential_ground_neighbors = getNeighbors(cell, type_ground, indices, 1);
        std::vector<Index3D> actual_ground_neighbors;

        for (const auto& gn : potential_ground_neighbors){
            std::vector<Index3D> explored_neighbors;
            std::queue<Index3D> q;
            q.push(gn);
            explored_neighbors = explore(q);

            if (explored_neighbors.size() < 0.5*total_ground_cells){
                continue;
            }
            actual_ground_neighbors.push_back(gn);
        }

        Eigen::Vector3d ground_normal;
        size_t nearest_index{0};
        std::vector<int> point_indices(1);
        std::vector<float> point_distances(1);

        if (actual_ground_neighbors.size() == 0){
            ground_normal = Eigen::Vector3d::UnitZ();
        }
        else{
            for (const auto& gp : actual_ground_neighbors){
                GridCell<PointT>& ground_cell = gridCells[gp.x][gp.y][gp.z];

                if (cell.normal.isApprox(Eigen::Vector3d::Zero())){
                    ground_normal = cell.eigenvectors.col(0);
                }
                else{
                    ground_normal = cell.normal;
                }

                double distance = computeDistance(cell.centroid, ground_cell.centroid);

                extract_ground.setInputCloud(ground_cell.points);
                extract_ground.setIndices(ground_cell.inliers);

                extract_ground.setNegative(false);
                extract_ground.filter(*ground_inliers);

                for (typename pcl::PointCloud<PointT>::iterator it = ground_inliers->points.begin(); it != ground_inliers->points.end(); ++it){
                    close_ground_points->points.push_back(*it);
                }

                ground_normal += (1/(distance+0.001)) * ground_cell.normal.normalized();
            }
            ground_normal /= actual_ground_neighbors.size();
            ground_normal = orientation * ground_normal;
            ground_normal.normalize(); 
        }

        if (close_ground_points->size() == 0){
            for (typename pcl::PointCloud<PointT>::iterator it = cell.points->points.begin(); it != cell.points->points.end(); ++it){
                non_ground_points->points.push_back(*it);
            }
            continue;    
        }

        kdtree.setInputCloud(close_ground_points);
        for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
            Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
            PointT search_point;
            search_point.x = it->x;
            search_point.y = it->y;
            search_point.z = it->z;

            if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
                nearest_index = point_indices[0];
            }
            else{
                //TODO: Handle this case
            }

            Eigen::Vector3d ground_point(close_ground_points->points.at(nearest_index).x,
                                            close_ground_points->points.at(nearest_index).y,
                                            close_ground_points->points.at(nearest_index).z);

            Eigen::Vector3d diff = obstacle_point - ground_point;

            double distance = std::abs(diff.dot(ground_normal)); 

            if (distance > grid_config.groundInlierThreshold){
                non_ground_points->points.push_back(*it);
            }
            //else{
            //    if (grid_config.returnGroundPoints){
            //        ground_points->points.push_back(*it);
            //    }
            //}
        }

        statistics.clear();
        statistics.ground_cells = ground_cells.size();
        statistics.non_ground_cells = non_ground_cells.size();
        statistics.undefined_cells = undefined_cells.size();
        statistics.unknown_cells = unknown_cells.size();
    
    }
    return std::make_pair(ground_points, non_ground_points);
}

template<typename PointT>
std::map<int, std::map<int, std::map<int, GridCell<PointT>>>>& PointCloudGrid<PointT>::getGridCells(){
    return gridCells;
}

//template<typename PointT>
//typename pcl::PointCloud<PointT>::Ptr PointCloudGrid<PointT>::extractHoles(){
//TODO
//}

} //namespace pointcloud_obstacle_detection
