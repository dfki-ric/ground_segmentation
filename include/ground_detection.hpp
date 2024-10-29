#pragma once

#include <omp.h>

#include "pointcloud_processor.hpp"
#include "ground_detection_types.hpp"
#include <nanoflann.hpp>

namespace pointcloud_obstacle_detection{

template <typename PointT>
struct PCLPointCloudAdaptor {
    typedef pcl::PointCloud<PointT> PointCloudType;

    PointCloudType& cloud;

    PCLPointCloudAdaptor(PointCloudType& cloud) : cloud(cloud) {}

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return cloud.points.size(); }

    // Returns the dim'th component of the idx'th point in the point cloud
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return cloud.points[idx].x;
        else if (dim == 1) return cloud.points[idx].y;
        else return cloud.points[idx].z;
    }

    // Optional bounding-box computation
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { return false; }
};

template<typename PointT> 
class PointCloudGrid {

public:
    PointCloudGrid(const GridConfig& config);
    void clear();
    typedef GridCell<PointT> CellType;
    typedef std::unordered_map<Index3D, CellType, Index3D::HashFunction> GridCellsType;
    void setInputCloud(typename pcl::PointCloud<PointT>::Ptr input, const Eigen::Quaterniond& R_body2World);
    std::pair<typename pcl::PointCloud<PointT>::Ptr,typename pcl::PointCloud<PointT>::Ptr> segmentPoints();
    GridCellsType& getGridCells() { return gridCells; }
    std::vector<Index3D> getSeedCells();

    // Build KD-Tree
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PCLPointCloudAdaptor<PointT>>,
                                                PCLPointCloudAdaptor<PointT>, 3> KDTree;
    bool checkIndex3DInGrid(const Index3D& index);
private:

    std::vector<Index3D> generateIndices(const uint16_t& z_threshold);
    void cleanUp();
    void addPoint(const PointT& point);
    std::vector<Index3D> getGroundCells();
    std::vector<Index3D> getNeighbors(const GridCell<PointT>& cell, const TerrainType& type, const std::vector<Index3D>& indices, const double& radius);
    double computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2);
    double computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const;
    double computeSlope(const Eigen::Vector3d& normal);
    bool neighborCheck(const GridCell<PointT>& cell, GridCell<PointT>& neighbor);
    Index3D findLowestCell(const std::vector<Index3D> ids);
    int computeMeanHeight(const std::vector<Index3D> ids);
    Index3D cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height);    
    bool fitGroundPlane(GridCell<PointT>& cell, const double& inlier_threshold);
    void selectStartCell(GridCell<PointT>& cell);
    std::pair<size_t,PointT> findLowestPoint(const GridCell<PointT>& cell);
    std::vector<Index3D> expandGrid(std::queue<Index3D> q);

    std::vector<Index3D> indices;
    std::vector<Index3D> obs_indices;

    GridCellsType gridCells;
    GridConfig grid_config;
    std::vector<Index3D> ground_cells;
    std::vector<Index3D> non_ground_cells;
    std::vector<Index3D> selected_cells_first_quadrant;
    std::vector<Index3D> selected_cells_second_quadrant;
    std::vector<Index3D> selected_cells_third_quadrant;
    std::vector<Index3D> selected_cells_fourth_quadrant;
    Eigen::Quaterniond orientation;
    GridCell<PointT> robot_cell;
    ProcessCloudProcessor<PointT> processor;

    std::vector<Index3D> seed_cells;

};
    
template<typename PointT>
PointCloudGrid<PointT>::PointCloudGrid(const GridConfig& config){
    grid_config = config;
    robot_cell.x = 0;
    robot_cell.y = 0;
    robot_cell.z = 0;
   
    if (grid_config.processing_phase == 2){
        indices = generateIndices(1);
    }
    else{
        indices = generateIndices(0);
    }
    obs_indices = generateIndices(0);
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::generateIndices(const uint16_t& z_threshold){
    std::vector<Index3D> idxs;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= z_threshold; ++dz) {
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
}

template<typename PointT>
void PointCloudGrid<PointT>::cleanUp(){
    ground_cells.clear();
    non_ground_cells.clear();
    selected_cells_first_quadrant.clear();
    selected_cells_second_quadrant.clear();
    selected_cells_third_quadrant.clear();
    selected_cells_fourth_quadrant.clear();
    seed_cells.clear();
}

template<typename PointT>
void PointCloudGrid<PointT>::addPoint(const PointT& point){
    double cell_x = point.x / grid_config.cellSizeX;
    double cell_y = point.y / grid_config.cellSizeY;
    double cell_z = point.z / grid_config.cellSizeZ;

    int x = static_cast<int>(std::floor(cell_x));
    int y = static_cast<int>(std::floor(cell_y));
    int z = static_cast<int>(std::floor(cell_z));

    CellType& cell = gridCells[{x,y,z}];
    // information is redundant:
    cell.x = x;
    cell.y = y;
    cell.z = z;
    cell.points->push_back(point);
}

template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Vector3d& normal)
{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = normal;
    planeNormal = orientation * planeNormal;
    planeNormal.normalize();
    return acos(std::abs(planeNormal.dot(zNormal)));    
}

template<typename PointT>
double PointCloudGrid<PointT>::computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = plane.normal();
    planeNormal = orientation * planeNormal;
    planeNormal.normalize();
    return acos(std::abs(planeNormal.dot(zNormal)));
}

template<typename PointT>
bool PointCloudGrid<PointT>::neighborCheck(const GridCell<PointT>& cell, GridCell<PointT>& neighbor){
    pcl::KdTreeFLANN<PointT> kdtree;

    nanoflann::SearchParams params;
    params.checks = 10;  // Minimal checks for speed
    params.eps = 0.5;    // Larger tolerance for faster results
    params.sorted = false; // No need to sort

    size_t nearest_index{0};
    std::vector<int> point_indices(1);
    std::vector<float> point_distances(1);
    uint16_t count = 0;

    bool cell_above = true;   

    typename pcl::PointCloud<PointT>::Ptr total_neighbor_points(new pcl::PointCloud<PointT>());

    int step = neighbor.z;

    do {
        Index3D neighbor_id;
        neighbor_id.x = neighbor.x;
        neighbor_id.y = neighbor.y;
        neighbor_id.z = step;

        if (!checkIndex3DInGrid(neighbor_id)){
            cell_above = false;
        }
        else{
            auto& neighbor_cell = gridCells[neighbor_id];
            if (neighbor_cell.points->size() == 0){
                cell_above = false;
            }
            else{
                *total_neighbor_points += *(neighbor_cell.points);
                step++;
            }
        }
    }
    while (cell_above == true);


    bool cell_below = true;  
    step = neighbor.z-1;

    do {
        Index3D neighbor_id;
        neighbor_id.x = neighbor.x;
        neighbor_id.y = neighbor.y;
        neighbor_id.z = step;

        if (!checkIndex3DInGrid(neighbor_id)){
            cell_below = false;
        }
        else{
            auto& neighbor_cell = gridCells[neighbor_id];
            if (neighbor_cell.points->size() == 0){
                cell_below = false;
            }
            else{
                *total_neighbor_points += *(neighbor_cell.points);
                step--;
            }
        }
    }
    while (cell_below == true);

    // Wrap the PCL point cloud with nanoflann adaptor
    PCLPointCloudAdaptor<PointT> pclAdaptor(*cell.points);
    KDTree tree(3, pclAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    tree.buildIndex();

    double out_dist_sqr;
    nanoflann::KNNResultSet<double> resultSet(1);
    resultSet.init(&nearest_index, &out_dist_sqr);

    for (typename pcl::PointCloud<PointT>::iterator it = total_neighbor_points->begin(); it != total_neighbor_points->end(); ++it){
        Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
        double query_pt[3] = {it->x, it->y, it->z};

        tree.findNeighbors(resultSet, query_pt, params);

        Eigen::Vector3d ground_point(cell.points->points.at(nearest_index).x,
                                     cell.points->points.at(nearest_index).y,
                                     cell.points->points.at(nearest_index).z);

        Eigen::Vector3d diff = obstacle_point - ground_point;
        
        double distance = std::abs(diff.dot(cell.normal)); 
        if (distance < grid_config.groundInlierThreshold){
            count++;
        }
        if ((count / total_neighbor_points->size()) > 0.95){   
            neighbor.terrain_type = TerrainType::GROUND;
            return true;
        }
    }
    return false;
}

template<typename PointT>
Index3D PointCloudGrid<PointT>::findLowestCell(const std::vector<Index3D> ids){

    int lowest_height = std::numeric_limits<int>::max();
    int lowest_index = 0;

    for (int i{0}; i < ids.size(); ++i){
        if (gridCells[ids[i]].z < lowest_height){
            lowest_index = i;
            lowest_height = gridCells[ids[i]].z;
        }
    }
    return ids[lowest_index];
}

template<typename PointT>
int PointCloudGrid<PointT>::computeMeanHeight(const std::vector<Index3D> ids){

    double total_height = 0.0;
    for (const Index3D& id : ids) {
        total_height += gridCells[id].z;
    }

    int mean_height = std::floor(total_height / ids.size());
    return mean_height;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getNeighbors(const GridCell<PointT>& cell, const TerrainType& type, const std::vector<Index3D>& idx, const double& radius){

    std::vector<Index3D> neighbors;

    Index3D cell_id;
    cell_id.x = cell.x;
    cell_id.y = cell.y;
    cell_id.z = cell.z;

    for (int i = 0; i < idx.size(); ++i){
        Index3D neighbor_id = cell_id + idx[i];

        if (!checkIndex3DInGrid(neighbor_id)){
            continue;
        }

        GridCell<PointT>& neighbor = gridCells[neighbor_id];

        if (neighbor.points->size() > 0 && computeDistance(cell.centroid,neighbor.centroid) < radius && neighbor.terrain_type == type){
            Index3D id;
            id.x = neighbor.x;
            id.y = neighbor.y;
            id.z = neighbor.z;
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
        const GridCell<PointT>& cell = gridCells[id];

        double height_difference = std::abs(cell.z - mean_height);
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
    seg.setMethodType(pcl::SAC_PROSAC);
    seg.setMaxIterations(1000);
    seg.setInputCloud(cell.points);
    seg.setDistanceThreshold(threshold); // Adjust this threshold based on your needs
    seg.segment(*inliers, *coefficients);
    cell.inliers = inliers;
    if (cell.inliers->indices.size() == 0){
        return false;
    }

    if (cell.inliers->indices.size() / cell.points->size() > 0.98){
        cell.confidence = Confidence::HIGH;
    }

    Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    double distToOrigin = coefficients->values[3];
    cell.plane = Eigen::Hyperplane<double, 3>(plane_normal, distToOrigin);
    cell.slope = computeSlope(cell.plane);
    return true;
}

template<typename PointT>
void PointCloudGrid<PointT>::selectStartCell(GridCell<PointT>& cell){

    Index3D id;
    id.x = cell.x;
    id.y = cell.y;
    id.z = cell.z;

    if (cell.z >= 0){return;}

    double distance = computeDistance(robot_cell.centroid, cell.centroid);

    if (distance <= grid_config.startCellDistanceThreshold){
        if (cell.x >= 0 && cell.y > 0){
            selected_cells_first_quadrant.push_back(id);
        }
        else if (cell.x <= 0 && cell.y > 0){
            selected_cells_second_quadrant.push_back(id);
        }
        else if (cell.x <= 0 && cell.y < 0){
            selected_cells_third_quadrant.push_back(id);
        }
        else if (cell.x >= 0 && cell.y < 0) {
            selected_cells_fourth_quadrant.push_back(id);
        }
    }
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getGroundCells(){

    if (gridCells.empty()){
        return ground_cells;
    }

    this->cleanUp();

    for (auto& cellPair : gridCells){
        GridCell<PointT>& cell = cellPair.second;

        //TODO: Why ignored?
        if ((cell.points->size() < 3)){continue;}

        Index3D id = cellPair.first;

        pcl::compute3DCentroid(*(cell.points), cell.centroid);

        if (cell.points->size() <= 5){
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
                cell.terrain_type = TerrainType::GROUND;
            }
            continue;
        }

        Eigen::Matrix3d covariance_matrix;
        pcl::computeCovarianceMatrixNormalized(*cell.points, cell.centroid, covariance_matrix);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
        cell.eigenvectors = eigen_solver.eigenvectors();
        cell.eigenvalues = eigen_solver.eigenvalues();

        Eigen::Vector3d normal = cell.eigenvectors.col(0);

        // Ensure all normals point upward
        if (normal(2) < 0) {
            normal *= -1; // flip the normal direction
        }

        normal.normalize();
        cell.normal = normal;

        double ratio = cell.eigenvalues[2] / cell.eigenvalues.sum();
        if (ratio > 0.950){ 
            cell.primitive_type = PrimitiveType::LINE; 
            
            Eigen::Vector3d v = cell.eigenvectors.col(2);
            if (v(2) < 0) {
                v *= -1; // flip the normal direction
            }
            v = orientation * v;
            v.normalize();

            double angle_rad = acos(std::abs(v.dot(Eigen::Vector3d::UnitZ())));

            if (angle_rad > ((90-grid_config.slopeThresholdDegrees) * (M_PI / 180))){
                cell.terrain_type = TerrainType::GROUND;
            }
            else{
                cell.terrain_type = TerrainType::OBSTACLE;
                non_ground_cells.push_back(id);  
            }
            continue;
        } 
        else 
        if (ratio > 0.4){
            cell.primitive_type = PrimitiveType::PLANE; 
            if (std::abs(computeSlope(cell.normal)) > grid_config.slopeThresholdDegrees ){
                cell.terrain_type = TerrainType::OBSTACLE;
                non_ground_cells.push_back(id);  
                continue;
            }
        } 
        else{
            cell.terrain_type = TerrainType::OBSTACLE;
            cell.primitive_type = PrimitiveType::NOISE;
            non_ground_cells.push_back(id);
            continue;
        }

        if (!fitGroundPlane(cell, grid_config.groundInlierThreshold)){
            cell.terrain_type = TerrainType::OBSTACLE;
            non_ground_cells.push_back(id);
            continue;
        }

        if (cell.slope < (grid_config.slopeThresholdDegrees * (M_PI / 180)) ){
            cell.terrain_type = TerrainType::GROUND;

            if (cell.confidence == Confidence::HIGH){
                selectStartCell(cell);
            }
        }
        else{
            cell.terrain_type = TerrainType::OBSTACLE;
            non_ground_cells.push_back(id);
        }
    }

    std::queue<Index3D> q;

    std::vector<Index3D> start_cells_front;
    std::vector<Index3D> start_cells_back;

    if (selected_cells_first_quadrant.size() > 0){
        int cells_q1_mean_height = computeMeanHeight(selected_cells_first_quadrant);
        Index3D q1 = cellClosestToMeanHeight(selected_cells_first_quadrant,cells_q1_mean_height);
        start_cells_front.push_back(q1);
    }

    if (selected_cells_third_quadrant.size() > 0){
        int cells_q3_mean_height = computeMeanHeight(selected_cells_third_quadrant);
        Index3D q3 = cellClosestToMeanHeight(selected_cells_third_quadrant,cells_q3_mean_height);
        start_cells_front.push_back(q3);
    }

    if (selected_cells_second_quadrant.size() > 0){
        int cells_q2_mean_height = computeMeanHeight(selected_cells_second_quadrant);
        Index3D q2 = cellClosestToMeanHeight(selected_cells_second_quadrant,cells_q2_mean_height);
        start_cells_back.push_back(q2);
    }

    if (selected_cells_fourth_quadrant.size() > 0){
        int cells_q4_mean_height = computeMeanHeight(selected_cells_fourth_quadrant);
        Index3D q4 = cellClosestToMeanHeight(selected_cells_fourth_quadrant,cells_q4_mean_height);
        start_cells_back.push_back(q4);
    }

    if (grid_config.num_seed_cells < 4){
        if (start_cells_front.size() > 0){
            Index3D seed_id_front = findLowestCell(start_cells_front);
            q.push(seed_id_front);
            seed_cells.push_back(seed_id_front);
        }
        if (start_cells_back.size() > 0){
            Index3D seed_id_back = findLowestCell(start_cells_back);
            q.push(seed_id_back);
            seed_cells.push_back(seed_id_back);
        }
    }
    else{
        for (const auto& cell : start_cells_front) {
            q.push(cell);
        }

        for (const auto& cell : start_cells_back) {
            q.push(cell);
        }
    }
 
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

        GridCell<PointT>& current_cell = gridCells[idx];
        if (current_cell.expanded || current_cell.points->size() == 0){
            continue;
        }
        current_cell.expanded = true;

        for (int i = 0; i < indices.size(); ++i){
            Index3D neighbor_id = idx + indices[i];
            if (!checkIndex3DInGrid(neighbor_id)){
                if (grid_config.processing_phase == 2){
                    continue;
                }
                int height = neighbor_id.z;
                neighbor_id.z = height + 1;
                if (!checkIndex3DInGrid(neighbor_id)){
                    neighbor_id.z = height - 1;
                    if (!checkIndex3DInGrid(neighbor_id)){
                        continue;
                    }
                }
            }    
    
            GridCell<PointT>& neighbor = gridCells[neighbor_id];
            if(neighbor.points->size() == 0 || neighbor.expanded || neighbor.terrain_type == TerrainType::OBSTACLE){
                continue;
            }

            if (indices[i].z != 0 && grid_config.processing_phase == 2){
                if (!neighborCheck(current_cell,neighbor)){
                    continue;
                }
            }

            if (neighbor.terrain_type == TerrainType::GROUND ){
                q.push(neighbor_id);
            }
        }
        result.emplace_back(idx);
    }
    return result;
}

template<typename PointT>
std::vector<Index3D> PointCloudGrid<PointT>::getSeedCells(){
    return seed_cells;
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
bool PointCloudGrid<PointT>::checkIndex3DInGrid(const Index3D& index){
    if (auto search = gridCells.find(index); search != gridCells.end()){
        return true;
    }
    else{
        return false;
    }
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

    nanoflann::SearchParams params;
    params.checks = 10;  // Minimal checks for speed
    params.eps = 0.5;    // Larger tolerance for faster results
    params.sorted = false; // No need to sort

    ground_cells = getGroundCells();
    for (auto& id : ground_cells){
        GridCell<PointT>& cell = gridCells[id];

        if ((cell.points->size() <= 5 || cell.primitive_type == PrimitiveType::LINE) && cell.terrain_type == TerrainType::GROUND){
            *ground_points += *cell.points; 
            *(cell.ground_points) = *cell.points; 
            continue; 
        }

        extract_ground.setInputCloud(cell.points);
        extract_ground.setIndices(cell.inliers);

        extract_ground.setNegative(false);
        extract_ground.filter(*ground_inliers);

        extract_ground.setNegative(true);
        extract_ground.filter(*non_ground_inliers);

        if (ground_inliers->size() == 0){
            continue;
        } 

        size_t nearest_index{0};
        std::vector<int> point_indices(1);
        std::vector<float> point_distances(1);

        // Wrap the PCL point cloud with nanoflann adaptor
        PCLPointCloudAdaptor<PointT> pclAdaptor(*ground_inliers);
        KDTree tree(3, pclAdaptor, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();

        double out_dist_sqr;
        nanoflann::KNNResultSet<double> resultSet(1);
        resultSet.init(&nearest_index, &out_dist_sqr);

        for (typename pcl::PointCloud<PointT>::iterator it = non_ground_inliers->begin(); it != non_ground_inliers->end(); ++it){
            Eigen::Vector3d obstacle_point(it->x,it->y,it->z);

            double query_pt[3] = {it->x, it->y, it->z};

            tree.findNeighbors(resultSet, query_pt, params);

            Eigen::Vector3d nearest_point(ground_inliers->points.at(nearest_index).x,
                                          ground_inliers->points.at(nearest_index).y,
                                          ground_inliers->points.at(nearest_index).z);

            Eigen::Vector3d diff = obstacle_point - nearest_point;
            double distance = std::abs(diff.dot(cell.normal)); 
           
            if (distance > grid_config.groundInlierThreshold){
                non_ground_points->points.push_back(*it);
            }
            else{
                ground_points->points.push_back(*it);
                cell.ground_points->points.push_back(*it);
            }
        }
        *ground_points += *ground_inliers;
        *(cell.ground_points) += *ground_inliers;
    }

    for (const auto& id : non_ground_cells){
        GridCell<PointT>& cell = gridCells[id];

        typename pcl::PointCloud<PointT>::Ptr close_ground_points(new pcl::PointCloud<PointT>());
        std::vector<Index3D> ground_neighbors = getNeighbors(cell, type_ground, obs_indices, 3);
        if (ground_neighbors.size() == 0){
            *non_ground_points += *cell.points;
            continue;    
        }
        else{
            for (const auto& gp : ground_neighbors){
                GridCell<PointT>& ground_cell = gridCells[gp];
                *close_ground_points += *(ground_cell.ground_points);
            }
        }

        std::vector<Index3D> potential_ground_neighbors = getNeighbors(cell, type_ground, indices, 1);
        std::vector<Index3D> actual_ground_neighbors;

        for (const auto& gn : potential_ground_neighbors){
            GridCell<PointT>& ground_cell = gridCells[gn];
            if (!ground_cell.expanded){
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
                GridCell<PointT>& ground_cell = gridCells[gp];

                double distance = computeDistance(cell.centroid, ground_cell.centroid);

               *close_ground_points += *(ground_cell.ground_points);
                ground_normal += (1/(distance+0.001)) * ground_cell.normal;
            }
            ground_normal /= actual_ground_neighbors.size();
            ground_normal = orientation * ground_normal;
            ground_normal.normalize(); 
        }

        if (close_ground_points->size() == 0){
            *non_ground_points += *cell.points;
            continue;    
        }

        kdtree.setInputCloud(close_ground_points);
        for (typename pcl::PointCloud<PointT>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
            Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
            PointT query_point;
            query_point.x = it->x;
            query_point.y = it->y;
            query_point.z = it->z;

            if (kdtree.nearestKSearch(query_point, 1, point_indices, point_distances) > 0) {
                nearest_index = point_indices[0];
            }


            Eigen::Vector3d nearest_point(close_ground_points->points.at(nearest_index).x,
                                          close_ground_points->points.at(nearest_index).y,
                                          close_ground_points->points.at(nearest_index).z);

            Eigen::Vector3d diff = obstacle_point - nearest_point;
            double distance = std::abs(diff.dot(ground_normal)); 
           
            if (distance > grid_config.groundInlierThreshold){
                non_ground_points->points.push_back(*it);
            }
            else{
                ground_points->points.push_back(*it);
            }
        }
    }

    return std::make_pair(ground_points, non_ground_points);
}

} //namespace pointcloud_obstacle_detection
