#include "pointcloud_obstacle_detection/ground_detection.hpp"
#include <queue>

namespace pointcloud_obstacle_detection{

PointCloudGrid::PointCloudGrid(const GridConfig& config){
    grid_config = config;
    robot_cell.row = 0;
    robot_cell.col = 0;
    robot_cell.height = 0;
    total_ground_cells = 0;
 
    for (int dx = -grid_config.neighborsRadius; dx <= grid_config.neighborsRadius; ++dx) {
        for (int dy = -grid_config.neighborsRadius; dy <= grid_config.neighborsRadius; ++dy) {
            //Do not use neighbors in z for now
            for (int dz = 0; dz <= 0; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0){
                    continue;
                }
                 Index3D idx;
                idx.x = dx;
                idx.y = dy;
                idx.z = dz;
                indices.push_back(idx);
            }
        }
    }
}

void PointCloudGrid::clear(){
    gridCells.clear();
    total_ground_cells = 0;
}

GroundDetectionStatistics& PointCloudGrid::getStatistics(){
    return statistics;
}

void PointCloudGrid::addPoint(const pcl::PointXYZ& point) {

    Eigen::Vector3d radial_vector(point.x,point.y,point.z);
    double radial_angle = std::atan2(point.y, point.x);

    int row = static_cast<int>(std::floor(radial_vector.norm() / grid_config.radialCellSize));
    int col = static_cast<int>(std::floor(radial_angle / grid_config.angularCellSize));
    int height = static_cast<int>(std::floor(point.z / grid_config.cellSizeZ));

    gridCells[row][col][height].row = row;
    gridCells[row][col][height].col = col;
    gridCells[row][col][height].height = height;
    gridCells[row][col][height].points->push_back(point);
}

// Function to calculate the slope from the normal vector
double PointCloudGrid::computeSlope(const Eigen::Vector3d& normal)
{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = normal;
    planeNormal = orientation * planeNormal;
    planeNormal.normalize(); //just in case
    return acos(planeNormal.dot(zNormal));
}

double PointCloudGrid::computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const
{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = plane.normal();
    planeNormal = orientation * planeNormal;
    planeNormal.normalize(); //just in case
    return acos(planeNormal.dot(zNormal));
}

Eigen::Vector3d PointCloudGrid::computeSlopeDirection(const Eigen::Hyperplane< double, int(3) >& plane) const
{
    /** The vector of maximum slope on a plane is the projection of (0,0,1) onto the plane.
     *  (0,0,1) is the steepest vector possible in the global frame, thus by projecting it onto
     *  the plane we get the steepest vector possible on that plane.
     */
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    const Eigen::Vector3d planeNormal(plane.normal().normalized());
    const Eigen::Vector3d projection = zNormal - zNormal.dot(planeNormal) * planeNormal;
    return projection;
}

double PointCloudGrid::computeGridDistance(const GridCell& cell1, const GridCell& cell2){

    double dx = cell1.row - cell2.row;
    double dy = cell1.col - cell2.col;
    double dz = cell1.height - cell2.height;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

int PointCloudGrid::computeMeanHeight(const std::vector<Index3D> ids){

    // Calculate the mean height of selected cells
    double total_height = 0.0;
    for (const Index3D& id : ids) {
        total_height += gridCells[id.x][id.y][id.z].height;
    }
    // Find the cell closest to the mean height
    int mean_height = std::floor(total_height / ids.size());
    return mean_height;
}

double PointCloudGrid::computeMeanPointsHeight(const std::vector<Index3D> ids){
    // Calculate the mean height of selected cells
    int total_points = 0;
    double mean_height = 0.0;
    for (const Index3D& id : ids) {
        GridCell& cell = gridCells[id.x][id.y][id.z];

        // Compute the transformation matrix
        Eigen::Affine3f transform = pcl::getTransFromUnitVectorsZY(Eigen::Vector3f::UnitZ(), cell.normal.cast<float>());

        // Apply the transformation to the point clou
        pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::transformPointCloud(*(cell.points), *transformed_cloud, transform.cast<double>());

        for (pcl::PointCloud<pcl::PointXYZ>::iterator it = transformed_cloud->begin(); it != transformed_cloud->end(); ++it)
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

std::vector<Index3D> PointCloudGrid::getNeighbors(const GridCell& cell, const TerrainType& type, const std::vector<Index3D>& idx){

    std::vector<Index3D> neighbors;
    for (int i = 0; i < idx.size(); ++i){
        int neighborX = cell.row + idx[i].x;
        int neighborY = cell.col + idx[i].y;
        int neighborZ = cell.height + idx[i].z;

        GridCell& neighbor = gridCells[neighborX][neighborY][neighborZ];
        if (neighbor.points->size() > 0 && computeDistance(cell.centroid,neighbor.centroid) < grid_config.neighborsRadius && neighbor.terrain_type == type){
            Index3D id;
            id.x = neighbor.row;
            id.y = neighbor.col;
            id.z = neighbor.height;
            neighbors.push_back(id);
        }
    }
    return neighbors;
}

Index3D PointCloudGrid::cellClosestToMeanHeight(const std::vector<Index3D>& ids, const int mean_height){

    int min_height_difference = std::numeric_limits<int>::max();
    int max_ground_neighbors = std::numeric_limits<int>::min();
    Index3D closest_to_mean_height;

    for (const Index3D& id : ids) {
        const GridCell& cell = gridCells[id.x][id.y][id.z];

        double height_difference = std::abs(cell.height - mean_height);
        int neighbor_count = getNeighbors(cell, TerrainType::GROUND, indices).size();

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

bool PointCloudGrid::fitGroundPlane(GridCell& cell, const double& threshold){

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_MSAC);
    seg.setMaxIterations(1000);
    seg.setInputCloud(cell.points);
    seg.setDistanceThreshold(threshold); // Adjust this threshold based on your needs
    seg.segment(*inliers, *coefficients);
    cell.inliers = inliers;
    pcl::compute3DCentroid(*(cell.points), cell.centroid);

    if (cell.inliers->indices.size() == 0){
        return false;
    }

    // Compute the covariance matrix
    Eigen::Matrix3d covariance_matrix;
    pcl::computeCovarianceMatrixNormalized(*cell.points, cell.centroid, covariance_matrix);

    // Compute eigenvectors and eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
    Eigen::Matrix3d eigenvectors = eigen_solver.eigenvectors();

    // Normal is the eigenvector corresponding to the smallest eigenvalue
    Eigen::Vector3d normal = eigenvectors.col(0);

    // Ensure all normals point upward
    if (normal(2) < 0) {
        normal *= -1; // flip the normal direction
    }

    Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    normal.normalize();
    cell.normal = normal;
    double distToOrigin = coefficients->values[3];
    cell.plane = Eigen::Hyperplane<double, 3>(plane_normal, distToOrigin);
    //const Eigen::Vector3d slopeDir = computeSlopeDirection(cell.plane);
    cell.slope = computeSlope(cell.plane);
    //cell.slopeDirection = slopeDir;
    //cell.slopeDirectionAtan2 = std::atan2(slopeDir.y(), slopeDir.x());
    return true;
}

void PointCloudGrid::selectStartCell(GridCell& cell){

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
        else if (cell.row >= 0 && cell.col < 0){
            selected_cells_second_quadrant.push_back(id);
        }
        else if (cell.row <= 0 && cell.col < 0){
            selected_cells_third_quadrant.push_back(id);
        }
        else if (cell.row >= 0 && cell.col > 0) {
            selected_cells_fourth_quadrant.push_back(id);
        }
    }
}

std::vector<Index3D> PointCloudGrid::getGroundCells() {

    if (gridCells.empty()){
        return ground_cells;
    }

    ground_cells.clear();
    non_ground_cells.clear();
    undefined_cells.clear();
    selected_cells_first_quadrant.clear();
    selected_cells_second_quadrant.clear();
    selected_cells_third_quadrant.clear();
    selected_cells_fourth_quadrant.clear();
    total_ground_cells = 0;

    for (auto& rowPair : gridCells) {
        for (auto& colPair : rowPair.second) {
            for (auto& heightPair : colPair.second) {
                GridCell& cell = heightPair.second;

                Index3D id;
                id.x = cell.row;
                id.y = cell.col;
                id.z = cell.height;

                if (cell.points->size() <= 1){
                    cell.terrain_type = TerrainType::UNDEFINED;
                    undefined_cells.push_back(id);
                    continue;
                }

                if (cell.points->size() <= grid_config.minPoints) {
                    Eigen::Vector4f centroid;
                    pcl::compute3DCentroid(*(cell.points), centroid);
                    Eigen::Vector4f squared_diff_sum(0, 0, 0, 0);

                    for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                        Eigen::Vector4f diff = (*it).getVector4fMap() - centroid;
                        squared_diff_sum += diff.array().square().matrix();                     
                    }

                    Eigen::Vector4f variance = squared_diff_sum / cell.points->size();

                    if (variance[0] > variance[2] || variance[1] > variance[2]){
                        cell.terrain_type = TerrainType::GROUND;
                    }
                    else{
                        cell.terrain_type = TerrainType::OBSTACLE;
                        non_ground_cells.push_back(id);
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
    ground_cells = expandGrid(q);
    return ground_cells;
}

std::vector<Index3D> PointCloudGrid::expandGrid(std::queue<Index3D> q){
    std::vector<Index3D> result;
    int count{0};
    while (!q.empty()){

        Index3D idx = q.front();
        q.pop();
        GridCell& current_cell = gridCells[idx.x][idx.y][idx.z];

        if (current_cell.expanded == true || current_cell.points->size() == 0 ){
            continue;
        }
        current_cell.expanded = true;

        for (int i = 0; i < indices.size(); ++i){

            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height + indices[i].z; 

            GridCell& neighbor = gridCells[neighborX][neighborY][neighborZ];

            if(neighbor.points->size() == 0 || neighbor.expanded){
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

std::vector<Index3D> PointCloudGrid::explore(std::queue<Index3D> q){
    std::vector<Index3D> result;
    std::map<int, std::map<int, std::map<int, GridCell>>> copy = gridCells;

    while (!q.empty()){

        Index3D& idx = q.front();
        q.pop();

        GridCell& current_cell = copy[idx.x][idx.y][idx.z];

        if (current_cell.explored == true){
            continue;
        }
        current_cell.explored = true;
      
        for (int i = 0; i < indices.size(); ++i){

            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height + indices[i].z;

            GridCell& neighbor = copy[neighborX][neighborY][neighborZ];

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

double PointCloudGrid::computeDistance(const Eigen::Vector4d& centroid1, const Eigen::Vector4d& centroid2) {
    Eigen::Vector3d diff = centroid1.head<3>() - centroid2.head<3>();
    return diff.norm();
}

void PointCloudGrid::setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr input, const Eigen::Quaterniond& R_body2World){

    this->clear();
    orientation = R_body2World;
    unsigned int index = 0;
    for (pcl::PointCloud<pcl::PointXYZ>::iterator it = input->begin(); it != input->end(); ++it)
    {
        this->addPoint(*it);
        index++;
    }
    ground_cells = getGroundCells();
}

std::pair<size_t,pcl::PointXYZ> PointCloudGrid::findLowestPoint(const GridCell& cell){
    double min_height = std::numeric_limits<float>::max();
    pcl::PointXYZ min_point;
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

std::pair<pcl::PointCloud<pcl::PointXYZ>::Ptr,pcl::PointCloud<pcl::PointXYZ>::Ptr> PointCloudGrid::segmentPoints() {

    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_inliers(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_points(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr non_ground_inliers(new pcl::PointCloud<pcl::PointXYZ>());

    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;
    const TerrainType type_ground = TerrainType::GROUND;
    const TerrainType type_obstacle = TerrainType::OBSTACLE;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    for (auto& id : ground_cells){
        GridCell& cell = gridCells[id.x][id.y][id.z];
        if (cell.points->size() < grid_config.minPoints){
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it)
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
        Eigen::Vector3d ground_normal = cell.normal;
        ground_normal.normalize();

        for (pcl::PointCloud<pcl::PointXYZ>::iterator it = non_ground_inliers->begin(); it != non_ground_inliers->end(); ++it){
                Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
                pcl::PointXYZ search_point;
                search_point.x = it->x;
                search_point.y = it->y;
                search_point.z = it->z;

                if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
                    nearest_index = point_indices[0];
                }

                Eigen::Vector3d ground_point(ground_inliers->points.at(nearest_index).x,
                                             ground_inliers->points.at(nearest_index).y,
                                             ground_inliers->points.at(nearest_index).z);

                double distance = std::abs(ground_normal.dot(obstacle_point - ground_point) / ground_normal.norm()); 

                if (distance > grid_config.groundInlierThreshold){
                    non_ground_points->points.push_back(*it);
                }
                else{
                    ground_points->points.push_back(*it);
                }
            }

        if (grid_config.returnGroundPoints){
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = ground_inliers->begin(); it != ground_inliers->end(); ++it)
            {
                ground_points->points.push_back(*it);
            }
        }
    }

   for (const auto& id : non_ground_cells){
        GridCell& cell = gridCells[id.x][id.y][id.z];
        std::vector<Index3D> potential_ground_neighbors = getNeighbors(cell, type_ground, indices);
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

        if (actual_ground_neighbors.size() == 0){
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it)
            {
                non_ground_points->points.push_back(*it);
            }
        }
        else{
            size_t nearest_index{0};
            std::vector<int> point_indices(1);
            std::vector<float> point_distances(1);

            Eigen::Vector3d ground_normal;
            pcl::PointCloud<pcl::PointXYZ>::Ptr close_ground_points(new pcl::PointCloud<pcl::PointXYZ>());
            for (const auto& gp : actual_ground_neighbors){
                GridCell& ground_cell = gridCells[gp.x][gp.y][gp.z];

                double distance = computeDistance(cell.centroid, ground_cell.centroid);

                extract_ground.setInputCloud(ground_cell.points);
                extract_ground.setIndices(ground_cell.inliers);

                extract_ground.setNegative(false);
                extract_ground.filter(*ground_inliers);

                for (pcl::PointCloud<pcl::PointXYZ>::iterator it = ground_inliers->points.begin(); it != ground_inliers->points.end(); ++it){
                    close_ground_points->points.push_back(*it);
                }

                ground_normal += (1/(distance+0.001)) * ground_cell.normal.normalized();
            }

            if (close_ground_points->size() == 0)
                continue;

            ground_normal /= actual_ground_neighbors.size();
            ground_normal = orientation * ground_normal;
            ground_normal.normalize(); 

            kdtree.setInputCloud(close_ground_points);
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
                pcl::PointXYZ search_point;
                search_point.x = it->x;
                search_point.y = it->y;
                search_point.z = it->z;

                if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
                    nearest_index = point_indices[0];
                }

                Eigen::Vector3d ground_point(close_ground_points->points.at(nearest_index).x,
                                             close_ground_points->points.at(nearest_index).y,
                                             close_ground_points->points.at(nearest_index).z);

                double distance = std::abs(ground_normal.dot(obstacle_point - ground_point) / ground_normal.norm()); 

                if (distance > grid_config.groundInlierThreshold){
                    non_ground_points->points.push_back(*it);
                }
                else{
                    ground_points->points.push_back(*it);
                }
            }
        }

        statistics.clear();
        statistics.ground_cells = ground_cells.size();
        statistics.non_ground_cells = non_ground_cells.size();
        statistics.undefined_cells = undefined_cells.size();
 
        /*
        for (const auto& id : undefined_cells){
            GridCell& cell = gridCells[id.x][id.y][id.z];
            std::vector< pcl::PointCloud<pcl::PointXYZ>::Ptr> result = processor.Clustering_euclideanCluster(cell.points,0.5,1,1000);    
            std::cout << "Clusters are : " << result.size() << std::endl;
            std::vector<GridCell> ground_neighbors = getNeighbors(cell, type_ground);
            std::vector<GridCell> obstacle_neighbors = getNeighbors(cell, type_obstacle);

            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                if (ground_neighbors.size() < obstacle_neighbors.size()){
                    non_ground_points->points.push_back(*it);
                }
                else{
                    ground_points->points.push_back(*it);
                }
            }                
        }
        */ 
    }
    return std::make_pair(ground_points, non_ground_points);
}

std::map<int, std::map<int, std::map<int, GridCell>>>& PointCloudGrid::getGridCells(){
    return gridCells;
}

//pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudGrid::extractHoles(){
//TODO
//}

} //namespace pointcloud_obstacle_detection