#include "pointcloud_obstacle_detection/ground_detection.hpp"
#include <queue>

PointCloudGrid::PointCloudGrid(){

    robot_cell.row = 0;
    robot_cell.col = 0;
    robot_cell.height = 0;
    total_ground_cells = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            //for (int dz = -1; dz <= 1; ++dz) {

                if (dx == 0 && dy == 0) // && dz == 0){
                    continue;
                //}
 
                Index3D idx;

                idx.x = dx;
                idx.y = dy;
                idx.z = 0;
                indices.push_back(idx);
            //}
        }
    }
}

PointCloudGrid::PointCloudGrid(const GridConfig& config){
    grid_config = config;
    robot_cell.row = 0;
    robot_cell.col = 0;
    robot_cell.height = 0;
    total_ground_cells = 0;

    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            //for (int dz = -1; dz <= 1; ++dz) {

                if (dx == 0 && dy == 0) // && dz == 0){
                    continue;
                //}
 
                Index3D idx;

                idx.x = dx;
                idx.y = dy;
                idx.z = 0;
                indices.push_back(idx);
            //}
        }
    }
}

void PointCloudGrid::clear(){
    gridCells.clear();
    total_ground_cells = 0;
}

void PointCloudGrid::addPoint(const pcl::PointXYZ& point) {
    int row = static_cast<int>(std::floor(point.x / grid_config.cellSizeX));
    int col = static_cast<int>(std::floor(point.y / grid_config.cellSizeY));
    int height = static_cast<int>(std::floor(point.z / grid_config.cellSizeZ));

    if(!(row >= -grid_config.gridSizeX    && row < grid_config.gridSizeX &&
         col >= -grid_config.gridSizeY    && col < grid_config.gridSizeY &&
         height >= -grid_config.gridSizeZ  && height < grid_config.gridSizeZ)){
        return;
    }

    gridCells[row][col][height].row = row;
    gridCells[row][col][height].col = col;
    gridCells[row][col][height].height = height;
    gridCells[row][col][height].points->push_back(point);
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

double PointCloudGrid::calculateDistance(const GridCell& cell1, const GridCell& cell2){

    double dx = cell1.row - cell2.row;
    double dy = cell1.col - cell2.col;
    double dz = cell1.height - cell2.height;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

int PointCloudGrid::calculateMeanHeight(const std::vector<GridCell> cells){

    // Calculate the mean height of selected cells
    double total_height = 0.0;
    for (const GridCell& cell : cells) {
        total_height += cell.height;
    }

    // Find the cell closest to the mean height
    int mean_height = std::floor(total_height / cells.size());
    return mean_height;
}

std::vector<GridCell> PointCloudGrid::getNeighbors(const GridCell& cell, const TerrainType& type){

    std::vector<GridCell> neighbors;
    for (int i = 0; i < indices.size(); ++i){
        int neighborX = cell.row + indices[i].x;
        int neighborY = cell.col + indices[i].y;
        int neighborZ = cell.height; // + indices[i].z;
        //int neighborZ = cell.height;

        // Check if the neighbor is within the grid boundaries
        if (neighborX >= -grid_config.gridSizeX  && neighborX < grid_config.gridSizeX &&
            neighborY >= -grid_config.gridSizeY  && neighborY < grid_config.gridSizeY &&
            neighborZ >= -grid_config.gridSizeZ  && neighborZ < grid_config.gridSizeZ){

            GridCell neighbor = gridCells[neighborX][neighborY][neighborZ];
            if (neighbor.points->points.size() > 0 && computeDistance(cell.centroid,neighbor.centroid) < 1 && neighbor.terrain_type == type){
                neighbors.push_back(neighbor);
            }
        }
    }
    return neighbors;
}

int PointCloudGrid::countGroundNeighbors(const GridCell& cell){

    int neighbors{0};
    for (int i = 0; i < indices.size(); ++i){
        int neighborX = cell.row + indices[i].x;
        int neighborY = cell.col + indices[i].y;
        int neighborZ = cell.height; // + indices[i].z;
        //int neighborZ = cell.height;

        // Check if the neighbor is within the grid boundaries
        if (neighborX >= -grid_config.gridSizeX  && neighborX < grid_config.gridSizeX &&
            neighborY >= -grid_config.gridSizeY  && neighborY < grid_config.gridSizeY &&
            neighborZ >= -grid_config.gridSizeZ  && neighborZ < grid_config.gridSizeZ){

            GridCell neighbor = gridCells[neighborX][neighborY][neighborZ];
            if (neighbor.points->points.size() > 0 && computeDistance(cell.centroid,neighbor.centroid) < 1 && neighbor.terrain_type == TerrainType::GROUND){
                neighbors++;
            }
        }
    }
    return neighbors;
}

GridCell PointCloudGrid::cellClosestToMeanHeight(const std::vector<GridCell>& cells, const int mean_height){

    int min_height_difference = std::numeric_limits<int>::max();
    int max_ground_neighbors = std::numeric_limits<int>::min();
    GridCell closest_to_mean_height;

    for (const GridCell& cell : cells) {

        double height_difference = std::abs(cell.height - mean_height);
        int neighbor_count = countGroundNeighbors(cell);

        if (height_difference <= min_height_difference) {
            if (neighbor_count >= max_ground_neighbors){
                closest_to_mean_height = cell;
                min_height_difference = height_difference;
                max_ground_neighbors = neighbor_count;
            }
        }
    }
    return closest_to_mean_height;
}

bool PointCloudGrid::fitPlane(GridCell& cell, const double& threshold){

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

    if (inliers->indices.size() / cell.points->size() > 0.95) {
        Eigen::Vector3d normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
        normal.normalize();
        double distToOrigin = coefficients->values[3];
        cell.plane = Eigen::Hyperplane<double, 3>(normal, distToOrigin);
        const Eigen::Vector3d slopeDir = computeSlopeDirection(cell.plane);
        cell.slope = computeSlope(cell.plane);
        cell.slopeDirection = slopeDir;
        cell.slopeDirectionAtan2 = std::atan2(slopeDir.y(), slopeDir.x());

        return true;
    }
    return false;
}

void PointCloudGrid::selectStartCell(GridCell& cell){

    double distance = calculateDistance(robot_cell, cell);
    if (distance <= grid_config.startCellDistanceThreshold) {
        // This grid cell is within the specified radius around the robot

        if (cell.row >= 0 && cell.col > 0){
            selected_cells_first_quadrant.push_back(cell);
        }
        else if (cell.row >= 0 && cell.col < 0){
            selected_cells_second_quadrant.push_back(cell);
        }
        else if (cell.row <= 0 && cell.col < 0){
            selected_cells_third_quadrant.push_back(cell);
        }
        else if (cell.row >= 0 && cell.col > 0) {
            selected_cells_fourth_quadrant.push_back(cell);
        }
    }
}

std::vector<GridCell> PointCloudGrid::getGroundCells() {

    if (gridCells.empty()){
        return ground_cells;
    }

    ground_cells.clear();
    non_ground_cells.clear();
    holes_cells.clear();
    selected_cells_first_quadrant.clear();
    selected_cells_second_quadrant.clear();
    selected_cells_third_quadrant.clear();
    selected_cells_fourth_quadrant.clear();
    total_ground_cells = 0;

    for (auto& rowPair : gridCells) {
        for (auto& colPair : rowPair.second) {
            for (auto& heightPair : colPair.second) {
                GridCell& cell = heightPair.second;

                if (cell.points->size() < 5) {
                    cell.terrain_type = TerrainType::UNDEFINED;
                    continue;
                }
                                
                if (fitPlane(cell, grid_config.groundInlierThreshold) && cell.slope < (grid_config.slopeThresholdDegrees * (M_PI / 180)) ){
                    cell.terrain_type = TerrainType::GROUND;
                    total_ground_cells += 1;
                    selectStartCell(cell);
                }
                else{
                    cell.terrain_type = TerrainType::OBSTACLE;
                    non_ground_cells.push_back(cell);
                }
            }
        }
    }

    std::queue<Index3D> q;

    if (selected_cells_first_quadrant.size() > 0){
        int cells_q1_mean_height = calculateMeanHeight(selected_cells_first_quadrant);
        GridCell closest_to_mean_height_q1 = cellClosestToMeanHeight(selected_cells_first_quadrant, cells_q1_mean_height);
        Index3D q1;
        q1.x = closest_to_mean_height_q1.row;
        q1.y = closest_to_mean_height_q1.col;
        q1.z = closest_to_mean_height_q1.height;
        q.push(q1);
    }

    if (selected_cells_second_quadrant.size() > 0){
        int cells_q2_mean_height = calculateMeanHeight(selected_cells_second_quadrant);
        GridCell closest_to_mean_height_q2 = cellClosestToMeanHeight(selected_cells_second_quadrant, cells_q2_mean_height);
        Index3D q2;
        q2.x = closest_to_mean_height_q2.row;
        q2.y = closest_to_mean_height_q2.col;
        q2.z = closest_to_mean_height_q2.height;
        q.push(q2);
    }

    if (selected_cells_third_quadrant.size() > 0){
        int cells_q3_mean_height = calculateMeanHeight(selected_cells_third_quadrant);
        GridCell closest_to_mean_height_q3 = cellClosestToMeanHeight(selected_cells_third_quadrant, cells_q3_mean_height);
        Index3D q3;
        q3.x = closest_to_mean_height_q3.row;
        q3.y = closest_to_mean_height_q3.col;
        q3.z = closest_to_mean_height_q3.height;
        q.push(q3);
    }

    if (selected_cells_fourth_quadrant.size() > 0){
        int cells_q4_mean_height = calculateMeanHeight(selected_cells_fourth_quadrant);
        GridCell closest_to_mean_height_q4 = cellClosestToMeanHeight(selected_cells_fourth_quadrant, cells_q4_mean_height);
        Index3D q4;
        q4.x = closest_to_mean_height_q4.row;
        q4.y = closest_to_mean_height_q4.col;
        q4.z = closest_to_mean_height_q4.height;
        q.push(q4);
    }
    ground_cells = expandGrid(q);
    return ground_cells;
}

std::vector<GridCell> PointCloudGrid::expandGrid(std::queue<Index3D> q){
    std::vector<GridCell> result;
    while (!q.empty()){

        Index3D& idx = q.front();
        q.pop();

        if (gridCells[idx.x][idx.y][idx.z].expanded == true){
            continue;
        }
        gridCells[idx.x][idx.y][idx.z].expanded = true;
        GridCell& current_cell = gridCells[idx.x][idx.y][idx.z];
      
        for (int i = 0; i < indices.size(); ++i){

            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height; //+ indices[i].z; 

            // Check if the neighbor is within the grid boundaries
            if (neighborX >= -grid_config.gridSizeX  && neighborX < grid_config.gridSizeX &&
                neighborY >= -grid_config.gridSizeY  && neighborY < grid_config.gridSizeY &&
                neighborZ >= -grid_config.gridSizeZ  && neighborZ < grid_config.gridSizeZ) {

                GridCell neighbor = gridCells[neighborX][neighborY][neighborZ];

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
        }
        result.emplace_back(current_cell);
    }
    return result;
}

std::vector<GridCell> PointCloudGrid::explore(std::queue<Index3D> q){
    std::vector<GridCell> result;
    std::map<int, std::map<int, std::map<int, GridCell>>> copy = gridCells;

    while (!q.empty()){

        Index3D& idx = q.front();
        q.pop();

        if (copy[idx.x][idx.y][idx.z].explored == true){
            continue;
        }
        copy[idx.x][idx.y][idx.z].explored = true;
        GridCell& current_cell = copy[idx.x][idx.y][idx.z];
      
        for (int i = 0; i < indices.size(); ++i){

            int neighborX = current_cell.row + indices[i].x;
            int neighborY = current_cell.col + indices[i].y;
            int neighborZ = current_cell.height; // + indices[i].z;

            // Check if the neighbor is within the grid boundaries
            if (neighborX >= -grid_config.gridSizeX  && neighborX < grid_config.gridSizeX &&
                neighborY >= -grid_config.gridSizeY  && neighborY < grid_config.gridSizeY &&
                neighborZ >= -grid_config.gridSizeZ  && neighborZ < grid_config.gridSizeZ) {

                GridCell neighbor = copy[neighborX][neighborY][neighborZ];

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
        }
        result.emplace_back(current_cell);
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

    for (size_t i = 0; i < cell.points->points.size(); ++i) {
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

    // Extract points based on indices
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;

    const TerrainType type_ground = TerrainType::GROUND;
    const TerrainType type_obstacle = TerrainType::OBSTACLE;

    for (auto& cellx : ground_cells){
        GridCell& cell = gridCells[cellx.row][cellx.col][cellx.height];
        std::vector<GridCell> obstacle_neighbors = getNeighbors(cell, type_obstacle);
        if (obstacle_neighbors.size() > 0){
            fitPlane(cell, 0.05);
            if (cell.slope > (grid_config.slopeThresholdDegrees * (M_PI / 180))){
                cell.terrain_type = TerrainType::OBSTACLE;
            }
        }

        extract_ground.setInputCloud(cell.points);
        extract_ground.setIndices(cell.inliers);

        if (cell.terrain_type == TerrainType::GROUND){
            extract_ground.setNegative(false);
            extract_ground.filter(*ground_inliers);
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = ground_inliers->begin(); it != ground_inliers->end(); ++it)
            {
                ground_points->points.push_back(*it);
            }
            extract_ground.setNegative(true);
            extract_ground.filter(*non_ground_inliers);
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = non_ground_inliers->begin(); it != non_ground_inliers->end(); ++it)
            {
                non_ground_points->points.push_back(*it);
            }
        }
        else{
            cell.terrain_type = TerrainType::OBSTACLE;
            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it)
            {
                non_ground_points->points.push_back(*it);
            }
        }
    }

    for (const auto& cellx : non_ground_cells){
        GridCell& cell = gridCells[cellx.row][cellx.col][cellx.height];
        std::vector<GridCell> potential_ground_neighbors = getNeighbors(cell, type_ground);
        std::vector<GridCell> actual_ground_neighbors;

        for (const auto& gn : potential_ground_neighbors){
            std::vector<GridCell> explored_neighbors;
            std::queue<Index3D> q;
            Index3D id;
            id.x = gn.row;
            id.y = gn.col;
            id.z = gn.height;
            q.push(id);
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
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            std::pair<size_t, pcl::PointXYZ> lowest_point = findLowestPoint(cell);

            Eigen::Vector3d e_lowest_point{lowest_point.second.x, lowest_point.second.y, lowest_point.second.z};

            pcl::PointXYZ search_point = lowest_point.second;
            std::vector<int> point_indices(1);
            std::vector<float> point_distances(1);

            double min_distance = std::numeric_limits<float>::max();
            size_t ground_cell_index{0};
            size_t index{0};
            for (const auto& gp : actual_ground_neighbors){
                double distance = (gp.centroid.head<3>() - e_lowest_point).norm();
                if (distance < min_distance) {
                    min_distance = distance;
                    ground_cell_index = index;
                }
                index++;
            }    
 
            kdtree.setInputCloud(actual_ground_neighbors.at(ground_cell_index).points);

            size_t nearest_index{0};
            if (kdtree.nearestKSearch(search_point, 1, point_indices, point_distances) > 0) {
                nearest_index = point_indices[0];
            }

            Eigen::Vector3d ground_point(actual_ground_neighbors.at(ground_cell_index).points->points.at(nearest_index).x,
                                         actual_ground_neighbors.at(ground_cell_index).points->points.at(nearest_index).y,
                                         actual_ground_neighbors.at(ground_cell_index).points->points.at(nearest_index).z);

            Eigen::Vector3d ground_normal = actual_ground_neighbors.at(ground_cell_index).plane.normal();
            ground_normal.normalize(); 
            ground_normal = orientation * ground_normal;

            for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
                Eigen::Vector3d obstacle_point(it->x,it->y,it->z);
                double distance = ground_normal.dot(obstacle_point - ground_point) / ground_normal.norm();                
                if (std::abs(distance) > 0.05){
                    non_ground_points->points.push_back(*it);
                }
                else{
                    ground_points->points.push_back(*it);
                }
            }
        }
    }
    return std::make_pair(ground_points, non_ground_points);
}

//pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudGrid::extractHoles(){
//TODO
//}
