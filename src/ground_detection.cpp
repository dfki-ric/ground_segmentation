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

int PointCloudGrid::calculateMeanHeight(const std::vector<Index3D> ids){

    // Calculate the mean height of selected cells
    double total_height = 0.0;
    for (const Index3D& id : ids) {
        total_height += gridCells[id.x][id.y][id.z].height;
    }
    // Find the cell closest to the mean height
    int mean_height = std::floor(total_height / ids.size());
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

std::vector<GridCell> PointCloudGrid::fitPlanes(GridCell& cell, const double& threshold){
    int neighbor_size = std::max(static_cast<std::size_t>(12),cell.points->size());

    //Efficient Ransac
    Point_set point_set;

    for (pcl::PointCloud<pcl::PointXYZ>::iterator it = cell.points->begin(); it != cell.points->end(); ++it){
        Point_3 pt(it->x, it->y, it->z);
        point_set.insert(pt);
    }

    // Add normal property and estimate normal values
    point_set.add_normal_map();
    CGAL::jet_estimate_normals<CGAL::Sequential_tag>
        (point_set,
        neighbor_size, // Number of neighbors
        point_set.parameters(). // Named parameters provided by Point_set_3
        degree_fitting(2));     // additional named parameter specific to jet_estimate_normals
    // Simplify point set
    CGAL::grid_simplify_point_set
        (point_set,
        0.1); // Size of grid cell
    // point_set.parameters() can be omitted if no additional named parameter is needed
    std::vector<std::string> properties = point_set.properties();
    // Detect sphere with RANSAC
    Efficient_ransac ransac;
    ransac.set_input(point_set,
                        point_set.point_map(),   // Call built-in property map
                        point_set.normal_map()); // Call built-in property map
    ransac.add_shape_factory<Plane>();
    Efficient_ransac::Parameters parameters;
    parameters.probability = 0.05;
    parameters.min_points = std::size_t(point_set.size() / 3);
    parameters.epsilon = 0.01;
    parameters.cluster_epsilon = 0.2;
    parameters.normal_threshold = 0.9;
    ransac.detect(parameters);
    Efficient_ransac::Plane_range planes = ransac.planes();

    std::vector<GridCell> fitted_planes;

    for (auto plane : planes){
        GridCell temp;
        auto indices = plane->indices_of_assigned_points();

        for (int i : indices){
            Point_set::Index id(i);
            const auto& point = point_set.point(id);
            temp.points->points.push_back(pcl::PointXYZ(point.x(),point.y(),point.z()));
        }

        Eigen::Vector3d normal{plane->plane_normal().x(), plane->plane_normal().y(), plane->plane_normal().z()};
        normal.normalize();
        double distToOrigin = plane->d();

        temp.plane = Eigen::Hyperplane<double, 3>(normal, distToOrigin);
        const Eigen::Vector3d slopeDir = computeSlopeDirection(temp.plane);
        temp.slope = computeSlope(temp.plane);
        temp.slopeDirection = slopeDir;
        temp.slopeDirectionAtan2 = std::atan2(slopeDir.y(), slopeDir.x());    
        fitted_planes.push_back(temp);
    }
    return fitted_planes;
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

void PointCloudGrid::train(Point_set_c& pts){
    Imap label_map;
    bool lm_found = false;
    boost::tie (label_map, lm_found) = pts.property_map<int> ("label");
    if (!lm_found)
    {
    std::cerr << "Error: \"label\" property not found in input file." << std::endl;
    return;
    }
    std::vector<int> ground_truth;
    ground_truth.reserve (pts.size());
    std::copy (pts.range(label_map).begin(), pts.range(label_map).end(),
                std::back_inserter (ground_truth));
    std::cerr << "Generating features" << std::endl;
    CGAL::Real_timer t;
    t.start();
    Feature_set features;
    std::size_t number_of_scales = 5;
    Feature_generator generator (pts, pts.point_map(), number_of_scales);
    #ifdef CGAL_LINKED_WITH_TBB
    features.begin_parallel_additions();
    #endif
    generator.generate_point_based_features (features);
    #ifdef CGAL_LINKED_WITH_TBB
    features.end_parallel_additions();
    #endif
    t.stop();
    std::cerr << features.size() << " feature(s) generated in " << t.time() << " second(s)" << std::endl;
    // Add types
    Label_set labels;
    Label_handle ground = labels.add ("ground");
    Label_handle vegetation = labels.add ("vegetation");
    Label_handle roof = labels.add ("roof");
    Classifier classifier (labels, features);
    std::cerr << "Training" << std::endl;
    t.reset();
    t.start();
    classifier.train<CGAL::Sequential_tag> (ground_truth, 800);
    t.stop();
    std::cerr << "Done in " << t.time() << " second(s)" << std::endl;
    t.reset();
    t.start();
    std::vector<int> label_indices(pts.size(), -1);
    Classification::classify_with_graphcut<CGAL::Sequential_tag>
    (pts, pts.point_map(), labels, classifier,
        generator.neighborhood().k_neighbor_query(12),
        0.2f, 10, label_indices);
    t.stop();
    std::cerr << "Classification with graphcut done in " << t.time() << " second(s)" << std::endl;
    std::cerr << "Precision, recall, F1 scores and IoU:" << std::endl;
    Classification::Evaluation evaluation (labels, ground_truth, label_indices);
    for (std::size_t i = 0; i < labels.size(); ++ i)
    {
    std::cerr << " * " << labels[i]->name() << ": "
                << evaluation.precision(labels[i]) << " ; "
                << evaluation.recall(labels[i]) << " ; "
                << evaluation.f1_score(labels[i]) << " ; "
                << evaluation.intersection_over_union(labels[i]) << std::endl;
    }
    std::cerr << "Accuracy = " << evaluation.accuracy() << std::endl
            << "Mean F1 score = " << evaluation.mean_f1_score() << std::endl
            << "Mean IoU = " << evaluation.mean_intersection_over_union() << std::endl;
      
}

void PointCloudGrid::classify(std::vector<Kernel_c::Point_3>& pts){
    float grid_resolution = 0.34f;
    unsigned int number_of_neighbors = 6;
    std::cerr << "Computing useful structures" << std::endl;
    Iso_cuboid_3 bbox = CGAL::bounding_box (pts.begin(), pts.end());
    Planimetric_grid grid (pts, IPmap(), bbox, grid_resolution);
    Neighborhood neighborhood (pts, IPmap());
    Local_eigen_analysis eigen
    = Local_eigen_analysis::create_from_point_set
    (pts, IPmap(), neighborhood.k_neighbor_query(number_of_neighbors));
    float radius_neighbors = 1.7f;
    float radius_dtm = 15.0f;
    std::cerr << "Computing features" << std::endl;
    Feature_set features;
    #ifdef CGAL_LINKED_WITH_TBB
    features.begin_parallel_additions();
    #endif
    Feature_handle distance_to_plane = features.add<Distance_to_plane> (pts, IPmap(), eigen);
    Feature_handle dispersion = features.add<Dispersion> (pts, IPmap(), grid,
                                                        radius_neighbors);
    Feature_handle elevation = features.add<Elevation> (pts, IPmap(), grid,
                                                        radius_dtm);
    #ifdef CGAL_LINKED_WITH_TBB
    features.end_parallel_additions();
    #endif
    Label_set labels;
    Label_handle ground = labels.add ("ground");
    Label_handle vegetation = labels.add ("vegetation");
    Label_handle roof = labels.add ("roof");
    std::cerr << "Setting weights" << std::endl;
    Classifier classifier (labels, features);
    classifier.set_weight (distance_to_plane, 6.75e-2f);
    classifier.set_weight (dispersion, 5.45e-1f);
    classifier.set_weight (elevation, 1.47e1f);
    std::cerr << "Setting effects" << std::endl;
    classifier.set_effect (ground, distance_to_plane, Classifier::NEUTRAL);
    classifier.set_effect (ground, dispersion, Classifier::NEUTRAL);
    classifier.set_effect (ground, elevation, Classifier::PENALIZING);
    classifier.set_effect (vegetation, distance_to_plane,  Classifier::FAVORING);
    classifier.set_effect (vegetation, dispersion, Classifier::FAVORING);
    classifier.set_effect (vegetation, elevation, Classifier::NEUTRAL);
    classifier.set_effect (roof, distance_to_plane,  Classifier::NEUTRAL);
    classifier.set_effect (roof, dispersion, Classifier::NEUTRAL);
    classifier.set_effect (roof, elevation, Classifier::FAVORING);
    // Run classification
    std::cerr << "Classifying" << std::endl;
    std::vector<int> label_indices (pts.size(), -1);
    CGAL::Real_timer t;
    t.start();
    Classification::classify<Concurrency_tag> (pts, labels, classifier, label_indices);
    t.stop();
    std::cerr << "Raw classification performed in " << t.time() << " second(s)" << std::endl;
    t.reset();    
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
        int cells_q1_mean_height = calculateMeanHeight(selected_cells_first_quadrant);
        Index3D closest_to_mean_height_q1 = cellClosestToMeanHeight(selected_cells_first_quadrant, cells_q1_mean_height);
        q.push(closest_to_mean_height_q1);
    }

    if (selected_cells_second_quadrant.size() > 0){
        int cells_q2_mean_height = calculateMeanHeight(selected_cells_second_quadrant);
        Index3D closest_to_mean_height_q2 = cellClosestToMeanHeight(selected_cells_second_quadrant, cells_q2_mean_height);
        q.push(closest_to_mean_height_q2);
    }

    if (selected_cells_third_quadrant.size() > 0){
        int cells_q3_mean_height = calculateMeanHeight(selected_cells_third_quadrant);
        Index3D closest_to_mean_height_q3 = cellClosestToMeanHeight(selected_cells_third_quadrant, cells_q3_mean_height);
        q.push(closest_to_mean_height_q3);
    }

    if (selected_cells_fourth_quadrant.size() > 0){
        int cells_q4_mean_height = calculateMeanHeight(selected_cells_fourth_quadrant);
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

    // Extract points based on indices
    pcl::ExtractIndices<pcl::PointXYZ> extract_ground;

    const TerrainType type_ground = TerrainType::GROUND;
    const TerrainType type_obstacle = TerrainType::OBSTACLE;

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

        for (pcl::PointCloud<pcl::PointXYZ>::iterator it = non_ground_inliers->begin(); it != non_ground_inliers->end(); ++it)
        {
            non_ground_points->points.push_back(*it);
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
            pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
            size_t nearest_index{0};
            std::vector<int> point_indices(1);
            std::vector<float> point_distances(1);
            pcl::ExtractIndices<pcl::PointXYZ> extract_ground;

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

                ground_normal += (1/(distance+0.001)) * ground_cell.plane.normal().normalized();
            }

            if (close_ground_points->size() == 0)
                continue;

            ground_normal /= actual_ground_neighbors.size();
            ground_normal.normalize(); 
            ground_normal = orientation * ground_normal;

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

                double distance = ground_normal.dot(obstacle_point - ground_point) / ground_normal.norm(); 

                if (distance > grid_config.groundInlierThreshold){
                    non_ground_points->points.push_back(*it);
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

//pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudGrid::extractHoles(){
//TODO
//}

} //namespace pointcloud_obstacle_detection