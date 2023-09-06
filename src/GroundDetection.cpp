#include "GroundDetection.h"

PointCloudGrid::PointCloudGrid(double cellSizeX, double cellSizeY,double cellSizeZ) : cellSizeX(cellSizeX), cellSizeY(cellSizeY), cellSizeZ(cellSizeZ){

    gridWidth = 100;
    gridDepth = 100;

    gridHeight = 100;

}

void PointCloudGrid::clear(){
    gridCells.clear();
}

void PointCloudGrid::addPoint(const pcl::PointXYZI& point, const unsigned int index) {
    int row = static_cast<int>(std::floor(point.x / cellSizeX));
    int col = static_cast<int>(std::floor(point.y / cellSizeY));
    int height = static_cast<int>(std::floor(point.z / cellSizeZ));

    if(!(row >= -gridWidth && row < gridWidth && col >= -gridDepth && col < gridDepth && height >= -gridHeight && height < gridHeight)){
        return;
    }

    gridCells[row][col][height].row = row;
    gridCells[row][col][height].col = col;
    gridCells[row][col][height].height = height;
    gridCells[row][col][height].points->push_back(point);
    gridCells[row][col][height].source_indices->indices.push_back(index);
}

double PointCloudGrid::computeSlope(const Eigen::Hyperplane< double, int(3) >& plane) const
{
    const Eigen::Vector3d zNormal(Eigen::Vector3d::UnitZ());
    Eigen::Vector3d planeNormal = plane.normal();
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

std::vector<GridCell> PointCloudGrid::getGroundCells() {
    std::vector<GridCell> ground_cells;

    for (auto& rowPair : gridCells) {
        for (auto& colPair : rowPair.second) {
            for (auto& heightPair : colPair.second) {        
                GridCell& cell = heightPair.second;

                if (cell.points->size() > 5) {

                    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
                    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                    pcl::SACSegmentation<pcl::PointXYZI> seg;
                    seg.setOptimizeCoefficients(true);
                    seg.setModelType(pcl::SACMODEL_PLANE);
                    seg.setMethodType(pcl::SAC_MSAC);
                    seg.setMaxIterations(1000);
                    seg.setInputCloud(cell.points);
                    seg.setDistanceThreshold(0.1); // Adjust this threshold based on your needs
                    seg.segment(*inliers, *coefficients); 

                    if (inliers->indices.size() > 5) {
                        
                        Eigen::Vector4d centroid;
                        pcl::compute3DCentroid(*(cell.points), centroid);

                        Eigen::Vector3d normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
                        normal.normalize();
                        double distToOrigin = coefficients->values[3];
                        cell.plane = Eigen::Hyperplane<double, 3>(normal, distToOrigin);

                        //adjust height of patch
                        Eigen::ParametrizedLine<double, 3> line(Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitZ());
                        Eigen::Vector3d newPos =  line.intersectionPoint(cell.plane);

                        if(newPos.x() > 0.0001 || newPos.y() > 0.0001)
                        {
                            std::cout << "TraversabilityGenerator3d: Error, adjustement height calculation is weird" << std::endl;
                            break;
                        }

                        if(newPos.allFinite())
                        {
                            cell.height = newPos.z();                            
                        }

                        const Eigen::Vector3d slopeDir = computeSlopeDirection(cell.plane);
                        cell.slope = computeSlope(cell.plane);
                        cell.slopeDirection = slopeDir;
                        cell.slopeDirectionAtan2 = std::atan2(slopeDir.y(), slopeDir.x());

                        if (cell.slope < 0.785398){
                            cell.isGround = true ;
                            ground_cells.push_back(cell);
                        }
                    }
                }
            }
        }
    }
    return ground_cells;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudGrid::getGroundPoints() {

    std::vector<GridCell> ground_cells = getGroundCells();
    pcl::PointCloud<pcl::PointXYZI>::Ptr ground_points(new pcl::PointCloud<pcl::PointXYZI>());

    for (auto& cell : ground_cells){
        for (pcl::PointCloud<pcl::PointXYZI>::iterator it = cell.points->begin(); it != cell.points->end(); ++it)
        {
            ground_points->points.push_back(*it);
        }
    }
    return ground_points;
}


pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudGrid::removeGroundPoints(pcl::PointCloud<pcl::PointXYZI>::Ptr source){

    this->clear();
    unsigned int index = 0;
    for (pcl::PointCloud<pcl::PointXYZI>::iterator it = source->begin(); it != source->end(); ++it)
    {
        this->addPoint(*it,index);
        index++;
    }

    std::vector<GridCell> ground_cells = getGroundCells();

    pcl::PointIndices::Ptr ground_indices(new pcl::PointIndices);

    for (auto& cell : ground_cells){
        for (int i{0}; i < cell.source_indices->indices.size(); ++i)
        {
            ground_indices->indices.push_back(cell.source_indices->indices.at(i));
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_points(new pcl::PointCloud<pcl::PointXYZI>());

    // Extract points based on indices
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setNegative (true);
    extract.setInputCloud(source);
    extract.setIndices(ground_indices);
    extract.filter(*non_ground_points);

    return non_ground_points;    
}
