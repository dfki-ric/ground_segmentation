
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/principal_curvatures.h>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/features/normal_3d.h>
#include <iostream>
#include <thread>

#include <ground_detection.hpp>

//using namespace std::chrono_literals;
using namespace pointcloud_obstacle_detection;


int main (int argc, char** argv)
{
    if (argc != 2)
        return (0);
    
    CloudXYZ cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    pcl::PLYReader PLYFileReader;
    const int offset=0;
    PLYFileReader.read<pcl::PointXYZ>(argv[1],*cloud,offset);

    GridConfig config;

    config.radialCellSize = 3;
    config.angularCellSize = 0.785398;
    config.cellSizeZ = 1;

    config.neighborsRadius = 1;

    config.startCellDistanceThreshold = 20;
    config.slopeThresholdDegrees = 3;
    config.groundInlierThreshold = 0.1;
    config.returnGroundPoints = true;
    config.minPoints = 5;


    PointCloudGrid* ground_detection = new PointCloudGrid(config);
    GroundDetectionStatistics statistics;

    Eigen::Quaterniond R_robot2World(1,0,0,0); 
    
    std::cout <<"Adding points" << std::endl;
    ground_detection->setInputCloud(cloud, R_robot2World);

    std::cout <<"Segmenting points " << std::endl;
    std::pair<CloudXYZ,CloudXYZ> result = ground_detection->segmentPoints();
    std::map<int, std::map<int, std::map<int, GridCell>>>& gridCells = ground_detection->getGridCells();

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    int count{0};


    for (auto& rowPair : gridCells) {
        for (auto& colPair : rowPair.second) {
            for (auto& heightPair : colPair.second) {
                GridCell& cell = heightPair.second;

                if (cell.points->size() == 0) continue;

                uint8_t r = rand() % 256;
                uint8_t g = rand() % 256;
                uint8_t b = rand() % 256;

                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cell.points, r, g, b);
                viewer->addPointCloud<pcl::PointXYZ>(cell.points, color, std::to_string(count));
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, std::to_string(count++)); 

                // Compute centroid of the point cloud
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*cell.points, centroid);

                // Compute the covariance matrix
                Eigen::Matrix3f covariance_matrix;
                pcl::computeCovarianceMatrixNormalized(*cell.points, centroid, covariance_matrix);

                // Compute eigenvectors and eigenvalues
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
                Eigen::Matrix3f eigenvectors = eigen_solver.eigenvectors();

                // Normal is the eigenvector corresponding to the smallest eigenvalue
                Eigen::Vector3f normal = eigenvectors.col(0);

                // Ensure all normals point upward
                if (normal(2) < 0) {
                    normal *= -1; // flip the normal direction
                }

                // Draw normal at the centroid
                pcl::PointXYZ centroid_point(centroid[0], centroid[1], centroid[2]);
                pcl::PointXYZ normal_point(centroid[0] + normal[0], centroid[1] + normal[1], centroid[2] + normal[2]);

                viewer->addArrow(normal_point, centroid_point, 1.0, 0, 0, false, "normal" + std::to_string(count));
  
            }
        }
    }


    // Start visualization loop
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    
    return (0);
}
