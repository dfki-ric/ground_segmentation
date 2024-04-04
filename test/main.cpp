
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

// Function to extract the file extension from a given filename
std::string getFileExtension(const std::string& filename) {
    size_t lastDotPos = filename.find_last_of(".");
    if (lastDotPos != std::string::npos) {
        return filename.substr(lastDotPos + 1);
    }
    // Return an empty string if there is no extension
    return "";
}


int main (int argc, char** argv)
{
    if (argc != 2)
        return (0);
    
    CloudXYZ cloud (new pcl::PointCloud<pcl::PointXYZ> ());

    std::string ext = getFileExtension(argv[1]);

    if (ext == "ply"){
        pcl::PLYReader PLYFileReader;
        const int offset=0;
        PLYFileReader.read<pcl::PointXYZ>(argv[1],*cloud,offset);
    }
    else
    if (ext == "pcd"){
        if (pcl::io::loadPCDFile<pcl::PointXYZ> (argv[1], *cloud) == -1) //* load the file
        {
            std::cerr << "Couldn't read file: " << argv[1] << std::endl;
            return 0;
        }
    }

    GridConfig config;

    //config.radialCellSize = 1;
    //config.angularCellSize = 0.3;
    config.cellSizeX = 1;
    config.cellSizeY = 1;
    config.cellSizeZ = 1;

    config.neighborsIndexThreshold = 1;

    config.startCellDistanceThreshold = 5;
    config.slopeThresholdDegrees = 30;
    config.groundInlierThreshold = 0.1;
    config.returnGroundPoints = true;
    config.minPoints = 5;
    config.grid_type = GridType::SQUARE;


    PointCloudGrid* ground_detection = new PointCloudGrid(config);
    GroundDetectionStatistics statistics;

    Eigen::Quaterniond R_robot2World(1,0,0,0); 
    
    std::cout <<"Adding points" << std::endl;
    ground_detection->setInputCloud(cloud, R_robot2World);

    std::cout <<"Segmenting points " << std::endl;
    std::pair<CloudXYZ,CloudXYZ> result = ground_detection->segmentPoints();

    pcl::visualization::PCLVisualizer::Ptr viewer2 (new pcl::visualization::PCLVisualizer ("3D Viewer 2"));
    viewer2->setBackgroundColor (0, 0, 0);
    viewer2->addCoordinateSystem (1.0);
    viewer2->initCameraParameters ();

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ground(result.first, 0, 255, 0);
    viewer2->addPointCloud<pcl::PointXYZ>(result.first, ground, "ground");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> obstacle(result.second, 255, 0, 0);
    viewer2->addPointCloud<pcl::PointXYZ>(result.second, obstacle, "obstacle");

    std::map<int, std::map<int, std::map<int, GridCell>>>& gridCells = ground_detection->getGridCells();

    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();

    int count{0};
    bool show_grid{true};
    for (auto& rowPair : gridCells) {
        for (auto& colPair : rowPair.second) {
            for (auto& heightPair : colPair.second) {
                GridCell& cell = heightPair.second;

                if (cell.points->size() == 0) continue;

                uint8_t r,g,b;

                if (show_grid){
                    r = rand() % 256;
                    g = rand() % 256;
                    b = rand() % 256;
                }
                else{
                    if (cell.terrain_type == TerrainType::GROUND){
                        r = 0; g = 255 ; b = 0;
                    }
                    else if (cell.terrain_type == TerrainType::OBSTACLE){
                        r = 255; g = 0 ; b = 0;
                    }

                }

                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color(cell.points, r, g, b);
                viewer->addPointCloud<pcl::PointXYZ>(cell.points, color, std::to_string(count));
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, std::to_string(count++)); 

                Eigen::Vector4d centroid = cell.centroid;
                Eigen::Vector3d normal = cell.normal;

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
        viewer2->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    
    return (0);
}
