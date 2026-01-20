#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/filter.h>   // REQUIRED for removeNaNFromPointCloud

#include <Eigen/Dense>

#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "ground_detection.hpp"

using namespace ground_segmentation;
using PointType = pcl::PointXYZ;

std::string getFileExtension(const std::string & filename)
{
  const auto pos = filename.find_last_of('.');
  return (pos == std::string::npos) ? "" : filename.substr(pos + 1);
}

void removeNaNs(pcl::PointCloud<PointType>::Ptr & cloud)
{
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
}

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <cloud.{pcd|ply}> [cell_size] [slope_deg]\n";
    return EXIT_FAILURE;
  }

  pcl::PointCloud<PointType>::Ptr cloud(
    new pcl::PointCloud<PointType>);

  const std::string filename = argv[1];
  const std::string ext = getFileExtension(filename);

  if (ext == "pcd") {
    if (pcl::io::loadPCDFile<PointType>(filename, *cloud) < 0) {
      std::cerr << "Failed to read PCD file: " << filename << "\n";
      return EXIT_FAILURE;
    }
  } else if (ext == "ply") {
    pcl::PLYReader reader;
    if (reader.read(filename, *cloud) < 0) {
      std::cerr << "Failed to read PLY file: " << filename << "\n";
      return EXIT_FAILURE;
    }
  } else {
    std::cerr << "Unsupported file type: " << ext << "\n";
    return EXIT_FAILURE;
  }

  removeNaNs(cloud);

  if (cloud->empty()) {
    std::cerr << "Input cloud is empty after NaN removal\n";
    return EXIT_FAILURE;
  }

  std::cout << "Loaded cloud with " << cloud->size() << " points\n";

  /* ---------------- Grid configuration ---------------- */

  GridConfig config;
  config.cellSizeX = (argc >= 3) ? std::stod(argv[2]) : 1.0;
  config.cellSizeY = config.cellSizeX;
  config.cellSizeZ = 1.0;

  config.slopeThresholdDegrees = (argc >= 4) ? std::stod(argv[3]) : 30.0;
  config.groundInlierThreshold = 0.1;
  config.centroidSearchRadius = 3.0;
  config.distToGround = 0.0;

  /* ---------------- Phase 1 ---------------- */

  config.processing_phase = 1;

  auto ground_detector =
    std::make_unique<PointCloudGrid<PointType>>(config);

  Eigen::Quaterniond q_sensor_to_world =
    Eigen::Quaterniond::Identity();

  std::cout << "Running ground segmentation phase 1...\n";

  ground_detector->setInputCloud(cloud, q_sensor_to_world);
  auto phase1 = ground_detector->segmentPoints();

  pcl::PointCloud<PointType>::Ptr ground_cloud = phase1.first;
  pcl::PointCloud<PointType>::Ptr obstacle_cloud = phase1.second;

  if (ground_cloud->empty()) {
    std::cerr << "No ground points after phase 1 – aborting\n";
    return EXIT_FAILURE;
  }

  /* ---------------- Phase 2 ---------------- */

  config.processing_phase = 2;

  ground_detector =
    std::make_unique<PointCloudGrid<PointType>>(config);

  std::cout << "Running ground segmentation phase 2...\n";

  ground_detector->setInputCloud(ground_cloud, q_sensor_to_world);
  auto phase2 = ground_detector->segmentPoints();

  // Accumulate obstacles from both phases
  ground_cloud = phase2.first;
  *obstacle_cloud += *phase2.second;

  std::cout << "Ground points:   " << ground_cloud->size() << "\n";
  std::cout << "Obstacle points: " << obstacle_cloud->size() << "\n";

  /* ---------------- Visualisation ---------------- */

  pcl::visualization::PCLVisualizer::Ptr viewer(
    new pcl::visualization::PCLVisualizer("Ground Segmentation Result"));

  viewer->setBackgroundColor(0.05, 0.05, 0.05);
  viewer->addCoordinateSystem(1.0);
  viewer->setUseVbos(true);

  pcl::visualization::PointCloudColorHandlerCustom<PointType>
    ground_color(ground_cloud, 0, 255, 0);
  viewer->addPointCloud(
    ground_cloud, ground_color, "ground");

  pcl::visualization::PointCloudColorHandlerCustom<PointType>
    obstacle_color(obstacle_cloud, 255, 0, 0);
  viewer->addPointCloud(
    obstacle_cloud, obstacle_color, "obstacles");

  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ground");
  viewer->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "obstacles");

  std::cout << "Close the viewer window to exit.\n";

  while (!viewer->wasStopped()) {
    viewer->spinOnce(50);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  return EXIT_SUCCESS;
}
