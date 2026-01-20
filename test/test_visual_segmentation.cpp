#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <Eigen/Dense>

#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include <string>

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
  cloud->is_dense = true;
}

int main(int argc, char ** argv)
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <cloud.{pcd|ply}>\n";
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

  std::cout << "Loaded cloud with "
            << cloud->size() << " points\n";

  GridConfig config;
  config.cellSizeX = 1.0;
  config.cellSizeY = 1.0;
  config.cellSizeZ = 1.0;
  config.slopeThresholdDegrees = 30.0;
  config.groundInlierThreshold = 0.1;
  config.centroidSearchRadius = 3.0;
  config.distToGround = 0.0;
  config.processing_phase = 1;

  auto ground_detector =
    std::make_unique<PointCloudGrid<PointType>>(config);

  Eigen::Quaterniond R_robot2World =
    Eigen::Quaterniond::Identity();

  std::cout << "Running ground segmentation...\n";
  ground_detector->setInputCloud(cloud, R_robot2World);

  auto result = ground_detector->segmentPoints();
  auto ground_cloud = result.first;
  auto obstacle_cloud = result.second;

  std::cout << "Ground points:   " << ground_cloud->size() << "\n";
  std::cout << "Obstacle points: " << obstacle_cloud->size() << "\n";

  pcl::visualization::PCLVisualizer::Ptr viewer_seg(
    new pcl::visualization::PCLVisualizer("Segmentation Result"));

  viewer_seg->setBackgroundColor(0.05, 0.05, 0.05);
  viewer_seg->addCoordinateSystem(1.0);

  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  ground_color(ground_cloud, 0, 255, 0);
  viewer_seg->addPointCloud(
    ground_cloud, ground_color, "ground");

  pcl::visualization::PointCloudColorHandlerCustom<PointType>
  obstacle_color(obstacle_cloud, 255, 0, 0);
  viewer_seg->addPointCloud(
    obstacle_cloud, obstacle_color, "obstacles");

  viewer_seg->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "ground");
  viewer_seg->setPointCloudRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "obstacles");

  pcl::visualization::PCLVisualizer::Ptr viewer_grid(
    new pcl::visualization::PCLVisualizer("Grid Cells & Normals"));

  viewer_grid->setBackgroundColor(0.05, 0.05, 0.05);
  viewer_grid->addCoordinateSystem(1.0);

  auto & gridCells = ground_detector->getGridCells();

  int id = 0;
  for (const auto & cellPair : gridCells) {
    const GridCell<PointType> & cell = cellPair.second;

    if (!cell.points || cell.points->empty()) {
      continue;
    }

    uint8_t r, g, b;
    if (cell.terrain_type == TerrainType::GROUND) {
      r = 0; g = 255; b = 0;
    } else {
      r = 255; g = 0; b = 0;
    }

    const std::string cid = "cell_" + std::to_string(id);

    pcl::visualization::PointCloudColorHandlerCustom<PointType>
    cell_color(cell.points, r, g, b);

    viewer_grid->addPointCloud(
      cell.points, cell_color, cid);

    viewer_grid->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cid);

    // Normal visualization (if available)
    if (cell.normal.norm() > 1e-6) {
      Eigen::Vector3d n = cell.normal;
      if (n.z() < 0.0) {n *= -1.0;}

      const Eigen::Vector4d & c = cell.centroid;
      PointType p0(c.x(), c.y(), c.z());
      PointType p1(
        c.x() + n.x(),
        c.y() + n.y(),
        c.z() + n.z());

      viewer_grid->addArrow(
        p1, p0,
        1.0, 0.0, 0.0,
        false,
        "normal_" + std::to_string(id));
    }

    ++id;
  }

  std::cout << "Close both windows to exit.\n";

  while (!viewer_seg->wasStopped() &&
    !viewer_grid->wasStopped())
  {
    viewer_seg->spinOnce(50);
    viewer_grid->spinOnce(50);
    std::this_thread::sleep_for(
      std::chrono::milliseconds(50));
  }

  return EXIT_SUCCESS;
}
