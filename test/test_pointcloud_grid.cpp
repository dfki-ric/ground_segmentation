#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "ground_detection.hpp"

using namespace ground_segmentation;
using PointT = pcl::PointXYZ;

GridConfig defaultConfig()
{
    GridConfig cfg;
    cfg.cellSizeX = 1.5;
    cfg.cellSizeY = 1.0;
    cfg.cellSizeZ = 1.5;
    cfg.slopeThresholdDegrees = 30.0;
    cfg.groundInlierThreshold = 0.1;
    cfg.centroidSearchRadius = 2.5;
    cfg.distToGround = 0.0;
    cfg.processing_phase = 1;
    return cfg;
}

pcl::PointCloud<PointT>::Ptr makeFlatGround(
    int n, double z = 0.0)
{
    auto cloud = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);

    for (int i = 0; i < n; ++i) {
        PointT p;
        p.x = static_cast<float>(i % 10);
        p.y = static_cast<float>(i / 10);
        p.z = z;
        cloud->push_back(p);
    }
    return cloud;
}

TEST(Index3DTest, HashAndEquality)
{
    Index3D a(1,2,3);
    Index3D b(1,2,3);
    Index3D c(3,2,1);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);

    Index3D::HashFunction h;
    EXPECT_EQ(h(a), h(b));
}

TEST(PointCloudGridTest, InsertAndLookupCell)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto cloud = makeFlatGround(10);
    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());

    EXPECT_FALSE(grid.getGridCells().empty());

    Index3D idx(0,0,0);
    EXPECT_TRUE(grid.checkIndex3DInGrid(idx));
}

TEST(PointCloudGridTest, FlatGroundIsClassifiedAsGround)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto cloud = makeFlatGround(100, 0.0);
    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());

    auto result = grid.segmentPoints();
    auto ground = result.first;
    auto nonground = result.second;

    EXPECT_GT(ground->size(), 0u);
    EXPECT_EQ(nonground->size(), 0u);
}

TEST(PointCloudGridTest, FlatLinePointsAreGround)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto cloud = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);

    for (int i = 0; i < 500; ++i) {
        PointT p;
        p.x = i * 0.01f;      
        p.y = i * 0.01f;
        p.z = 0.0f;
        cloud->push_back(p);
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto result = grid.segmentPoints();

    EXPECT_EQ(result.first->size(), 500);
    EXPECT_EQ(result.second->size(), 0);
}

TEST(PointCloudGridTest, VerticalLinePointsAreObstacle)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto cloud = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);

    for (int i = 0; i < 500; ++i) {
        PointT p;
        p.x = 5.0f;      
        p.y = 0.0f;
        p.z = i * 0.01f;
        cloud->push_back(p);
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto result = grid.segmentPoints();

    EXPECT_EQ(result.first->size(), 0);
    EXPECT_EQ(result.second->size(), 500);
}

TEST(PointCloudGridTest, VerticalWallIsObstacle)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto cloud = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);

    for (int iy = 0; iy < 20; ++iy) {
        for (int iz = 0; iz < 20; ++iz) {
            PointT p;
            p.x = 0.0f;                // vertical plane
            p.y = iy * 0.1f;
            p.z = iz * 0.1f;
            cloud->push_back(p);
        }
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto result = grid.segmentPoints();

    //EXPECT_EQ(result.first->size(), 0u);
    EXPECT_GT(result.second->size(), 0u);
}

TEST(PointCloudGridTest, EmptyInputDoesNotCrash)
{
    PointCloudGrid<PointT> grid(defaultConfig());

    auto empty = pcl::PointCloud<PointT>::Ptr(
        new pcl::PointCloud<PointT>);

    EXPECT_NO_THROW({
        grid.setInputCloud(empty, Eigen::Quaterniond::Identity());
        auto res = grid.segmentPoints();
    });
}

TEST(PointCloudGridTest, Phase1_AllowsVerticalPropagationFromRobot)
{
    auto cfg = defaultConfig();

    PointCloudGrid<PointT> grid(cfg);
    auto cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

    // Vertical column through robot
    for (int i = 0; i < 500; ++i) {
        cloud->push_back({0.0f, 0.0f, i * 0.01f});
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto [ground, obstacle] = grid.segmentPoints();

    EXPECT_GT(ground->size(), 0u);
    EXPECT_LT(ground->size(), cloud->size());
}

TEST(PointCloudGridTest, Phase1_AllowsSlantedLine)
{
    PointCloudGrid<PointT> grid(defaultConfig());
    auto cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

    for (int i = 0; i < 200; ++i) {
        cloud->push_back({0.0f, i * 0.05f, i * 0.05f});
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto [ground, obstacle] = grid.segmentPoints();

    EXPECT_GT(ground->size(), 0u);
}

TEST(PointCloudGridTest, Phase2_StillSeedsRobotCell)
{
    auto cfg = defaultConfig();
    cfg.processing_phase = 2;

    PointCloudGrid<PointT> grid(cfg);
    auto cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

    for (int i = 0; i < 200; ++i) {
        cloud->push_back({0.0f, 0.0f, i * 0.01f});
    }

    grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
    auto [ground, obstacle] = grid.segmentPoints();

    EXPECT_GT(ground->size(), 0u);
}

TEST(PointCloudGridTest, OrientationAffectsGroundClassification)
{
    auto cfg = defaultConfig();
    cfg.slopeThresholdDegrees = 30.0;
    cfg.cellSizeX = 1.0;
    cfg.cellSizeY = 1.0;
    cfg.cellSizeZ = 0.5;
    cfg.processing_phase = 1;

    auto cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>);

    // Create a sloped plane: ~20°
    const double slope_rad = 20.0 * M_PI / 180.0;
    for (int x = -5; x <= 5; ++x) {
        for (int y = -5; y <= 5; ++y) {
            PointT p;
            p.x = x * 0.2f;
            p.y = y * 0.2f;
            p.z = std::tan(slope_rad) * p.y;
            cloud->push_back(p);
        }
    }

    // Case 1: identity orientation (should be ground)
    {
        PointCloudGrid<PointT> grid(cfg);
        grid.setInputCloud(cloud, Eigen::Quaterniond::Identity());
        auto [ground, obstacle] = grid.segmentPoints();

        EXPECT_GT(ground->size(), 0u);
        EXPECT_LT(obstacle->size(), ground->size());
    }

    // Case 2: rotated gravity (should become obstacle)
    {
        PointCloudGrid<PointT> grid(cfg);

        // Rotate gravity frame by +30° around X axis
        Eigen::AngleAxisd tilt(30.0 * M_PI / 180.0, Eigen::Vector3d::UnitX());
        Eigen::Quaterniond rotated_orientation(tilt);

        grid.setInputCloud(cloud, rotated_orientation);
        auto [ground, obstacle] = grid.segmentPoints();

        EXPECT_GT(obstacle->size(), ground->size());
    }
}
