#include <boost/test/unit_test.hpp>
#include <pointcloud_obstacle_detection/Dummy.hpp>

using namespace pointcloud_obstacle_detection;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    pointcloud_obstacle_detection::DummyClass dummy;
    dummy.welcome();
}
