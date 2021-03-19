cmake_minimum_required(VERSION 3.0.2)
project(yolo)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  pcl_conversions
  pcl_ros
  sensor_msgs
  tf2
  tf2_ros
	cv_bridge
	image_transport
)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_DIR})

catkin_package(
	INCLUDE_DIRS
)

include_directories(
	${catkin_INCLUDE_DIRS}
)


