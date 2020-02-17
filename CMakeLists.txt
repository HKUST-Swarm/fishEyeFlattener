cmake_minimum_required(VERSION 2.8.3)
project(fisheye_flattener)

# Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-march=native -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wall -fPIC -fopenmp")
set(ENABLE_BACKWARD false)

SET("OpenCV_DIR"  "/usr/local/share/OpenCV/")
find_package(OpenCV 3.4 REQUIRED)

# Find catkin macros and libraries if COMPONENTS list like find_package(catkin
# REQUIRED COMPONENTS xyz) is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS roscpp std_msgs cv_bridge camera_model)
find_package(Ceres REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)


catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES 
  CATKIN_DEPENDS roscpp std_msgs cv_bridge
  # DEPENDS system_lib
)

if(ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
endif()

include_directories(inc ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_nodelet src/fisheyeFlattener_node.cpp)

# Rename C++ executable without prefix The above recommended prefix causes long
# target names, the following renames the target back to the shorter version for
# ease of user use e.g. "rosrun someones_pkg node" instead of "rosrun
# someones_pkg someones_pkg_node" set_target_properties(${PROJECT_NAME}_node
# PROPERTIES OUTPUT_NAME node PREFIX "")

# Add cmake target dependencies of the executable same as for the library above
add_dependencies(${PROJECT_NAME}_nodelet ${${PROJECT_NAME}_EXPORTED_TARGETS}
                 ${catkin_EXPORTED_TARGETS})

# Specify libraries to link a library or executable target against
if(ENABLE_BACKWARD)
  target_link_libraries(${PROJECT_NAME}_nodelet ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  camera_model
  dw
  stdc++fs)
else()
  target_link_libraries(${PROJECT_NAME}_nodelet ${catkin_LIBRARIES} 
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
  camera_model
  stdc++fs)
endif()


# ##############################################################################
# Install ##
# ##############################################################################


install(TARGETS ${PROJECT_NAME}_nodelet
DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)


