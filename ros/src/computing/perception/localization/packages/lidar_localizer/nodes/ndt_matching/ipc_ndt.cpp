/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 Localization program using Normal Distributions Transform

 Yuki KITSUKAWA
 */

#include <pthread.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include <boost/filesystem.hpp>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <velodyne_pointcloud/point_types.h>
#include <velodyne_pointcloud/rawdata.h>

#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/TwistStamped.h>

#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <pcl/registration/ndt.h>
#ifdef CUDA_FOUND
#include <ndt_gpu/NormalDistributionsTransform.h>
#endif
#ifdef USE_PCL_OPENMP
#include <pcl_omp_registration/ndt.h>
#endif

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

#include <autoware_config_msgs/ConfigNDT.h>

#include <autoware_msgs/NDTStat.h>

//headers in Autoware Health Checker
#include <autoware_health_checker/node_status_publisher.h>

#define PREDICT_POSE_THRESHOLD 0.5

#define Wa 0.4
#define Wb 0.3
#define Wc 0.3

static std::shared_ptr<autoware_health_checker::NodeStatusPublisher> node_status_publisher_ptr_;

struct pose
{
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

enum class MethodType
{
  PCL_GENERIC = 0,
  PCL_ANH = 1,
  PCL_ANH_GPU = 2,
  PCL_OPENMP = 3,
};
static MethodType _method_type = MethodType::PCL_GENERIC;

static pose initial_pose, predict_pose, predict_pose_imu, predict_pose_odom, predict_pose_imu_odom, previous_pose,
    ndt_pose, current_pose, current_pose_imu, current_pose_odom, current_pose_imu_odom, localizer_pose;

static double offset_x, offset_y, offset_z, offset_yaw;  // current_pos - previous_pose
static double offset_imu_x, offset_imu_y, offset_imu_z, offset_imu_roll, offset_imu_pitch, offset_imu_yaw;
static double offset_odom_x, offset_odom_y, offset_odom_z, offset_odom_roll, offset_odom_pitch, offset_odom_yaw;
static double offset_imu_odom_x, offset_imu_odom_y, offset_imu_odom_z, offset_imu_odom_roll, offset_imu_odom_pitch,
    offset_imu_odom_yaw;

// Can't load if typed "pcl::PointCloud<pcl::PointXYZRGB> map, add;"
static pcl::PointCloud<pcl::PointXYZ> map, add;

// If the map is loaded, map_loaded will be 1.
static int map_loaded = 0;
static int _use_gnss = 1;
static int init_pos_set = 0;

static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
static cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> anh_ndt;
#ifdef CUDA_FOUND
static std::shared_ptr<gpu::GNormalDistributionsTransform> anh_gpu_ndt_ptr =
    std::make_shared<gpu::GNormalDistributionsTransform>();
#endif
#ifdef USE_PCL_OPENMP
static pcl_omp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> omp_ndt;
#endif

// Default values
static int max_iter = 30;        // Maximum iterations
static float ndt_res = 1.0;      // Resolution
static double step_size = 0.1;   // Step size
static double trans_eps = 0.01;  // Transformation epsilon

static ros::Publisher predict_pose_pub;
static geometry_msgs::PoseStamped predict_pose_msg;

static ros::Publisher predict_pose_imu_pub;
static geometry_msgs::PoseStamped predict_pose_imu_msg;

static ros::Publisher predict_pose_odom_pub;
static geometry_msgs::PoseStamped predict_pose_odom_msg;

static ros::Publisher predict_pose_imu_odom_pub;
static geometry_msgs::PoseStamped predict_pose_imu_odom_msg;

static ros::Publisher ndt_pose_pub;
static geometry_msgs::PoseStamped ndt_pose_msg;

// current_pose is published by vel_pose_mux
/*
static ros::Publisher current_pose_pub;
static geometry_msgs::PoseStamped current_pose_msg;
 */

static ros::Publisher localizer_pose_pub;
static geometry_msgs::PoseStamped localizer_pose_msg;

static ros::Publisher estimate_twist_pub;
static geometry_msgs::TwistStamped estimate_twist_msg;

static ros::Duration scan_duration;

static double exe_time = 0.0;
static bool has_converged;
static int iteration = 0;
static double fitness_score = 0.0;
static double trans_probability = 0.0;

static double diff = 0.0;
static double diff_x = 0.0, diff_y = 0.0, diff_z = 0.0, diff_yaw;

static double current_velocity = 0.0, previous_velocity = 0.0, previous_previous_velocity = 0.0;  // [m/s]
static double current_velocity_x = 0.0, previous_velocity_x = 0.0;
static double current_velocity_y = 0.0, previous_velocity_y = 0.0;
static double current_velocity_z = 0.0, previous_velocity_z = 0.0;
// static double current_velocity_yaw = 0.0, previous_velocity_yaw = 0.0;
static double current_velocity_smooth = 0.0;

static double current_velocity_imu_x = 0.0;
static double current_velocity_imu_y = 0.0;
static double current_velocity_imu_z = 0.0;

static double current_accel = 0.0, previous_accel = 0.0;  // [m/s^2]
static double current_accel_x = 0.0;
static double current_accel_y = 0.0;
static double current_accel_z = 0.0;
// static double current_accel_yaw = 0.0;

static double angular_velocity = 0.0;

static int use_predict_pose = 0;

static ros::Publisher estimated_vel_mps_pub, estimated_vel_kmph_pub, estimated_vel_pub;
static std_msgs::Float32 estimated_vel_mps, estimated_vel_kmph, previous_estimated_vel_kmph;

static std::chrono::time_point<std::chrono::system_clock> matching_start, matching_end;

static ros::Publisher time_ndt_matching_pub;
static std_msgs::Float32 time_ndt_matching;

static int _queue_size = 1000;

static ros::Publisher ndt_stat_pub;
static autoware_msgs::NDTStat ndt_stat_msg;

static double predict_pose_error = 0.0;

static double _tf_x, _tf_y, _tf_z, _tf_roll, _tf_pitch, _tf_yaw;
static Eigen::Matrix4f tf_btol;

static std::string _localizer = "velodyne";
static std::string _offset = "linear";  // linear, zero, quadratic

static ros::Publisher ndt_reliability_pub;
static std_msgs::Float32 ndt_reliability;

static bool _get_height = false;
static bool _use_local_transform = false;
static bool _use_imu = false;
static bool _use_odom = false;
static bool _imu_upside_down = false;
static bool _output_log_data = false;

static std::string _imu_topic = "/imu_raw";

static std::ofstream ofs;
static std::string filename;

static sensor_msgs::Imu imu;
static nav_msgs::Odometry odom;

// static tf::TransformListener local_transform_listener;
static tf::StampedTransform local_transform;

static unsigned int points_map_num = 0;

pthread_mutex_t mutex;

std_msgs::Bool bl1, bl2;

ros::Publisher ack_pub;

void handleCallback(const autoware_msgs::gpu_handle msg)
{
  unsigned char handle[65];
  for (int i = 0; i < 64; i++)
  {
    handle[i] = msg.data[i];
  }

  anh_gpu_ndt_ptr->getHandle(handle);

  sleep(1);

  ack_pub.publish(bl2);
  ROS_INFO("publish ready");

  sub.shutdown();
}

void dataCallback(const std_msgs::Int32 msg)
{
  anh_gpu_ndt_ptr->debug(msg.data);

  std::cout << "finish" << std::end;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ndt_matching");
  pthread_mutex_init(&mutex, NULL);

  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");
  node_status_publisher_ptr_ = std::make_shared<autoware_health_checker::NodeStatusPublisher>(nh,private_nh);
  node_status_publisher_ptr_->ENABLE();
  node_status_publisher_ptr_->NODE_ACTIVATE();

  // Set log file name.
  private_nh.getParam("output_log_data", _output_log_data);
  if(_output_log_data)
  {
    char buffer[80];
    std::time_t now = std::time(NULL);
    std::tm* pnow = std::localtime(&now);
    std::strftime(buffer, 80, "%Y%m%d_%H%M%S", pnow);
    std::string directory_name = "/tmp/Autoware/log/ndt_matching";
    filename = directory_name + "/" + std::string(buffer) + ".csv";
    boost::filesystem::create_directories(boost::filesystem::path(directory_name));
    ofs.open(filename.c_str(), std::ios::app);
  }

  // Geting parameters
  int method_type_tmp = 0;
  private_nh.getParam("method_type", method_type_tmp);
  _method_type = static_cast<MethodType>(method_type_tmp);
  private_nh.getParam("use_gnss", _use_gnss);
  private_nh.getParam("queue_size", _queue_size);
  private_nh.getParam("offset", _offset);
  private_nh.getParam("get_height", _get_height);
  private_nh.getParam("use_local_transform", _use_local_transform);
  private_nh.getParam("use_imu", _use_imu);
  private_nh.getParam("use_odom", _use_odom);
  private_nh.getParam("imu_upside_down", _imu_upside_down);
  private_nh.getParam("imu_topic", _imu_topic);

  if (nh.getParam("localizer", _localizer) == false)
  {
    std::cout << "localizer is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_x", _tf_x) == false)
  {
    std::cout << "tf_x is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_y", _tf_y) == false)
  {
    std::cout << "tf_y is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_z", _tf_z) == false)
  {
    std::cout << "tf_z is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_roll", _tf_roll) == false)
  {
    std::cout << "tf_roll is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_pitch", _tf_pitch) == false)
  {
    std::cout << "tf_pitch is not set." << std::endl;
    return 1;
  }
  if (nh.getParam("tf_yaw", _tf_yaw) == false)
  {
    std::cout << "tf_yaw is not set." << std::endl;
    return 1;
  }

  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "Log file: " << filename << std::endl;
  std::cout << "method_type: " << static_cast<int>(_method_type) << std::endl;
  std::cout << "use_gnss: " << _use_gnss << std::endl;
  std::cout << "queue_size: " << _queue_size << std::endl;
  std::cout << "offset: " << _offset << std::endl;
  std::cout << "get_height: " << _get_height << std::endl;
  std::cout << "use_local_transform: " << _use_local_transform << std::endl;
  std::cout << "use_odom: " << _use_odom << std::endl;
  std::cout << "use_imu: " << _use_imu << std::endl;
  std::cout << "imu_upside_down: " << _imu_upside_down << std::endl;
  std::cout << "imu_topic: " << _imu_topic << std::endl;
  std::cout << "localizer: " << _localizer << std::endl;
  std::cout << "(tf_x,tf_y,tf_z,tf_roll,tf_pitch,tf_yaw): (" << _tf_x << ", " << _tf_y << ", " << _tf_z << ", "
            << _tf_roll << ", " << _tf_pitch << ", " << _tf_yaw << ")" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

#ifndef CUDA_FOUND
  if (_method_type == MethodType::PCL_ANH_GPU)
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_ANH_GPU is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif
#ifndef USE_PCL_OPENMP
  if (_method_type == MethodType::PCL_OPENMP)
  {
    std::cerr << "**************************************************************" << std::endl;
    std::cerr << "[ERROR]PCL_OPENMP is not built. Please use other method type." << std::endl;
    std::cerr << "**************************************************************" << std::endl;
    exit(1);
  }
#endif

  Eigen::Translation3f tl_btol(_tf_x, _tf_y, _tf_z);                 // tl: translation
  Eigen::AngleAxisf rot_x_btol(_tf_roll, Eigen::Vector3f::UnitX());  // rot: rotation
  Eigen::AngleAxisf rot_y_btol(_tf_pitch, Eigen::Vector3f::UnitY());
  Eigen::AngleAxisf rot_z_btol(_tf_yaw, Eigen::Vector3f::UnitZ());
  tf_btol = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();

  // Updated in initialpose_callback or gnss_callback
  initial_pose.x = 0.0;
  initial_pose.y = 0.0;
  initial_pose.z = 0.0;
  initial_pose.roll = 0.0;
  initial_pose.pitch = 0.0;
  initial_pose.yaw = 0.0;

  bl1.msg = true;
  bl2.msg = false;

  ack_pub = n.advertise<std_msgs::Bool>("voxel_node_ack", 1);

  sleep(1);

  ack_pub.publish(bl1);

  ros::Subscriber handle_sub = nh.subscribe("/voxel_handler", 10, handleCallback);
  ros::Subscriber data_sub = nh.subscribe("voxel_data_size", 10, dataCallback);

  ros::spin();

  return 0;
}
  