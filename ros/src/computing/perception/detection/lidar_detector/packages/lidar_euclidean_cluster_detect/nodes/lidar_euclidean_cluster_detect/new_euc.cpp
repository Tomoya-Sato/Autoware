#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>

#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/don.h>
#include <pcl/features/fpfh_omp.h>

#include <pcl/kdtree/kdtree.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <pcl/common/common.h>

#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>

#include <pcl/segmentation/extract_clusters.h>

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>

#include "autoware_msgs/Centroids.h"
#include "autoware_msgs/CloudCluster.h"
#include "autoware_msgs/CloudClusterArray.h"
#include "autoware_msgs/DetectedObject.h"
#include "autoware_msgs/DetectedObjectArray.h"

#include <vector_map/vector_map.h>

#include <tf/tf.h>

#include <yaml-cpp/yaml.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/version.hpp>

#if (CV_MAJOR_VERSION == 3)

#include "gencolors.cpp"

#else

#include <opencv2/contrib/contrib.hpp>
#include <autoware_msgs/DetectedObjectArray.h>

#endif

#include "cluster.h"

#ifdef GPU_CLUSTERING

#include "gpu_euclidean_clustering.h"

#endif

#define __APP_NAME__ "euclidean_clustering"

using namespace cv;

ros::Publisher _pub_cluster_cloud;
ros::Publisher _pub_ground_cloud;
ros::Publisher _centroid_pub;

ros::Publisher _pub_clusters_message;

ros::Publisher _pub_points_lanes_cloud;

ros::Publisher _pub_detected_objects;

std_msgs::Header _velodyne_header;

std::string _output_frame;

static bool _velodyne_transform_available;
static bool _downsample_cloud;
static bool _pose_estimation;
static double _leaf_size;
static int _cluster_size_min;
static int _cluster_size_max;

static bool _remove_ground;  // only ground

static bool _using_sensor_cloud;
static bool _use_diffnormals;

static double _clip_min_height;
static double _clip_max_height;

static bool _keep_lanes;
static double _keep_lane_left_distance;
static double _keep_lane_right_distance;

static double _max_boundingbox_side;
static double _remove_points_upto;
static double _cluster_merge_threshold;
static double _clustering_distance;

static bool _use_gpu;
static std::chrono::system_clock::time_point _start, _end;

std::vector<std::vector<geometry_msgs::Point>> _way_area_points;
std::vector<cv::Scalar> _colors;
pcl::PointCloud<pcl::PointXYZ> _sensor_cloud;
visualization_msgs::Marker _visualization_marker;

static bool _use_multiple_thres;
std::vector<double> _clustering_distances;
std::vector<double> _clustering_ranges;

tf::StampedTransform *_transform;
tf::StampedTransform *_velodyne_output_transform;
tf::TransformListener *_transform_listener;
tf::TransformListener *_vectormap_transform_listener;

#include <std_msgs/Bool.h>

std_msgs::Bool bl1, bl2;

ros::Publisher ack_pub;

GpuEuclideanCluster gecl_cluster;

void handleCallback(const autoware_msgs::gpu_handle msg)
{
    unsigned char handle[65];
    for (int i = 0; i < 64; i++)
    {
        handle[i] = msg.data[i];
    }

    gecl_cluster.getHandle(handle);

    sleep(1);

    ack_pub.publish(bl2);
    ROS_INFO("publish ready");

    sub.shutdown();
}

void dataCallback(const std_msgs::Int32 msg)
{
    gecl_cluster.debug(msg.data);

    std::cout << "finish" << std::endl;
}

int main(int argc, char* argv[])
{
    // Initialize ROS
    ros::init(argc, argv, "euclidean_cluster");

    ros::NodeHandle h;
    ros::NodeHandle private_nh("~");

    tf::StampedTransform transform;
    tf::TransformListener listener;
    tf::TransformListener vectormap_tf_listener;

    _vectormap_transform_listener = &vectormap_tf_listener;
    _transform = &transform;
    _transform_listener;

#if (CV_MAJOR_VERSION == 3)
    generateColors(_colors, 255);
#else
    cv::generateColors(_colors, 255);
#endif

    _pub_cluster_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_cluster", 1);
    _centroid_pub = h.advertise<autoware_msgs::Centroids>("/cluster_centroids", 1);

    std::string points_topic, gridmap_topic;

    _using_sensor_cloud = false;

    /* Initialize tuning parameter */
    private_nh.param("downsample_cloud", _downsample_cloud, false);
    ROS_INFO("[%s] downsample_cloud: %d", __APP_NAME__, _downsample_cloud);
    private_nh.param("remove_ground", _remove_ground, true);
    ROS_INFO("[%s] remove_ground: %d", __APP_NAME__, _remove_ground);
    private_nh.param("leaf_size", _leaf_size, 0.1);
    ROS_INFO("[%s] leaf_size: %f", __APP_NAME__, _leaf_size);
    private_nh.param("cluster_size_min", _cluster_size_min, 20);
    ROS_INFO("[%s] cluster_size_min %d", __APP_NAME__, _cluster_size_min);
    private_nh.param("cluster_size_max", _cluster_size_max, 100000);
    ROS_INFO("[%s] cluster_size_max: %d", __APP_NAME__, _cluster_size_max);
    private_nh.param("pose_estimation", _pose_estimation, false);
    ROS_INFO("[%s] pose_estimation: %d", __APP_NAME__, _pose_estimation);
    private_nh.param("clip_min_height", _clip_min_height, -1.3);
    ROS_INFO("[%s] clip_min_height: %f", __APP_NAME__, _clip_min_height);
    private_nh.param("clip_max_height", _clip_max_height, 0.5);
    ROS_INFO("[%s] clip_max_height: %f", __APP_NAME__, _clip_max_height);
    private_nh.param("keep_lanes", _keep_lanes, false);
    ROS_INFO("[%s] keep_lanes: %d", __APP_NAME__, _keep_lanes);
    private_nh.param("keep_lane_left_distance", _keep_lane_left_distance, 5.0);
    ROS_INFO("[%s] keep_lane_left_distance: %f", __APP_NAME__, _keep_lane_left_distance);
    private_nh.param("keep_lane_right_distance", _keep_lane_right_distance, 5.0);
    ROS_INFO("[%s] keep_lane_right_distance: %f", __APP_NAME__, _keep_lane_right_distance);
    private_nh.param("max_boundingbox_side", _max_boundingbox_side, 10.0);
    ROS_INFO("[%s] max_boundingbox_side: %f", __APP_NAME__, _max_boundingbox_side);
    private_nh.param("cluster_merge_threshold", _cluster_merge_threshold, 1.5);
    ROS_INFO("[%s] cluster_merge_threshold: %f", __APP_NAME__, _cluster_merge_threshold);
    private_nh.param<std::string>("output_frame", _output_frame, "velodyne");
    ROS_INFO("[%s] output_frame: %s", __APP_NAME__, _output_frame.c_str());

    private_nh.param("remove_points_upto", _remove_points_upto, 0.0);
    ROS_INFO("[%s] remove_points_upto: %f", __APP_NAME__, _remove_points_upto);

    private_nh.param("clustering_distance", _clustering_distance, 0.75);
    ROS_INFO("[%s] clustering_distance: %f", __APP_NAME__, _clustering_distance);

    private_nh.param("use_gpu", _use_gpu, false);
    ROS_INFO("[%s] use_gpu: %d", __APP_NAME__, _use_gpu);

    private_nh.param("use_multiple_thres", _use_multiple_thres, false);
    ROS_INFO("[%s] use_multiple_thres: %d", __APP_NAME__, _use_multiple_thres);

    std::string str_distances;
    std::string str_ranges;
    private_nh.param("clustering_distances", str_distances, std::string("[0.5,1.1,1.6,2.1,2.6]"));
    ROS_INFO("[%s] clustering_distances: %s", __APP_NAME__, str_distances.c_str());
    private_nh.param("clustering_ranges", str_ranges, std::string("[15,30,45,60]"));
    ROS_INFO("[%s] clustering_ranges: %s", __APP_NAME__, str_ranges.c_str());

    _velodyne_transform_available = false;

    bl1.msg = true;
    bl2.msg = false;

    ack_pub = n.advertise<std_msgs::Bool>("node_ack", 1);

    sleep(1);

    ack_pub.publish(bl1);

    ros::Subscriber sub = n.subscribe("/gpu_handler", 10, handleCallback);
    ros::Subscriber data_sub = n.subscribe("/data_size", 10, dataCallback);

    ros::spin();

    return 0;
}