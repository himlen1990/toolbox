#ifndef ANNOTATION_TOOL_H
#define ANNOTATION_TOOL_H

#include <QWidget>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QFileDialog>
#include <QLineEdit>

#include <boost/filesystem.hpp>
#include <interactive_markers/interactive_marker_server.h>
#include <tf/tf.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <ros/ros.h>
#include <rviz/tool_manager.h>
#include "rviz/visualization_manager.h"
#include "rviz/display.h"
#include "rviz/render_panel.h"
#include <sensor_msgs/PointCloud2.h>
#include <ros/package.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/ply_io.h>
#include <fstream>

#define HSR 0

#define COLOR true; //true for colored point cloud
#ifdef COLOR
typedef pcl::PointXYZRGB PCType;
#else
typedef pcl::PointXYZ PCType;
#endif


namespace rviz
{
  class Display;
  class RenderPanel;
  class VisualizationManager;
}

class AnnotationTool: public QWidget
{
Q_OBJECT
 public:
 AnnotationTool( QWidget* parent = 0 );
 virtual ~AnnotationTool();


 public:
   visualization_msgs::Marker makeBox( visualization_msgs::InteractiveMarker &msg );
   visualization_msgs::InteractiveMarkerControl& makeBoxControl( visualization_msgs::InteractiveMarker &msg );
   void make6DofMarker( std::string name, unsigned int interaction_mode, bool show_6dof );
   void markerFeedback( const visualization_msgs::InteractiveMarkerFeedbackConstPtr &feedback);
   void PublishPointCloud(pcl::PointCloud<PCType>::Ptr cloud);   
   void loadPointCloud();

 private Q_SLOTS:
   void loadPointCloudDir();
   void addMarker();
   void removeMarker();
   void saveAnnotation();
   void switch_marker();
   void moveToFrame();
   void loadAnnotation();

 private:
   rviz::VisualizationManager* manager_;
   rviz::RenderPanel* render_panel_;

   QLineEdit* move_to_frame;   

   //parameters
   pcl::PointCloud<PCType>::Ptr current_cloud;
   int device;
   int num_marker;
   float pre_marker_x;
   float pre_marker_y;
   float pre_marker_z;
   int num_annotated_cloud;
   float marker_scale;


   std::string marker_mesh_resource;
   std::string current_marker_type;
   std::string base_dir;
   std::vector<std::string> pointcloud_files;

   ros::NodeHandle nh_;
   boost::shared_ptr<interactive_markers::InteractiveMarkerServer> server;
   std::vector<std::vector<float> > label;
   ros::Publisher marker_pub;
   ros::Publisher pointcloud_dataset_pub;
   ros::Subscriber pointcloud_dataset_sub;
};

#endif // ANNOTATION_TOOL_H
