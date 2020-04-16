# point_cloud_annotation_tool

This interface is built upon librviz, for more detail please check http://docs.ros.org/indigo/api/librviz_tutorial/html/index.html

Usage
====

0.build the package under ros workspace.

1.prepare your point cloud dataset.

2.click load point cloud directory button and open your dataset direction.
(default point cloud format is ply and PointXYZRGB, you can change it to pcd by editing the source code).
![image](https://github.com/himlen1990/toolbox/blob/master/annotation_tool/IMG/1.png)
![image](https://github.com/himlen1990/toolbox/blob/master/annotation_tool/IMG/2.png)

3.click add marker and start annotation (you can also change the marker type by clicking swith marker). 
![image](https://github.com/himlen1990/toolbox/blob/master/annotation_tool/IMG/3.png)

4.click save label and move to next frame after you finishing annotate the current frame.

5.you can jump to arbitrary frame by giving a frame number and click move to frame.

6.before click "load annotation for checking", you should asign the point cloud directory first (follow step2)