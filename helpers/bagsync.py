import rospy

import message_filters
from sensor_msgs.msg import Image, CameraInfo

def callback(lidar, cam0, cam1):


lidar_sub   = message_filters.Subscriber('os_sensor', Image)
cam0_sub    = message_filters.Subscriber('/stereo/left/image_raw/compressed', Image)
cam1_sub    = message_filters.Subscriber('/stereo/right/image_raw/compressed', Image)

ts = message_filters.TimeSynchronizer([lidar_sub, cam0_sub, cam1_sub], 10)
ts.registerCallback(callback)
rospy.spin()