import pdb
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

# Path to the ROS bag file
# bag_file = "/robodata/husky_logs/calibrations/02222023/Feb222023.bag"
bag_file = "coda_slam_saver.bag"
# Topic name for the PointCloud2
topic_name = "/ouster/points"

# Open the ROS bag file
bag = rosbag.Bag(bag_file)

idx=0
# Iterate over each message in the bag
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    print("topic ", topic, " ts ", t)

    # Check if the message is of type PointCloud2
    if topic == topic_name:
        # Process the PointCloud2 message

        # Access point cloud data
        for point in point_cloud2.read_points(msg, field_names=("x", "y", "z", "t"), skip_nans=True):
            x, y, z, t = point
            # Do something with the x, y, z values
            print(f"X: {x}, Y: {y}, Z: {z}, T: {t}")


# Close the bag file
bag.close()