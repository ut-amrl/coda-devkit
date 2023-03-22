#!/usr/bin/env python
import os
import sys
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu

from queue import Queue
from threading import Thread

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bag_out', default="./coda_slam_saver.bag",
                    help="Point Cloud 2 ROS Topic")
parser.add_argument('--lidar_topic', default="/coda/ouster/lidar_packets",
                    help="Point Cloud 2 ROS Topic")
parser.add_argument('--imu_topic', default="/coda/vectornav/IMU",
                    help="Vectornav IMU ROS Topic")

# For imports
sys.path.append(os.getcwd())

recorder = None
lidar_queue = Queue(maxsize=10)
imu_queue = Queue(maxsize=10)

class BagRecorder(object):
    def __init__(self, bag_out, lidar_topic, imu_topic):
        self._bag_out = bag_out
        self._lidar_topic = lidar_topic
        self._imu_topic = imu_topic

        rospy.init_node('coda_slam_saver')
        self._bag = rosbag.Bag(self._bag_out, 'w')

        self._pc_sub = rospy.Subscriber(self._lidar_topic, PointCloud2, self.pc_callback, queue_size=10)
        self._imu_sub = rospy.Subscriber(self._imu_topic, Imu, self.imu_callback, queue_size=10)

    def pc_callback(self, msg):
        lidar_queue.put(msg)
        print("Length of ouster queue %i"% lidar_queue.qsize() )

    def imu_callback(self, msg):
        imu_queue.put(msg)
        print("Length of imu queue %i" % imu_queue.qsize() )

    def process_queues(self):
        #Fine earliest timestamp from queues
        while(1):
            earliest_lidar_topic = float('inf')
            tmp_lidar_q = lidar_queue.queue
            if lidar_queue.qsize() > 0:
                earliest_lidar_topic = tmp_lidar_q[0].header.stamp.to_sec()
            
            earliest_imu_topic = float('inf')
            tmp_imu_q = imu_queue.queue
            if imu_queue.qsize() > 0:
                earliest_imu_topic = tmp_imu_q[0].header.stamp.to_sec()

            if earliest_lidar_topic<float('inf') or earliest_imu_topic<float('inf'):
                if earliest_lidar_topic < earliest_imu_topic:
                    print("Write lidar ")
                    self._bag.write('/ouster/points', lidar_queue.get())
                else:
                    print("Wrote imu ")
                    self._bag.write('/vectornav/IMU', imu_queue.get())

    def close(self):
        self._pc_sub.unregister()
        self._imu_sub.unregister()
        self._bag.close()

if __name__ == '__main__':
    cli_args = parser.parse_args()

    recorder = BagRecorder(cli_args.bag_out, cli_args.lidar_topic, cli_args.imu_topic)

    worker = Thread(target=recorder.process_queues, args=())
    worker.setDaemon(True)
    worker.start()
    rospy.spin() # while(1)
    rospy.on_shutdown(recorder.close)