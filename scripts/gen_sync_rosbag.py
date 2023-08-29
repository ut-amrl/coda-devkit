import os
import os.path as osp
import rospy
import rosbag
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, CompressedImage, Imu, MagneticField, Image, NavSatFix
from message_filters import ApproximateTimeSynchronizer, Subscriber

from queue import Queue
from threading import Thread

recorder = None
sync_queues = []

# parser = argparse.ArgumentParser()

class BagRecorder(object):
    def __init__(self, outdir, topic_list, rostype_list):
        global sync_queues

        self._outdir = outdir
        self._topic_list = topic_list
        self._rostype_list = rostype_list

        self._bag = None

        rospy.init_node('topic_synchronizer', anonymous=True)

        # Create subscribers for the three topics
        self._sub_list = []
        for idx, topic_name in enumerate(self._topic_list):
            print("Added topic ", topic_name)
            self._sub_list.append(
                Subscriber(topic_name, self._rostype_list[idx])
            )
            sync_queues.append(Queue(maxsize=10))
            
        # Create the TimeSynchronizer with a list of the subscribers
        ts = ApproximateTimeSynchronizer(self._sub_list, queue_size=10, slop=0.1)

        # Register the callback function
        ts.registerCallback(self.callback)

        self._signal_sub = rospy.Subscriber("bagdecoder_signal", String, self.decoder_signal_callback)

        print("Callback registered")

    def decoder_signal_callback(self, signal):
        global sync_queues
        print("Received signal ", signal.data)
        if "START" in signal.data:
            # Retrieve next bag date and name from message
            _, self._bag_date, self._bag_file = signal.data.split(" ")

            bag_out_dir = osp.join(self._outdir, self._bag_date)
            if not osp.exists(bag_out_dir):
                os.makedirs(bag_out_dir)
            bag_out_path = osp.join(bag_out_dir, self._bag_file)

            # Close previous if it exists
            if self._bag is not None:
                self._bag.close()

            # Open next bag for writing
            self._bag = rosbag.Bag(bag_out_path, 'w')

            # Reset sync queues
            for idx, queue in enumerate(sync_queues):
                while sync_queues[idx].qsize() > 0:
                    sync_queues[idx].get()
                print("queue length ", sync_queues[idx].qsize())

    def callback(self, *args):
        # This callback function will be called when all three messages are synchronized
        # Process the synchronized data here
        # rospy.loginfo("Received synchronized messages:\n%s\n%s\n%s",
        #             pc.header, cam0.header, cam1.header)
        for idx, arg in enumerate(args):
            sync_queues[idx].put(arg)

    def process_queues(self):
        #Fine earliest timestamp from queues
        while(not rospy.is_shutdown()):
            are_queues_full = True
            for queue in sync_queues:
                if queue.qsize() <= 0:
                    are_queues_full = False
            
            if not are_queues_full:
                continue

            # rospy.loginfo("Writing to bag...")
            # Write to synchronized bag file
            for idx, topic in enumerate(self._topic_list):
                rospy.loginfo("Writing topic %s to bag..." % topic)
                self._bag.write(topic.replace("/coda", ""), sync_queues[idx].get())

    def close(self):
        global sync_queues
        for idx, sub in enumerate(self._sub_list):
            self._sub_list[idx].unregister()

        self._signal_sub.unregister()
        self._bag.close()

        sync_queues = []

def save_sync_bag(outdir): #, date, bag_name):
    global has_received_once
    global sync_queues
    global recorder

    # bag_out_dir = osp.join(outdir, date) 
    # if not osp.exists(bag_out_dir):
    #     os.makedirs(bag_out_dir)

    # bag_out_path = osp.join(bag_out_dir, bag_name)
    topic_list = [
        "/coda/ouster/lidar_packets",
        "/coda/stereo/left/image_raw/compressed",
        "/coda/stereo/right/image_raw/compressed"
    ]

    rostype_list = [
        PointCloud2,
        CompressedImage,
        CompressedImage
    ]

    recorder = BagRecorder(outdir, topic_list, rostype_list) #cli_args.bag_out, cli_args.lidar_topic, cli_args.imu_topic)

    worker = Thread(target=recorder.process_queues, args=())
    worker.setDaemon(True)
    worker.start()
    rospy.spin() # while(1)

    # while not rospy.is_shutdown():
    #     # if not has_received_once:
    #     #     continue

    #     try:
    #         _, publishers, _ = master.getSystemState()
    #         if not publishers:
    #             rospy.loginfo("No publishers connected. Exiting...")
    #             rospy.signal_shutdown("No publishers connected.")
    #             break
    #     except Exception as e:
    #         rospy.logerr("Error occurred while checking publishers: %s", str(e))
    rospy.on_shutdown(recorder.close)

if __name__ == '__main__':
    # outdir = "/robodata/arthurz/Datasets/CODa_sync_bags"
    # orig_bag_dir = "/robodata/husky_logs/CODa_bags"
    # orig_bag_subdirs = [entry.path for entry in os.scandir(orig_bag_dir) if entry.is_dir()]

    # save_sync_bag(outdir)
    # for day_subdir in orig_bag_subdirs:
    #     orig_bags = [entry.path for entry in os.scandir(day_subdir) if entry.path.endswith(".bag") and 
    #         "calibration" not in entry.path]

    #     date = day_subdir.split('/')[-1]
    #     for bag_name in orig_bags:
    #         print("Starting date %s bag %s" % (date, bag_name))
            
    #         save_sync_bag(outdir, date, bag_name)
    #         import pdb; pdb.set_trace()
            
# def callback(pc, cam0, cam1):
#         # This callback function will be called when all three messages are synchronized
#         # Process the synchronized data here
#         rospy.loginfo("Received synchronized messages:\n%s\n%s\n%s",
#                     pc.header, cam0.header, cam1.header)
#         lidar_queue.put(pc)
#         cam0_queue.put(cam0)
#         cam1_queue.put(cam1)

# def listener():
#     rospy.init_node('topic_synchronizer', anonymous=True)

#     # Create subscribers for the three topics
#     sub1 = Subscriber("/coda/ouster/lidar_packets", PointCloud2)
#     sub2 = Subscriber("/coda/stereo/left/image_raw/compressed", CompressedImage)
#     sub3 = Subscriber("/coda/stereo/right/image_raw/compressed", CompressedImage)

#     # Create the TimeSynchronizer with a list of the subscribers
#     ts = ApproximateTimeSynchronizer([sub1, sub2, sub3], queue_size=10, slop=0.1)

#     # Register the callback function
#     ts.registerCallback(callback)

#     rospy.spin()

# if __name__ == '__main__':
#     listener()
