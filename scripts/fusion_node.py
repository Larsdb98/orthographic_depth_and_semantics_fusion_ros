#!/usr/bin/env python3
########################################################################

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv_bridge
# import cv2
# import datetime
# import os

# In order to ignore all python warnings.
# This will not suppress ROS warnings
import warnings
warnings.filterwarnings("ignore")

class FuseDepthAndSemantics():
    def __init__(self):
        rospy.init_node("orthographic_depth_and_semantics_fusion", anonymous=False)
        self.bridge = cv_bridge.CvBridge()

        self.__depth_image_topic = rospy.get_param("~depth_image_topic", "/vrglasses_for_robots_ros/depth_map")
        self.__semantic_image_topic = rospy.get_param("~semantic_image_topic", "/vrglasses_for_robots_ros/semantic_map")

        self.__fused_depth_out_topic = rospy.get_param("~fused_depth_out", "/vrglasses_for_robots_ros/depth_fused")
        self.__fused_depth_confidence_out_topic = rospy.get_param("~fused_depth_confidence_out", "/vrglasses_for_robots_ros/depth_fused_confidence")

        self.__fused_semantic_out_topic = rospy.get_param("~fused_semantic_out", "/vrglasses_for_robots_ros/semantic_fused")
        self.__fused_semantic_confidence_out_topic = rospy.get_param("~fused_semantic_confidence_out", "/vrglasses_for_robots_ros/semantic_fused_confidence")

        self.__image_count = rospy.get_param("~max_image_count", 10)
        # Publish ?
        self.publish_fused_images = rospy.get_param("~publish_fused_images", True)


        # Depth image attributes
        self.fused_depth_image = None
        self.depth_accumulator = None
        self.depth_confidence_accumulator = None
        self.depth_image_count = 0

        # Semantic image attributes
        self.fused_semantic_image = None
        self.semantic_accumulator = None
        self.semantic_confidence_accumulator = None
        self.semantic_image_count = 0

        # Publishers
        self.fused_depth_pub = rospy.Publisher(self.__fused_depth_out_topic, Image, queue_size=10)
        self.fused_depth_confidence_pub = rospy.Publisher(self.__fused_depth_confidence_out_topic, Image, queue_size=10)

        self.fused_semantic_pub = rospy.Publisher(self.__fused_semantic_out_topic, Image, queue_size=10)
        self.fused_semantic_confidence_pub = rospy.Publisher(self.__fused_semantic_confidence_out_topic, Image, queue_size=10)


        # Subcribers
        rospy.Subscriber(self.__depth_image_topic, Image, self.depth_callback)
        rospy.Subscriber(self.__semantic_image_topic, Image, self.semantic_callback)

        print("DEBUG: max_image_count: {}".format(self.__image_count))
        print("DEBUG: publish_fused_images: {}".format(self.publish_fused_images))


    def depth_callback(self, depth_msg):
        depth_msg_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_image = np.array(depth_msg_np)

        h, w = depth_image.shape

        valid_px = depth_image != 0.0
        depth_image[~valid_px] = np.nan

        # Reset index counter when max value reached
        if self.depth_image_count >= self.__image_count:
            self.depth_image_count = 0 

        # print("DEBUG: Depth Image Counter: {}".format(self.depth_image_count))

        # Create array for first image:
        if self.depth_accumulator is None:
            rospy.loginfo("First rendered depth image received ! Fusion process started. Waiting for new messages...")
            self.depth_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)

            self.depth_accumulator[:] = np.nan # replace all values with np.nan values (will be useful for ignoring them when computing means)
            
            self.depth_confidence_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)
            self.depth_confidence_accumulator[:] = np.nan
            # print("Depth accumulator shape: {}".format(self.depth_accumulator.shape))

        # Place depth values in accumulator:
        self.depth_accumulator[:, :, self.depth_image_count] = depth_image
        self.depth_image_count += 1 # Only increase after placing new image in accumulator

        # Compute mean of depth accumulator & place this in self.fused_depth_image
        # Then, if necessary, publish this fused image
        self.fused_depth_image = np.nanmean(self.depth_accumulator, axis=2) # compute pixelwise mean for all images stored in accumulator
        

        # Publish image
        if self.publish_fused_images:
            fused_depth_image_msg = self.bridge.cv2_to_imgmsg(self.fused_depth_image, encoding="passthrough")
            fused_depth_image_msg.header = depth_msg.header
            self.fused_depth_pub.publish(fused_depth_image_msg)



    def semantic_callback(self, semantic_msg):
        semantic_msg_np = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding="mono8")
        semantic_image = np.array(semantic_msg_np)

        h, w = semantic_image.shape
        # No need to filter out 0.0 since this is a legit value to designate background

        # Reset index counter when max value reached
        if self.semantic_image_count >= self.__image_count:
            self.semantic_image_count = 0

        # print("DEBUG: Semantic Image Counter: {}".format(self.semantic_image_count))

        # Create array for first image:
        if self.semantic_accumulator is None:
            rospy.loginfo("First rendered semantic image received ! Fusion process started. Waiting for new messages...")
            self.semantic_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)

            # As the accumulator fills up, it's important to compute the mean without the missing
            # images since we haven't given enough time to reiceve all of them.
            # So we still initialize self.semantic_accumulator with np.nan values.
            self.semantic_accumulator[:] = np.nan

            self.semantic_confidence_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)
            self.semantic_confidence_accumulator[:] = np.nan

        # Place semantic values from recieved message into accumulator:
        self.semantic_accumulator[:, :, self.semantic_image_count] = semantic_image
        self.semantic_image_count += 1 # increment semantic image count

        # Compute pixel-wise mean of semantic accumulator:
        fused_semantic_image_mean = np.nanmean(self.semantic_accumulator, axis=2)
        self.fused_semantic_image = np.round(fused_semantic_image_mean).astype("uint8")



        # Publish image
        if self.publish_fused_images:
            fused_semantic_image_msg = self.bridge.cv2_to_imgmsg(self.fused_semantic_image, encoding="mono8")
            fused_semantic_image_msg.header = semantic_msg.header
            self.fused_semantic_pub.publish(fused_semantic_image_msg)
            


    def run(self):
        rospy.spin()




def main():
    try:
        node = FuseDepthAndSemantics()
        node.run()
    except rospy.ROSInterruptException:
        pass


#########################################################
if __name__ == "__main__":
    main()
