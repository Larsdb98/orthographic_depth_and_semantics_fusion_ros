#!/usr/bin/env python3
########################################################################

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv_bridge
# import cv2
# import datetime
# import os

class FuseDepthAndSemantics():
    def __init__(self):
        rospy.init_node("orthographic_depth_and_semantics_fusion", anonymous=False)
        self.bridge = cv_bridge.CvBridge()

        self.__depth_image_topic = rospy.get_param("~depth_image_topic", "/vrglasses_for_robots_ros/depth_map")
        self.__semantic_image_topic = rospy.get_param("~semantic_image_topic", "/vrglasses_for_robots_ros/semantic_map")
        self.__fused_depth_out_topic = rospy.get_param("~fused_depth_out", "/vrglasses_for_robots_ros/depth_fused")
        self.__fused_depth_confidence_out_topic = rospy.get_param("~fused_depth_confidence_out", "/vrglasses_for_robots_ros/depth_fused_confidence")
        self.__image_count = rospy.get_param("~max_image_count", 10)

        # Subcribers
        rospy.Subscriber(self.__depth_image_topic, Image, self.depth_callback)
        rospy.Subscriber(self.__semantic_image_topic, Image, self.semantic_callback)

        self.fused_depth_image = None
        self.fused_semantic_image = None

        self.depth_accumulator = None
        self.depth_confidence_accumulator = None
        self.depth_image_count = 0

        # Publishers
        self.fused_depth_pub = rospy.Publisher(self.__fused_depth_out_topic, Image, queue_size=10)
        self.fused_depth_confidence_pub = rospy.Publisher(self.__fused_depth_confidence_out_topic, Image, queue_size=10)


    def depth_callback(self, depth_msg):
        depth_msg_np = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        depth_image = np.array(depth_msg_np)

        h, w = depth_image.shape

        print("Depth height: {}".format(h))
        print("Depth width: {}".format(w))

        # Reset index counter when max value reached
        if self.depth_image_count >= self.__image_count:
            self.depth_image_count = 0 

        # Create array for first image:
        if self.depth_accumulator is None:
            self.depth_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)
            self.depth_accumulator[:] = np.nan # replace all values with np.nan values (will be useful for ignoring them when computing means)
            
            self.depth_confidence_accumulator = np.zeros((h, w, self.__image_count), dtype=np.float32)
            print("Depth accumulator shape: {}".format(self.depth_accumulator.shape))

        # Place depth values in accumulator:
        self.depth_accumulator[:, :, self.depth_image_count] = depth_image
        self.depth_image_count += 1


        # Compute mean of depth accumulator & place this in self.fused_depth_image
        # Then, if necessary, publish this fused image
        




        # self.depth_accumulator = np.zeros_like(depth_image, dtype=np.float32)
        

        # Mask pixels without depth value:
        valid_px = depth_image != 0.0
        depth_image[~valid_px] = np.nan

        # Accumulate depth values:
        



    def semantic_callback(self, semantic_msg):
        semantic_image = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding="mono8")


    




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