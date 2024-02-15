#!/usr/bin/env python3
##################################################################

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv_bridge

import numpy as np
import cv2
import matplotlib.pyplot as plt

# IMPORTANT NOTE:
# This confidence processor is in charge of detecting blobs
# where the confidence value is lower in a certain region.
# The center coordinates of these blobs is computed and published
# as an odometry message.


class ConfidenceProcessorNode():
    def __init__(self):
        rospy.init_node("confidence_processor_node", anonymous=True)

        self.bridge = cv_bridge.CvBridge()

        # ROS params:
        self.__confidence_image_topic = rospy.get_param("~confidence_image_topic", "/vrglasses_for_robots_ros/semantic_fused_confidence")
        self.__publish_debug_image = rospy.get_param("~publish_debug_image", False)
        self.__publish_debug_image_topic = rospy.get_param("~publish_debug_image_topic", "/vrglasses_for_robots_ros/semantic_fused_confidence_processed")

        self.__publish_uncertain_location_odom_topic = rospy.get_param("~publish_uncertain_location_odom", "/vrglasses_for_robots_ros/low_confidence_region")

        # Params for blob detection in confidence maps
        self.gaussian_kernel_size       = (15, 15) # (15, 15)
        self.threshold_value            = 0.1 # 0.1
        self.neighborhood_size_value    = 2 # 2
        self.minimum_area               = 2000 # 2000

        # Publishers
        # Odometry message with blob location on x & y axis
        self.odom_pub = rospy.Publisher(self.__publish_uncertain_location_odom_topic, Odometry, queue_size=1)

        # Debug image publishing
        self.debug_publisher = None 
        if self.__publish_debug_image: 
            self.debug_publisher = rospy.Publisher(self.__publish_debug_image_topic, Image, queue_size=1)



        # Subscribers
        rospy.Subscriber(self.__confidence_image_topic, Image, self.callback)




    def callback(self, img_msg):
        image = self.bridge.imgmsg_to_cv2(img_msg=img_msg, desired_encoding="passthrough")
        image_np = np.array(image)

        # Perform blob detection
        detected_centers, detected_contours = self.detect_blobs(image_np, 
                                                                self.threshold_value, 
                                                                self.neighborhood_size_value, 
                                                                self.gaussian_kernel_size, 
                                                                self.minimum_area)

        # get maximum size blob
        if detected_centers:

            contour_areas = []
            for cnt in detected_contours:
                cnt_area = cv2.contourArea(cnt)
                contour_areas.append(cnt_area)

            max_area_id = np.argmax(contour_areas)

            # TODO: Process here coordinates of blob with max area
            # From pixel coordinates, compute x, y position with respect to
            # orthographic camera.
            # 
            # Publish computed position w.r.t. ortho. camera as odometry message. 
            


    def publish_debug_img(self, img_np, msg_header):
        img_msg = self.bridge.cv2_to_imgmsg(img_np, encoding="bgr8")
        img_msg.header = msg_header
        self.debug_publisher.publish(img_msg)





    def plot_image_with_blobs(self, image, centers, contours):
        try:
            plt.imshow(image, cmap='gray')

            # Plot contours
            for contour in contours:
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue', linewidth=1)

            # Plot blob centers
            for i, center in enumerate(centers):
                plt.scatter(center[0], center[1], color='red', marker='x')
                plt.text(center[0] - 10, center[1], str(i + 1), color='red', fontsize=18, ha='right', va='bottom')

            plt.title('Image with Blob Detections')
            plt.show()

        except Exception as e:
            rospy.logerr("An error occurred when trying to plot the detected blobs ! : {}".format(e))





    def detect_blobs(self, image, threshold, neighborhood_size, kernel_size= (5, 5), min_area=50):
        # Blur the image to reduce noise
        blurred = cv2.GaussianBlur(image, kernel_size, 0)

        # Calculate the difference between the blurred image and the original
        diff = np.abs(blurred - image)

        # Threshold the difference image to get binary mask
        _, binary_mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)

        # Dilate the binary mask to capture surrounding pixels
        kernel = np.ones((neighborhood_size, neighborhood_size), np.uint8)
        dilated_mask = cv2.dilate(binary_mask, kernel)

        # Find contours in the dilated mask
        contours, _ = cv2.findContours(dilated_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to remove small detections
        # min_area = 200
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Get the center coordinates of the filtered contours
        centers = [tuple(np.mean(cnt, axis=0, dtype=int)[0]) for cnt in filtered_contours]

        return centers, filtered_contours





    def run(self):
        rospy.spin()



#########################################

def main():
    try:
        node = ConfidenceProcessorNode()
        node.run()

    except rospy.ROSInterruptException:
        pass

#########################################
if __name__ == "__main__":
    main()
