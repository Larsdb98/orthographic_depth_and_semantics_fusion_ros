#!/usr/bin/env python3
##################################################################

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
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

        self.__image_width = rospy.get_param("~image_width", 2.0) # image width in camera frame. Should be the same as what is set for the vulkan render engine.

        gaussian_kernel_size_value = rospy.get_param("~gaussian_kernel_size", 15)
        self.threshold_value = rospy.get_param("~threshold_value", 0.1)
        self.neighborhood_size_value = rospy.get_param("~neighborhood_size_value", 2)
        self.median_filter_kernel = rospy.get_param("~median_filter_kernel", 25)
        self.minimum_area = rospy.get_param("~minimum_area", 2000)

        print("BLOB DETECTION PARAMETERS")
        print("------------------------------")
        print("Gaussian Kernel Size: {}".format(gaussian_kernel_size_value))
        print("Threshold Value: {}".format(self.threshold_value))
        print("Neighborhood Value: {}".format(self.neighborhood_size_value))
        print("Median Filter Kernel: {}".format(self.median_filter_kernel))
        print("Minimum area: {}".format(self.minimum_area))

        self.gaussian_kernel_size = (gaussian_kernel_size_value, gaussian_kernel_size_value)


        # Params for blob detection in confidence maps
        # self.gaussian_kernel_size       = (15, 15) # (15, 15)
        # self.threshold_value            = 0.1 # 0.1
        # self.neighborhood_size_value    = 2 # 2
        # self.minimum_area               = 2000 # 2000

        # Publishers
        # Odometry message with blob location on x & y axis
        self.pose_pub = rospy.Publisher(self.__publish_uncertain_location_odom_topic, PoseStamped, queue_size=1)

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
                                                                self.minimum_area,
                                                                self.median_filter_kernel)

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
            rospy.loginfo("Low confidence blob detected at {}".format(detected_centers[max_area_id]))
            rospy.loginfo("Low confidence region has a pixel area of {}".format(contour_areas[max_area_id]))

            coord_x, coord_y = self.pixel_to_world_coords(detected_centers[max_area_id], image_np.shape)
            self.publish_pose_msg(coord_x, coord_y, img_msg.header)

            if self.__publish_debug_image:
                debug_image = self.plot_image_with_blobs(image_np, detected_centers, detected_contours)
                
                if debug_image is None: # 
                    rospy.logwarn("Debug image has not been generated correctly ! Ignoring task...")
                    return
                else:
                    self.publish_debug_img(img_np = debug_image, msg_header= img_msg.header)
            


    def publish_debug_img(self, img_np, msg_header):
        img_msg = self.bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
        img_msg.header = msg_header
        self.debug_publisher.publish(img_msg)


    def publish_pose_msg(self, x, y, header):
        msg = PoseStamped()
        msg.header = header
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        # rotation: set to identity quaternion (no rotation)
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        # publish message
        self.pose_pub.publish(msg)



    def pixel_to_world_coords(self, pixel_coords, image_shape):
        # Compute left, right, top, bottom coords in camera frame:
        # it is assumed that pixel_coords and image_shape are of the
        # following structure: [Height, Width]
        # Flip in case this assumption isn't valid:
        pixel_coords = [pixel_coords[1], pixel_coords[0]]
        height = float(self.__image_width * image_shape[0]) / image_shape[1] # w * (H/W)

        left = float(-1.0 * self.__image_width) / 2.0
        # right = float(self.__image_width) / 2.0
        bottom = float(-1.0 * height) / 2.0
        # top = float(height) / 2.0

        

        coord_x = float(pixel_coords[1]) / image_shape[1] * self.__image_width + left
        coord_y = float(pixel_coords[0]) / image_shape[0] * height + bottom

        return coord_x, coord_y


    def plot_image_with_blobs(self, image, centers, contours):
        try:
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray')

            # Plot contours
            for contour in contours:
                ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='blue', linewidth=1)

            # Plot blob centers
            for i, center in enumerate(centers):
                ax.scatter(center[0], center[1], color='red', marker='x')
                ax.text(center[0] - 10, center[1], str(i + 1), color='red', fontsize=18, ha='right', va='bottom')

            ax.set_title('Image with Blob Detections')

            # Convert the plot to a numpy array
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)  # Close the figure to avoid displaying it

            return image_array

        except Exception as e:
            rospy.logerr("An error occurred when trying to plot the detected blobs ! : {}".format(e))
            return None





    def detect_blobs(self, image, threshold, neighborhood_size, kernel_size= (5, 5), min_area=50, median_kernel=25):
        # Blur the image to reduce noise>
        blurred = cv2.GaussianBlur(image, kernel_size, 0)

        # Calculate the difference between the blurred image and the original
        # diff = blurred - image
        diff = np.ones_like(blurred) - blurred

        # Convert to opencv 8bit grayscale image:
        diff = np.array(diff * 255, dtype=np.uint8)

        diff = cv2.medianBlur(diff, ksize=median_kernel)

        # Threshold the difference image to get binary mask
        _, binary_mask = cv2.threshold(diff, threshold * 255, 255, cv2.THRESH_BINARY)

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
