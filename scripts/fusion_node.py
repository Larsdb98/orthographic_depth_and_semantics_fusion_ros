#!/usr/bin/env python3
########################################################################

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv_bridge
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


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
        self.__fused_depth_confidence_amplifier = rospy.get_param("~fused_depth_confidence_amplifier", 500.0)

        self.__fused_semantic_out_topic = rospy.get_param("~fused_semantic_out", "/vrglasses_for_robots_ros/semantic_fused")
        self.__fused_semantic_confidence_out_topic = rospy.get_param("~fused_semantic_confidence_out", "/vrglasses_for_robots_ros/semantic_fused_confidence")
        self.__fused_semantic_threshold = rospy.get_param("~fused_semantic_threshold", 0.8)
        self.__fused_semantic_confidence_amplifier = rospy.get_param("~fused_semantic_confidence_amplifier", 500.0)

        self.__show_confidence_score_graph = rospy.get_param("~show_confidence_score_graph", False)
        self.__score_graph_data_to_show = rospy.get_param("~score_graph_data_to_show", "semantics") # "semantics" or "depth"

        self.__save_maps_dir = rospy.get_param("~save_maps_dir", "/root/moma_ws/src/mapping_pipeline_packages/orthographic_depth_and_semantics_fusion_ros/saved_maps")
        self.__save_numpy_maps = rospy.get_param("~save_numpy_maps", False)

        self.__image_count = rospy.get_param("~max_image_count", 10)
        # Publish ?
        self.publish_fused_images = rospy.get_param("~publish_fused_images", True)

        # make directory for saving maps:
        if not os.path.isdir(self.__save_maps_dir):
            os.mkdir(self.__save_maps_dir)

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


        #################################### VISUALIZER OF CONFIDENCE SCORE ################################
        if self.__show_confidence_score_graph:
            if self.__score_graph_data_to_show == "semantics" or self.__score_graph_data_to_show == "depth":
                self.fig, self.ax = plt.subplots()
                self.ln, = self.ax.plot([], [], '-b', label="Semantic Confidence Score")
                self.ax.set_xlabel('Frame')
                self.ax.set_ylabel('Confidence Score')
                self.ax.legend()
                self.x_data, self.y_data = [] , []
                # rospy.Timer(rospy.Duration(1), self.update_plot)
                self.animation = FuncAnimation(self.fig, self.update_plot, interval=1000)
            else:
                raise Exception("Type of data to show not recognized ! Please review ~score_graph_data_to_show ROS Parameter.")




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
        fused_depth_image_mean = np.nanmean(self.depth_accumulator, axis=2) # compute pixelwise mean for all images stored in accumulator
        self.fused_depth_image = fused_depth_image_mean
        fused_depth_confidence = self.get_depth_confidence(depth_accumulator=self.depth_accumulator, depth_mean=fused_depth_image_mean)


        # Publish image
        if self.publish_fused_images:
            # Publish fused depth
            fused_depth_image_msg = self.bridge.cv2_to_imgmsg(self.fused_depth_image, encoding="passthrough")
            fused_depth_image_msg.header = depth_msg.header
            self.fused_depth_pub.publish(fused_depth_image_msg)

            if self.__save_numpy_maps:
                depth_img_filepath = os.path.join(self.__save_maps_dir, "depth_fused.npy")
                np.save(depth_img_filepath, self.fused_depth_image)

            # Publish fused depth confidence map
            fused_depth_confidence_msg = self.bridge.cv2_to_imgmsg(fused_depth_confidence, encoding="passthrough")
            fused_depth_confidence_msg.header = depth_msg.header
            self.fused_depth_confidence_pub.publish(fused_depth_confidence_msg)

            if self.__save_numpy_maps:
                depth_confidence_filepath = os.path.join(self.__save_maps_dir, "depth_confidence.npy")
                np.save(depth_confidence_filepath, fused_depth_confidence)




    def semantic_callback(self, semantic_msg):
        semantic_msg_np = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding="mono8")
        semantic_image = np.array(semantic_msg_np)

        h, w = semantic_image.shape
        # No need to filter out 0.0 since this is a legit value to designate background

        semantic_image_normalized = semantic_image / 255.0

        # print("Recieved rendered semantic image max value: {}".format(np.max(semantic_image_normalized)))
        # print("Recieved rendered semantic image min value: {}".format(np.min(semantic_image_normalized)))

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
        self.semantic_accumulator[:, :, self.semantic_image_count] = semantic_image_normalized
        self.semantic_image_count += 1 # increment semantic image count

        # Compute pixel-wise mean of semantic accumulator:
        fused_semantic_image_mean = np.nanmean(self.semantic_accumulator, axis=2)
        
        # Threshold semantic image with chosen threshold in ROS params
        self.fused_semantic_image = np.where(fused_semantic_image_mean >= self.__fused_semantic_threshold, 1, 0).astype("uint8")
        
        self.fused_semantic_image = self.fused_semantic_image * 255

        # Compute semantic confidence based on recieved data
        fused_semantic_confidence = self.get_semantic_confidence(semantic_accumulator=self.semantic_accumulator)



        # Publish image
        if self.publish_fused_images:
            # Publish fused semantics
            fused_semantic_image_msg = self.bridge.cv2_to_imgmsg(self.fused_semantic_image, encoding="mono8")
            fused_semantic_image_msg.header = semantic_msg.header
            self.fused_semantic_pub.publish(fused_semantic_image_msg)

            if self.__save_numpy_maps:
                semantic_img_filepath = os.path.join(self.__save_maps_dir, "semantic_fused.npy")
                np.save(semantic_img_filepath, self.fused_semantic_image)

            # Publish fused semantic confidence map
            fused_semantic_confidence_msg = self.bridge.cv2_to_imgmsg(fused_semantic_confidence, encoding = "passthrough")
            fused_semantic_confidence_msg.header = semantic_msg.header
            self.fused_semantic_confidence_pub.publish(fused_semantic_confidence_msg)

            if self.__save_numpy_maps:
                semantic_confidence_filepath = os.path.join(self.__save_maps_dir, "semantic_confidence.npy")
                np.save(semantic_confidence_filepath, fused_semantic_confidence)




    def get_semantic_confidence(self, semantic_accumulator):
        # Need to scale down all semantic values to fit between 0 and 10 instead of 0 and 255
        # ortherwise we get extreme values for variance
        semantic_accumulator_normalized = semantic_accumulator / 255.0
        semantic_accumulator_normalized = semantic_accumulator_normalized * self.__fused_semantic_confidence_amplifier
        
        # Compute pixelwise variance
        fused_semantic_var = np.nanstd(semantic_accumulator_normalized, axis=2)
        # fused_semantic_var = np.nanvar(semantic_accumulator_normalized, axis=2)

        # print("DEBUG: Maximum of Semantic Variance: {}".format(np.nanmax(fused_semantic_var)))
        # print("DEBUG: Minimum of Semantic Variance: {}".format(np.nanmin(fused_semantic_var)))

        # We want to convert variance to confidence. To do so, we use the following formula:
        # 1 / (1 + var) for each pixel. 

        normalized_var = 1 / (1 + fused_semantic_var)
        # print("Min max variance for semantic accumulation: {}, {}".format(np.nanmin(fused_semantic_var), np.nanmax(fused_semantic_var)))
        # print("Normalized variance of semantic accumulation: {}".format(normalized_var))
        # print("Minimum variance computed: {}".format(np.nanmin(normalized_var)))

        if self.__show_confidence_score_graph and self.__score_graph_data_to_show == "semantics":
            sum_confidence = np.nansum(normalized_var)
            self.update_data(sum_confidence)



        return normalized_var


    def get_depth_confidence(self, depth_accumulator, depth_mean):
        # To get a better variance results, we scale up the depth  accumulator by a factor of 10
        depth_accumulator_scaled = depth_accumulator * self.__fused_depth_confidence_amplifier
        # Compute pixelwise variance
        fused_depth_var = np.nanstd(depth_accumulator_scaled, axis=2)
        # fused_depth_var = np.nanvar(depth_accumulator_scaled, axis=2)

        # print("DEBUG: Maximum of Depth Variance: {}".format(np.nanmax(fused_depth_var)))
        # print("DEBUG: Minimum of Depth Variance: {}".format(np.nanmin(fused_depth_var)))

        # Want to convert variance to confidence. To do so, we use the following formula:
        # 1 / (1 + var) for each pixel value (depth)
        normalized_var = 1 / (1 + fused_depth_var)

        # Invalid depth values need to have minimal confidence values
        normalized_var[depth_mean == 0.0] = 0.0

        if self.__show_confidence_score_graph and self.__score_graph_data_to_show == "depth":
            sum_confidence = np.nansum(normalized_var)
            self.update_data(sum_confidence)

        return normalized_var


    def run(self):
        if self.__show_confidence_score_graph:
            plt.show(block=True) # Run matplotlib loop
        else:
            rospy.spin() # run ROS loop

    ##################################################

    def update_data(self, uncertaintyScore):
        self.y_data.append(uncertaintyScore)
        x_index = len(self.y_data)
        self.x_data.append(x_index + 1)



    def update_plot(self, frame):
        self.ln.set_data(self.x_data, self.y_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        plt.pause(0.001)
        return self.ln
    




def main():
    try:
        node = FuseDepthAndSemantics()

        # Animated score graph ?
        # https://stackoverflow.com/questions/31174497/dynamic-updating-a-matplotlib-3d-plot-when-working-with-ros-callback
        # if node.show_confidence_score_graph:
        #     anim = FuncAnimation(node.fig, node.update_plot, init_func=node.plot_init)
        #     plt.show(block=True)

        node.run()
    except rospy.ROSInterruptException:
        pass


#########################################################
if __name__ == "__main__":
    main()
