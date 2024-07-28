#!/usr/bin/env python3

import os
import cv2
import rclpy
from rclpy.node import Node
#we will send messages in form of images consequently, we need to import Image
from sensor_msgs.msg import Image
#CvBridge object helps to convert OpenCV Images to ROS image messages
from cv_bridge import CvBridge

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer_ = self.create_timer(1.0 / 30, self.publish_frames)
        self.cv_bridge = CvBridge()
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 30 FPS
        self.cv_bridge = CvBridge()
        self.get_logger().info("Image publisher node.")

    
    def publish_frames(self):
        ret, frame = self.video_capture.read()
        self.get_logger().info("Image capturing...")
        if ret:
            # Convert the frame to sensor_msgs.Image format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame_bgr,encoding='rgb8')

            # Publish the image message
            self.publisher_.publish(image_msg)
            self.get_logger().info("Image publishing successfully...")
        else:
            self.get_logger().warn('Failed to read frame from camera.')
            
def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisherNode()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
