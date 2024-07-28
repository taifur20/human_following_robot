#!/usr/bin/env python3

import os
import re
import cv2
import glob
import rclpy
import pathlib
import argparse
import subprocess
from PIL import Image as Img

from rclpy.node import Node
#we will send messages in form of images consequently, we need to import Image
from sensor_msgs.msg import Image
#CvBridge object helps to convert OpenCV Images to ROS image messages
from cv_bridge import CvBridge

import numpy as np

import sys
sys.path.append('/usr/lib/python3.10/site-packages')
sys.path.append('/usr/local/share/pynq-venv/lib/python3.10/site-packages')
# the above path is needed by pynq_dpu
from pynq import allocate, Overlay

new_height = 420
new_width = 410 

class CameraPublisherNode(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer_ = self.create_timer(1.0 / 30, self.publish_frames)
        
        self.resize_design = Overlay("resizer.bit")
        self.dma = self.resize_design.axi_dma_0
        self.resizer = self.resize_design.resize_accel_0
        
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
            old_height, old_width, color = frame.shape
            self.get_logger().info("Image size: {}x{} pixels.".format(old_width, old_height))
            self.in_buffer = allocate(shape=(old_height, old_width, 3), 
                           dtype=np.uint8, cacheable=1)
            self.out_buffer = allocate(shape=(new_height, new_width, 3), 
                            dtype=np.uint8, cacheable=1)
            self.in_buffer[:] = np.array(frame)
            self.resizer.register_map.src_rows = old_height
            self.resizer.register_map.src_cols = old_width
            self.resizer.register_map.dst_rows = new_height
            self.resizer.register_map.dst_cols = new_width
            self.run_kernel()
            resized_image = Img.fromarray(self.out_buffer)
            self.get_logger().info("Image size: {}x{} pixels.".format(new_width, new_height))
            # Convert image from PIL to OpenCV
            cv2_image = np.array(resized_image)
            # Convert the frame to sensor_msgs.Image format
            frame_bgr = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)
            image_msg = self.cv_bridge.cv2_to_imgmsg(frame_bgr,encoding='rgb8')

            # Publish the image message
            self.publisher_.publish(image_msg)
            self.get_logger().info("Image publishing successfully...")
        else:
            self.get_logger().warn('Failed to read frame from camera.')
            
    def run_kernel(self):
        self.dma.sendchannel.transfer(self.in_buffer)
        self.dma.recvchannel.transfer(self.out_buffer)    
        self.resizer.write(0x00,0x81) # start
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
            
def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisherNode()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
