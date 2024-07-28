#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial


class PublishResultSerial(Node):
    received_msg=""
    def __init__(self):
        super().__init__('publish_result_serial_port')
        self.subscription = self.create_subscription(String,'detection/result', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning
        self.serial_port = serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1)
        self.get_logger().info('UART port is started')
        self.timer = self.create_timer(1, self.write_serial_data)

    def listener_callback(self, msg):
        PublishResultSerial.received_msg = msg.data
        self.get_logger().info('I heard: "%s"' % msg.data)
        
    def write_serial_data(self):
        msg_to_write = PublishResultSerial.received_msg + "\n\r" 
        self.serial_port.write(msg_to_write.encode())
        self.get_logger().info('Data transmitted to serial port: %s' % msg_to_write)


def main(args=None):
    
    rclpy.init(args=args)
    result_subscriber = PublishResultSerial()
    rclpy.spin(result_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    result_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
