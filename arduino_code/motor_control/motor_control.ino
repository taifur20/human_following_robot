/*************************************
 * Author: Md. Khairul Alam
 * 
 */
#include <Servo.h>

#define LEFT_MOTOR_IN1 4
#define LEFT_MOTOR_IN2 5
#define RIGHT_MOTOR_IN1 7
#define RIGHT_MOTOR_IN2 8
#define LEFT_MOTOR_EN 6
#define RIGHT_MOTOR_EN 9

Servo horizontal_servo, vertical_servo;

String inputString = "";         // a string to hold incoming data
boolean stringComplete = false;  // whether the string is complete

int horizontal_pin = 2;
int vertical_pin = 3;

int servo_angle = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  horizontal_servo.attach(horizontal_pin);
  vertical_servo.attach(vertical_pin);
  horizontal_servo.write(0); 
  pinMode(LEFT_MOTOR_IN1, OUTPUT);
  pinMode(LEFT_MOTOR_IN2, OUTPUT);
  pinMode(RIGHT_MOTOR_IN1, OUTPUT);
  pinMode(RIGHT_MOTOR_IN2, OUTPUT);
  pinMode(LEFT_MOTOR_EN, OUTPUT);
  pinMode(RIGHT_MOTOR_EN, OUTPUT);
  
  inputString.reserve(200);
}

void loop() {
  if(servo_angle>=180){
     rotate_servo_left();
    }
  else if(servo_angle<=0){
     rotate_servo_right();
    }
  if (stringComplete) {
    if(inputString == "person_detected"){//if received message = pos1
        if(servo_angle > 100){
           //person detected to the right side of the robot, 
           //turn robot right based on the servo angle, where person is detected
           move_robot(100+servo_angle, 100-servo_angle); 
           delay(2);
          }
         else if(servo_angle < 80){
           //person detected to the left side of the robot, 
           //turn robot left based on the servo angle, where person is detected
           move_robot(100 - (180 - servo_angle), 100+ (180 - servo_angle)); 
           delay(2);
          }
         else if(servo_angle >= 80 && servo_angle <= 100){
          //person detected to the front side of the robot, 
          //go straight
          move_robot(100 + servo_angle, 100 + servo_angle); 
          delay(2);
          }
      } 
   else{
       move_robot(0, 0); 
      }
      
    inputString = "";
    stringComplete = false;
  }
}

void rotate_servo_right(){
    for(int i=0; i<180; i++){
      horizontal_servo.write(i);
      servo_angle = i; 
      delay(1); //rotate servo slowley 
      }
 }

void rotate_servo_left(){
    for(int i=180; i>0; i--){
      horizontal_servo.write(i); 
      servo_angle = i; 
      delay(1); 
      }
 }

void move_robot(int left_speed, int right_speed){
  analogWrite(LEFT_MOTOR_EN, left_speed);
  analogWrite(RIGHT_MOTOR_EN, right_speed);
  digitalWrite(LEFT_MOTOR_IN1, HIGH);
  digitalWrite(LEFT_MOTOR_IN2, LOW);
  digitalWrite(RIGHT_MOTOR_IN1, HIGH);
  digitalWrite(RIGHT_MOTOR_IN2, LOW);
}

void serialEvent() {
  while (Serial.available()) {    
    // get the new byte:
    char inChar = (char)Serial.read();     
    // if the incoming character is a newline, set a flag
    // so the main loop can do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
    else
    // add it to the inputString:  
      inputString += inChar;
  }
}
