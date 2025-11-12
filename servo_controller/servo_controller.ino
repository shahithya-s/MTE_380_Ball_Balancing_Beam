#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
#define SERVOMIN  85 // This is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  520 // This is the 'maximum' pulse length count (out of 4096)
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

const int MIN_ANGLE = 40;
const int MAX_ANGLE = 112;
const int NEUTRAL_ANGLE = 90;

// Communication variables
int targetAngles[3] = {NEUTRAL_ANGLE, NEUTRAL_ANGLE, NEUTRAL_ANGLE};
bool newCommand = false;
int servoToUpdate = -1;


void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setOscillatorFrequency(23500000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

  // Initialize all servos to neutral
  for (int i = 0; i < 3; i++) {
    setServoAngle(i, targetAngles[i]);
  }

  delay(10);
}

// 180deg move arm down
// 0deg moves arm up
// Servo 0
// 30 deg is max upwards
// 112 deg is max downwards

// Servo 1
// 40 deg is max upwards
// 112 deg is max downwards

// Servo 2
// 40 deg is max upwards
// 116 deg is max downwards

void loop() {

  setServoAngle(0, 90);
  setServoAngle(1, 90);

  // Check for incoming serial data
  if (Serial.available() > 0) {
    // Read the incoming byte
    String input = Serial.readStringUntil('\n');  // read until newline

    // Expect the format "servo:angle"
    int colonIndex = input.indexOf(":");

    if (colonIndex > 0) {
      int servoNum = input.substring(0, colonIndex).toInt();
      int receivedAngle = input.substring(colonIndex+1).toInt();

      // validate servo index and angle
      if (servoNum >= 0 && servoNum <= 2 &&
        receivedAngle >= MIN_ANGLE && receivedAngle <= MAX_ANGLE) {
          servoToUpdate = servoNum;
          targetAngles[servoNum] = receivedAngle;
          newCommand = true; 
        }
    }
  }
  
  // Update servo if new command received
  if (newCommand) {
    setServoAngle(servoToUpdate, targetAngles[servoToUpdate]);
    Serial.print("Servo ");
    Serial.print(servoToUpdate);
    Serial.print(" set to ");
    Serial.println(targetAngles[servoToUpdate]);
    newCommand = false;
  }
}

void setServoAngle(uint8_t channel_num, float angle){
  uint16_t off_to_write = map(angle, 0.0, 180.0, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel_num, 0, off_to_write);
}