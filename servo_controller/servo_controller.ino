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
int targetAngle = NEUTRAL_ANGLE;
bool newCommand = false;


void setup() {
  Serial.begin(115200);
  pwm.begin();
  pwm.setOscillatorFrequency(23500000);
  pwm.setPWMFreq(SERVO_FREQ);  // Analog servos run at ~50 Hz updates

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
    int receivedAngle = input.toInt();            // convert string to integer
    // Validate angle range
    if (receivedAngle >= MIN_ANGLE && receivedAngle <= MAX_ANGLE) {
      targetAngle = receivedAngle;
      newCommand = true;
    }
  }
  
  // Update servo if new command received
  if (newCommand) {
    setServoAngle(2, targetAngle);
    newCommand = false;
    
    // Optional: Echo back the angle for debugging
    Serial.print("Angle set to: ");
    Serial.println(targetAngle);
  }

  // delay(10);
}

void setServoAngle(uint8_t channel_num, float angle){
  uint16_t off_to_write = map(angle, 0.0, 180.0, SERVOMIN, SERVOMAX);
  pwm.setPWM(channel_num, 0, off_to_write);
}