#include <Servo.h>

// Servo configuration
Servo beamServo;
const int servoPin = 9;
const int MIN_ANGLE = 105;
const int MAX_ANGLE = 175;
const int NEUTRAL_ANGLE = 140;

// Communication variables
int targetAngle = NEUTRAL_ANGLE;
bool newCommand = false;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  
  // Attach servo
  beamServo.attach(servoPin);
  
  // Move to neutral position
  beamServo.write(NEUTRAL_ANGLE);
  
  // Initialize variables
  targetAngle = NEUTRAL_ANGLE;
  
  // Optional: Send ready message
  Serial.println("Arduino servo controller ready");

}

void loop() {
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
    beamServo.write(targetAngle);
    newCommand = false;
    
    // Optional: Echo back the angle for debugging
    Serial.print("Angle set to: ");
    Serial.println(targetAngle);
  }
  
  // Small delay for stability
  delay(10);
}