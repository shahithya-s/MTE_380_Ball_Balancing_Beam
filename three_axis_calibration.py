# Simple Auto Calibration System for Ball and Beam Control
# Interactive calibration tool for color detection, geometry, and servo limits
# Generates config.json file for use with ball tracking controller

import cv2
import numpy as np
import json
import math
import serial
import time
from datetime import datetime

class SimpleAutoCalibrator:
    """Interactive calibration system for ball and beam control setup."""
    
    def __init__(self):
        """Initialize calibration parameters and default values."""
        # User input on which servo is being calibrated
        self.servo_id = int(input("Enter servo axis to calibrate (0, 1, or 2): "))

        # Physical system parameters
        self.BEAM_LENGTH_M = 0.30  # Known beam length in meters
        
        # Camera configuration
        self.CAM_INDEX = 0  # Default camera index
        self.FRAME_W, self.FRAME_H = 640, 360  # Frame dimensions
        
        # Calibration state tracking
        self.current_frame = None  # Current video frame
        self.phase = "color"  # Current phase: "color", "geometry", "limits", "complete"
        
        # Color calibration data
        self.hsv_samples = []  # Collected HSV color samples
        self.lower_hsv = None  # Lower HSV bound for ball detection
        self.upper_hsv = None  # Upper HSV bound for ball detection
        
        # Geometry calibration data
        self.peg_points = []  # Beam endpoint pixel coordinates
        self.pixel_to_meter_ratio = None  # Conversion ratio from pixels to meters
        
        # Servo hardware configuration
        self.servo = None  # Serial connection to servo
        self.servo_port = "/dev/cu.usbmodem31301"  # Servo communication port
        self.neutral_angle = 76  # Servo neutral position angle
        
        # Position limit results
        self.position_min = None  # Minimum ball position in meters
        self.position_max = None  # Maximum ball position in meters

    def connect_servo(self):
        """Establish serial connection to servo motor for automated limit finding.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.servo = serial.Serial(self.servo_port, 115200)
            time.sleep(2)  # Allow time for connection to stabilize
            print("[SERVO] Connected")
            return True
        except:
            print("[SERVO] Failed to connect - limits will be estimated")
            return False

    def send_servo_angle(self, angle):
        """Send angle command to selected servo."""
        if self.servo:
            angle = int(np.clip(angle, 40, 112))
            cmd = f"{self.servo_id}:{angle}\n"
            self.servo.write(cmd.encode("utf-8"))
            print(f"[SERVO] Sent {self.servo_id}:{angle}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events for interactive calibration.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse click coordinates
            flags: Additional event flags
            param: User data (unused)
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.phase == "color":
                # Color sampling phase - collect HSV samples at click point
                self.sample_color(x, y)
            elif self.phase == "geometry" and len(self.peg_points) < 2:
                # Geometry phase - collect beam endpoint coordinates
                self.peg_points.append((x, y))
                print(f"[GEO] Peg {len(self.peg_points)} selected")
                if len(self.peg_points) == 2:
                    self.calculate_geometry()

    def sample_color(self, x, y):
        """Sample HSV color values in a 5x5 region around click point.
        
        Args:
            x, y: Center coordinates for color sampling
        """
        if self.current_frame is None:
            return
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        
        # Sample 5x5 region around click point
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                px, py = x + dx, y + dy
                # Check bounds and collect valid samples
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    self.hsv_samples.append(hsv[py, px])
        
        # Update HSV bounds based on collected samples
        if self.hsv_samples:
            samples = np.array(self.hsv_samples)
            
            # Calculate adaptive margins for each HSV channel
            # h_margin = max(5, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            # s_margin = max(10, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            # v_margin = max(10, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)
            h_margin = max(2, (np.max(samples[:, 0]) - np.min(samples[:, 0])) * 0.1)
            s_margin = max(5, (np.max(samples[:, 1]) - np.min(samples[:, 1])) * 0.15)
            v_margin = max(5, (np.max(samples[:, 2]) - np.min(samples[:, 2])) * 0.15)

            
            # Set lower bounds with margin
            self.lower_hsv = [
                max(0, np.min(samples[:, 0]) - h_margin),
                max(0, np.min(samples[:, 1]) - s_margin),
                max(0, np.min(samples[:, 2]) - v_margin)
            ]
            
            # Set upper bounds with margin
            self.upper_hsv = [
                min(179, np.max(samples[:, 0]) + h_margin),
                min(255, np.max(samples[:, 1]) + s_margin),
                min(255, np.max(samples[:, 2]) + v_margin)
            ]
            
            print(f"[COLOR] Samples: {len(self.hsv_samples)}")

    def calculate_geometry(self):
        """Calculate beam axis geometry + pixel→meter conversion."""
        p1, p2 = self.peg_points  # (x, y)

        # Beam vector in pixels
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]

        # Pixel length of beam
        pixel_distance = math.sqrt(vx*vx + vy*vy)

        # Unit vector along beam axis
        self.u_beam = np.array([vx, vy]) / pixel_distance   # (ux, uy)

        # Unit vector perpendicular to beam (not used now but useful later)
        self.u_perp = np.array([-self.u_beam[1], self.u_beam[0]])

        # Pixel → meter scale
        self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / pixel_distance
        
        print(f"[GEO] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
        print(f"[GEO] Beam unit vector: {self.u_beam}")

        # Advance to limits phase
        self.phase = "limits"


    def detect_ball_position(self, frame):
        """Detect ball in frame and return position in meters from center.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            float or None: Ball position in meters from center, None if not detected
        """
        if not self.lower_hsv:
            return None
        
        # Convert to HSV and create color mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_hsv, dtype=np.uint8)
        upper = np.array(self.upper_hsv, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up mask with morphological operations
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour (assumed to be ball)
        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)
        
        # Filter out very small detections
        if radius < 5:
            return None
        
        # Convert pixel position to meters from center
        ball = np.array([x, y], dtype=float)
        p1 = np.array(self.peg_points[0], dtype=float)

        # Vector from peg1 → ball
        v = ball - p1

        # Projection onto beam axis
        t = np.dot(v, self.u_beam)   # pixel distance along beam

        # Perpendicular distance to beam
        d = np.dot(v, self.u_perp)   # pixel distance off the beam

        # Convert to meters
        t_m = t * self.pixel_to_meter_ratio
        d_m = d * self.pixel_to_meter_ratio

        t_m_centered = t_m - (self.BEAM_LENGTH_M / 2)

        return float(t_m_centered), float(d_m)

    def find_limits_automatically(self):
        """Use servo motor to automatically find ball position limits."""
        if not self.servo:
            # Estimate limits without servo if connection failed
            self.position_min = -self.BEAM_LENGTH_M / 2
            self.position_max = self.BEAM_LENGTH_M / 2
            print("[LIMITS] Estimated without servo")
            return
        
        print("[LIMITS] Finding limits with servo...")
        positions = []
        
        # Test servo at different angles to find position range
        test_angles = [40, self.neutral_angle, 112]
        
        for angle in test_angles:
            # Move servo to test angle
            self.send_servo_angle(angle)
            time.sleep(2)  # Wait for ball to settle
            
            # Collect multiple position measurements
            angle_positions = []
            start_time = time.time()
            while time.time() - start_time < 1.0:
                ret, frame = self.cap.read()
                if ret:
                    pos = self.detect_ball_position(frame)
                    if pos is not None:
                        angle_positions.append(pos)
                time.sleep(0.05)
            
            # Calculate average position for this angle
            if angle_positions:
                avg_pos = np.mean(angle_positions)
                positions.append(avg_pos)
                print(f"[LIMITS] Angle {angle}: {avg_pos:.4f}m")
        
        # Return servo to neutral position
        self.send_servo_angle(self.neutral_angle)
        
        # Determine position limits from collected data
        if len(positions) >= 2:
            self.position_min = min(positions)
            self.position_max = max(positions)
            print(f"[LIMITS] Range: {self.position_min:.4f}m to {self.position_max:.4f}m")
        else:
            print("[LIMITS] Failed to find limits")

    def save_config(self):
        """Save all calibration results to config.json file."""
        config = {
            "timestamp": datetime.now().isoformat(),
            "beam_length_m": float(self.BEAM_LENGTH_M),
            "camera": {
                "index": int(self.CAM_INDEX),
                "frame_width": int(self.FRAME_W),
                "frame_height": int(self.FRAME_H)
            },
            "ball_detection": {
                "lower_hsv": [float(x) for x in self.lower_hsv] if self.lower_hsv else None,
                "upper_hsv": [float(x) for x in self.upper_hsv] if self.upper_hsv else None
            },
            "calibration": {
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None,
                "position_min_m": float(self.position_min) if self.position_min else None,
                "position_max_m": float(self.position_max) if self.position_max else None
            },
            "servo": {
                "port": str(self.servo_port),
                "servo_id": int(self.servo_id),
                "neutral_angle": int(self.neutral_angle)
            },
            "geometry": {
                "peg1": [int(self.peg_points[0][0]), int(self.peg_points[0][1])] if len(self.peg_points) == 2 else None,
                "peg2": [int(self.peg_points[1][0]), int(self.peg_points[1][1])] if len(self.peg_points) == 2 else None,
                "u_beam": [float(self.u_beam[0]), float(self.u_beam[1])] if hasattr(self, 'u_beam') else None,
                "u_perp": [float(self.u_perp[0]), float(self.u_perp[1])] if hasattr(self, 'u_perp') else None,
                "pixel_to_meter_ratio": float(self.pixel_to_meter_ratio) if self.pixel_to_meter_ratio else None
            },
        }
        
        # Write configuration to JSON file
        filename = f"config_axis_{self.servo_id}.json"
        with open(filename, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[SAVE] Configuration saved to {filename}")

    def draw_overlay(self, frame):
        """Draw calibration status and instructions overlay on frame.
        
        Args:
            frame: Input BGR image frame
            
        Returns:
            numpy.ndarray: Frame with overlay graphics and text
        """
        overlay = frame.copy()
        
        # Phase-specific instruction text
        phase_text = {
            "color": "Click on ball to sample colors. Press 'c' when done.",
            "geometry": "Click on beam endpoints (2 points)",
            "limits": "Press 'l' to find limits automatically",
            "complete": "Calibration complete! Press 's' to save"
        }
        
        # Draw current phase and instructions
        cv2.putText(overlay, f"Phase: {self.phase}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, phase_text[self.phase], (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show color calibration progress
        if self.hsv_samples:
            cv2.putText(overlay, f"Color samples: {len(self.hsv_samples)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show geometry calibration points
        for i, peg in enumerate(self.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+10, peg[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw line between beam endpoints if both are selected
        if len(self.peg_points) == 2:
            cv2.line(overlay, self.peg_points[0], self.peg_points[1], (255, 0, 0), 2)
        
        # Show real-time ball detection if color calibration is complete
        if self.lower_hsv:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array(self.lower_hsv, dtype=np.uint8)
            upper = np.array(self.upper_hsv, dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up mask
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)
            
            # Find and draw detected ball
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(largest)
                if radius > 5:
                    # Draw detection circle
                    cv2.circle(overlay, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(overlay, (int(x), int(y)), 3, (0, 255, 255), -1)
                    
                    # Show position if geometry calibration is complete
                    if self.pixel_to_meter_ratio:
                        pos = self.detect_ball_position(frame)
                        if pos is not None:
                            t_m, d_m = pos
                            cv2.putText(
                                overlay,
                                f"Pos: {t_m:.4f} m",
                                (int(x)+20, int(y)+20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1
                            )
        
        # Show final results if limit calibration is complete
        if self.position_min is not None and self.position_max is not None:
            cv2.putText(overlay, f"Limits: {self.position_min:.4f}m to {self.position_max:.4f}m",
                       (10, overlay.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return overlay

    def run(self):
        """Main calibration loop with interactive GUI."""
        # Initialize camera capture
        self.cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_AVFOUNDATION)
        # self.cap = cv2.VideoCapture(self.CAM_INDEX, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # --- ADJUST EXPOSURE AND BRIGHTNESS ---
        # These values depend on your camera model; experiment with them
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # Turn off auto exposure
        # self.cap.set(cv2.CAP_PROP_EXPOSURE, -10)          # Lower exposure (try values -4 to -8)
        # self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)        # Adjust brightness if needed
        # self.cap.set(cv2.CAP_PROP_CONTRAST, 30)
        # self.cap.set(cv2.CAP_PROP_SATURATION, 70)
        
        # Setup OpenCV window and mouse callback
        cv2.namedWindow("Auto Calibration")
        cv2.setMouseCallback("Auto Calibration", self.mouse_callback)
        
        # Attempt servo connection
        self.connect_servo()
        
        # Display instructions
        print("[INFO] Simple Auto Calibration")
        print("Phase 1: Click on ball to sample colors, press 'c' when done")
        print("Phase 2: Click on beam endpoints")
        print("Phase 3: Press 'l' to find limits")
        print("Press 's' to save, 'q' to quit")
        
        # Main calibration loop
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            self.current_frame = frame
            
            # Draw overlay and display frame
            display = self.draw_overlay(frame)
            cv2.imshow("Auto Calibration", display)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit calibration
                break
            elif key == ord('c') and self.phase == "color":
                # Complete color calibration phase
                if self.hsv_samples:
                    self.phase = "geometry"
                    print("[INFO] Color calibration complete. Click on beam endpoints.")
            elif key == ord('l') and self.phase == "limits":
                # Start automatic limit finding
                self.find_limits_automatically()
                self.phase = "complete"
            elif key == ord('s') and self.phase == "complete":
                # Save configuration and exit
                self.save_config()
                break
        
        # Clean up resources
        self.cap.release()
        cv2.destroyAllWindows()
        if self.servo:
            self.servo.close()

if __name__ == "__main__":
    """Run calibration when script is executed directly."""
    calibrator = SimpleAutoCalibrator()
    calibrator.run()