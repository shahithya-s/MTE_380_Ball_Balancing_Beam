import cv2
import numpy as np
import json
import serial
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from threading import Thread
import queue
from PIL import Image, ImageTk
from ball_detection import detect_ball_x

import threading

class ServoWriter(threading.Thread):
    def __init__(self, port, baud=9600, neutral=140, min_abs=105, max_abs=175, rate_hz=20, eol="\n"):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.neutral = int(neutral)
        self.min_abs = int(min_abs)
        self.max_abs = int(max_abs)
        self.period = 1.0/max(1.0, rate_hz)
        self.eol = eol
        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rel_deg = 0

    def set_angle(self, rel_deg):
        with self._lock:
            self._rel_deg = float(rel_deg)

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1, write_timeout=0.2)
            time.sleep(2.0)
        except Exception as e:
            print(f"[SERVO] open failed: {e}")
            return

        last_sent = None
        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                rel = int(round(self._rel_deg))
            abs_deg = int(self.neutral + rel)
            abs_deg = max(self.min_abs, min(self.max_abs, abs_deg))
            if abs_deg != last_sent:
                try:
                    self._ser.write(f"{abs_deg}{self.eol}".encode("utf-8"))
                    # print(f"[SERVO] abs={abs_deg} (rel={rel:+d})")
                    last_sent = abs_deg
                except Exception as e:
                    print(f"[SERVO] write err: {e}")
            dt = self.period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        # send neutral once on exit
        try:
            self._ser.write(f"{self.neutral}{self.eol}".encode("utf-8"))
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass

    def stop(self):
        self._stop.set()

class BasicPIDController:
    def __init__(self, config_file="config.json"):
        """Initialize controller, load config, set defaults and queues."""
        # Load experiment and hardware config from JSON file
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        # PID gains (controlled by sliders in GUI)
        self.Kp = 3.8
        self.Ki = 0.226
        self.Kd = 1.92
        # Scale factor for converting from pixels to meters
        self.scale_factor = self.config['calibration']['pixel_to_meter_ratio'] * self.config['camera']['frame_width'] / 2
        # Servo port name and center angle
        self.servo_port = self.config['servo']['port']
        self.neutral_angle = self.config['servo']['neutral_angle']
        self.servo = None
        # Controller-internal state
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        # Data logs for plotting results
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.start_time = None
        # Thread-safe queue for most recent ball position measurement
        self.position_queue = queue.Queue(maxsize=1)
        self.running = False    # Main run flag for clean shutdown
        self.preview_queue = queue.Queue(maxsize=1)
        self._tk_img = None
        self.writer = None
        self.control_period = 0.05     # 20 Hz
        self.min_angle_step = 1.0      # only send if >= 1 deg change
        self.last_send_time = 0.0
        self.last_sent_angle = None
        self.servo_dir = -1      # +1 normal, -1 invert servo mapping
        self.output_alpha = 0.3        # low-pass smoothing 0..1 (higher=less smoothing)
        self._y_smooth = 0.0           # internal for smoothing


    def connect_servo(self):
        """Try to open serial connection to servo, return True if success."""
        try:
            self.servo = serial.Serial(self.servo_port, 9600)
            time.sleep(2)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Failed: {e}")
            return False

    def send_servo_angle(self, angle_rel_deg):
        """Non-blocking: hand the target to the servo writer thread."""
        if self.writer:
            self.writer.set_angle(angle_rel_deg)

    def test_servo(self):
        """Send a sweep of dummy angles to the servo to verify connection and movement."""
        if not self.connect_servo():
            print("[ERROR] Cannot test servo: connection failed.")
            return

        print("[TEST] Starting servo test sweep")
        try:
            for angle in range(-15, 16, 5):  # From -15 to 15 in steps of 5 (matches clamp range)
                self.send_servo_angle(angle)
                time.sleep(0.5)  # Wait 0.5 seconds to observe movement
        finally:
            # Return to neutral angle and close connection
            self.send_servo_angle(0)
            time.sleep(1)
            if self.servo:
                self.servo.close()
        print("[TEST] Servo test sweep complete")

    def update_pid(self, position, dt=0.033):
        """Perform PID calculation and return control output."""
        error = self.setpoint - position  # Compute error
        error = error * 100  # Scale error for easier tuning (if needed)
        # Proportional term
        P = self.Kp * error
        # Integral term accumulation
        self.integral += error * dt
        I = self.Ki * self.integral
        # Derivative term calculation
        derivative = (error - self.prev_error) / dt
        D = self.Kd * derivative
        self.prev_error = error
        # PID output (limit to safe beam range)
        output = P + I + D
        output = np.clip(output, -25, 25)
        print(f"Error: {error}")
        return output

    def camera_thread(self):
        """Dedicated thread for video capture and ball detection."""
        cap = cv2.VideoCapture(self.config['camera']['index'], cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (320, 180))
            # Detect ball position in frame
            found, x_normalized, vis_frame = detect_ball_x(frame)
            if found:
                # Convert normalized to meters using scale
                position_m = x_normalized * self.scale_factor
                # Always keep latest measurement only
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except Exception:
                    pass
            # Show processed video with overlays
            try:
                if self.preview_queue.full():
                    self.preview_queue.get_nowait()
                # Convert BGR -> RGB for Tk
                rgb = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                self.preview_queue.put_nowait(rgb)
            except Exception:
                pass
        cap.release()
        

    def control_thread(self):
        """Runs PID control loop in parallel with GUI and camera."""
        self.start_time = time.time()
        last_send_time = 0

        while self.running:
            try:
                # Wait for latest ball position from camera
                position = self.position_queue.get(timeout=0.1)
                # Compute control output using PID
                control_output = self.update_pid(position)
                control_output *= self.servo_dir
                # Send control command to servo (real or simulated)
                current_send_time = time.time()
                if current_send_time - last_send_time > 0: # 0 placeholder for now, remove later
                    self.send_servo_angle(control_output)
                    last_send_time = current_send_time
                # Log results for plotting
                current_time = time.time() - self.start_time
                self.time_log.append(current_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(control_output)
                print(f"Pos: {position:.3f}m, Output: {control_output:.1f}°")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break

    def create_gui(self):
        """Build Tkinter GUI with large sliders and labeled controls."""
        self.root = tk.Tk()
        self.root.title("Basic PID Controller")
        self.root.geometry("520x400")

        # Title label
        ttk.Label(self.root, text="PID Gains", font=("Arial", 18, "bold")).pack(pady=10)

        # Kp slider
        ttk.Label(self.root, text="Kp (Proportional)", font=("Arial", 12)).pack()
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0, to=100, variable=self.kp_var,
                              orient=tk.HORIZONTAL, length=500)
        kp_slider.pack(pady=5)
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.1f}", font=("Arial", 11))
        self.kp_label.pack()

        # Ki slider
        ttk.Label(self.root, text="Ki (Integral)", font=("Arial", 12)).pack()
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0, to=1, variable=self.ki_var,
                              orient=tk.HORIZONTAL, length=500)
        ki_slider.pack(pady=5)
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.3f}", font=("Arial", 11))
        self.ki_label.pack()

        # Kd slider
        ttk.Label(self.root, text="Kd (Derivative)", font=("Arial", 12)).pack()
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0, to=10, variable=self.kd_var,
                              orient=tk.HORIZONTAL, length=500)
        kd_slider.pack(pady=5)
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.3f}", font=("Arial", 11))
        self.kd_label.pack()

        # Setpoint slider
        ttk.Label(self.root, text="Setpoint (meters)", font=("Arial", 12)).pack()
        pos_min = self.config['calibration']['position_min_m']
        pos_max = self.config['calibration']['position_max_m']
        self.setpoint_var = tk.DoubleVar(value=self.setpoint)
        setpoint_slider = ttk.Scale(self.root, from_=pos_min, to=pos_max,
                                   variable=self.setpoint_var,
                                   orient=tk.HORIZONTAL, length=500)
        setpoint_slider.pack(pady=5)
        self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}m", font=("Arial", 11))
        self.setpoint_label.pack()

        # --- Video preview area (Tkinter, main thread) ---
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Button group for actions
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="Reset Integral",
                   command=self.reset_integral).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Plot Results",
                   command=self.plot_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Stop",
                   command=self.stop).pack(side=tk.LEFT, padx=5)

        # Schedule periodic GUI update
        self.update_gui()

    def update_gui(self):
        """Reflect latest values from sliders into program and update display."""
        if self.running:
            # PID parameters
            self.Kp = self.kp_var.get()
            self.Ki = self.ki_var.get()
            self.Kd = self.kd_var.get()
            self.setpoint = self.setpoint_var.get()
            # Update displayed values
            self.kp_label.config(text=f"Kp: {self.Kp:.1f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.3f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.3f}")
            self.setpoint_label.config(text=f"Setpoint: {self.setpoint:.3f}m")
            # Update video preview if a new frame is available
            try:
                frame_rgb = self.preview_queue.get_nowait()  # HxWx3 RGB uint8
                img = Image.fromarray(frame_rgb)
                self._tk_img = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self._tk_img)
            except queue.Empty:
                pass
            # Call again after 50 ms (if not stopped)
            self.root.after(50, self.update_gui)

    def reset_integral(self):
        """Clear integral error in PID (button handler)."""
        self.integral = 0.0
        print("[RESET] Integral term reset")

    def plot_results(self):
        """Show matplotlib plots of position and control logs."""
        if not self.time_log:
            print("[PLOT] No data to plot")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        # Ball position trace
        ax1.plot(self.time_log, self.position_log, label="Ball Position", linewidth=2)
        ax1.plot(self.time_log, self.setpoint_log, label="Setpoint",
                 linestyle="--", linewidth=2)
        ax1.set_ylabel("Position (m)")
        ax1.set_title(f"Basic PID Control (Kp={self.Kp:.1f}, Ki={self.Ki:.1f}, Kd={self.Kd:.1f})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Control output trace
        ax2.plot(self.time_log, self.control_log, label="Control Output",
                 color="orange", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Beam Angle (degrees)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def stop(self):
        """Stop everything and clean up threads and GUI."""
        self.running = False
        # Stop writer
        try:
            if self.writer:
                self.writer.stop()
        except Exception:
            pass

        # Try to safely close all windows/resources
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass

    def run(self):
        """Entry point: starts threads, launches GUI mainloop."""
        print("[INFO] Starting Basic PID Controller")
        print("Use sliders to tune PID gains in real-time")
        print("Close camera window or click Stop to exit")
        self.running = True

        self.writer = ServoWriter(
            port=self.servo_port,
            baud=9600,
            neutral=int(self.neutral_angle),
            min_abs=115,
            max_abs=165,
            rate_hz=20,
            eol="\n",
        )
        self.writer.start()
        time.sleep(0.1)
        self.writer.set_angle(0)

        # Start camera and control threads, mark as daemon for exit
        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        cam_thread.start()
        ctrl_thread.start()

        # Build and run GUI in main thread
        self.create_gui()
        self.root.mainloop()

        # After GUI ends, stop everything
        self.running = False
        print("[INFO] Controller stopped")

if __name__ == "__main__":
    try:
        controller = BasicPIDController()
        # Run servo test before starting controller
        # controller.test_servo()

        # Then run the full controller
        controller.run()
    except FileNotFoundError:
        print("[ERROR] config.json not found. Run simple_autocal.py first.")
    except Exception as e:
        print(f"[ERROR] {e}")