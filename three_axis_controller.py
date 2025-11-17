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
import threading

# Use your existing 3-axis detector (unchanged)
from three_axis_ball_detection import BallDetector


# ============================================================
# Multi-servo writer: ONE Arduino, 3 servos, "id:angle\n" protocol
# ============================================================

class MultiServoWriter(threading.Thread):
    """
    Background thread that sends servo commands to a single Arduino
    with a PCA9685 shield. We send ASCII lines like "0:90\n", "1:85\n".
    """
    def __init__(self,
                 port,
                 baud=115200,
                 neutral_angles=(76, 76, 76),
                 min_angles=(40, 40, 40),
                 max_angles=(112, 112, 112),
                 rate_hz=20,
                 eol="\n"):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.neutral = np.array(neutral_angles, dtype=float)
        self.min_angles = np.array(min_angles, dtype=float)
        self.max_angles = np.array(max_angles, dtype=float)
        self.period = 1.0 / max(1.0, rate_hz)
        self.eol = eol

        self._ser = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # relative target angles [deg] for 3 servos
        self._rel_deg = np.zeros(3, dtype=float)

    def set_angles(self, rel_deg_vec):
        """Update desired relative angles for the 3 servos (in degrees)."""
        with self._lock:
            self._rel_deg[:] = rel_deg_vec

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud,
                                      timeout=0.1, write_timeout=0.2)
            time.sleep(2.0)  # let Arduino reset
            print(f"[SERVO] Connected on {self.port}")
        except Exception as e:
            print(f"[SERVO] open failed: {e}")
            return

        last_sent = np.full(3, np.nan)

        while not self._stop.is_set():
            t0 = time.time()
            with self._lock:
                rel = self._rel_deg.copy()

            # convert relative -> absolute and clamp
            abs_deg = self.neutral + rel
            abs_deg = np.maximum(self.min_angles, np.minimum(self.max_angles, abs_deg))
            abs_int = abs_deg.astype(int)

            # send only changed servos
            for i in range(3):
                if abs_int[i] != last_sent[i]:
                    cmd = f"{i}:{abs_int[i]}{self.eol}"
                    try:
                        self._ser.write(cmd.encode("utf-8"))
                        last_sent[i] = abs_int[i]
                        # Debug print; comment out later if noisy
                        print(f"[SERVO] {cmd.strip()}")
                    except Exception as e:
                        print(f"[SERVO] write err: {e}")
                        break

            dt = self.period - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)

        # on exit: send neutrals once
        try:
            for i in range(3):
                cmd = f"{i}:{int(self.neutral[i])}{self.eol}"
                self._ser.write(cmd.encode("utf-8"))
        except Exception:
            pass
        try:
            self._ser.close()
        except Exception:
            pass
        print("[SERVO] Stopped")

    def stop(self):
        self._stop.set()


# ============================================================
# 3-AXIS PER-AXIS PID CONTROLLER
# ============================================================

class StewartAxisPIDController:
    """
    3-axis Stewart-platform ball balancer.

    - Uses BallDetector(3 configs) to get axis projections (t0,t1,t2)
    - One PID per axis on t_i (meters)
    - PID outputs u_i mapped to servo relative angles (deg) per axis
    - Sends commands to Arduino + PCA9685 as "id:angle\n"
    """

    def __init__(self,
                 config_files=None):

        if config_files is None:
            config_files = [
                "config_axis_0.json",
                "config_axis_1.json",
                "config_axis_2.json"
            ]

        # ---------------- Load geometry & detector ----------------
        self.detector = BallDetector(config_files)

        # Use first config for camera & servo info if present
        cfg0 = self.detector.configs[0]

        camera_cfg = cfg0.get("camera", {})
        self.cam_index = camera_cfg.get("index", 0)
        self.cam_width = camera_cfg.get("frame_width", 640)
        self.cam_height = camera_cfg.get("frame_height", 360)

        servo_cfg = cfg0.get("servo", {})
        self.servo_port = servo_cfg.get("port", "/dev/cu.usbmodem31301")
        neutral_angle = servo_cfg.get("neutral_angle", 76)

        # If per-servo neutrals are provided, use them; else all 90
        na = servo_cfg.get("neutral_angles", [neutral_angle, neutral_angle, neutral_angle])
        if len(na) < 3:
            na = [neutral_angle, neutral_angle, neutral_angle]
        self.neutral_angles = (na[0], na[1], na[2])

        self.min_angles = (40, 40, 40)
        self.max_angles = (112, 112, 112)

        # ---------------- PID parameters (shared across axes) ----------------
        self.Kp = 3.0
        self.Ki = 0.0
        self.Kd = 0.2

        # Per-axis PID state
        self.integral = np.zeros(3, dtype=float)
        self.prev_error = np.zeros(3, dtype=float)

        # Per-axis setpoints (in meters along each beam)
        # Default 0; you can read actual center t_i from overlay and type them into GUI
        self.setpoints = np.zeros(3, dtype=float)

        # Gains for mapping PID output "tilt units" -> degrees
        # 1.0 tilt unit ≈ 10 degrees of servo motion
        self.tilt_to_deg_gain = 80.0

        # Servo direction per axis (flip sign if a servo reacts backwards)
        # 80 tilts plate UP, 100 tilts plate DOWN for all three:
        #   positive rel_deg -> tilt DOWN (toward 100)
        # If you find an axis is inverted, change its sign to -1.0
        self.servo_signs = np.array([1.0, -1.0, -1.0], dtype=float)

        # logging
        self.time_log = []
        self.t_log = []      # shape (N,3)
        self.u_log = []      # shape (N,3)
        self.servo_log = []  # shape (N,3)

        self.start_time = None

        # queues and threading flags
        self.position_queue = queue.Queue(maxsize=1)  # stores np.array([t0,t1,t2])
        self.preview_queue = queue.Queue(maxsize=1)   # RGB frames for GUI
        self.running = False
        self._tk_img = None

        # servo writer
        self.writer = None

    # ---------------- Per-axis PID ----------------

    def pid_step_axes(self, t_vec, dt):
        """
        t_vec: np.array shape (3,) with current t0,t1,t2 in meters.
        dt:   elapsed time [s]
        Returns:
            u_vec: np.array shape(3,) PID outputs per axis.
        """
        # error per axis
        e = self.setpoints - t_vec

        # integrate
        self.integral += e * dt
        # simple anti-windup clamp
        self.integral = np.clip(self.integral, -0.5, 0.5)

        # derivative
        de = (e - self.prev_error) / dt if dt > 0 else np.zeros_like(e)
        self.prev_error = e

        # PID (same Kp/Ki/Kd for all axes)
        u = self.Kp * e + self.Ki * self.integral + self.Kd * de

        # Optional clamp on raw PID outputs
        # u = np.clip(u, -0.3, 0.3)
        u = np.clip(u, -1.0, 1.0)

        return u

    # ---------------- PID output -> servo relative degrees ----------------

    def pid_to_servo_rel_deg(self, u_vec):
        """
        u_vec: shape(3,) PID outputs.
        Returns:
            rel_deg: shape(3,) relative servo angles in degrees.
            (neutral + rel_deg goes to the Arduino)
        """
        rel_deg = u_vec * self.tilt_to_deg_gain  # 1 tilt unit -> 10 deg
        rel_deg *= self.servo_signs

        # Keep within +/-36 degrees from neutral
        rel_deg = np.clip(rel_deg, -36.0, 36.0)

        return rel_deg

    # ---------------- Threads ----------------

    def camera_thread(self):
        """Captures frames, runs 3-axis ball detection, pushes t0,t1,t2 to queue."""
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_AVFOUNDATION)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Detect ball and get axis projections
            found, ball_xy, radius, axis_positions = self.detector.detect_ball(frame)

            if found:
                # axis_positions is [(t0,d0),(t1,d1),(t2,d2)] in meters
                t_vec = np.array([axis_positions[i][0] for i in range(3)], dtype=float)

                # Push into queue (keep latest only)
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(t_vec)
                except Exception:
                    pass

            # Build overlay for GUI (uses your BallDetector drawing)
            try:
                overlay, _, _ = self.detector.draw(frame)
                if self.preview_queue.full():
                    self.preview_queue.get_nowait()
                rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                self.preview_queue.put_nowait(rgb)
            except Exception as e:
                print("[CAM] overlay error:", e)

        cap.release()
        print("[CAM] Stopped")

    def control_thread(self):
        """Runs 3 per-axis PIDs and sends servo commands."""
        self.start_time = time.time()
        last_time = self.start_time

        while self.running:
            try:
                t_vec = self.position_queue.get(timeout=0.2)  # shape(3,)
            except queue.Empty:
                continue

            now = time.time()
            dt = now - last_time
            if dt <= 0:
                dt = 1e-3
            last_time = now

            # 3 PIDs
            u_vec = self.pid_step_axes(t_vec, dt)

            # Map to servo relative degrees
            rel_deg = self.pid_to_servo_rel_deg(u_vec)

            # Send to writer
            if self.writer:
                self.writer.set_angles(rel_deg)

            # Logging
            t_rel = now - self.start_time
            self.time_log.append(t_rel)
            self.t_log.append(t_vec.copy())
            self.u_log.append(u_vec.copy())
            self.servo_log.append(rel_deg.copy())

            print(f"[CTRL] t = {t_vec},  u = {u_vec},  rel_deg = {rel_deg}")

        print("[CTRL] Stopped")

    # ---------------- GUI ----------------

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("3-Axis Stewart Axis PID Controller")
        self.root.geometry("650x560")

        ttk.Label(self.root, text="3-Axis PID Gains", font=("Arial", 16, "bold")).pack(pady=10)

        # Kp
        ttk.Label(self.root, text="Kp", font=("Arial", 12)).pack()
        self.kp_var = tk.DoubleVar(value=self.Kp)
        kp_slider = ttk.Scale(self.root, from_=0.0, to=20.0, variable=self.kp_var,
                              orient=tk.HORIZONTAL, length=600)
        kp_slider.pack(pady=4)
        self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.2f}")
        self.kp_label.pack()

        # Ki
        ttk.Label(self.root, text="Ki", font=("Arial", 12)).pack()
        self.ki_var = tk.DoubleVar(value=self.Ki)
        ki_slider = ttk.Scale(self.root, from_=0.0, to=5.0, variable=self.ki_var,
                              orient=tk.HORIZONTAL, length=600)
        ki_slider.pack(pady=4)
        self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.3f}")
        self.ki_label.pack()

        # Kd
        ttk.Label(self.root, text="Kd", font=("Arial", 12)).pack()
        self.kd_var = tk.DoubleVar(value=self.Kd)
        kd_slider = ttk.Scale(self.root, from_=0.0, to=10.0, variable=self.kd_var,
                              orient=tk.HORIZONTAL, length=600)
        kd_slider.pack(pady=4)
        self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.3f}")
        self.kd_label.pack()

        # Setpoints per axis
        ttk.Label(self.root, text="Axis Setpoints t0,t1,t2 (meters along each beam)",
                  font=("Arial", 11)).pack(pady=6)

        sp_frame = ttk.Frame(self.root)
        sp_frame.pack()

        self.sp_vars = []
        for i in range(3):
            ttk.Label(sp_frame, text=f"t{i}:").grid(row=0, column=2*i, padx=3)
            var = tk.DoubleVar(value=float(self.setpoints[i]))
            self.sp_vars.append(var)
            tk.Entry(sp_frame, textvariable=var, width=8).grid(row=0, column=2*i+1, padx=3)

        # Video preview
        self.video_label = ttk.Label(self.root)
        self.video_label.pack(pady=10)

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Reset Integral",
                   command=self.reset_integral).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Plot Logs",
                   command=self.plot_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop",
                   command=self.stop).pack(side=tk.LEFT, padx=5)

        self.update_gui()

    def update_gui(self):
        if self.running:
            # gains (shared)
            self.Kp = self.kp_var.get()
            self.Ki = self.ki_var.get()
            self.Kd = self.kd_var.get()
            self.kp_label.config(text=f"Kp: {self.Kp:.2f}")
            self.ki_label.config(text=f"Ki: {self.Ki:.3f}")
            self.kd_label.config(text=f"Kd: {self.Kd:.3f}")

            # setpoints per axis
            for i in range(3):
                self.setpoints[i] = self.sp_vars[i].get()

            # video frame
            try:
                frame_rgb = self.preview_queue.get_nowait()
                img = Image.fromarray(frame_rgb)
                self._tk_img = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=self._tk_img)
            except queue.Empty:
                pass

            self.root.after(50, self.update_gui)

    # ---------------- Utility methods ----------------

    def reset_integral(self):
        self.integral[:] = 0.0
        print("[RESET] Integral terms reset")

    def plot_results(self):
        if not self.time_log:
            print("[PLOT] No data to plot yet")
            return

        t = np.array(self.time_log)
        t_axes = np.array(self.t_log)      # N x 3
        u_axes = np.array(self.u_log)      # N x 3
        servo_axes = np.array(self.servo_log)  # N x 3

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

        # Axis positions
        ax1.plot(t, t_axes[:, 0], label="t0")
        ax1.plot(t, t_axes[:, 1], label="t1")
        ax1.plot(t, t_axes[:, 2], label="t2")
        ax1.set_ylabel("t (m)")
        ax1.set_title("Axis Positions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # PID outputs
        ax2.plot(t, u_axes[:, 0], label="u0")
        ax2.plot(t, u_axes[:, 1], label="u1")
        ax2.plot(t, u_axes[:, 2], label="u2")
        ax2.set_ylabel("PID output")
        ax2.set_title("PID Outputs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Servo commands
        ax3.plot(t, servo_axes[:, 0], label="servo0")
        ax3.plot(t, servo_axes[:, 1], label="servo1")
        ax3.plot(t, servo_axes[:, 2], label="servo2")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("rel angle (deg)")
        ax3.set_title("Servo Commands (rel to neutral)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def stop(self):
        self.running = False
        try:
            if self.writer:
                self.writer.stop()
        except Exception:
            pass
        try:
            self.root.quit()
            self.root.destroy()
        except Exception:
            pass
        print("[INFO] Stop requested")

    def run(self):
        print("[INFO] Starting 3-axis Axis-PID Stewart Controller")

        self.running = True

        # start servo writer
        self.writer = MultiServoWriter(
            port=self.servo_port,
            baud=115200,
            neutral_angles=self.neutral_angles,
            min_angles=self.min_angles,
            max_angles=self.max_angles,
            rate_hz=20,
            eol="\n"
        )
        self.writer.start()
        time.sleep(0.05)
        self.writer.set_angles([0, 0, 0])

        # camera + control threads
        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        cam_thread.start()
        ctrl_thread.start()

        # GUI (main thread)
        self.create_gui()
        self.root.mainloop()

        # after GUI exits
        self.running = False
        print("[INFO] Controller mainloop ended")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    try:
        controller = StewartAxisPIDController(
            config_files=[
                "config_axis_0.json",
                "config_axis_1.json",
                "config_axis_2.json"
            ]
        )
        controller.run()
    except FileNotFoundError as e:
        print(f"[ERROR] Config file missing: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")
