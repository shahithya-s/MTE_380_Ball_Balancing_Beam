import cv2
import numpy as np
import json
import os

class BallDetector:
    """Ball detector supporting 3-axis geometry loaded from 3 config files."""

    def __init__(self, config_files):
        """
        Args:
            config_files (list of str): list of 3 JSON config paths
        """
        if len(config_files) != 3:
            raise ValueError("BallDetector requires exactly 3 config files.")

        self.configs = []
        self.axes = []   # Stores peg1, peg2, u_beam, u_perp, scale

        # Default HSV if configs missing
        self.lower_hsv = np.array([5,150,150], dtype=np.uint8)
        self.upper_hsv = np.array([20,255,255], dtype=np.uint8)

        for cfg_path in config_files:
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"Config not found: {cfg_path}")

            with open(cfg_path, 'r') as f:
                cfg = json.load(f)

            self.configs.append(cfg)

            # Load geometry
            g = cfg.get("geometry", {})

            peg1 = np.array(g.get("peg1", [0,0]), dtype=float)
            peg2 = np.array(g.get("peg2", [1,0]), dtype=float)
            u_beam = np.array(g.get("u_beam", [1,0]), dtype=float)
            u_perp = np.array(g.get("u_perp", [0,1]), dtype=float)
            u_beam = np.array(u_beam, dtype=float)
            u_beam /= np.linalg.norm(u_beam)

            u_perp = np.array(u_perp, dtype=float)
            u_perp /= np.linalg.norm(u_perp)
            scale = g.get("pixel_to_meter_ratio", None)

            if scale is None:
                raise ValueError(f"{cfg_path} missing pixel_to_meter_ratio")

            self.axes.append({
                "peg1": peg1,
                "peg2": peg2,
                "u_beam": u_beam,
                "u_perp": u_perp,
                "scale": scale
            })

            # Load HSV bounds from the FIRST config only
            if cfg.get("ball_detection"):
                self.lower_hsv = np.array(cfg["ball_detection"].get("lower_hsv", self.lower_hsv), dtype=np.uint8)
                self.upper_hsv = np.array(cfg["ball_detection"].get("upper_hsv", self.upper_hsv), dtype=np.uint8)

        print("[3AXIS] Loaded configs for axes 0,1,2")
        print("[3AXIS] HSV bounds:", self.lower_hsv, "to", self.upper_hsv)

        midpoints = []
        scales = []
        for axis in self.axes:
            midpoints.append(0.5 * (axis["peg1"] + axis["peg2"]))
            scales.append(axis["scale"])
        self.center_px = np.mean(np.stack(midpoints, axis=0), axis=0)   # (cx, cy) in pixels
        self.avg_scale = float(np.mean(scales))  # m per pixel (approx)
        print("[3AXIS] Estimated plate center (px):", self.center_px)
        print("[3AXIS] Avg scale (m/px):", self.avg_scale)

    # ---------------- BALL FINDING ------------------------

    def detect_ball_pixel(self, frame):
        """Return (found, (x,y), radius). Pure pixel detection."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None

        largest = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest)

        if radius < 5:
            return False, None, None

        return True, np.array([x, y], dtype=float), radius

    # ---------------- PROJECTION ON ALL AXES ------------------------

    def project_to_axis(self, ball_xy, axis):
        """Compute (t_m, d_m) for one axis."""
        p1 = axis["peg1"]
        u_beam = axis["u_beam"]
        u_perp = axis["u_perp"]
        s = axis["scale"]  # pixel_to_meter_ratio

        v = ball_xy - p1       # vector from peg1 to ball (pixels)
        t = np.dot(v, u_beam)  # pixel distance along beam
        d = np.dot(v, u_perp)  # perpendicular pixel distance

        return float(t*s), float(d*s)

    def detect_ball(self, frame):
        """
        Returns:
            found (bool)
            ball_xy (tuple)
            radius (float)
            axis_positions: [(t0,d0), (t1,d1), (t2,d2)]
        """
        found, ball_xy, radius = self.detect_ball_pixel(frame)
        if not found:
            return False, None, None, [(0,0),(0,0),(0,0)]

        axis_positions = []
        for axis in self.axes:
            axis_positions.append(self.project_to_axis(ball_xy, axis))

        return True, tuple(ball_xy), radius, axis_positions
    
    def solve_global_xy(self, ball_xy):
        """
        Compute global (x,y) in meters relative to plate center.

        ball_xy: np.array([x_px, y_px]) or tuple, pixel coordinates.
        Returns (x_m, y_m).
        """
        ball_xy = np.array(ball_xy, dtype=float)

        # Vector from center (pixels)
        delta_px = ball_xy - self.center_px  # [dx_px, dy_px]

        # Convert to meters (x: right +, y: up +)
        x_m = delta_px[0] * self.avg_scale
        y_m = -delta_px[1] * self.avg_scale   # invert because image y goes down

        return float(x_m), float(y_m)

    # ---------------- DRAWING ------------------------
    def draw(self, frame):
        found, ball_xy, radius, axis_positions = self.detect_ball(frame)
        overlay = frame.copy()

        # ---------------- BALL DRAWING ----------------
        if found:
            # Green circle on ball
            cv2.circle(overlay, (int(ball_xy[0]), int(ball_xy[1])), int(radius), (0,255,0), 2)

            # Axis t,d text
            for i, (t_m, d_m) in enumerate(axis_positions):
                text = f"A{i}: t={t_m:.3f}m  d={d_m:.3f}m"
                cv2.putText(overlay, text, (10, 30 + 25*i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # ---------------- AXIS LINES ----------------
        for i, axis in enumerate(self.axes):
            p1 = axis["peg1"]
            p2 = axis["peg2"]

            # Beam line
            cv2.line(
                overlay,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 180, 255),   # orange-ish
                2
            )

            # Axis label near peg1
            cv2.putText(
                overlay, f"A{i}",
                (int(p1[0]) + 10, int(p1[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2
            )

            # Draw perpendicular arrow from peg1 (just for visualization)
            u_perp = axis["u_perp"]
            perp_end = p1 + 40 * u_perp  # 40 px length
            cv2.arrowedLine(
                overlay,
                (int(p1[0]), int(p1[1])),
                (int(perp_end[0]), int(perp_end[1])),
                (255, 0, 0),
                2,
                tipLength=0.3
            )

        # ---------------- GLOBAL XY & RED DOT ----------------
        if found:
            x_m, y_m = self.solve_global_xy(ball_xy)

            # Text for global position
            cv2.putText(
                overlay,
                f"Global: x={x_m:.3f}m  y={y_m:.3f}m",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # Red dot exactly at the ball center (it's the same point)
            cv2.circle(overlay, (int(ball_xy[0]), int(ball_xy[1])), 6, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                "XY",
                (int(ball_xy[0]) + 10, int(ball_xy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Optional: draw the estimated center of plate
        cv2.circle(overlay,
                   (int(self.center_px[0]), int(self.center_px[1])),
                   5, (255, 255, 0), -1)

        return overlay, found, axis_positions

    


def main():
    # CHANGE TO YOUR REAL CONFIG FILES
    config_files = [
        "config_axis_0.json",
        "config_axis_1.json",
        "config_axis_2.json"
    ]

    detector = BallDetector(config_files)

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    print("3-Axis Ball Detection Running (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break
        
        overlay, found, axis_positions = detector.draw(frame)

        # Debug print
        if found:
            print("POS:", axis_positions)

        cv2.imshow("3 Axis Ball Tracking", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
