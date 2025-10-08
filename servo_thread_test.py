#!/usr/bin/env python3
"""
servo_thread_test.py
Simple threaded servo sanity test:
 - Opens serial
 - Starts a writer thread that sends angle commands at a fixed rate
 - Runs one of: sweep / steps / dither
 - Clean shutdown on Ctrl+C

Your Arduino sketch should read a line like "145\n" and use Servo.write(angle).
"""

import time
import threading
import queue
import argparse
import sys

try:
    import serial
except ImportError:
    print("pyserial not found. Install with: pip install pyserial")
    sys.exit(1)


class ServoWriter(threading.Thread):
    """
    A worker thread that rate-limits serial writes and applies neutral+clamps.
    Feed it *relative* angles via .set_angle(relative_deg).
    """
    def __init__(
        self,
        port: str,
        baud: int = 9600,
        neutral_deg: int = 145,     # center value your Arduino expects
        min_abs_deg: int = 130,     # absolute clamp lower bound
        max_abs_deg: int = 160,     # absolute clamp upper bound
        period_s: float = 0.05,     # 20 Hz
        eol: str = "\n",
        open_delay_s: float = 2.0    # let Arduino reset after opening serial
    ):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.neutral_deg = int(neutral_deg)
        self.min_abs_deg = int(min_abs_deg)
        self.max_abs_deg = int(max_abs_deg)
        self.period_s = float(period_s)
        self.eol = eol
        self.open_delay_s = float(open_delay_s)

        self._ser = None
        self._target_rel_deg = 0
        self._last_sent_abs = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

    def set_angle(self, relative_deg: float):
        """Set the *relative* angle (e.g., -15..+15) that the thread should send."""
        with self._lock:
            self._target_rel_deg = float(relative_deg)

    def stop(self):
        self._stop_evt.set()

    def _compute_abs(self, rel_deg: float) -> int:
        abs_deg = int(round(self.neutral_deg + rel_deg))
        if abs_deg < self.min_abs_deg:
            abs_deg = self.min_abs_deg
        if abs_deg > self.max_abs_deg:
            abs_deg = self.max_abs_deg
        return abs_deg

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1, write_timeout=0.2)
        except Exception as e:
            print(f"[SERVO] Failed to open serial on {self.port}: {e}")
            return

        # Give Arduino time to reset after opening serial
        time.sleep(self.open_delay_s)
        try:
            self._ser.reset_input_buffer()
            self._ser.reset_output_buffer()
        except Exception:
            pass

        print(f"[SERVO] Connected on {self.port} @ {self.baud} baud.")
        t_next = time.time()

        try:
            while not self._stop_evt.is_set():
                # pace the loop
                now = time.time()
                if now < t_next:
                    time.sleep(t_next - now)
                t_next = max(t_next + self.period_s, time.time() + 0.0)

                # read current target safely
                with self._lock:
                    rel = self._target_rel_deg

                abs_deg = self._compute_abs(rel)

                # Only send if it changed (avoid spamming duplicates)
                if abs_deg != self._last_sent_abs:
                    try:
                        line = f"{abs_deg}{self.eol}".encode("utf-8")
                        self._ser.write(line)
                        self._last_sent_abs = abs_deg
                        # Optional: print to observe the stream
                        print(f"[SEND] rel={rel:+.1f} => abs={abs_deg}")
                    except Exception as e:
                        print(f"[SERVO] Write error: {e}")
                        # small backoff if serial hiccups
                        time.sleep(self.period_s)
        finally:
            # Return to neutral once on exit
            try:
                neutral = self._compute_abs(0.0)
                self._ser.write(f"{neutral}{self.eol}".encode("utf-8"))
                print(f"[SERVO] Neutral {neutral} sent. Closing.")
            except Exception:
                pass
            try:
                self._ser.close()
            except Exception:
                pass


def pattern_sweep(writer: ServoWriter, amp=15, step=1, dwell=0.2):
    """-amp..+amp..-amp sweep with dwell per step."""
    while True:
        for rel in range(-amp, amp + 1, step):
            writer.set_angle(rel)
            time.sleep(dwell)
        for rel in range(amp, -amp - 1, -step):
            writer.set_angle(rel)
            time.sleep(dwell)


def pattern_steps(writer: ServoWriter, positions=(-10, 0, 10, 0), dwell=1.0):
    """Jump through a tuple of relative positions, holding each for dwell seconds."""
    idx = 0
    while True:
        writer.set_angle(positions[idx % len(positions)])
        time.sleep(dwell)
        idx += 1


def pattern_dither(writer: ServoWriter, amp=3.0, freq=0.5):
    """Small sinusoid around neutral to test smooth motion."""
    t0 = time.time()
    import math
    while True:
        t = time.time() - t0
        rel = amp * math.sin(2 * math.pi * freq * t)
        writer.set_angle(rel)
        time.sleep(0.01)  # 100 Hz compute; writer will rate-limit to its period


def main():
    ap = argparse.ArgumentParser(description="Threaded servo sanity test (no camera/PID).")
    ap.add_argument("--port", required=True, help="Serial port (e.g., /dev/tty.usbmodem1101 or COM3)")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--neutral", type=int, default=145, help="Neutral absolute angle your Arduino expects")
    ap.add_argument("--min", dest="min_abs", type=int, default=130)
    ap.add_argument("--max", dest="max_abs", type=int, default=160)
    ap.add_argument("--rate", type=float, default=20.0, help="Send rate in Hz (default 20)")
    ap.add_argument("--mode", choices=["sweep", "steps", "dither"], default="sweep")
    ap.add_argument("--amp", type=float, default=15.0, help="Relative amplitude (deg) for sweep/dither")
    ap.add_argument("--step", type=int, default=1, help="Step size (deg) for sweep")
    ap.add_argument("--dwell", type=float, default=0.2, help="Hold time per step (s) for sweep/steps")
    args = ap.parse_args()

    writer = ServoWriter(
        port=args.port,
        baud=args.baud,
        neutral_deg=args.neutral,
        min_abs_deg=args.min_abs,
        max_abs_deg=args.max_abs,
        period_s=1.0 / max(1e-3, args.rate),
        eol="\n",
    )
    writer.start()

    try:
        if args.mode == "sweep":
            print("[MODE] sweep")
            pattern_sweep(writer, amp=int(args.amp), step=int(args.step), dwell=args.dwell)
        elif args.mode == "steps":
            print("[MODE] steps")
            pattern_steps(writer, positions=(-int(args.amp), 0, int(args.amp), 0), dwell=args.dwell)
        else:
            print("[MODE] dither")
            pattern_dither(writer, amp=float(args.amp), freq=0.5)
    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C received, stopping...")
    finally:
        writer.stop()
        writer.join(timeout=2.0)
        print("[MAIN] Done.")


if __name__ == "__main__":
    main()
