import serial, time

PORT = "/dev/cu.usbmodem1401"  # <- change to your cu device
BAUD = 9600

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # allow Arduino to reset
    print("✅ Opened", PORT)
    ser.write(bytes([15]))   # go to neutral
    time.sleep(0.2)
    ser.write(bytes([0]))    # to 0°
    time.sleep(0.5)
    ser.write(bytes([30]))   # to 30°
    time.sleep(0.5)
    ser.write(bytes([15]))   # back to neutral
    ser.close()
    print("✅ Wrote angles OK")
except Exception as e:
    print("❌ Failed to open/write:", e)
