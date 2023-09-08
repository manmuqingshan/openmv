# Single Color Grayscale Blob Tracking Example
#
# This example shows off single color grayscale tracking using the OpenMV Cam using the FLIR LEPTON.

# FLIR Lepton Shutter Note: FLIR Leptons with radiometry and a shutter will pause the video often
# as they heatup to re-calibrate. This will happen less and less often as the sensor temperature
# stablizes. You can force the re-calibration to not happen if you need to via the lepton API.
# However, it is not recommended because the image will degrade overtime.

import sensor
import time
import lcd

# Color Tracking Thresholds (Grayscale Min, Grayscale Max)
threshold_list = [(220, 255)]

print("Resetting Lepton...")
# These settings are applied on reset
sensor.reset()
print(
    "Lepton Res (%dx%d)"
    % (
        sensor.ioctl(sensor.IOCTL_LEPTON_GET_WIDTH),
        sensor.ioctl(sensor.IOCTL_LEPTON_GET_HEIGHT),
    )
)
print(
    "Radiometry Available: "
    + ("Yes" if sensor.ioctl(sensor.IOCTL_LEPTON_GET_RADIOMETRY) else "No")
)

sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.LCD)
sensor.skip_frames(time=5000)
clock = time.clock()
lcd.init()

# Only blobs that with more pixels than "pixel_threshold" and more area than "area_threshold" are
# returned by "find_blobs" below. Change "pixels_threshold" and "area_threshold" if you change the
# camera resolution. "merge=True" merges all overlapping blobs in the image.

while True:
    clock.tick()
    img = sensor.snapshot()
    for blob in img.find_blobs(
        threshold_list, pixels_threshold=200, area_threshold=200, merge=True
    ):
        img.draw_rectangle(blob.rect(), color=127)
        img.draw_cross(blob.cx(), blob.cy(), color=127)
    lcd.display(img)
    print(clock.fps())