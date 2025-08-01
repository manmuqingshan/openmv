# This work is licensed under the MIT license.
# Copyright (c) 2013-2024 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# Pure Thermal Example Script
#
# Thanks for buying the Pure Thermal OpenMV! This example script shows
# off thermal video overlay onto the color camera image and driving
# the attached LCD screen and HDMI output.

import csi
import image
import time
import display
import math
import tfp410


# Color Tracking Thresholds (Grayscale Min, Grayscale Max)
threshold_list = [(200, 255)]

# Set the target temp range here
min_temp_in_celsius = 20.0
max_temp_in_celsius = 40.0

csi0 = csi.CSI(cid=csi.OV5640)
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.WVGA)
time.sleep_ms(50)

csi1 = csi.CSI(cid=csi.LEPTON)
csi1.reset(hard=False)
csi1.pixformat(csi.GRAYSCALE)
csi1.framesize(csi.QQVGA)

# Enables exact temperature measurments from the flir lepton.
# The second argument turns high gain mode on for high temp reading.
csi1.ioctl(csi.IOCTL_LEPTON_SET_MODE, True, False)
csi1.ioctl(
    csi.IOCTL_LEPTON_SET_RANGE, min_temp_in_celsius, max_temp_in_celsius
)

print(
    "Lepton Res (%dx%d)"
    % (
        csi1.ioctl(csi.IOCTL_LEPTON_GET_WIDTH),
        csi1.ioctl(csi.IOCTL_LEPTON_GET_HEIGHT),
    )
)

print(
    "Radiometry Available: "
    + ("Yes" if csi1.ioctl(csi.IOCTL_LEPTON_GET_RADIOMETRY) else "No")
)

fir_img = image.Image(csi1.width(), csi1.height(), image.GRAYSCALE)
time.sleep_ms(50)

lcd = display.RGBDisplay(framesize=display.FWVGA, refresh=60)
lcd.backlight(True)
hdmi = tfp410.TFP410()
time.sleep_ms(50)

alpha_pal = image.Image(256, 1, image.GRAYSCALE)
for i in range(256):
    alpha_pal[i] = int(math.pow((i / 255), 2) * 255)

to_min = None
to_max = None


def map_g_to_temp(g):
    return (
        (g * (max_temp_in_celsius - min_temp_in_celsius)) / 255.0
    ) + min_temp_in_celsius


# Kickstart thermal camera capture.
csi1.snapshot(update=False, blocking=True, image=fir_img)

while True:
    img = csi0.snapshot()

    # Capture the thermal image without blocking.
    csi1.snapshot(update=False, blocking=False, image=fir_img)

    fir_img_size = fir_img.width() * fir_img.height()

    # Find IR Blobs
    blobs = fir_img.find_blobs(threshold_list,
                               pixels_threshold=(fir_img_size // 100),
                               area_threshold=(fir_img_size // 100),
                               merge=True)

    # Collect stats into a list of tuples
    blob_stats = []
    for b in blobs:
        blob_stats.append(
            (b.rect(), map_g_to_temp(fir_img.get_statistics(
                thresholds=threshold_list, roi=b.rect()
            ).mean()))
        )

    x_scale = img.width() / fir_img.width()
    y_scale = img.height() / fir_img.height()
    img.draw_image(fir_img, 0, 0, x_scale=x_scale, y_scale=y_scale,
                   color_palette=image.PALETTE_IRONBOW,
                   alpha_palette=alpha_pal,
                   hint=image.BICUBIC)

    # Draw stuff on the colored image
    for b in blobs:
        img.draw_rectangle(int(b.rect()[0] * x_scale),
                           int(b.rect()[1] * y_scale),
                           int(b.rect()[2] * x_scale),
                           int(b.rect()[3] * y_scale))
        img.draw_cross(int(b.cx() * x_scale), int(b.cy() * y_scale))

    for blob_stat in blob_stats:
        img.draw_string(int((blob_stat[0][0] * x_scale) + 4),
                        int((blob_stat[0][1] * y_scale) + 1),
                        '%.2f C' % blob_stat[1], mono_space=False, scale=2)

    lcd.write(img, hint=(
        image.BILINEAR | image.CENTER | image.SCALE_ASPECT_KEEP
    ))
