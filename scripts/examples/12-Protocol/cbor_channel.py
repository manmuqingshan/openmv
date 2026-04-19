# This work is licensed under the MIT license.
# Copyright (c) 2026 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# CBOR Channel Example (ToF + Face Detection)
#
# This example runs face detection on the main camera while sending
# CBOR-encoded sensor data and ToF depth data over custom protocol
# channels.
#
# CBOR record keys (SenML-compatible for scalars):
#   -2 = bn (base name)    0 = n (name)       1 = u (unit)
#    2 = v  (value)         3 = vs (string)    4 = vb (bool)
#    8 = vd (data/binary)   6 = t  (time)
#
# Custom 2D data extension keys:
#   -20 = width    -21 = height    -22 = format
#   -23 = min      -24 = max

import csi
import tof
import cbor2
import struct
import random
import protocol
import ml
from ml.postprocessing.mediapipe import BlazeFace

# SenML-compatible CBOR integer keys.
_BN = -2  # base name
_N = 0    # record name
_U = 1    # unit
_V = 2    # numeric value
_VD = 8   # data value (binary)

# Custom 2D data extension keys.
_W = -20    # width
_H = -21    # height
_FMT = -22  # format string
_MIN = -23  # min value
_MAX = -24  # max value


class CBORChannel:
    """A read-only protocol channel that serves CBOR-encoded data."""

    def __init__(self):
        self._buf = b""

    def poll(self):
        return len(self._buf) > 0

    def size(self):
        return len(self._buf)

    def read(self, offset, size):
        end = min(offset + size, len(self._buf))
        return bytes(self._buf[offset:end])

    def update(self, records):
        self._buf = cbor2.dumps(records)


# Initialize the camera.
csi0 = csi.CSI()
csi0.reset()
csi0.pixformat(csi.RGB565)
csi0.framesize(csi.QVGA)

# Load built-in face detection model.
model = ml.Model(
    "/rom/blazeface_front_128.tflite",
    postprocess=BlazeFace(threshold=0.5),
)
print(model)

# Initialize ToF sensor.
tof.init()
tof_w = tof.width()
tof_h = tof.height()

# Register two CBOR channels.
sensors_backend = CBORChannel()
depth_backend = CBORChannel()
sensors_channel = protocol.register(name="sensors", backend=sensors_backend)
depth_channel = protocol.register(name="ToF depth", backend=depth_backend)

# Simulated sensor state.
temp = 25.0
humidity = 50.0

while True:
    # Capture frame and run face detection.
    img = csi0.snapshot()

    for r, score, keypoints in model.predict([img]):
        ml.utils.draw_predictions(img, [r], ("face",), ((0, 0, 255),), format=None)
        ml.utils.draw_keypoints(img, keypoints, color=(255, 0, 0))

    # Simulate sensor readings with random drift.
    temp += random.uniform(-0.3, 0.3)
    temp = max(15.0, min(35.0, temp))
    humidity += random.uniform(-0.5, 0.5)
    humidity = max(20.0, min(80.0, humidity))

    # Read ToF depth map.
    try:
        depth, dmin, dmax = tof.read_depth(vflip=True, hmirror=True)
    except RuntimeError:
        continue

    # Pack depth floats into binary (4 bytes per float).
    depth_bytes = struct.pack("<%df" % len(depth), *depth)

    # Update sensor channel.
    sensors_backend.update([
        {_N: "temp", _U: "Cel", _V: round(temp, 1)},
        {_N: "humidity", _U: "%RH", _V: round(humidity, 1)},
    ])

    # Update depth channel.
    depth_backend.update([
        {
            _N: "depth",
            _VD: depth_bytes,
            _W: tof_w,
            _H: tof_h,
            _FMT: "depth",
            _MIN: dmin,
            _MAX: dmax,
        },
    ])

    # Send notifications on custom channels.
    depth_channel.send_event(0xFFFF)
    sensors_channel.send_event(0xFFFF)

    print(
        "temp=%.1f | humidity=%.1f | depth=%d bytes"
        % (temp, humidity, len(depth_bytes))
    )
