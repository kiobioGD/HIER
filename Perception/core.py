import cv2
import numpy as np
#import pyrealsense2 as rs
import depthai as dai

# Select camera type: 'realsense' or 'luxonis'
CAMERA_TYPE = 'luxonis'  # Change to 'luxonis' for OAK-D

def get_distance(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_frame, depth_scale = param
        if depth_frame is not None:
            distance = depth_frame[y, x] * depth_scale
            print(f"Distance at ({x}, {y}): {distance:.3f} meters")

def run_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    cv2.namedWindow("Depth Stream (RealSense)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth Stream (RealSense)", depth_colormap)
            cv2.setMouseCallback("Depth Stream (RealSense)", get_distance, (depth_image, depth_scale))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def run_luxonis():
    pipeline = dai.Pipeline()

    depth = pipeline.create(dai.node.StereoDepth)

    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam.out.link(depth.left)

    cam2 = pipeline.create(dai.node.MonoCamera)
    cam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam2.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam2.out.link(depth.right)

    depth_out = pipeline.create(dai.node.XLinkOut)
    depth.depth.link(depth_out.input)
    depth_out.setStreamName("depth")

    device_info = dai.DeviceInfo("169.254.1.222")
    with dai.Device(pipeline, device_info) as device:
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        cv2.namedWindow("Depth Stream (Luxonis)")

        while True:
            depth_frame = depth_queue.get()
            depth_image = depth_frame.getFrame()
            depth_scale = 0.001  # OAK-D typically provides depth in millimeters

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Depth Stream (Luxonis)", depth_colormap)
            cv2.setMouseCallback("Depth Stream (Luxonis)", get_distance, (depth_image, depth_scale))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if CAMERA_TYPE == 'realsense':
    run_realsense()
elif CAMERA_TYPE == 'luxonis':
    run_luxonis()
else:
    print("Invalid camera type! Choose 'realsense' or 'luxonis'.")
