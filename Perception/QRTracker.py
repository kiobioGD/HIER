import cv2
import numpy as np
# import pyrealsense2 as rs
import depthai as dai

# Choose your camera: 'realsense' or 'luxonis'
CAMERA_TYPE = 'luxonis'

def detect_qr_and_get_distance(color_frame, depth_frame, depth_scale):
    qr_detector = cv2.QRCodeDetector()

    gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    enhanced_frame = cv2.equalizeHist(gray_frame)

    scaled_frame = cv2.resize(enhanced_frame, None, fx=1.5, fy=1.5)

    data, points, _ = qr_detector.detectAndDecode(scaled_frame)

    distance = None

    if points is not None and len(points) > 0:
        points = (points/1.5).astype(int)

        # Draw the QR code outline
        #for i in range(len(points)):
        #    pt1 = tuple(points[i])
         #   pt2 = tuple(points[(i + 1) % len(points)])
         #   cv2.line(color_frame, pt1, pt2, (0, 255, 0), 2)

        # Get the center point of the QR code
        cx = int(np.mean(points[:, 0]))
        cy = int(np.mean(points[:, 1]))

        # Estimate the distance
        distance = depth_frame[cy, cx] * depth_scale

        # Show the QR data (if any)
        if data:
            cv2.putText(
                color_frame, 
                f"QR Code: {data}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 0, 0), 
                2
            )
        else:
            cv2.putText(color_frame, "No QR code detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .07, (0, 0, 255), 2)

        #if distance is not None:
            cv2.putText(color_frame, f"Distance: {distance:.2f} meters",
                        (10, color_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, .07, (0, 255, 255), 2)

        cv2.imshow("Preprocessed Frame (QR Detection)", enhanced_frame)

    return color_frame

def run_realsense():
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    cv2.namedWindow("QR Tracking (RealSense)")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Get depth scale
            depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # Detect QR and get distance
            output_frame = detect_qr_and_get_distance(color_image, depth_image, depth_scale)

            # Show the output frame
            cv2.imshow("QR Tracking (RealSense)", output_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

def run_luxonis():
    pipeline = dai.Pipeline()

    # Create camera nodes
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_depth = pipeline.create(dai.node.XLinkOut)

    cam = pipeline.create(dai.node.MonoCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
    cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam.out.link(stereo.left)

    cam2 = pipeline.create(dai.node.MonoCamera)
    cam2.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    cam2.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    cam2.out.link(stereo.right)

    # Configure camera settings
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # Link outputs
    xout_rgb.setStreamName("rgb")
    xout_depth.setStreamName("depth")
    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

        cv2.namedWindow("QR Tracking (Luxonis)")

        while True:
            rgb_frame = rgb_queue.get()
            depth_frame = depth_queue.get()

            # Convert frames
            color_image = rgb_frame.getCvFrame()
            depth_image = depth_frame.getFrame()

            # Depth scale for Luxonis (depth in mm)
            depth_scale = 0.001

            # Detect QR and get distance
            output_frame = detect_qr_and_get_distance(color_image, depth_image, depth_scale)

            # Show the output frame
            cv2.imshow("QR Tracking (Luxonis)", output_frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    if CAMERA_TYPE == 'realsense':
        run_realsense()
    elif CAMERA_TYPE == 'luxonis':
        run_luxonis()
    else:
        print("Invalid camera type! Choose 'realsense' or 'luxonis'.")
