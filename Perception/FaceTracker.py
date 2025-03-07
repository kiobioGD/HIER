import cv2
import depthai as dai
import numpy as np
import blobconverter


def create_pipeline():
    pipeline = dai.Pipeline()

    # Create RGB and Depth cameras
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    stereo = pipeline.create(dai.node.StereoDepth)

    # Set up the RGB camera
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

    # Load the face detection model
    face_det_nn.setBlobPath("face-detection-retail-0004.blob")
    face_det_nn.setConfidenceThreshold(0.5)

    # Configure stereo depth
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # Linking the nodes
    cam_rgb.preview.link(face_det_nn.input)
    left = pipeline.create(dai.node.MonoCamera)
    right = pipeline.create(dai.node.MonoCamera)
    
    left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    
    left.out.link(stereo.left)
    right.out.link(stereo.right)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    
    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")

    cam_rgb.preview.link(xout_rgb.input)
    stereo.depth.link(xout_depth.input)
    face_det_nn.out.link(xout_nn.input)

    return pipeline


def main():
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
        detection_queue = device.getOutputQueue(name="detections", maxSize=1, blocking=False)

        while True:
            rgb_frame = rgb_queue.get().getCvFrame()
            depth_frame = depth_queue.get().getFrame()
            detections = detection_queue.get().detections

            for detection in detections:
                x1 = int(detection.xmin * rgb_frame.shape[1])
                y1 = int(detection.ymin * rgb_frame.shape[0])
                x2 = int(detection.xmax * rgb_frame.shape[1])
                y2 = int(detection.ymax * rgb_frame.shape[0])

                # Get depth at the center of the face rectangle
                face_center_x = (x1 + x2) // 2
                face_center_y = (y1 + y2) // 2
                distance = depth_frame[face_center_y, face_center_x]

                # Draw the rectangle and distance
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(rgb_frame, f"Distance: {distance/1000:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Face Tracker", rgb_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()