from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import argparse
import cv2
import sys
import time
# import utils

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool):
    counter, fps = 0,0
    start_time = time.time()

    frame = cv2.VideoCapture(camera_id)
    frame.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    frame.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


    base_options = core.BaseOptions(
        file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
    detection_options = processor.DetectionOptions(
        max_results=3, score_threshold=0.3)
    options = vision.ObjectDetectorOptions(
        base_options=base_options, detection_options=detection_options)
    detector = vision.ObjectDetector.create_from_options(options)

    while frame.isOpened():
        success, image = frame.read()
        if not success:
            sys.exit()

        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        input_tensor = vision.TensorImage.create_from_array(rgb_image)

        detections = detector.detect(input_tensor)

        for detection in detections.detections:
            print(detection.categories[0].category_name)

        cv2.imshow('image', image)
        cv2.waitKey(50)

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='efficientdet_lite0.tflite')
    parser.add_argument(
        '--cameraId', help='Id of camera.', required=False, type=int, default=0)
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=8)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    args = parser.parse_args()

    run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
        int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()