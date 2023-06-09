import sys
import socket
import imutils 
import time
import cv2
from imutils.video import VideoStream, FileVideoStream, FPS
sys.path.append('.\\models')
from SampleModel import SampleModel


def inference(model,video_source):
    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = VideoStream(video_source).start()  # default camera
    time.sleep(1.0)
    fps = FPS().start()
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        frame = stream.read()
        key = cv2.waitKey(1) & 0xFF

        
        frame, _ = model.detect_mask(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('Face Mask Detector', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('Face Mask Detector', frame)
        # update FPS counter
        fps.update()
        if key == 27:  # exit
            break 
    fps.stop()
    print(f'Ave. FPS: {fps.fps()}')
video_source = 'http://192.168.100.3:4747/video'  #droidcam
# video_source = 0                              #webcam
model = SampleModel()
inference(model,video_source)