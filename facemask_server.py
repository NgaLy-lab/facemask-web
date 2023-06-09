from flask_socketio import emit, send, SocketIO
import json
import sys
import os
from flask import Flask, render_template, Response
import cv2
import imutils
import time
import timeit
import base64
from imutils.video import WebcamVideoStream, VideoStream, FileVideoStream, FPS
import ast
import torch
import numpy as np


"""server.py support the main loop of the app
	get handlers
	get
	while the program is still running:
		get frame
		run frame through model
		receive new frame and results
		call respective socketio's emit function
"""

# VIDEO_SOURCE = 'http://192.168.87.166:4747/video'
VIDEO_SOURCE = 0
# VIDEO_SOURCE = 'video/vid1.mp4'
model = torch.hub.load(
    './models/yolov5/', 'custom', path='./weights/new_pwmfd/yolov5_mix/best.pt', source='local', force_reload=True)
    # './', 'custom', path='weights/w_pretrained/best.pt', source='local')

app = Flask(__name__,
            static_url_path='/',
            static_folder='./client/static/', template_folder='./client/static/')
app.config["TEMPLATES_AUTO_RELOAD"] = True


socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def test_root():
    return 'hello'


@app.route("/video")
def display():
    return render_template('view-video.html')

@app.route('/video_feed')
def video_feed():
    return Response(inference_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('message')
def handle_message(data):
    """Testing socket io"""
    print('received message: ', data)


def broadcast_result(result):
    print('enter broadcast result', result)
    socketio.emit('Sending result', result, broadcast=True, json=True)

@socketio.on('connected')
def connected():
    print('Connected')


@socketio.on('disconnected')
def disconnected():
    print('Disconnected')


# save for later use
@socketio.on('new_user')
def broadcast_new_user(new_user):
    socketio.emit('Connecting new user', new_user.username)


def inference_video():
    print('Entered inference')
    stream = VideoStream(VIDEO_SOURCE).start()  # default camera
    # stream = FileVideoStream(VIDEO_SOURCE).start()  # video
    while True:
        # grab next frame
        frame = stream.read()
        starttime = timeit.default_timer()
        results = model(frame)
        endtime = timeit.default_timer()
        # Results
        results.display(render=True)
        frame = results.imgs[0].astype(np.uint8)
        result = export_image_info(results.pred[0], endtime-starttime)

        broadcast_result(result)
        ret, buffer = cv2.imencode('.jpg', frame)
        serialized = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + serialized + b'\r\n')

def export_image_info(pred, inference_time):
    total = len(pred)
    result = [0, 0, 0]
    # must be checked with more people
    for c in pred[:, -1].unique():
        n = int((pred[:, -1] == c).sum())
        result[int(c)] = n

    return {
        'total': total,
        'mask': result[1],
        'no_mask': result[0],
        'incorrect': result[2],
        'inference_time': round(inference_time,3)
    }


if __name__ == '__main__':
    socketio.run(app, debug=True)
