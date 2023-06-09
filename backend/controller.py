# import imutils
# import time
# import cv2
# from imutils.video import WebcamVideoStream, VideoStream, FileVideoStream, FPS
# import logging
# from backend import model_handler, socket_handler

# logger = logging.Logger(__name__)


# class Controller:
#     async def inference(model, video_source):
#         # start video stream thread, allow buffer to fill
#         logger.info('Starting threaded video stream...')
#         stream = VideoStream(video_source).start()  # default camera
#         time.sleep(1.0)
#         # fps = FPS().start()
#         # start fps timer
#         # loop over frames from the video file stream
#         while True:
#             # grab next frame
#             frame = stream.read()
#             key = cv2.waitKey(1) & 0xFF
#             # print(frame)
#             # update FPS counter
#             # fps.update()
#             frame, result = await model.run_detection(frame)
#             socket_handler.broadcast_result(result)
#             socket_handler.broadcast_frame(frame)

#             if key == 27:  # exit
#                 break

#     def load_model():
#         self.model = model_handler.ModelHandler()
