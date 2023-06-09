# import json
# from flask_socketio import emit, send, SocketIO


# @socketio.on('message')
# def handle_message(data):
#     """Testing socket io"""
#     print('received message: ', data)


# def broadcast_result(result):
#     emit('Sending result', json.dumps(result), broadcast=True)


# def broadcast_frame(frame):
#     emit('Sending frames', frame, broadcast=True)


# def connected():
#     print('Connected')


# def disconnected():
#     print('Disconnected')


# # save for later use
# def broadcast_new_user(new_user):
#     emit('Connecting new user', new_user.username)
