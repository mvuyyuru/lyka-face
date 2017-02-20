#finds faces and draws a box around them using opencv

#see http://opencv.org/
import cv2
import os

base_path = os.path.dirname(os.path.realpath(__file__))
video_path = base_path + '/vid/'
haarcascade_path = base_path + '/haarcascades/'

#webcam
#video_stream = cv2.VideoCapture(0)

video_name = "familyguy-short.mp4"
video_stream = cv2.VideoCapture(video_path + video_name)

face_cascade = cv2.CascadeClassifier(haarcascade_path + 'haarcascade_frontalface_default.xml')

def find_faces(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found_faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    return found_faces

def draw_face_bounding_boxes(faces, frame):
        #color in BGR
        box_color = (255, 0, 0)
        for (x_coord, y_coord, width, height) in faces:
            cv2.rectangle(frame, (x_coord, y_coord), (x_coord+width, y_coord+height), box_color)

while video_stream.isOpened():

    ret, frame = video_stream.read()
    faces = find_faces(frame)
    draw_face_bounding_boxes(faces, frame)

    cv2.imshow('video stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

#cleanup
video_stream.release()
cv2.destroyAllWindows()
