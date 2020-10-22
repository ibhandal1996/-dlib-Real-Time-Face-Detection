from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# command line arg: python LandmarkDetection.py --shape-predictor shape_predictor_68_face_landmarks.dat


# argument parser
ap = argparse.ArgumentParser()
# --shape-predictor: dlib's pre-trained facial landmark detector
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize face detector
detect = dlib.get_frontal_face_detector()
# create the facial landmark predictor
pred = dlib.shape_predictor(args["shape_predictor"])

# initialize video stream
vidStream = VideoStream().start()
time.sleep(2.0)

while True:
    # grabs frame and manipulates it
    frame = vidStream.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detects face in grayscale
    rect = detect(gray, 0)

    for r in rect:
        # determines the facial landmark
        sh = pred(gray, r)
        # converts coordinates to a Numpy array
        sh = face_utils.shape_to_np(sh)

        # draws trackers on frame
        for x, y in sh:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    cv2.imshow("Cam", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vidStream.stop()