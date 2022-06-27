import cv2
# import the im utils library..
# imutils library used to resize a frame screen.
import imutils
# define a video capture object
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2

print("[INFO] sampling THREADED frames from webcam...")
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
# loop over some frames...this time using the threaded stream
while fps._numFrames < 100:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=640, height=480)
	# check to see if the frame should be displayed to our screen
	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# update the FPS counter
	fps.update()
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()