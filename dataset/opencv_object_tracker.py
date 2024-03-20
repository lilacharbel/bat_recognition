import os
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
ap.add_argument("-i", "--interval", type=int, default=8, help="interval between frames to save")
args = vars(ap.parse_args())


# create folder to save raw frames and bbox
video_name = args["video"].split("/")[-1].split(".")[0]
imgs_savedir = f'../data/processed_data/{video_name}/imgs'
bboxes_savedir = f'../data/processed_data/{video_name}/bboxes'

os.makedirs(imgs_savedir, exist_ok=True)
os.makedirs(bboxes_savedir, exist_ok=True)

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    # "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    # "tld": cv2.TrackerTLD_create,
    # "medianflow": cv2.TrackerMedianFlow_create,
    # "mosse": cv2.TrackerMOSSE_create
}
# grab the appropriate object tracker using our dictionary of
# OpenCV object tracker objects
tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])
# initialize the FPS throughput estimator
fps = None

frame_idx = 0
# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()[1]
    org_frame = frame.copy()
    # frame = frame[1] if args.get("video", False) else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions

    scale = 2
    frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
        else:
            initBB = None

        # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args["tracker"]),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track

    if initBB is None:
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        box = initBB
        tracker.init(frame, initBB)
        fps = FPS().start()

    if key == ord("s"):
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                               showCrosshair=True)
        box = initBB
        tracker.init(frame, initBB)
        fps = FPS().start()

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

    # save img and bbox
    if frame_idx % args["interval"] == 0:
        cv2.imwrite(os.path.join(imgs_savedir, f'{video_name}_frame_{frame_idx}.png'), org_frame)
    np.save(os.path.join(bboxes_savedir, f'{video_name}_frame_{frame_idx}.npy'), np.array(box)*2)

    frame_idx += 1


# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
