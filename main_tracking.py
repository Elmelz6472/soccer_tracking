# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
from enum import Enum
import mediapipe as mpb
import numpy as np
import argparse
import cv2
import imutils
import time


class Collision(Enum):
    NO_COLLISION = 0
    COLLISION = 1


class CollisionParts(Enum):
    NULL = None
    RIGHT_HAND__RIGHT_SHOULDER = [
        ("right_shoulder", "right_hand"),
        ("right_hand", "right_shoulder"),
    ]
    HEAD__RIGHT_HAND = [("head", "right_hand"), ("right_hand", "head")]
    LEFT_HAND__LEFT_SHOULDER = [
        ("left_shoulder", "left_hand"),
        ("left_hand", "left_shoulder"),
    ]
    HEAD__LEFT_HAND = [("head", "left_hand"), ("left_hand", "head")]


state_collision = Collision.NO_COLLISION
parts_collision = CollisionParts.NULL


# SHOW_BODY = True
SHOW_BALL = True
SHOW_BODY = False


def show_body(frame):
    # PART HUMAN TRACKING
    # Convert the frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = pose.process(frame)

    # Check if pose detection succeeded
    if results.pose_landmarks is not None:
        # Get keypoint locations
        keypoint_dict = {}
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            keypoint_dict[mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y)

        # Extract keypoint coordinates
        right_shoulder = (
            int(keypoint_dict["RIGHT_SHOULDER"][0] * frame.shape[1]),
            int(keypoint_dict["RIGHT_SHOULDER"][1] * frame.shape[0]),
        )
        left_shoulder = (
            int(keypoint_dict["LEFT_SHOULDER"][0] * frame.shape[1]),
            int(keypoint_dict["LEFT_SHOULDER"][1] * frame.shape[0]),
        )
        head = (
            int(keypoint_dict["NOSE"][0] * frame.shape[1]),
            int(keypoint_dict["NOSE"][1] * frame.shape[0]),
        )
        right_foot = (
            int(keypoint_dict["RIGHT_ANKLE"][0] * frame.shape[1]),
            int(keypoint_dict["RIGHT_ANKLE"][1] * frame.shape[0]),
        )
        left_foot = (
            int(keypoint_dict["LEFT_ANKLE"][0] * frame.shape[1]),
            int(keypoint_dict["LEFT_ANKLE"][1] * frame.shape[0]),
        )
        right_hand = (
            int(keypoint_dict["RIGHT_WRIST"][0] * frame.shape[1]),
            int(keypoint_dict["RIGHT_WRIST"][1] * frame.shape[0]),
        )
        left_hand = (
            int(keypoint_dict["LEFT_WRIST"][0] * frame.shape[1]),
            int(keypoint_dict["LEFT_WRIST"][1] * frame.shape[0]),
        )
        right_knee = (
            int(keypoint_dict["RIGHT_KNEE"][0] * frame.shape[1]),
            int(keypoint_dict["RIGHT_KNEE"][1] * frame.shape[0]),
        )
        left_knee = (
            int(keypoint_dict["LEFT_KNEE"][0] * frame.shape[1]),
            int(keypoint_dict["LEFT_KNEE"][1] * frame.shape[0]),
        )

        # Draw circles around the keypoints
        for keypoint in keypoint_dict.values():
            x, y = int(keypoint[0] * frame.shape[1]), int(keypoint[1] * frame.shape[0])
            cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

        # Draw lines between the keypoints
        cv2.line(frame, right_shoulder, left_shoulder, (255, 0, 0), 2)
        cv2.line(frame, right_shoulder, head, (255, 0, 0), 2)
        cv2.line(frame, left_shoulder, head, (255, 0, 0), 2)
        cv2.line(frame, right_foot, right_shoulder, (255, 0, 0), 2)
        cv2.line(frame, left_foot, left_shoulder, (255, 0, 0), 2)
        cv2.line(frame, right_hand, right_shoulder, (255, 0, 0), 2)
        cv2.line(frame, left_hand, left_shoulder, (255, 0, 0), 2)
        cv2.line(frame, right_knee, right_foot, (255, 0, 0), 2)
        cv2.line(frame, left_knee, left_foot, (255, 0, 0), 2)

        # Add boxes with part labels
        for part, coords in {
            "right_shoulder": right_shoulder,
            "left_shoulder": left_shoulder,
            "head": head,
            "right_foot": right_foot,
            "left_foot": left_foot,
            "right_hand": right_hand,
            "left_hand": left_hand,
            "right_knee": right_knee,
            "left_knee": left_knee,
        }.items():
            x, y = coords
            x0, y0 = max(x - box_size // 2, 0), max(y - box_size // 2, 0)
            x1, y1 = min(x + box_size // 2, frame.shape[1]), min(
                y + box_size // 2, frame.shape[0]
            )
            cv2.rectangle(frame, (x0, y0), (x1, y1), box_color, -1)
            label = part_labels[part]
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
            )
            label_x, label_y = (
                x - label_size[0] // 2,
                y + box_size // 2 + label_size[1] // 2,
            )
            cv2.putText(
                frame,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )

    # Convert the frame back to BGR color space for displaying the original colors
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def show_ball(frame):
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame)
    blurred = cv2.GaussianBlur(frame, (15, 15), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask

    # Create a 5x5 elliptical structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=6)
    mask = cv2.erode(mask, kernel, iterations=6)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Mask", mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > 30:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
        except ZeroDivisionError:
            pass

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), thickness)


# Set up the MediaPipe Pose model
mp_pose = mpb.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)


part_labels = {
    "right_shoulder": "RS",
    "left_shoulder": "LS",
    "head": "H",
    "right_foot": "RF",
    "left_foot": "LF",
    "right_hand": "RH",
    "left_hand": "LH",
    "right_knee": "RK",
    "left_knee": "LK",
}

# Define the box size and color
box_size = 100
box_color = (255, 255, 255)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64, help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (5, 100, 100)
greenUpper = (20, 255, 255)
pts = deque(maxlen=args["buffer"])


# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
collision_count = 0


box_collided = []


while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    if SHOW_BALL:
        show_ball(frame)

    if SHOW_BODY:
        # PART HUMAN TRACKING
        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = pose.process(frame)

        # Check if pose detection succeeded
        if results.pose_landmarks is not None:
            # Get keypoint locations
            keypoint_dict = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoint_dict[mp_pose.PoseLandmark(idx).name] = (landmark.x, landmark.y)

            # Extract keypoint coordinates
            right_shoulder = (
                int(keypoint_dict["RIGHT_SHOULDER"][0] * frame.shape[1]),
                int(keypoint_dict["RIGHT_SHOULDER"][1] * frame.shape[0]),
            )
            left_shoulder = (
                int(keypoint_dict["LEFT_SHOULDER"][0] * frame.shape[1]),
                int(keypoint_dict["LEFT_SHOULDER"][1] * frame.shape[0]),
            )
            head = (
                int(keypoint_dict["NOSE"][0] * frame.shape[1]),
                int(keypoint_dict["NOSE"][1] * frame.shape[0]),
            )
            right_foot = (
                int(keypoint_dict["RIGHT_ANKLE"][0] * frame.shape[1]),
                int(keypoint_dict["RIGHT_ANKLE"][1] * frame.shape[0]),
            )
            left_foot = (
                int(keypoint_dict["LEFT_ANKLE"][0] * frame.shape[1]),
                int(keypoint_dict["LEFT_ANKLE"][1] * frame.shape[0]),
            )
            right_hand = (
                int(keypoint_dict["RIGHT_WRIST"][0] * frame.shape[1]),
                int(keypoint_dict["RIGHT_WRIST"][1] * frame.shape[0]),
            )
            left_hand = (
                int(keypoint_dict["LEFT_WRIST"][0] * frame.shape[1]),
                int(keypoint_dict["LEFT_WRIST"][1] * frame.shape[0]),
            )
            right_knee = (
                int(keypoint_dict["RIGHT_KNEE"][0] * frame.shape[1]),
                int(keypoint_dict["RIGHT_KNEE"][1] * frame.shape[0]),
            )
            left_knee = (
                int(keypoint_dict["LEFT_KNEE"][0] * frame.shape[1]),
                int(keypoint_dict["LEFT_KNEE"][1] * frame.shape[0]),
            )

            # Draw circles around the keypoints
            for keypoint in keypoint_dict.values():
                x, y = int(keypoint[0] * frame.shape[1]), int(
                    keypoint[1] * frame.shape[0]
                )
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

            # Draw lines between the keypoints
            cv2.line(frame, right_shoulder, left_shoulder, (255, 0, 0), 2)
            cv2.line(frame, right_shoulder, head, (255, 0, 0), 2)
            cv2.line(frame, left_shoulder, head, (255, 0, 0), 2)
            cv2.line(frame, right_foot, right_shoulder, (255, 0, 0), 2)
            cv2.line(frame, left_foot, left_shoulder, (255, 0, 0), 2)
            cv2.line(frame, right_hand, right_shoulder, (255, 0, 0), 2)
            cv2.line(frame, left_hand, left_shoulder, (255, 0, 0), 2)
            cv2.line(frame, right_knee, right_foot, (255, 0, 0), 2)
            cv2.line(frame, left_knee, left_foot, (255, 0, 0), 2)

            # Add boxes with part labels
            overlapping_boxes = []
            part_coords = {
                "right_shoulder": right_shoulder,
                "left_shoulder": left_shoulder,
                "head": head,
                "right_foot": right_foot,
                "left_foot": left_foot,
                "right_hand": right_hand,
                "left_hand": left_hand,
                "right_knee": right_knee,
                "left_knee": left_knee,
            }
            for part1, coords1 in part_coords.items():
                x1, y1 = coords1
                x10, y10 = max(x1 - box_size // 2, 0), max(y1 - box_size // 2, 0)
                x11, y11 = min(x1 + box_size // 2, frame.shape[1]), min(
                    y1 + box_size // 2, frame.shape[0]
                )
                cv2.rectangle(frame, (x10, y10), (x11, y11), box_color, -1)
                label = part_labels[part1]
                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4
                )
                label_x, label_y = (
                    x1 - label_size[0] // 2,
                    y1 + box_size // 2 + label_size[1] // 2,
                )
                cv2.putText(
                    frame,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                for part2, coords2 in part_coords.items():
                    if part1 != part2:
                        x2, y2 = coords2
                        x20, y20 = max(x2 - box_size // 2, 0), max(
                            y2 - box_size // 2, 0
                        )
                        x21, y21 = min(x2 + box_size // 2, frame.shape[1]), min(
                            y2 + box_size // 2, frame.shape[0]
                        )
                        cv2.rectangle(frame, (x20, y20), (x21, y21), box_color, -1)

                        if (
                            x11 >= x20 and x10 <= x21 and y11 >= y20 and y10 <= y21
                        ):  # THIS IF STATEMENT MEANS THERES A COLLISION
                            overlapping_boxes.append((part1, part2))
                            cv2.putText(
                                frame,
                                "Overlap",
                                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,
                                (0, 0, 255),
                                4,
                                cv2.LINE_AA,
                            )
                            print(overlapping_boxes)
                            if state_collision == Collision.COLLISION:
                                if box_collided != overlapping_boxes:
                                    state_collision = Collision.NO_COLLISION

                            elif state_collision == Collision.NO_COLLISION:
                                if box_collided != overlapping_boxes:
                                    if len(box_collided) <= 2:
                                        collision_count += 1
                                box_collided = overlapping_boxes
                                state_collision = Collision.COLLISION

            # Print which two boxes have collided
            if len(box_collided) > 0:
                for parts in overlapping_boxes:
                    part1, part2 = parts
                    text = f"Overlap: {part1}, {part2}"
                    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                    cv2.rectangle(
                        frame,
                        (0, 0),
                        (text_size[0] + 10, text_size[1] + 10),
                        (0, 0, 255),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        text,
                        (10, text_size[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.putText(
                frame,
                f"Collisions: {collision_count}",
                (frame.shape[1] - 300, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Convert the frame back to BGR color space for displaying the original colors
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break


# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
    vs.stop()

# otherwise, release the camera
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()
