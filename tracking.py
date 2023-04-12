import cv2
import mediapipe as mpb
import numpy as np
import math
import time



class ObjectTracker:
    def __init__(self, video_path, tracker_type=cv2.TrackerCSRT_create, min_detection_confidence=0.8,min_tracking_confidence=0.8):
        self.video = cv2.VideoCapture(video_path)
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        if self.fps == 0: self.fps = 30
        self.delay = int(1000 / self.fps)
        self.tracker_type = tracker_type
        self.tracker = None
        self.bbox = None

        self.pose = mpb.solutions.pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
        self.box_size = 20
        self.box_color = (255, 255, 255)
        self.part_labels = {'right_shoulder': 'RS', 'left_shoulder': 'LS', 'head': 'H', 'right_foot': 'RF',
                        'left_foot': 'LF', 'right_wrist': 'RH', 'left_hand': 'LH', 'right_knee': 'RK',
                        'left_knee': 'LK'}

        self.num_kickups = 0
        self.prev_center_y = 0


    def draw_fps(self, frame):
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


    @staticmethod
    def check_collision(rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)
        return results


    def show_video(self):
        self.prev_time = time.time()

        while True:
            k, frame = self.video.read()

            if not k:
                break

            self.draw_fps(frame)
            cv2.imshow("Tracking_show_video_first_one", frame)
            k = cv2.waitKey(self.delay) & 0xFF
            if k == ord("r"):
                break

        self.bbox = cv2.selectROI(frame, False)
        self.tracker = self.tracker_type()
        ok = self.tracker.init(frame, self.bbox)
        cv2.destroyWindow("ROI selector")


    def track_object(self):
        self.prev_time = time.time()
        # initialize is_falling to False
        is_falling = False

        while True:
            ok, frame = self.video.read()
            if not ok:
                break

            found, bbox = self.tracker.update(frame)

            if found:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 2) #rectangle

            else:
                cv2.putText(
                    frame,
                    "Object no longer found",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

             # Convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame
            results = self.pose.process(frame)

            # Check if pose detection succeeded
            if results.pose_landmarks is not None:

                # Get keypoint locations
                keypoint_dict = {}
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    keypoint_dict[mpb.solutions.pose.PoseLandmark(idx).name] = (landmark.x, landmark.y)

                # Extract keypoint coordinates
                # right_shoulder = (int(keypoint_dict['RIGHT_SHOULDER'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_SHOULDER'][1] * frame.shape[0]))
                # left_shoulder = (int(keypoint_dict['LEFT_SHOULDER'][0] * frame.shape[1]), int(keypoint_dict['LEFT_SHOULDER'][1] * frame.shape[0]))
                # head = (int(keypoint_dict['NOSE'][0] * frame.shape[1]), int(keypoint_dict['NOSE'][1] * frame.shape[0]))
                right_foot = (int(keypoint_dict['RIGHT_ANKLE'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_ANKLE'][1] * frame.shape[0]))
                left_foot = (int(keypoint_dict['LEFT_ANKLE'][0] * frame.shape[1]), int(keypoint_dict['LEFT_ANKLE'][1] * frame.shape[0]))
                # right_wrist = (int(keypoint_dict['RIGHT_WRIST'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_WRIST'][1] * frame.shape[0]))
                # left_hand = (int(keypoint_dict['LEFT_WRIST'][0] * frame.shape[1]), int(keypoint_dict['LEFT_WRIST'][1] * frame.shape[0]))
                # right_knee = (int(keypoint_dict['RIGHT_KNEE'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_KNEE'][1] * frame.shape[0]))
                # left_knee = (int(keypoint_dict['LEFT_KNEE'][0] * frame.shape[1]), int(keypoint_dict['LEFT_KNEE'][1] * frame.shape[0]))

                # Draw circles around the keypoints
                for keypoint in keypoint_dict.values():
                    x, y = int(keypoint[0] * frame.shape[1]), int(keypoint[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

                # Draw lines between the keypoints
                # cv2.line(frame, right_shoulder, left_shoulder, (255, 0, 0), 2)
                # cv2.line(frame, right_shoulder, head, (255, 0, 0), 2)
                # cv2.line(frame, left_shoulder, head, (255, 0, 0), 2)
                # cv2.line(frame, right_foot, right_shoulder, (255, 0, 0), 2)
                # cv2.line(frame, left_foot, left_shoulder, (255, 0, 0), 2)
                # cv2.line(frame, right_wrist, right_shoulder, (255, 0, 0), 2)
                # cv2.line(frame, left_hand, left_shoulder, (255, 0, 0), 2)
                # cv2.line(frame, right_knee, right_foot, (255, 0, 0), 2)
                # cv2.line(frame, left_knee, left_foot, (255, 0, 0), 2)

                # Add boxes with part labels
                # for part, coords in {'right_shoulder': right_shoulder, 'left_shoulder': left_shoulder, 'head': head, 'right_foot': right_foot, 'left_foot': left_foot, 'right_knee': right_knee, 'left_knee': left_knee}.items():
                for part, coords in {'right_foot': right_foot, 'left_foot': left_foot,}.items():

                    x, y = coords
                    x0, y0 = max(x - self.box_size // 2, 0), max(y - self.box_size // 2, 0)
                    x1, y1 = min(x + self.box_size // 2, frame.shape[1]), min(y + self.box_size // 2, frame.shape[0])
                    cv2.rectangle(frame, (x0, y0), (x1, y1), self.box_color, -1) #limb
                    label = self.part_labels[part]
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 4)
                    label_x, label_y = x - label_size[0] // 2, y + self.box_size // 2 + label_size[1] // 2
                    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)



                   # check if the ball is falling or not
                if found:
                    x, y, w, h = bbox
                    center_y = y + h // 2
                    if center_y > self.prev_center_y:
                        is_falling = True
                    else:
                        is_falling = False
                    self.prev_center_y = center_y



                # Add a list to keep track of limbs that have already collided with the ball in the current kickup attempt
                current_collision_lims = []


                # # Check for collisions between the ball and the limbs
                # collision_detected = False

                # for limb_name, limb_bbox in {'right_foot': right_foot, 'left_foot': left_foot,}.items():
                #     x0, y0 = limb_bbox
                #     x1, y1 = x0 + self.box_size, y0 + self.box_size
                #     limb_bbox = (x0, y0, x1 - x0, y1 - y0)

                #     if self.check_collision(bbox, limb_bbox):
                #         print(f"Collision detected with {limb_name}")
                #         collision_detected = True
                #         self.num_kickups += 1
                #         current_collision_lims.append(limb_name)
                #         break

                # if not collision_detected:
                #     # print("No collision")
                #     current_collision_lims = []



                # Check for collisions between the ball and the limbs
                collision_detected = False

                for keypoint_name, keypoint_coords in keypoint_dict.items():
                    # Calculate the distance between the keypoint and the center of the ball
                    dist = np.sqrt((bbox[0] + bbox[2] / 2 - keypoint_coords[0] * frame.shape[1])**2 + (bbox[1] + bbox[3] / 2 - keypoint_coords[1] * frame.shape[0])**2)

                    # Calculate the sum of the radii of the two circles
                    radius_sum = self.box_size / 2 + 4  # 4 is the radius of the circle drawn around the keypoint

                    # Check if there is a collision
                    if keypoint_name in ["LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]:
                        if dist < radius_sum:
                            print(f"Collision detected with {keypoint_name}")
                            collision_detected = True
                            self.num_kickups += 1
                            break

                if not collision_detected:
                    # print("No collision")
                    pass




            # Convert the frame back to BGR color space for displaying the original colors
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


            self.draw_fps(frame)
            cv2.putText(frame, f"Kickups: {self.num_kickups}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


            cv2.imshow("Tracking", frame)
            k = cv2.waitKey(self.delay) & 0xFF
            if k == 27:
                break
            elif k == ord("r"):
                self.bbox = cv2.selectROI(frame, False)
                self.tracker = self.tracker_type()
                ok = self.tracker.init(frame, self.bbox)
                cv2.destroyWindow("ROI selector")

        self.video.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    tracker = ObjectTracker("foot_3.mp4")
    # tracker = ObjectTracker(0)
    tracker.show_video()
    tracker.track_object()
