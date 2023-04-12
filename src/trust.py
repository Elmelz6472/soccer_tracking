import cv2
import mediapipe as mpb


class PoseDetector:
    def __init__(self,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.pose = mpb.solutions.pose.Pose(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence)
        self.box_size = 50
        self.box_color = (255, 255, 255)
        self.part_labels = {'right_shoulder': 'RS', 'left_shoulder': 'LS', 'head': 'H', 'right_foot': 'RF',
                        'left_foot': 'LF', 'right_wrist': 'RH', 'left_hand': 'LH', 'right_knee': 'RK',
                        'left_knee': 'LK'}

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)
        return results

    def draw_keypoints(self, frame, results):
        if results.pose_landmarks is not None:
            keypoint_dict = {}
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoint_dict[mpb.solutions.pose.PoseLandmark(idx).name] = (landmark.x, landmark.y)

            # (The rest of the original while loop)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

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
                    right_shoulder = (int(keypoint_dict['RIGHT_SHOULDER'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_SHOULDER'][1] * frame.shape[0]))
                    left_shoulder = (int(keypoint_dict['LEFT_SHOULDER'][0] * frame.shape[1]), int(keypoint_dict['LEFT_SHOULDER'][1] * frame.shape[0]))
                    head = (int(keypoint_dict['NOSE'][0] * frame.shape[1]), int(keypoint_dict['NOSE'][1] * frame.shape[0]))
                    right_foot = (int(keypoint_dict['RIGHT_ANKLE'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_ANKLE'][1] * frame.shape[0]))
                    left_foot = (int(keypoint_dict['LEFT_ANKLE'][0] * frame.shape[1]), int(keypoint_dict['LEFT_ANKLE'][1] * frame.shape[0]))
                    right_wrist = (int(keypoint_dict['RIGHT_WRIST'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_WRIST'][1] * frame.shape[0]))
                    left_hand = (int(keypoint_dict['LEFT_WRIST'][0] * frame.shape[1]), int(keypoint_dict['LEFT_WRIST'][1] * frame.shape[0]))
                    right_knee = (int(keypoint_dict['RIGHT_KNEE'][0] * frame.shape[1]), int(keypoint_dict['RIGHT_KNEE'][1] * frame.shape[0]))
                    left_knee = (int(keypoint_dict['LEFT_KNEE'][0] * frame.shape[1]), int(keypoint_dict['LEFT_KNEE'][1] * frame.shape[0]))

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
                    cv2.line(frame, right_wrist, right_shoulder, (255, 0, 0), 2)
                    cv2.line(frame, left_hand, left_shoulder, (255, 0, 0), 2)
                    cv2.line(frame, right_knee, right_foot, (255, 0, 0), 2)
                    cv2.line(frame, left_knee, left_foot, (255, 0, 0), 2)

                    # Add boxes with part labels
                    for part, coords in {'right_shoulder': right_shoulder, 'left_shoulder': left_shoulder, 'head': head, 'right_foot': right_foot, 'left_foot': left_foot, 'right_wrist': right_wrist, 'left_hand': left_hand, 'right_knee': right_knee, 'left_knee': left_knee}.items():
                        x, y = coords
                        x0, y0 = max(x - self.box_size // 2, 0), max(y - self.box_size // 2, 0)
                        x1, y1 = min(x + self.box_size // 2, frame.shape[1]), min(y + self.box_size // 2, frame.shape[0])
                        cv2.rectangle(frame, (x0, y0), (x1, y1), self.box_color, -1)
                        label = self.part_labels[part]
                        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
                        label_x, label_y = x - label_size[0] // 2, y + self.box_size // 2 + label_size[1] // 2
                        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA)

                # Convert the frame back to BGR color space for displaying the original colors
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Display the frame
                cv2.imshow("MediaPipe Pose Demo", frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) == ord('q'):
                    break


            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame





if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    pose_detector = PoseDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_detector.process_frame(frame)
        frame_with_keypoints = pose_detector.draw_keypoints(frame, results)

        cv2.imshow("MediaPipe Pose Demo", frame_with_keypoints)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




