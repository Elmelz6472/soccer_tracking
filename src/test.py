import cv2
import mediapipe as mpb

class PoseDetector:
    def __init__(self):
        self.mp_pose = mpb.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.part_labels = {'right_shoulder': 'RS', 'left_shoulder': 'LS', 'head': 'H', 'right_foot': 'RF', 'left_foot': 'LF', 'right_wrist': 'RH', 'left_hand': 'LH', 'right_knee': 'RK', 'left_knee': 'LK'}

    def process_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame)
        return results.pose_landmarks

    def draw_pose(self, frame, landmarks):
        keypoint_dict = {self.mp_pose.PoseLandmark(idx).name: (landmark.x, landmark.y) for idx, landmark in enumerate(landmarks.landmark)}

        for part, label in self.part_labels.items():
            x, y = int(keypoint_dict[part][0] * frame.shape[1]), int(keypoint_dict[part][1] * frame.shape[0])
            cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)

        for part1, part2 in [('right_shoulder', 'left_shoulder'), ('right_shoulder', 'head'), ('left_shoulder', 'head'),
                             ('right_foot', 'right_shoulder'), ('left_foot', 'left_shoulder'), ('right_wrist', 'right_shoulder'),
                             ('left_hand', 'left_shoulder'), ('right_knee', 'right_foot'), ('left_knee', 'left_foot')]:
            cv2.line(frame, (int(keypoint_dict[part1][0] * frame.shape[1]), int(keypoint_dict[part1][1] * frame.shape[0])),
                     (int(keypoint_dict[part2][0] * frame.shape[1]), int(keypoint_dict[part2][1] * frame.shape[0])), (255, 0, 0), 2)
        return frame

def main():
    pose_detector = PoseDetector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = pose_detector.process_frame(frame)
        if landmarks is not None:
            frame = pose_detector.draw_pose(frame, landmarks)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("MediaPipe Pose Demo", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
