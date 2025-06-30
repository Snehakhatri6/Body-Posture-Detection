import cv2
import mediapipe as mp
import numpy as np
import time

import subprocess

def convert_to_browser_mp4(input_path, output_path):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vcodec', 'libx264', '-acodec', 'aac', '-strict', 'experimental', output_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Converted {input_path} to browser-compatible MP4: {output_path}")
    except Exception as e:
        print(f"ffmpeg conversion failed: {e}")

def analyze_squat(input_video_path, output_video_path):
    import cv2
    import mediapipe as mp
    import numpy as np
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        feedback = "No pose detected"
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            angle = np.degrees(np.arctan2(left_ankle.y - left_knee.y, left_ankle.x - left_knee.x) -
                               np.arctan2(left_hip.y - left_knee.y, left_hip.x - left_knee.x))
            angle = abs(angle)
            if 80 <= angle <= 100:
                feedback = "Correct squat"
            else:
                feedback = "Incorrect squat"
        # Draw feedback on frame
        cv2.rectangle(frame, (0, 0), (frame_width, 40), (255, 255, 255), -1)
        cv2.putText(frame, feedback, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if feedback != "Correct squat" else (0, 200, 0), 2)
        out.write(frame)
    cap.release()
    out.release()
    pose.close()

    print(f"Opening video: {input_video_path}")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video properties: width={frame_width}, height={frame_height}, fps={fps}")
    if fps == 0 or frame_width == 0 or frame_height == 0:
        raise ValueError("Invalid video file or properties: FPS, width, or height is zero.")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    def is_correct_squat(landmarks):
        if landmarks:
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            hip_knee_vec = np.array([left_knee.x - left_hip.x, left_knee.y - left_hip.y])
            knee_ankle_vec = np.array([left_ankle.x - left_knee.x, left_ankle.y - left_knee.y])
            dot_product = np.dot(hip_knee_vec, knee_ankle_vec)
            mag_hip_knee = np.linalg.norm(hip_knee_vec)
            mag_knee_ankle = np.linalg.norm(knee_ankle_vec)
            angle = np.arccos(dot_product / (mag_hip_knee * mag_knee_ankle)) * 180.0 / np.pi
            if 90 <= angle <= 130:
                return True
        return False

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break
        frame_count += 1
        # Flip frame for correct view (optional)
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            print(f"Frame {frame_count}: Pose detected!")
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            if is_correct_squat(results.pose_landmarks.landmark):
                cv2.putText(frame, 'Correct Squat', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Incorrect Squat', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            print(f"Frame {frame_count}: No pose detected.")
        # Only write the frame with squat feedback, no debug overlays
        out.write(frame)
    print(f"Total frames processed: {frame_count}")
    cap.release()
    out.release()
    pose.close()
    return output_video_path

    def is_correct_squat(landmarks):
        if landmarks:
            # Get landmarks for left leg (Hip, Knee, Ankle)
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]

            # Calculate the angle using cosine rule or simple vector math
            hip_knee_vec = np.array([left_knee.x - left_hip.x, left_knee.y - left_hip.y])
            knee_ankle_vec = np.array([left_ankle.x - left_knee.x, left_ankle.y - left_knee.y])

            dot_product = np.dot(hip_knee_vec, knee_ankle_vec)
            mag_hip_knee = np.linalg.norm(hip_knee_vec)
            mag_knee_ankle = np.linalg.norm(knee_ankle_vec)

            angle = np.arccos(dot_product / (mag_hip_knee * mag_knee_ankle)) * 180.0 / np.pi

        # Calculate the angle using cosine rule or simple vector math
        hip_knee_vec = np.array([left_knee.x - left_hip.x, left_knee.y - left_hip.y])
        knee_ankle_vec = np.array([left_ankle.x - left_knee.x, left_ankle.y - left_knee.y])

        dot_product = np.dot(hip_knee_vec, knee_ankle_vec)
        mag_hip_knee = np.linalg.norm(hip_knee_vec)
        mag_knee_ankle = np.linalg.norm(knee_ankle_vec)

        angle = np.arccos(dot_product / (mag_hip_knee * mag_knee_ankle)) * 180.0 / np.pi

        # Check if the angle is between 90° and 130° for correct squat posture
        if 90 <= angle <= 130:
            return True
    return False
