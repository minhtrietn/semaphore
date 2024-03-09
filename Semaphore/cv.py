import cv2
import mediapipe
import numpy as np
from math import degrees, atan2

current_semaphore = ''
last_frames = FRAME_HISTORY * [empty_frame.copy()]
last_keys = []


# Hàm tính toán góc độ từ vị trí của cánh tay
def angle_between(p1, p2, p3, closest_degrees=45):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    deg1 = (360 + degrees(atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + degrees(atan2(x3 - x2, y3 - y2))) % 360

    angle = deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

    if angle > 180:
        angle -= 180

    mod_close = angle % closest_degrees
    angle -= mod_close
    if mod_close > closest_degrees / 2:
        angle += closest_degrees

    if angle == 270:
        angle = -90

    return angle


def type_semaphore(armL_angle, armR_angle, image, numerals, allow_repeat):
    global current_semaphore

    arm_match = SEMAPHORES.get((armL_angle, armR_angle), '')
    if arm_match:
        current_semaphore = arm_match.get('n', '') if numerals else arm_match.get('a', '')
        type_and_remember(image, allow_repeat)
        return current_semaphore

    return False


semaphore = {
    "A": (225, -90),
    "B": (180, -90),
    "C": (135, -90),
    "D": (90, -90),
    "E": (-90, 45),
    "F": (-90, 0),
    "G": (-90, -45),
    "H": (180, 225),
    "I": (135, 225),
    "J": (90, 0),
    "K": (225, 90),
    "L": (225, 45),
    "M": (225, 0),
    "N": (225, -45),
    "O": (180, -135),
    "P": (180, 90),
    "Q": (180, 45),
    "R": (180, 0),
    "S": (180, -45),
    "T": (135, 90),
    "U": (135, 45),
    "V": (90, -45),
    "W": (45, 0),
    "X": (45, -45),
    "Y": (135, 0),
    "Z": (0, -45),
    " ": (-90, -90)}


# Hàm chạy chương trình chính
def main():
    # Khởi tạo camera hoặc đọc video
    cap = cv2.VideoCapture("semaphore.mp4")  # Sử dụng camera mặc định (index 0)

    with mediapipe.solutions.pose.Pose() as pose_model:
        image = cv2.imread("exemple2.PNG")
        # while True:
        # Đọc frame từ camera hoặc video
        # ret, image = cap.read()
        # if not ret:
        # break

        pose_results = pose_model.process(image)

        mediapipe.solutions.drawing_utils.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mediapipe.solutions.pose.POSE_CONNECTIONS,
            mediapipe.solutions.drawing_styles.get_default_pose_landmarks_style())

        # Xử lý frame để nhận dạng và tính toán góc độ của cánh tay
        # (ví dụ: sử dụng các kỹ thuật như xử lý ảnh, phát hiện đối tượng, v.v.)
        # Trong phần này, bạn cần thực hiện xử lý hình ảnh để nhận dạng vị trí của cánh tay

        body = []
        for point in pose_results.pose_landmarks.landmark:
            body.append({
                'x': 1 - point.x,
                'y': 1 - point.y,
                'visibility': point.visibility
            })

        print(body[11]["visibility"])

        shoulderL, elbowL, wristL = (body[11]["x"], body[11]["y"]), (body[13]["x"], body[13]["y"]), (
        body[15]["x"], body[15]["y"])
        shoulderR, elbowR, wristR = (body[12]["x"], body[12]["y"]), (body[14]["x"], body[14]["y"]), (
        body[16]["x"], body[16]["y"])
        armL = shoulderL, elbowL, wristL
        armR = shoulderR, elbowR, wristR

        # Tính toán góc độ từ vị trí của cánh tay
        angle = [angle_between(shoulderR, shoulderL, elbowL), angle_between(shoulderL, shoulderR, elbowR)]
        print(collinear(shoulderR, shoulderL, elbowL, wristL))
        print(collinear(shoulderL, shoulderR, elbowR, wristR))

        # Hiển thị góc độ trên frame
        cv2.putText(image, "ArmLeft Angle: {:.2f} degrees".format(angle[0]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        cv2.putText(image, "ArmRight Angle: {:.2f} degrees".format(angle[1]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0), 2)

        while True:
            # Hiển thị frame
            cv2.imshow('Arm Angle Detection', image)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == 27:
                break

    # Giải phóng camera hoặc video
    cap.release()
    cv2.destroyAllWindows()


# Gọi hàm chạy chương trình chính
if __name__ == "__main__":
    main()
