from datetime import datetime
from math import atan2, degrees, sqrt, acos

import cv2
import mediapipe as mp
from scipy.spatial import distance as dist

DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()
DEFAULT_HAND_CONNECTIONS_STYLE = mp.solutions.drawing_styles.get_default_hand_connections_style()

RECORDING_FILENAME = str(datetime.now()).replace('.', '').replace(':', '') + '.avi'
FPS = 24

VISIBILITY_THRESHOLD = .8
STRAIGHT_LIMB_MARGIN = 20
EXTENDED_LIMB_MARGIN = .8

SEMAPHORES = {
    (-45, 0): {'a': "a", 'n': "1"},
    (-90, 0): {'a': "b", 'n': "2"},
    (-135, 0): {'a': "c", 'n': "3"},
    (180, 0): {'a': "d", 'n': "4"},
    (0, 135): {'a': "e", 'n': "5"},
    (0, 90): {'a': "f", 'n': "6"},
    (0, 45): {'a': "g", 'n': "7"},
    (-90, -45): {'a': "h", 'n': "8"},
    (-135, -45): {'a': "i", 'n': "9"},
    (180, 90): {'a': "j"},
    (-45, 180): {'a': "k", 'n': "0"},
    (-45, 135): {'a': "l"},
    (-45, 90): {'a': "m"},
    (-45, 45): {'a': "n"},
    (-135, -90): {'a': "o"},
    (-90, 180): {'a': "p"},
    (-90, 135): {'a': "q"},
    (-90, 90): {'a': "r"},
    (-90, 45): {'a': "s"},
    (180, -135): {'a': "t"},
    (-135, 135): {'a': "u"},
    (180, 45): {'a': "v"},
    (135, 90): {'a': "w"},
    (45, 135): {'a': "x"},
    (-135, 90): {'a': "y"},
    (45, 90): {'a': "z"},
}

FRAME_HISTORY = 5

empty_frame = {
    'signed': False,
}
last_frames = FRAME_HISTORY * [empty_frame.copy()]
temp_keys = []

frame_midpoint = (0, 0)

current_semaphore = ''
last_keys = []
delay_time = 1


def get_angle(a, b, c):
    ang = degrees(atan2(c['y'] - b['y'], c['x'] - b['x']) - atan2(a['y'] - b['y'], a['x'] - b['x']))
    return ang + 360 if ang < 0 else ang


def is_missing(part):
    return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)


def is_limb_pointing(upper, mid, lower):
    if is_missing([upper, mid, lower]):
        return False
    limb_angle = get_angle(upper, mid, lower)
    is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
    if is_in_line:
        upper_length = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
        lower_length = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
        # print(lower_length, upper_length * EXTENDED_LIMB_MARGIN)
        is_extended = lower_length > EXTENDED_LIMB_MARGIN * upper_length
        return is_extended
    return False


def get_limb_direction(p1, p2, p3, closest_degrees=45):
    angle = get_angle(p1, p2, p3)

    mod_close = angle % closest_degrees
    if mod_close < closest_degrees / 2:
        angle -= mod_close
    else:
        angle += closest_degrees - mod_close

    angle = int(angle)

    if angle > 180:
        angle = -(360 - angle)

    return angle


def type_semaphore(armL_angle, armR_angle, image, numerals=False, allow_repeat=None):
    global current_semaphore

    arm_match = SEMAPHORES.get((armR_angle, armL_angle), '') or SEMAPHORES.get((armL_angle, armR_angle), '')
    if arm_match:
        current_semaphore = arm_match.get('n', '') if numerals else arm_match.get('a', '')
        type_and_remember(image, allow_repeat)
        return current_semaphore

    return False


def type_and_remember(image=None, allow_repeat=False):
    global current_semaphore, last_keys

    if len(current_semaphore) == 0:
        return

    keys = [current_semaphore]

    if allow_repeat or (keys != last_keys):
        last_keys = keys.copy()
        current_semaphore = ''
        output(keys, image)


def get_key_text(keys):
    if not (len(keys) > 0):
        return ''

    semaphore = keys[-1]
    keystring = ''

    keystring += semaphore
    return keystring


def output(keys, image):
    keystring = '+'.join(keys)
    if len(keystring):
        to_display = get_key_text(keys)
        cv2.putText(image, to_display, frame_midpoint,
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 10)


def render_and_maybe_exit(image, recording):
    cv2.imshow('Semaphore', image)
    if recording:
        recording.write(image)
    return cv2.waitKey(5) & 0xFF == 27


def main():
    global last_frames, frame_midpoint

    INPUT = "2.mp4"
    FLIP = None
    DRAW_LANDMARKS = True
    RECORDING = None
    ALLOW_REPEAT = None

    cap = cv2.VideoCapture(INPUT)

    frame_size = (int(cap.get(3)), int(cap.get(4)))
    frame_midpoint = (int(frame_size[0] / 2), int(frame_size[1] / 2))

    recording = cv2.VideoWriter(RECORDING_FILENAME,
                                cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size) if RECORDING else None

    with mp.solutions.pose.Pose() as pose_model:
        with mp.solutions.hands.Hands(max_num_hands=2) as hands_model:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pose_results = pose_model.process(image)
                hand_results = hands_model.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # draw pose
                if DRAW_LANDMARKS:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image,
                        pose_results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        DEFAULT_LANDMARKS_STYLE)

                hands = []
                hand_index = 0
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # draw hands
                        if DRAW_LANDMARKS:
                            mp.solutions.drawing_utils.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp.solutions.hands.HAND_CONNECTIONS,
                                DEFAULT_LANDMARKS_STYLE,
                                DEFAULT_HAND_CONNECTIONS_STYLE)
                        hands.append([])
                        for point in hand_landmarks.landmark:
                            hands[hand_index].append({
                                'x': 1 - point.x,
                                'y': 1 - point.y
                            })
                        hand_index += 1

                if FLIP:
                    image = cv2.flip(image, 1)

                if pose_results.pose_landmarks:
                    last_frames = last_frames[1:] + [empty_frame.copy()]
                    if any(frame['signed'] for frame in last_frames):
                        if render_and_maybe_exit(image, recording):
                            break
                        else:
                            continue

                    body = []
                    for point in pose_results.pose_landmarks.landmark:
                        body.append({
                            'x': 1 - point.x,
                            'y': 1 - point.y,
                            'visibility': point.visibility
                        })

                    elbowL, shoulderL, hipL = body[15], body[11], body[23]
                    armL = (elbowL, shoulderL, hipL)

                    elbowR, shoulderR, hipR = body[16], body[12], body[24]
                    armR = (elbowR, shoulderR, hipR)

                    # print(get_limb_direction(*armR), get_limb_direction(*armL))
                    armL_angle = get_limb_direction(*armL)
                    armR_angle = get_limb_direction(*armR)

                    # print(get_key_text(last_keys))

                    if type_semaphore(armL_angle, armR_angle, image,
                                      False, ALLOW_REPEAT):
                        last_frames[-1]['signed'] = True

                        print(get_key_text(last_keys))
                        output(get_key_text(last_keys), image)

                if render_and_maybe_exit(image, recording):
                    break

    if RECORDING:
        recording.release()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
