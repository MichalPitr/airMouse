import cv2
import mediapipe as mp
import pyautogui
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles


class Airpad:
    def __init__(self):
        self.width = 640
        self.height = 480
        self.min_x = 60
        self.max_x = 540
        self.min_y = 60
        self.max_y = 410
        self.monitor_width = 1920
        self.monitor_height = 1080

    def inside(self, x, y):
        if self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y:
            return True
        else:
            return False

    def rescale(self, x, y, curr_width, curr_height, target_width, target_height):
        return int((x / curr_width) * target_width), int((y / curr_height) * target_height)

    def scale_to_monitor(self, x, y, target_width, target_height):
        return int(((x - self.min_x) / (self.max_x - self.min_x)) * target_width), int((
                    (y - self.min_y) / (self.max_y - self.min_y)) * target_height)


class CoordinateCircularBuffer:
    def __init__(self, length):
        self.buffer = [None]*length
        self.idx = 0
        self.length = length

    def append(self, item):
        self.buffer[self.idx] = item
        self.idx = (self.idx + 1) % self.length

    def mean(self):
        i = 0
        x = 0
        y = 0
        while i < self.length and self.buffer[i] is not None:
            x += self.buffer[i][0]
            y += self.buffer[i][1]
            i += 1
        if i == 0:
            return 0, 0
        else:
            return x/(i), y/(i)

    def reset(self):
        self.buffer = [None]*self.length
        self.idx = 0


airpad = Airpad()
position_buffer = CoordinateCircularBuffer(2)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    drawing_styles.get_default_hand_landmark_style(),
                    drawing_styles.get_default_hand_connection_style())
                x = hand_landmarks.landmark[8].x
                y = hand_landmarks.landmark[8].y
                index_x, index_y = airpad.rescale(x, y, 1, 1, 640, 480)
                if airpad.inside(index_x, index_y):
                    x, y = airpad.scale_to_monitor(index_x, index_y, 1920, 1080)
                    position_buffer.append([x, y])
                    x, y = position_buffer.mean()
                    # pyautogui.moveTo(x, y, _pause=False)
                    thumb_x, thumb_y = hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y

                    thumb_x, thumb_y = airpad.rescale(thumb_x, thumb_y, 1, 1, 640, 480)
                    distance_thumb_index = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)**0.5
                    if distance_thumb_index < 20:
                        cv2.circle(image, (thumb_x, thumb_y), 20, (0, 255, 0))
                        # pyautogui.click(_pause=False)
                        pyautogui.click(x, y, button='left', _pause=False)

                    else:
                        pyautogui.moveTo(x, y, 0.01, _pause=False)
                        cv2.circle(image, (thumb_x, thumb_y), 20, (0, 0, 255))
                    cv2.circle(image, (index_x, index_y), 10, (0, 0, 255))

                else:
                    position_buffer.reset()

        cv2.rectangle(image, (60, 60), (540, 410), (255, 0, 0), 2)
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
