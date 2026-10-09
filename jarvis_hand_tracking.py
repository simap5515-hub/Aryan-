import cv2
import mediapipe as mp
import math
import time

# =========================
# JARVIS HAND TRACKING
# =========================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera open nahi ho raha.")
    exit()

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def distance(a, b):
    return math.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )

def fingers_up(hand):
    """
    Returns:
    [thumb, index, middle, ring, pinky]
    """

    lm = hand.landmark
    result = []

    # Thumb
    if lm[4].x < lm[3].x:
        result.append(1)
    else:
        result.append(0)

    # Other fingers
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        if lm[tip].y < lm[pip].y:
            result.append(1)
        else:
            result.append(0)

    return result


while True:

    success, frame = cap.read()

    if not success:
        continue

    # Mirror camera
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    status = "SEARCHING..."

    if results.multi_hand_landmarks:

        status = "HAND DETECTED"

        for hand in results.multi_hand_landmarks:

            # Draw hand skeleton
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(
                    color=(255, 180, 0),
                    thickness=2,
                    circle_radius=3
                ),
                mp_draw.DrawingSpec(
                    color=(0, 255, 255),
                    thickness=2
                )
            )

            lm = hand.landmark

            # Finger tips
            tips = [4, 8, 12, 16, 20]

            for tip in tips:

                x = int(lm[tip].x * frame.shape[1])
                y = int(lm[tip].y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (255, 0, 255),
                    -1
                )

                cv2.circle(
                    frame,
                    (x, y),
                    13,
                    (255, 255, 255),
                    1
                )

            # Finger state
            fingers = fingers_up(hand)

            total = sum(fingers)

            if total == 5:
                gesture = "OPEN PALM"

            elif total == 0:
                gesture = "FIST"

            elif fingers[1] == 1 and total == 1:
                gesture = "SELECT / POINT"

            elif fingers[1] == 1 and fingers[2] == 1 and total == 2:
                gesture = "TWO FINGER MODE"

            elif fingers[0] == 1 and total == 1:
                gesture = "THUMBS UP"

            else:
                gesture = "GESTURE DETECTED"

            # Display gesture
            cv2.putText(
                frame,
                gesture,
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 220, 0),
                2
            )

    # =========================
    # JARVIS UI
    # =========================

    h, w = frame.shape[:2]

    # Top line
    cv2.line(
        frame,
        (30, 35),
        (350, 35),
        (255, 180, 0),
        2
    )

    cv2.putText(
        frame,
        "J.A.R.V.I.S",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 220, 0),
        2
    )

    # Status
    cv2.putText(
        frame,
        "STATUS: " + status,
        (30, h - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Crosshair
    cx = w // 2
    cy = h // 2

    cv2.line(
        frame,
        (cx - 25, cy),
        (cx + 25, cy),
        (255, 180, 0),
        1
    )

    cv2.line(
        frame,
        (cx, cy - 25),
        (cx, cy + 25),
        (255, 180, 0),
        1
    )

    # FPS
    cv2.putText(
        frame,
        "HAND TRACKING ONLINE",
        (w - 350, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )

    cv2.imshow("JARVIS - Hand Tracking System", frame)

    # Q / ESC = exit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break


cap.release()
cv2.destroyAllWindows()
hands.close()
