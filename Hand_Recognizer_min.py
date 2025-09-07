import cv2 as cv
import mediapipe as mp

cap=cv.VideoCapture(0)
mp_hand=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hand.Hands()

while True:
    isTrue,frame=cap.read()
    img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result=hands.process(img)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,hand,mp_hand.HAND_CONNECTIONS)

    cv.imshow("hand",cv.flip(frame,1))
    if cv.waitKey(10) & 0xFF==ord('q'):
        break

cap.release()

