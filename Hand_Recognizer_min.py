import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)
mp_hand=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hand.Hands()

ptime=0
ctime=0

while True:
    isTrue,frame=cap.read()
    img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    result=hands.process(img)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            for id,lm in enumerate(hand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                if id==8:
                    cv.circle(frame,(cx,cy),15,(200,0,0),cv.FILLED)
            mp_draw.draw_landmarks(frame,hand,mp_hand.HAND_CONNECTIONS)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    frame=cv.flip(frame,1)
    cv.putText(frame,
               str(int(fps)),
               (10, 70),
               cv.FONT_HERSHEY_SIMPLEX,
               3,
               (200,0, 90),
               3)
    cv.imshow("hand",frame)
    if cv.waitKey(10) & 0xFF==27:
        break

cap.release()

