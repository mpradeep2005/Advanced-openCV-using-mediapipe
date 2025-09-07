import cv2 as cv
import mediapipe as mp
import time

mp_pose=mp.solutions.pose
poses=mp_pose.Pose()

mp_draw=mp.solutions.drawing_utils

vid=cv.VideoCapture("E:/Advanced OpenCV/res/vid5.mp4")
ptime=0

while vid.isOpened():
    isTrue,frame=vid.read()
    frame = cv.resize(frame, (640, 480))
    img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    result=poses.process(img)
    print(result.pose_landmarks)
    """ unlike the hand there is no multi hand so no need for loops"""
    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id,lm in enumerate(result.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            if id==0:
                cv.circle(frame,(cx,cy),10,(0,100,200),-1)
    ctime=time.time()
    fps=1/(ctime-ptime)
    ptime=ctime

    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(200,0,200),3)

    cv.imshow("pose",frame)

    cv.waitKey(10)