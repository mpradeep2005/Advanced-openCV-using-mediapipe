import cv2 as cv
import mediapipe as mp
import time

class detector():
    def __init__(self,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_pose=mp.solutions.pose
        self.poses=self.mp_pose.Pose(self.static_image_mode,
                                        self.model_complexity,
                                        self.smooth_landmarks,
                                        self.enable_segmentation,
                                        self.smooth_segmentation,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.mp_draw=mp.solutions.drawing_utils

    def find_pose(self,img,draw=True):

        img_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)

        self.result=self.poses.process(img_rgb)
        if self.result.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img, self.result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def draw_con(self,img,draw=True):
        lis=[]
        if self.result.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            for id,lm in enumerate(self.result.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lis.append([id,cx,cy])
                #if id==0:
                    #cv.circle(img,(cx,cy),10,(0,100,200),-1)
            return lis



def main():
    vid=cv.VideoCapture(0)
    ptime=0
    dec=detector()
    while vid.isOpened():
        isTrue,img=vid.read()
        img = cv.resize(img, (640, 480))
        img=dec.find_pose(img)
        lis=dec.draw_con(img,False)
        if len(lis)!=0:
            print(lis[14])
            cv.circle(img,(lis[14][1],lis[14][2]),20,(250,0,0),-1)

        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv.putText(img,str(int(fps)),(10,70),cv.FONT_HERSHEY_SIMPLEX,3,(200,0,200),3)
        cv.imshow("pose",img)
        cv.waitKey(10)

if __name__=="__main__":
    main()
