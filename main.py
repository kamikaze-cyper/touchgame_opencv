import mediapipe as mp
import cv2
from mediapipe.framework.formats import landmark_pb2
import numpy
import random
import time



mediapipe_drawing = mp.solutions.drawing_utils
mediapipe_hands = mp.solutions.hands 
score = 0


x_enemy = random.randint(50,600)
y_enemy = random.randint(50,400)


def enemy():
    global score,x_enemy,y_enemy
    cv2.circle(image,(x_enemy,y_enemy),25,(0,200,0), 5)
    
    
video = cv2.VideoCapture(0)



with mediapipe_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5) as hands:
    _,frame = video.read()
    image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image = cv2.flip(image,1)
    
    imageHeight,imageWidth,_=image.shape
    results = hands.process(image)
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    font= cv2.FONT_HERSHEY_SIMPLEX
    color =(255,0,255)
    text = cv2.putText(image,"score",(480,30),font,1,color,4,cv2.LINE_AA)
    text = cv2.putText(image,str(score),(590,30),font,1,color,4,cv2.LINE_AA)
    
    
    enemy()
    
    if results.multi_hand_landmarks :
        for num,hand in enumerate(results.multi_hand_landmarks):
            mediapipe_drawing.draw_landmarks(image,hand,mediapipe_hands.HAND_CONNECTIONS,
                                             mediapipe_drawing.DrawingSpec(color=(250,44,250),thickness=2,circle_raidus=2),)
            
    if results.multi_hand_landmarks !=None:
        for handLandmarks in results.multi_hand_landmarks:
            for point in mediapipe_hands.HandLandmark:
                normalizedLandmark = handLandmarks.landmark[point]
                pixelCoordinatesLandmark=mediapipe_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x,normalizedLandmark.y,imageWidth,imageHeight)
                try:
                    cv2.circle(image,(pixelCoordinatesLandmark[0],pixelCoordinatesLandmark[1]),25,(0,200,0),5)
                    if pixelCoordinatesLandmark[0]==x_enemy or pixelCoordinatesLandmark[0] ==x_enemy+10 or pixelCoordinatesLandmark[0]==x_enemy-10:
                        x_enemy = random.randint(50,600)
                        y_enemy= random.randint(50,400)
                        score = score=+1
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = cv2.putText(frame,"Score",(100,100),font,1,color,4,cv2.LINE_AA)
                        enemy()
                except:
                    pass
                        
            cv2.imshow('Hand Tracking',image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print(score)
                break
            
                
                
    video.release()
    cv2.destroyAllWindows()            
            




