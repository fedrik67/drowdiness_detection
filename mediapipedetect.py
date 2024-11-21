# This file is written in python. This code records the video and detects the face and marks
#eyes and mouth parts. Further it uses a pretrained model to check drowsiness by detecting 
#eye states and yawning.



import time
import cv2 as cv
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist 
from keras.models import load_model
from tensorflow.keras.utils import img_to_array
import matplotlib.pyplot as plt
from playsound import playsound
from threading import Thread

# load model
model = load_model("C:/Users/fspar/Downloads/driver_state_original_mixed_wd.h5")
# load alarm_sound file
alarm_sound = "C:/Users/fspar/Downloads/Drowsiness detection/Drowsiness detection/alarm.wav"
# mediapipe detects face
mp_face_detection = mp.solutions.face_detection
count = 0 # count for eye closure time
alarm_on = False # initial state of alarm
def start_alarm(sound):#sound
    playsound(sound)

cap = cv.VideoCapture(0) #video capture from webcam
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detector:
    frame_counter = 0
    fonts = cv.FONT_HERSHEY_PLAIN
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = cap.read()# reading image
        if ret is False:
            break
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)# forming color frame

        results = face_detector.process(rgb_frame)
        frame_height, frame_width, c = frame.shape
        
        if results.detections:
            for face in results.detections:
                x = face.location_data.relative_bounding_box.xmin
                y = face.location_data.relative_bounding_box.ymin
                w = face.location_data.relative_bounding_box.width
                h = face.location_data.relative_bounding_box.height
                face_react = np.multiply(
                    [x, y, w, h],
                    [frame_width, frame_height, frame_width, frame_height]).astype(int)# calculating coordinates of frame
                draw_points = []
                cv.rectangle(frame, face_react, color=(255, 255, 255), thickness=2)# drawing rectangle on face
                key_points = np.array([(p.x, p.y) for p in face.location_data.relative_keypoints])
                #key_points are two eyes , two ears. nose, mouth
                #distance between two eye middle pts
                distance = dist.euclidean(key_points[1],key_points[0])
                
                v = [(distance/2),(distance/2)]
                # mouth detection points
                a_point = [x, key_points[2][1]]
                b_point =[x + w, y + h]
                #left eye points
                draw_points.append(key_points[0] + v)
                draw_points.append(key_points[0] - v)
                #right eye pts
                draw_points.append(key_points[1] + v)
                draw_points.append(key_points[1] - v)
                
                draw_points.append(a_point)
                draw_points.append(b_point)
                # finding the coordinates of above pts
                key_points_coords = np.multiply(draw_points,[frame_width,frame_height]).astype(int)
                #drawing rectangles on thse coordinates
                cv.rectangle(frame, key_points_coords[0],key_points_coords[1], (255, 0, 0), 2)
                cv.rectangle(frame, key_points_coords[2],key_points_coords[3], (255, 0, 0), 2)
                cv.rectangle(frame, key_points_coords[4],key_points_coords[5], (255, 0, 0), 2)
                
                #left eye
                eye1 = frame[key_points_coords[1][1]:key_points_coords[0][1], 
                                 key_points_coords[1][0]:key_points_coords[0][0]]
                
                
                eye1 = cv.resize(eye1, (140, 140))
                
                eye1 = img_to_array(eye1)
                
                eye1 = np.expand_dims(eye1, axis=0)
                x_left = (model.predict(eye1))
                pred1 = np.argmax(x_left)
                
                #right eye
                eye2 = frame[key_points_coords[3][1]:key_points_coords[2][1], 
                                 key_points_coords[3][0]:key_points_coords[2][0]]
                
                eye2 = cv.resize(eye2, (140, 140))
               
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                x_right = (model.predict(eye2))
                pred2 = np.argmax(x_right)
                
                
                # mouth = frame[key_points_coords[4][1]:key_points_coords[5][1], 
                #                   key_points_coords[4][0]:key_points_coords[5][0]]# points from nose to underneath
                mouth = frame[face_react[1]:face_react[1] + face_react[3],
                               face_react[0]:face_react[0]+face_react[2]]# full face
                
                mouth = cv.resize(mouth, (140, 140))
               
                mouth = img_to_array(mouth)
                mouth = np.expand_dims(mouth, axis=0)
                x_mouth = (model.predict(mouth))
                pred3 = np.argmax(x_mouth)
                
                # alert system
                if pred1 == 3 and pred2 == 3:# if eyes are closed
                    count += 1
                    cv.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    # if eyes are closed for 10 consecutive frames, start the alarm

                    if count >= 5:
                        cv.putText(frame, "Drowsiness Alert!!!", (100, face_react[3]-20), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        if not alarm_on:
                            alarm_on = True
                            # play the alarm sound in a new thread
                            t = Thread(target=start_alarm, args=(alarm_sound,))
                            t.daemon = True
                            t.start()
                            
                elif pred3 == 2:# if yawning
                    cv.putText(frame, "Yawning Drowsiness Alert!!!", (100, face_react[3]-10), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                else:# eyes open and not yawning
                    cv.putText(frame, "Eyes Open", (10, 30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                    count = 0
                
               
                    
                
              
                
        fps = frame_counter / (time.time() - start_time)
        cv.putText(frame,f"FPS: {fps:.2f}",(200,200),cv.FONT_HERSHEY_DUPLEX,0.7,(0, 255, 255),2,)
        cv.imshow("frame", frame)
        key = cv.waitKey(1)
        if key == ord("q"):
            break
    cap.release()
    cv.destroyAllWindows() 