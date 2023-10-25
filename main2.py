import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import pygame 
from pygame import mixer
import controller as cnt

mp_face_mesh = mp.solutions.face_mesh

start_voice= False
counter_right=0
counter_left =0
counter_center =0 

# เริ่มต้นรวม.init ไฟร์เสียง 
mixer.init()
# ที่เก็บไฟร์ voices/sounds 
voice_left = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\left.wav')
voice_right = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\Right.wav')
voice_center = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\center.wav')

# ดัชนีตา ซ้าย 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

# ดัชนีตา ขวา
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

RIGHT_IRIS =[ 474, 475, 476, 477 ] # ม่านตา ขวา
LEFT_IRIS =[ 469, 470, 471, 472 ] # ม่านตา ซ้าย
L_H_LEFT = [33] # จุดเริ่มต้น ตาซ้าย
L_H_RIGHT = [133] # จุดสิ้นสุด ตาซ้าย
R_H_LEFT = [362] # จุดเริ่มต้น ตาขวา
R_H_RIGHT = [263] # จุดสิ้นสุด ตาขวา
 
def euclidean_distance(point1, point2): #ฟังชั่นหาระยะทางแบบ ยุคลีด
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    #print(distance)
    return distance
def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(left_point, right_point)
    retio = center_to_right_dist/total_distance
    iris_position =""
    #print(retio)
    if retio <= 0.42:
        iris_position="Right"
    elif retio > 0.42 and  retio <= 0.57:
        iris_position="Center"
    else:
        iris_position="Left"
    return iris_position, retio

camera = cv.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1, 
    refine_landmarks=True, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB) #เปลี่ยนจาก BGR เป็น RGB
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
           # print(results.multi_face_landmarks[0].landmark) # ปริ้น landmark ค่า x,y,z ออกมา
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]) # กำหนดจุด mesh_points โดยให้ให้ไม่มีเลขทศนิยม
           # print(mesh_points.shape)
            cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,0,255), 1, cv.LINE_AA) # วาดสี่เหลี่ยมรอบจุด mesh_points รอบตา ซ้าย
            cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,0,255), 1, cv.LINE_AA) # วาดสี่เหลี่ยมรอบจุด mesh_points รอบตา ขวา

            # คำนวนการ วาดวงกลม mesh_points รอบตา ซ้าย,ขวา
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS]) 
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS]) 
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            #cv.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv.LINE_AA) # วาดวงกลม mesh_points รอบตา ซ้าย
            #cv.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv.LINE_AA) # วาดวงกลม mesh_points รอบตา ขวา
            #cv.circle(frame, mesh_points, [R_H_RIGHT][0], 3, (255,255,255), -1, cv.LINE_AA) # วาดวงกลม mesh_points รอบตา ขวา
            #cv.circle(frame, mesh_points, [R_H_LEFT][0], 3, (0,255,255), -1, cv.LINE_AA) # วาดวงกลม mesh_points รอบตา ขวา

            cv.polylines(frame, [mesh_points[LEFT_EYE]], True, (0,255,0), 1, cv.LINE_AA) # วาดเส้นรอบจุด mesh_points รอบตา ซ้าย
            cv.polylines(frame, [mesh_points[RIGHT_EYE]], True, (0,255,0), 1, cv.LINE_AA) # วาดเส้นรอบจุด mesh_points รอบตา ขวา

            iris_pos, retio = iris_position(center_right, mesh_points[R_H_RIGHT], mesh_points[R_H_LEFT][0])
            #print(iris_pos) #ปริ้นค่าสายตามาดู ว่าเป็น ซ้าย หรือ ขวา

            #แสดงตัวหนังสือหน้าจอ 
            cv.putText(frame,f"IRIS POS: {iris_pos} {retio:.2f}", (30,30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 255 ,0),1, cv.LINE_AA)

            eye_position_right = iris_pos
            #print(eye_position_right) #ปริ้นค่าสายตามาดู ว่าเป็น ซ้าย หรือ ขวา

            # ใช้เสียงเริ่มต้น 
            if eye_position_right=="Right" and pygame.mixer.get_busy()==0 and counter_right<2:
                # เคาน์เตอร์เริ่มต้น 
                counter_right+=1
                # การรีเซ็ตตัวนับ 
                counter_center=0
                counter_left=0 
                                       
                total=0                      # แก้ตรงนี้ 
                cnt.led(total)
                if total==0:
                    print(total) 
                # เล่นเสียง        
                voice_right.play()
            


            if eye_position_right=="Center" and pygame.mixer.get_busy()==0 and counter_center <2:
                # เคาน์เตอร์เริ่มต้น 
                counter_center +=1
                # การรีเซ็ตตัวนับ
                counter_right=0
                counter_left=0              

                total=1                      # แก้ตรงนี้
                cnt.led(total)
                if total==1:
                    print(total)
                # เล่นเสียง 
                voice_center.play()
            

           
            if eye_position_right=="Left" and pygame.mixer.get_busy()==0 and counter_left<2: 
                # เคาน์เตอร์เริ่มต้น
                counter_left +=1 
                # การรีเซ็ตตัวนับ
                counter_center=0
                counter_right=0
                
                total=2                      # แก้ตรงนี้
                cnt.led(total)
                if total==2:
                    print(total) 
                # เล่นเสียง
                voice_left.play()


        cv.imshow('img',frame)
        key = cv.waitKey(1)
        if key == ord('q') or key ==ord('Q'):
            break
camera.release()
cv.destroyAllWindows()