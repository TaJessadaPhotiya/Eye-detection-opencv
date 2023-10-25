
import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import pygame 
from pygame import mixer
import controller as cnt

# ตัวแปร 
frame_counter =0
CEF_COUNTER =0
TOTAL_BLINKS =0
start_voice= False
counter_right=0
counter_left =0
counter_center =0 
# ค่าคงที่
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# เริ่มต้นรวม.init ไฟร์เสียง 
mixer.init()
# ที่เก็บไฟร์ voices/sounds 
voice_left = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\left.wav')
voice_right = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\Right.wav')
voice_center = mixer.Sound('C:\Eyes-Jassada2\Eye-Test\Voice\center.wav')

# ดัชนีขอบเขตใบหน้า
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# ดัชนีริมฝีปากสำหรับจุดสังเกต
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# ดัชนีตา ซ้าย 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# ดัชนีตา ขวา
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# กล้องสำหรับรันโปรแกรม 
camera = cv.VideoCapture(0)

_, frame = camera.read()
img = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
img_hieght, img_width = img.shape[:2]
print(img_hieght, img_width)


# ฟังก์ชั่นการตรวจจับจุดสังเกต 

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # รายการ[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # ส่งคืนค่าสำหรับแต่ละจุดสังเกต 
    return mesh_coord

# Euclaidean ระยะห่าง 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# ฟังก์ชั่นแยกดวงตา
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # การแปลงภาพสีเป็นอัตตราส่วน (แปลงภาพเป็นสีเทา)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # ได้มิติของภาพ 
    dim = gray.shape

    # การสร้างหน้ากากจากสเกลสีเทาสลัว
    mask = np.zeros(dim, dtype=np.uint8)

    # วาด รูปร่างตา บนหน้ากากด้วยสีขาว 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # แสดงหน้ากาก 
    #cv.imshow('mask', mask) # โชว์ ดวงตาแบบขาวดำ //////////////////////////////////////////////////
    
    # วาดภาพดวงตาบนหน้ากากที่มีสีขาว 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # เปลี่ยนสีดำเป็นสีเทานอกเหนือจากดวงตา 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # รับ x และ y ต่ำสุดและสูงสุดสำหรับตาขวาและซ้าย 
    # สำหรับตา ขวา 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # สำหรับตา ซ้าย
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # ครอบตาจากหน้ากาก 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # คืนค่าสายตา 
    return cropped_right, cropped_left

# ประมาณตำแหน่งของดวงตา 
def positionEstimator(cropped_eye):
    # รับความสูงและความกว้างของดวงตา 
    h, w =cropped_eye.shape
    
    # รับความสูงและความกว้างของดวงตา
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # ใช้เกนณ์เพื่อแปลงภาพไบนารี่
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # สร้างส่วนคงที่ให้กับดวงตาด้วย 
    piece = int(w/3) 

    # ผ่าดวงตาออกเป็นสามส่วน 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # การเรียกใช้ฟังก์ชันตัวนับพิกเซล
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# การสร้างฟังก์ชันตัวนับพิกเซล 
def pixelCounter(first_piece, second_piece, third_piece):
    # นับพิกเซลสีดำในแต่ละส่วน 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # สร้างรายการพิกเซล รวมพิกเซลในแต่ละส่วน
    eye_parts = [right_part, center_part, left_part]

    # รับดัชนีของค่าสูงสุดในรายการ 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # เริ่มต้นนับเวลา 
    start_time = time.time()
    # เริ่มวนรอบวิดีโอ
    while True:
        frame_counter +=1 # เคาน์เตอร์เฟรม
        ret, frame = camera.read() # รับเฟรมจากกล้อง 
        if not ret: 
            break # ไม่มีเฟรมแตก
        #  ปรับขนาดกรอบ
        
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)

            cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

            # ตัวนับเครื่องตรวจจับการกะพริบเสร็จสมบูรณ์
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            cv.imshow('right', crop_right)
            cv.imshow('left', crop_left)
            eye_position_right, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            

            # ใช้เสียงเริ่มต้น 
            if eye_position_right=="RIGHT" and pygame.mixer.get_busy()==0 and counter_right<2:
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
            


            if eye_position_right=="CENTER" and pygame.mixer.get_busy()==0 and counter_center <2:
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
            

           
            if eye_position_right=="LEFT" and pygame.mixer.get_busy()==0 and counter_left<2: 
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



        # การคำนวณเฟรมต่อวินาที FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time
        
        frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # การเขียนภาพสำหรับการวาดภาพขนาดย่อ
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        # เขียนวิดีโอเพื่อวัตถุประสงค์ในการสาธิต 
        #       
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key==ord('q') or key ==ord('Q'):
            break
    cv.destroyAllWindows()
    camera.release()
