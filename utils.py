
import cv2 as cv 
import numpy as np

# สี 
# ค่า = (สีน้ำเงิน, สีเขียว, สีแดง) opencv ยอมรับค่า BGR ไม่ใช่ RGB
BLACK = (0,0,0)       #สีดำ
WHITE = (255,255,255) #สีขาว
BLUE = (255,0,0)      #สีน้ำเงิน
RED = (0,0,255)       #สีแดง
CYAN = (255,255,0)    #สีฟ้า
YELLOW =(0,255,255)   #สีเหลือง
MAGENTA = (255,0,255) #สีม่วงแดง
GRAY = (128,128,128)  #สีเทา
GREEN = (0,255,0)     #เขียว
PURPLE = (128,0,128)  #สีม่วง
ORANGE = (0,165,255)  #ส้ม
PINK = (147,20,255)   #สีชมพู
points_list =[(200, 300), (150, 150), (400, 200)]
def drawColor(img, colors):
    x, y = 0,10
    w, h = 20, 30
    
    for color in colors:
        x += w+5 
        # y += 10 
        cv.rectangle(img, (x-6, y-5 ), (x+w+5, y+h+5), (10, 50, 10), -1)
        cv.rectangle(img, (x, y ), (x+w, y+h), color, -1)
    
def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):

    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # รับขนาดตัวอักษร
    x, y = textPos
    cv.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # วาดรูปสี่เหลี่ยมผืนผ้า 
    cv.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # วาดเป็นข้อความ

    return img

def textWithBackground(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3, bgOpacity=0.5):

    (t_w, t_h), _= cv.getTextSize(text, font, fontScale, textThickness) # รับขนาดตัวอักษร
    x, y = textPos
    overlay = img.copy() # การจัดการภาพ
    cv.rectangle(overlay, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # วาดรูปสี่เหลี่ยมผืนผ้า 
    new_img = cv.addWeighted(overlay, bgOpacity, img, 1 - bgOpacity, 0) # ซ้อนทับสี่เหลี่ยมบนภาพ
    cv.putText(new_img,text, textPos,font, fontScale, textColor,textThickness ) # วาดเป็นข้อความ
    img = new_img

    return img

