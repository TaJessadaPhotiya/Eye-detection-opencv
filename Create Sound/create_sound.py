import pyttsx3
engine = pyttsx3.init() # การสร้างวัตถุ
text = 'Looking at center!'

""" RATE"""
engine.setProperty('rate', 180)   
rate = engine.getProperty('rate')   # รับรายละเอียดของอัตราการพูดในปัจจุบัน
print (rate)                        # การพิมพ์อัตราเสียงปัจจุบัน
  # การตั้งค่าอัตราเสียงใหม่


"""VOLUME"""
volume = engine.getProperty('volume')   # ทำความรู้จักกับระดับเสียงปัจจุบัน (ต่ำสุด=0 และ สูงสุด=1)
print (volume)                          #การพิมพ์ระดับเสียงปัจจุบัน
engine.setProperty('volume',1.0)    # การตั้งค่าระดับเสียงระหว่าง 0 ถึง 1

"""VOICE"""
voices = engine.getProperty('voices')       #รับรายละเอียดของเสียงปัจจุบัน
engine.setProperty('voice', voices[0].id)  #เปลี่ยนดัชนีเปลี่ยนเสียง o สำหรับผู้หญิง
#engine.setProperty('voice', voices[1].id)   #เปลี่ยนดัชนีเปลี่ยนเสียง 1 สำหรับผู้ชาย

engine.say(text)
engine.runAndWait()
engine.stop()

"""Saving Voice to a file"""
# บน linux ตรวจสอบให้แน่ใจว่าได้ติดตั้ง 'espeak' และ 'ffmpeg' แล้ว
engine.save_to_file(text, 'center.wav')
engine.runAndWait()