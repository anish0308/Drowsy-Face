import numpy
from pygame import mixer
import time
import cv2
from tkinter import *
import tkinter.messagebox
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tkinter import ttk  
from ttkthemes import ThemedTk
import tkinter.font as tkFont
new_model = tf.keras.models.load_model('my_model.h5')
root=ThemedTk(theme="itkt1")
root.geometry('1280x734')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
root.title('Drowsy camera')
frame.config(background='gray')
label = Label(frame, text="Drowsiness Detection",bg='gray',font=('Times 25 bold'))
label.pack(side=TOP)
filename = PhotoImage(file="eye.png")
background_label = Label(frame,image=filename)
background_label.pack(side=TOP)



def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.Anish Chandrasekaran\n2. Shreyas Bhat \n3. Charmika Tankala \n4. Manasi Tambade")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Driver Cam version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n In Python 3')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="Drowsy camera",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)



def exitt():
   exit()

  

def web():
   capture =cv2.VideoCapture(0)
   while True:
      ret,frame=capture.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
         break
   capture.release()
   cv2.destroyAllWindows()   

def webdet():
   capture =cv2.VideoCapture(0)
   face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
   eye_glass = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
   

   while True:
       ret, frame = capture.read()
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray)
    

       for (x,y,w,h) in faces:
           font = cv2.FONT_HERSHEY_COMPLEX
           cv2.putText(frame,'Face',(x+w,y+h),font,1,(250,250,250),2,cv2.LINE_AA)
           cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = frame[y:y+h, x:x+w]
        
          
           eye_g = eye_glass.detectMultiScale(roi_gray)
           for (ex,ey,ew,eh) in eye_g:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

       
       cv2.imshow('frame',frame)
       if cv2.waitKey(1) & 0xff == ord('q'):
          break
   capture.release()
   cv2.destroyAllWindows()


   
def alert():
   mixer.init()
   alert=mixer.Sound('beep-07.wav')
   alert.play()
   time.sleep(0.1)
   alert.play()   
   
def blink():
   capture =cv2.VideoCapture(0)
   face_cascade = cv2.CascadeClassifier('haar_face.xml')
   eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
   #blink_cascade = cv2.CascadeClassifier('CustomBlinkCascade.xml')
   counter = 0
   cnt = 0
   while True:
      ret, frame = capture.read()
      gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray,1.3,5)

      for (x,y,w,h) in faces:
         font = cv2.FONT_HERSHEY_COMPLEX
         cv2.putText(frame,'Face',(x+w,y+h),font,1,(250,250,250),2,cv2.LINE_AA)
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]

         eyes = eye_cascade.detectMultiScale(roi_gray)
         for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
         for(ex,ey,ew,eh) in eyes:
            eyes_roi = roi_color[ey:ey+eh,ex:ex+ew]
         final_image = cv2.resize(eyes_roi,(224,224))
         final_image = numpy.expand_dims(final_image,axis=0)
         final_image=final_image/255.0	
         prediction =new_model.predict(final_image)
         p1=prediction*10   	        
         if(p1>1):
             counter = 0
             cnt = cnt + 1
             status = 'Open Eyes' 
             if cnt>46:
                status = 'Error: Static Image Detected'
         else:
             counter = counter +1
             cnt = 0
             status = 'Closed Eyes'
             if counter>3:
                status = 'Drowsy'
                alert()
             
                 
         font1 = cv2.FONT_HERSHEY_SIMPLEX
         font2 = cv2.FONT_HERSHEY_PLAIN
         if status=='Error: Static Image Detected':
            cv2.putText(frame,status,(50,50),font2,2,(0,69,255),2,cv2.LINE_4)
         elif status=='Open Eyes':
            cv2.putText(frame,status,(50,50),font2,2.5,(0,255,0),2,cv2.LINE_4)
         elif status=='Closed Eyes':
            cv2.putText(frame,status,(50,50),font2,2.5,(0,255,255),2,cv2.LINE_4)
         else:
            cv2.putText(frame,status,(50,50),font2,2.5,(0,0,255),2,cv2.LINE_4)
            
            
            
         
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF ==ord('q'):
          break
         
  
   capture.release()
   cv2.destroyAllWindows()
   
helv12 = tkFont.Font(family="Helvetica",size=10,weight="bold",slant="italic")
s = ttk.Style()
s.configure('my.TButton', font=helv12)
but1=ttk.Button(frame,width=39,text='Open Cam',command=web,style='my.TButton')
but1.place(x=50,y=124)

but2=ttk.Button(frame,width=39,command=webdet,text='Open Cam & Detect',style='my.TButton')
but2.place(x=50,y=196)

but3=ttk.Button(frame,width=39,command=blink,text='Detect Drowsiness  With Sound',style='my.TButton')
but3.place(x=50,y=268)

but4=ttk.Button(frame,width=10,command=exitt,text='Exit',style='my.TButton')
but4.place(x=255,y=340)


root.mainloop()



