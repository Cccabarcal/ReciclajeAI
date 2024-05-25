from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math

def clean_lbl():
    
    lblimg.config(image='')
    lblimgtxt.config(image='')

def images(img, imgtxt):
    img = img
    imgtxt = imgtxt

    
    img = np.array(img, dtype="uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = Image.fromarray(img)

    img_ = ImageTk.PhotoImage(image=img)
    lblimg.configure(image=img_)
    lblimg.image = img_

   
    imgtxt = np.array(imgtxt, dtype="uint8")
    imgtxt = cv2.cvtColor(imgtxt, cv2.COLOR_BGR2RGB)
    imgtxt = Image.fromarray(imgtxt)

    img_txt = ImageTk.PhotoImage(image=imgtxt)
    lblimgtxt.configure(image=img_txt)
    lblimgtxt.image = img_txt


def Scanning():
    global img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla
    global lblimg, lblimgtxt

    
    lblimg = Label(pantalla)
    lblimg.place(x=75, y=260)

    lblimgtxt = Label(pantalla)
    lblimgtxt.place(x=995, y=310)
    detect = False

    
    if cap is not None:
        ret, frame = cap.read()
        frame_show =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        if ret == True:
            
            results = model(frame, stream=True, verbose=False)
            for res in results:
                
                boxes = res.boxes
                for box in boxes:
                    detect = True
                    
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    
                    if x1 < 0: x1 = 0
                    if y1 < 0: y1 = 0
                    if x2 < 0: x2 = 0
                    if y2 < 0: y2 = 0

                    
                    cls = int(box.cls[0])

                   
                    conf = math.ceil(box.conf[0])
                    
                    if cls == 0:
                        
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline), (0, 0, 0),cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                        
                        images(img_metal, img_metaltxt)

                    if cls == 1:
                        
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 255, 255), 2)
                       
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                      
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                        
                        images(img_glass, img_glasstxt)

                    if cls == 2:
                       
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                       
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        
                        images(img_plastic, img_plastictxt)

                    if cls == 3:
                        
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (150, 150, 150), 2)
                        
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)

                       
                        images(img_carton, img_cartontxt)

                    if cls == 4:
                        
                        cv2.rectangle(frame_show, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        text = f'{clsName[cls]} {int(conf) * 100}%'
                        sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                        dim = sizetext[0]
                        baseline = sizetext[1]
                        
                        cv2.rectangle(frame_show, (x1, y1 - dim[1] - baseline), (x1 + dim[0], y1 + baseline),
                                      (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame_show, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                       
                        images(img_medical, img_medicaltxt)

            if detect == False:
                
                clean_lbl()


         
            frame_show = imutils.resize(frame_show, width=640)

            
            im = Image.fromarray(frame_show)
            img = ImageTk.PhotoImage(image=im)

           
            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, Scanning)

        else:
            cap.release()


def ventana_principal():
    global cap, lblVideo, model, clsName, img_metal, img_glass, img_plastic, img_carton, img_medical
    global img_metaltxt, img_glasstxt, img_plastictxt, img_cartontxt, img_medicaltxt, pantalla
   
    pantalla = Tk()
    pantalla.title("RECICLAJE INTELIGENTE")
    pantalla.geometry("1280x720")

   
    imagenF = PhotoImage(file="setUp/Canva.png")
    background = Label(image=imagenF, text="Inicio")
    background.place(x=0, y=0, relwidth=1, relheight=1)

   
    model = YOLO('Modelos/best.pt')

    
    clsName = ['Metal', 'Glass', 'Plastic', 'Carton', 'Medical']

    
    img_metal = cv2.imread("setUp/metal.png")
    img_glass = cv2.imread("setUp/vidrio.png")
    img_plastic = cv2.imread("setUp/plastico.png")
    img_carton = cv2.imread("setUp/carton.png")
    img_medical = cv2.imread("setUp/medical.png")
    img_metaltxt = cv2.imread("setUp/metaltxt.png")
    img_glasstxt = cv2.imread("setUp/vidriotxt.png")
    img_plastictxt = cv2.imread("setUp/plasticotxt.png")
    img_cartontxt = cv2.imread("setUp/cartontxt.png")
    img_medicaltxt = cv2.imread("setUp/medicaltxt.png")


    lblVideo = Label(pantalla)
    lblVideo.place(x=320, y=180)

    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Scanning()

    
    pantalla.mainloop()

if __name__ == "__main__":
    ventana_principal()
