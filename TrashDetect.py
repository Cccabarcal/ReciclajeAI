from ultralytics import YOLO
import cv2
import math


model = YOLO('Modelos/best.pt')


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


clsName = ['Metal', 'Glass', 'Plastic', 'Carton', 'Medical']


while True:
    
    ret, frame = cap.read()

   
    results = model(frame, stream=True, verbose=False)
    for res in results:
       
        boxes = res.boxes
        for box in boxes:
           
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

          
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 < 0: x2 = 0
            if y2 < 0: y2 = 0

           
            cls = int(box.cls[0])

           
            conf = math.ceil(box.conf[0])
            print(f"Clase: {cls} Confidence: {conf}")

            if conf > 0:
               
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{clsName[cls]} {int(conf * 100)}%', (x1, y1 - 20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

   
    cv2.imshow("Waste Detect", frame)

  
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()