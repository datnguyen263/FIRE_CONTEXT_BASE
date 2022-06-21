import cv2
import numpy as np
import tensorflow as tf
import time

def camera_detect(model, fire_cascade):
    live_Camera = cv2.VideoCapture(0)
    live_Camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    live_Camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    live_Camera.set(cv2.CAP_PROP_FPS, 30)
    new_frame_time = 0
    prev_frame_time = 0
    while(live_Camera.isOpened()):
        ret, frame = live_Camera.read()
        
        fire = fire_cascade.detectMultiScale2(frame, 1.1, 5) #1.2, 5
        print(fire[0])
        for (x,y,w,h) in fire[0]:
            roi_color = frame[y:y+h, x:x+w]
            roi_color = cv2.resize(roi_color,(256,256))
            data = []
            image = np.array(roi_color)
            data.append(image)
            data = np.array(data, dtype="float32") / 255.0

            ypred = model.predict(data)
            if np.argmax(ypred) == 0:
                cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
                print(int(ypred[0][0]*100))
                print("fire detected!")
        # img = cv2.resize(frame,(256,256))
        # data = []
        # image = np.array(img)
        # data.append(image)
        # data = np.array(data, dtype="float32") / 255.0
        # ypred = model.predict(data)
        # # print(ypred)
        # if np.argmax(ypred) == 0:
        #     #cv2.rectangle(frame,(x-20,y-20),(x+w+20,y+h+20),(255,0,0),2)
        #     print(ypred)
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        fps = str(int(fps)) + "FPS"
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    fire_cascade = cv2.CascadeClassifier('model/fire_detection.xml')
    alexnet = tf.keras.models.load_model("model/model.h5")
    

    camera_detect(alexnet, fire_cascade)

if __name__ == "__main__":
    main()