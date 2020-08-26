import cv2
import numpy as np
import pandas as pd

from utils import CFEVideoConf, image_resize

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_mcs_nose.xml')

if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

if eye_cascade.empty():
	raise IOError('Unable to load the eye cascade classifier xml file')

img = cv2.imread('Before.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
sunglasses_img = cv2.imread('glasses.png',-1)
mustache=cv2.imread('mustache.png',-1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

centers = []
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (x_eye,y_eye,w_eye,h_eye) in eyes:
        #cv2.rectangle(roi_color, (x_eye,y_eye), (x_eye+w_eye,y_eye+h_eye), (0,255,0), 3)
        centers.append((x + int(x_eye + 0.5*w_eye), y + int(y_eye + 0.5*h_eye)))

    if len(centers) > 0:
        # Overlay sunglasses
        sunglasses_width = 2.12 * abs(centers[1][0] - centers[0][0])
        overlay_img = np.zeros(img.shape, np.uint8)
        h, w = sunglasses_img.shape[:2]
        scaling_factor = sunglasses_width / w
        overlay_sunglasses = cv2.resize(sunglasses_img, None, fx=scaling_factor, 
                fy=scaling_factor, interpolation=cv2.INTER_AREA)

        x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
        x -= 0.26*overlay_sunglasses.shape[1]
        y += 0.55*overlay_sunglasses.shape[0] 
        h, w = overlay_sunglasses.shape[:2]
        print(y,h,x,w)
        for i in range(0, h):
                for j in range(0, w):
                    #print(glasses[i, j]) #RGBA
                    if overlay_sunglasses[i, j][3] != 0: # alpha 0
                        img[int(y+i), int(x+j)] = overlay_sunglasses[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
        for (nx, ny, nw, nh) in nose:
            #cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = image_resize(mustache.copy(), width=nw+4)

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    #print(glasses[i, j]) #RGBA
                    if mustache2[i, j][3] != 0: # alpha 0
                        roi_color[ny + int(nh/1.8) + i, nx + j] = mustache2[i, j]

        cv2.imshow('Eye Detector', img)
        # cv2.imshow('Sunglasses', final_img)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
img_px=img.flatten().reshape(-1,3)
df=pd.DataFrame(data=img_px,columns=["Channel 1","Channel 2","Channel 3"])
df.to_csv('y_prediction.csv',index=False)
cv2.waitKey()
cv2.destroyAllWindows()