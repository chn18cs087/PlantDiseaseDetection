import numpy as np
import cv2
import pickle

###################
width = 640
height = 480
threshold = 0.65
camNo = 0
imageList = ['Apple_healthy', 'Pepper_healthy', 'Potato_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Leaf_mold', 'Tomato_target_spot']
#########################

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)

pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOrginal = cap.read()
    img = np.asarray(imgOrginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)
    img = img.reshape(1,32,32,1)

    # predict
    classIndex = model.predict_classes(img)
    # print(classIndex)
    predictions = model.predict(img)
    # print(predictions)
    probVal = np.amax(predictions)
    print(imageList[int(classIndex)],probVal)

    if probVal > threshold :
        cv2.putText(imgOrginal,imageList[int(classIndex)]+ "  " +str(probVal),
                    (50,50),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,(225,225,255),1)

    cv2.imshow("Orginal Image",imgOrginal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
