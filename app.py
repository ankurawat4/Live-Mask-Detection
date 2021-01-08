# importing libraries 
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 


def about():
	st.write(
		'''
		**In this pandemic mask is not a choice but need. This webapp here tells you whether
        a person is wearing a mask or not in live stream**

		The model has three stages:
            1. Haarcascade for face detection on image
			2. sending face to CNN model
			3. predicting whether a person is wearing mask or not
			 
			
		''')


def detect_on_image(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_copy = image.copy()     # copy of image
    
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5) 
    if faces == ():
#         print(' No face in frame.')
        pass
        
    else:
        for (x,y,w,h) in faces:
            
            roi = image_copy[y:y+h, x:x+w]
            #roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB )   # bgr to rgb 
            test_img = cv2.resize(roi, (200,200))             # resizing
            img = np.expand_dims(test_img, 0)                 # adding dimension
            test_img = img/255

            pred_prob = model.predict(test_img)
            pred = np.argmax(pred_prob)
            

            if pred==0:
                cv2.rectangle(image_copy, (x,y), (x+w, y+h), (0,255,0), 3)
                cv2.putText(image_copy, 'MASK', (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                prob = pred_prob[0][0] * 100
                flag = True
                print('person wearing Mask : prob -',prob)

            else:
                cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
                cv2.putText(image_copy, 'Please wear a Mask', (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,0), 2)
                prob = pred_prob[0][1] * 100
                flag = False
                print('No Mask : prob -', prob)
        return image_copy, prob, flag
        
def detect_live_stream():
    cap = cv2.VideoCapture(0)
    cap.set(10, 300)

    while True:

        ret, frame = cap.read()

        if ret==False:

            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_copy = frame.copy()     # copy of frame


        faces = face_cascade.detectMultiScale(frame_gray, 1.3, 5) 

        if faces == ():
    #         print(' No face in frame.')
            pass

        else:
            for (x,y,w,h) in faces:
                roi = frame_copy[y:y+h, x:x+w]


                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB )   # bgr to rgb 
                test_img = cv2.resize(roi_rgb, (200,200))             # resizing
                img = np.expand_dims(test_img, 0)                 # adding dimension
                test_img = img/255

                pred_prob = model.predict(test_img)
                pred = np.argmax(pred_prob)

                if pred==0:
                    cv2.rectangle(frame_copy, (x,y), (x+w, y+h), (0,255,0), 3)
                    cv2.putText(frame_copy, 'MASK', (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
                    print('person wearing Mask : prob -',pred_prob[0][0])

                else:
                    cv2.rectangle(frame_copy, (x,y), (x+w, y+h), (0,0,255), 3)
                    cv2.putText(frame_copy, 'Please wear a Mask', (x,y-5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)
                    print('No Mask : prob -', pred_prob[0][1])

        #     cv2.imshow('roi', roi)
        cv2.imshow('detect', frame_copy)
	
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
  

def main():
    st.title("Mask detection App :mask: ")
    st.write("**Using CNN and Haarcascade Classifiers**")

    activities = ["Home","Live", "About"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":
    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:
            image = Image.open(image_file)
            image = np.asarray(image)
            
            if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
                result_img, probability , flag = detect_on_image(image=image)
                st.image(result_img , use_column_width = True)
                if flag==True:
                    st.success("Person is wearing a mask, I am {:0.2f} % sure\n".format(probability))
                elif flag==False:
                    st.success("Person is not wearing a mask, I am {:0.2f} % sure\n".format(probability))
    
    if choice=="Live":
    	st.write("Click on below button to detect on live stream.")
    	if st.button("LIVE MASK DETECTION"):
            detect_live_stream()
    	st.write("Go to the About section from the sidebar to learn more about it.")
     
    elif choice == "About":
    	about()




if __name__ == "__main__":
    main()
