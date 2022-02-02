# Importing required libraries, obviously
import logging
import logging.handlers
import threading
from pathlib import Path
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

# Loading pre-trained parameters for the cascade classifier
try:
    # Face Detection
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Load model (h5 file)
    classifier =load_model('Face_Emotion_detection.h5')
    # Emotion that will be predicted
    emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
except Exception:
    st.write("Error loading cascade classifiers")
    
    
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        label=[]
        img = frame.to_ndarray(format="bgr24")
        face_detect = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(gray, 1.3,1)

        for (x,y,w,h) in faces:
            a=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            # Face Cropping for prediction
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            # reshaping the cropped face image for prediction
            roi = np.expand_dims(roi,axis=0)
            # Prediction part
            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            b=cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
               
        return b




def face_detect():
    class VideoTransformer(VideoTransformerBase):
        # `transform()` is running in another thread, then a lock object is used here for thread-safety.
        frame_lock: threading.Lock
        in_image: Union[np.ndarray, None]
        out_image: Union[np.ndarray, None]

        def __init__(self) -> None:
            self.frame_lock = threading.Lock()
            self.in_image = None
            self.out_image = None

        def transform(self, frame: av.VideoFrame) -> np.ndarray:
            in_image = frame.to_ndarray(format="bgr24")

            out_image = in_image[:, ::-1, :]  # Simple flipping for example.

            with self.frame_lock:
                self.in_image = in_image
                self.out_image = out_image

            return in_image

    ctx = webrtc_streamer(key="snapshot", video_transformer_factory=VideoTransformer)

    while ctx.video_transformer:
        
        
            with ctx.video_transformer.frame_lock:
                in_image = ctx.video_transformer.in_image
                out_image = ctx.video_transformer.out_image

            if in_image is not None :
                gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray)
                for (x,y,w,h) in faces:
                    a=cv2.rectangle(in_image,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray = gray[y:y+h,x:x+w]
                    roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                    if np.sum([roi_gray])!=0:
                        roi = roi_gray.astype('float')/255.0
                        roi = img_to_array(roi)
                        # reshaping the cropped face image for prediction
                        roi = np.expand_dims(roi,axis=0)
                        # Prediction part
                        prediction = classifier.predict(roi)[0]
                        label=emotion_labels[prediction.argmax()]
                        label_position = (x,y)
                        # Text Adding
                        b = cv2.putText(a,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        st.image(b,channels="BGR")

  
    



    



from streamlit_webrtc import (  ClientSettings, VideoTransformerBase, WebRtcMode,webrtc_streamer,)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)





WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": True},
)


def main():
    activities = ["Introduction","Home","Real-Time Snapshot"]

    # Face Analysis Application #
    st.title("Real Time Face Emotion Detection Application")

    choice = st.sidebar.selectbox("Select Activity", activities)
    st.sidebar.markdown( """<a href="https://www.linkedin.com/in/avisikta-majumdar//">Aviskta Majumdar LinkedIn</a>""", unsafe_allow_html=True,)


    if choice == "Real-Time Snapshot":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.title(":angry::dizzy_face::fearful::smile::pensive::open_mouth::neutral_face:")
        st.write("**Using the Haar cascade Classifiers**")
        st.write("Go to the About section from the sidebar to learn more about it.")
        st.write("**Instructions while using the APP**")
        st.write(
		'''1. Click on the Start button to start.
                 
		 
                  2. WebCam will ask permission for camera & microphone permission.
		    
		    
		  3. It will automatically throw the image with the prediction at that instant.
                 
		 
                  4. Make sure that camera shouldn't be used by any other app.
                
		
                  5. For live recognition the app is getting slow and takes more time to predict.
		  
		  
		  6. Easy to know what was or what is the emotion at a particular time.
                  
		  
                  7. Click on  Stop  to end.
                  
		  
                  8. Still webcam window didnot open,  refresh the page.''')
      
# calling face_detect() to detect the emotion
        face_detect()
        
    elif choice =="Home":
        html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Emotion Recognition WebApp</h2>
    </div>
    </body>
        """
        st.markdown(html_temp, unsafe_allow_html=True)
        st.title(":angry::dizzy_face::fearful::smile::pensive::open_mouth::neutral_face:")
        st.write("**Using the Haar cascade Classifiers**")
        st.write("Go to the About section from the sidebar to learn more about it.")
        st.write("**Instructions while using the APP**")
        st.write('''  
                  1. Click on the Start button to start.
                  
                  2. WebCam window will ask permission for camera & microphone permission.
                  
		  3. It will automatically  predict at that instant.
		          
                  4. Make sure that camera shouldn't be used by any other app.
                  
                  5. Click on  Stop  to end.
                  
		  6. Still webcam window didnot open,  go to Check Camera from the sidebar.''')

        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)


    elif choice=="Introduction":
         html_temp = """
    <body style="background-color:blue;">
    <div style="background-image: url('https://tinyurl.com/backgrou');padding:150px">
    <h2 style="color:red;text-align:center;">Emotions can get in the way or get you on the way. 
    
                                                                  -Mavis Mazhura.</h2>
    <h2 style="color:white;text-align:center;">To Know your emotion proceed to Home from the side bar.</h2>
    </div>
    </body>
        """
         st.markdown(html_temp, unsafe_allow_html=True)
        
        
  

if __name__ == "__main__":
    main()
