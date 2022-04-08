import streamlit as st
import numpy as np
import tensorflow
from tensorflow import keras
import cv2
from pydub import AudioSegment
from pydub.playback import play

def load_model(path):
    model = keras.models.load_model(path)
    return model

def realTime(model):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    capture = cv2.VideoCapture(0)
    Score=0
    placeHolder = st.empty()
    while True:
        ret, frame = capture.read()
        height,width = frame.shape[0:2]
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (80,80))
        faces= face_cascade.detectMultiScale(gray, scaleFactor= 1.2, minNeighbors=5)
        eyes= eye_cascade.detectMultiScale(gray, scaleFactor= 1.1, minNeighbors=1)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,pt1=(x,y),pt2=(x+w,y+h), color= (255,0,0), thickness=3 )
        
        for (ex,ey,ew,eh) in eyes:
            eye= frame[ey:ey+eh,ex:ex+w]
            eye= cv2.resize(eye,(80,80))
            eye= eye/255
            eye= eye.reshape(80,80,3)
            eye= np.expand_dims(eye,axis=0)
            prediction = model.predict(eye)
            if prediction[0][0]>0.25:
                Score=Score+1
                if(Score>7):
                    try:
                        cv2.putText(frame,'drowsy',(10,height-100),fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,fontScale=1,color=(255,0,0),
                       thickness=1,lineType=cv2.LINE_AA)
                        song = AudioSegment.from_wav('./res/alarm.wav')
                        play(song)
                        Score=1
                        continue
                    except:
                        pass

            elif prediction[0][1]>0.80:
                Score = Score-1
                if (Score<0):
                    Score=0


        placeHolder.image(frame, use_column_width=True, channels='BGR')
        key = cv2.waitKey(1)

        if key == 27:
            break

    st.stop()
    capture.release()
    cv2.destroyAllWindows()

def drowsinessDet(model):
    html="""
    <style>
    .element-container:nth-child(9) {
      left: 240px;
      top: 0px;
    }
    </style>
    """

    st.title("Drowsiness Detector")
    st.markdown("")

    st.subheader("The app expects the camera to be placed at eye level of the driver. This angle would be recommended.")
    st.subheader("Press the button below to get started. When you're done, click the Stop button on the top right. Your webcam should turn off in a few moments.")
    st.markdown(html, unsafe_allow_html=True)

    if st.button("Launch Application"):
        realTime(model)


def main():

    html = """
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#36cf9c, #27aedb);
        color: white;
    }
    </style>
    """

    st.markdown(html, unsafe_allow_html=True)

    menu = ['Welcome', 'Drowsiness Detector']
    with st.sidebar:
        st.subheader("MAJOR PROJECT- GROUP 17")
    with st.sidebar.expander("Menu", expanded=False):
        option = st.selectbox('', menu)
        st.subheader("GROUP- 17 DROWSINESS DETECTION SYSTEM")
    with st.sidebar:
        st.subheader("DROWSINESS DETECTION SYSTEM USING INCEPTION V3")
        st.subheader("Submitted By:")
        st.write("Asad Ahmed Khan- 0801CS181016")
        st.write("Batul Sabir Tahir- 0801CS181022")
        st.write("Farah Khan- 0801CS181027")
        st.write("Prabhakar Singh- 0801CS181058")
        st.write("Vijay Talviya- 0801CS181092")
    if option == 'Welcome':
        html = """
        <style>
        .element-container:nth-child(4)
        {
            color: #40E0D0;
        }
        </style>
        """

        st.markdown(html, unsafe_allow_html=True)
        st.title("Major Project")
        st.header("DRIVER DROWSINESS DETECTION SYSTEM")
        st.image('res/traffic.jpg', use_column_width=True)

        st.subheader("Over 400 people die tragically every day in India in road accidents, and more than 1500 are hospitalized with injuries. This is a tragic reality that we all need to try and set right.")

        st.markdown("Our goal is very simple: to ensure that everybody gets home safely. More than 40% of traffic accidents occur not due to technical malfunctions or driving errors, but simply due to fatigue. ")

        

        st.title("Our aim is to try and rectify this in our own small way.")

        st.subheader("Submitted By:")
        st.markdown("Asad Ahmed Khan- 0801CS181016")
        st.markdown("Batul Sabir Tahir- 0801CS181022")
        st.markdown("Farah Khan- 0801CS181027")
        st.markdown("Prabhakar Singh- 0801CS181058")
        st.markdown("Vijay Talviya- 0801CS181092")

    elif option == 'Drowsiness Detector':
        html = """
        <style>
        .element-container:nth-child(4)
        {
            color: #40E0D0;
        }
        </style>
        """

        st.markdown(html, unsafe_allow_html=True)

        st.image('res/driver.png', use_column_width=True)
        model = load_model('./models/model.h5')
        drowsinessDet(model)
    
    else:
        st.error('Please enter a valid input')



if __name__ == "__main__":
    main()
