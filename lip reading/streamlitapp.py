# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    # st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.image("lip_img1.jpg")
    st.title('Vision$peak')
    st.info('"WhisperSync leverages cutting-edge computer vision and machine learning to advance lip reading. Enhancing accuracy and efficiency, it promotes accessibility for individuals with hearing impairments through seamless synchronization of visual lip movements and transcription."')

st.title('Vision $peak App') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Select video', options)
st.text("Selected Video "+selected_video)
# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The selected video is displayed below !')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        # st.info('This is all the machine learning model sees when making a prediction')
        # if options="webcam":
            # video
        video= load_data(tf.convert_to_tensor(file_path))
        
        # st.text(imageio.help(".gif"))
        # imageio.mimsave('animation.gif', video, duration=2)
        # st.image('animation.gif', width=400) 

        # st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        # st.text(decoder)

        # Convert prediction to text
        st.info('This is what you heard and what the machine saw !')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(" ->    "+converted_prediction)
        
