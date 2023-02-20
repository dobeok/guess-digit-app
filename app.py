import tensorflow as tf
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from skimage.transform import resize
import numpy as np
import plotly.express as px


model = tf.keras.models.load_model('./predict-model/results/keras_mnist_cnn.h5')

st.set_page_config(layout="wide")


def predict(image, model):
    # grayscale and resize to 28 x 28
    image_bw = image[:, :, 3]
    image_bw_28 = resize(image_bw, (28, 28))
    
    predicted_probs = model.predict(np.array([image_bw_28]), verbose=0)
    predicted_label = np.argmax(predicted_probs, axis=1)

    return predicted_probs, predicted_label


with st.sidebar:
    drawing_mode = st.radio(
        label='Choose drawing mode',
        options=["freedraw", "line", "transform"]
    )

    # only show prediction if probability > THRESHOLD
    THRESHOLD = st.slider(
        label='Choose a confidence threshold',
        min_value=.5,
        max_value=.95,
        value=.6,
        step=.05,
        )


col_1, col_2 = st.columns([2, 1], gap='medium')

with col_1:
    st.markdown('### Draw in this box')
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=14,
        update_streamlit=True,
        height=225,
        width=225,
        drawing_mode=drawing_mode,
        key="canvas",
    )

 
    img_data = canvas_result.image_data



    if img_data is not None:
        probabilities, guess = predict(img_data, model)

        fig = px.bar(
            x=range(10),
            y=[0.01 for _ in range(10)]
            )

        sorted_probabilities = sorted(probabilities[0], reverse=True)
        
        if not np.all(img_data == img_data[0]):
            if sorted_probabilities[0] >= THRESHOLD:
                st.markdown(f'### Prediction = {guess[0]}')
            else:
                st.markdown(f'### Please draw again..')

        st.markdown('### Probabilities')
        if not np.all(img_data == img_data[0]):
            if probabilities[0].max() >= THRESHOLD:
                fig = px.bar(x=range(10), y=probabilities[0])
                
        
        fig.update_layout(
            yaxis={
                'title': None,
                'range': [0, 1]
                },
            xaxis={
                'title':None,
                'dtick': 1,
            }
            )


        st.plotly_chart(
            fig,
            theme='streamlit',
            use_container_width=True,
            config={
                'displayModeBar': False,
                }
            )


with st.sidebar:
    if img_data is not None:
        result = Image.fromarray(img_data)
        buf = BytesIO()
        result.save(buf, format='PNG')
        byte_im = buf.getvalue()

        st.download_button(
            label='download image',
            data=byte_im,
            file_name="downloaded_image.png",
            mime="image/png",
            )
