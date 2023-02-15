import tensorflow as tf
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from io import BytesIO
from skimage.transform import resize
import numpy as np


# model = tf.keras.models.load_model('./predict-model/results/keras_mnist_linear_stack.h5')
model = tf.keras.models.load_model('./predict-model/results/keras_mnist_cnn.h5')

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


col_1, col_2 = st.columns(2)


with col_1:
    st.title('Draw in this box')
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=14,
        update_streamlit=True,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    


def predict(image, model):
    # bw only
    image_bw = image[:, :, 3]

    # downsize image to 28 x 28
    image_bw_28 = resize(image_bw, (28, 28))
    # image_bw_28 = image_bw_28.reshape(784, -1)

    predicted_probs = model.predict(np.array([image_bw_28]))
    predicted_label = np.argmax(predicted_probs, axis=1)

    return predicted_probs, predicted_label


img_data = canvas_result.image_data
if img_data is not None:
    with col_2:
        if not np.all(img_data == img_data[0]):
            probabilities, guess = predict(img_data, model)
            
            if probabilities[0].max() >= THRESHOLD:
                st.title(f'Predicted = {guess[0]}')    
            else:
                st.title(f"I'm not very sure..")

            st.bar_chart(
                probabilities[0]
            )

    # download button
    result = Image.fromarray(img_data)
    buf = BytesIO()
    result.save(buf, format='PNG')
    byte_im = buf.getvalue()

    st.download_button(
        label='download img',
        data=byte_im,
        file_name="downloaded_image.png",
        mime="image/png",
        )