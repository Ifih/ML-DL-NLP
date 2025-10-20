import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import io

# --- 1. Load the Saved Model ---
@st.cache_resource
def load_model():
    """Load the trained Keras model."""
    try:
        # Assumes the model was saved as 'mnist_cnn_model.h5' in the same directory
        model = keras.models.load_model('mnist_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure the trained model saved is named as 'mnist_cnn_model.h5' and in the same folder as this script.")
        return None

model = load_model()

# --- 2. Streamlit App Layout ---
st.set_page_config(page_title="MNIST CNN Classifier", layout="wide")

# Sidebar for settings
st.sidebar.title("Settings")
canvas_size = 400  # Fixed canvas size
stroke_width = st.sidebar.slider("Stroke Width", min_value=5, max_value=20, value=10, step=1)

# Initialize session state for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize session state for canvas clearing
if 'clear_canvas' not in st.session_state:
    st.session_state.clear_canvas = 0

st.title("🔢 Handwritten Digit Recognition")
st.markdown("Draw a single digit (0-9) on the canvas below and let the CNN predict it!")

# Instructions expander
with st.expander("Instructions"):
    st.write("""
    - Draw a digit on the canvas.
    - Click 'Predict' to get the result.
    - Use 'Clear Canvas' to reset.
    - Or upload an image for prediction.
    """)

if model is not None:
    # --- 3. Drawing Canvas Setup ---
    col1, col2 = st.columns([2, 1])

    with col1:
        # Create the drawing canvas
        canvas_result = st_canvas(
            fill_color="#000000",  # Background color (black)
            stroke_width=stroke_width,
            stroke_color="#FFFFFF",  # Drawing color (white)
            background_color="#000000",
            height=canvas_size,
            width=canvas_size,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.clear_canvas}",
        )

    with col2:
        st.subheader("Controls")
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        col_clear, col_predict = st.columns(2)
        with col_clear:
            if st.button("Clear Canvas"):
                st.session_state.clear_canvas += 1
                st.rerun()
        with col_predict:
            predict_button = st.button("Predict", type="primary")

    # --- 4. Prediction Logic ---
    def preprocess_image(img):
        """Preprocess image for model prediction."""
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = img.convert('L')
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        # Normalize
        img_array = np.array(img).astype('float32') / 255.0
        # Reshape for model
        input_tensor = img_array.reshape(1, 28, 28, 1)
        return img_array, input_tensor

    def display_prediction(final_img, prediction, predicted_digit):
        """Display prediction results."""
        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Processed Image (28x28)")
            st.image(final_img, width=100, clamp=True)

        with col2:
            st.subheader("🔮 Prediction:")
            st.metric(label="Predicted Digit", value=f"**{predicted_digit}**")

            st.write("Confidence:")
            prob_df = pd.DataFrame(
                prediction[0],
                index=[str(i) for i in range(10)],
                columns=['Probability']
            )
            st.bar_chart(prob_df)

        # Add to history
        st.session_state.prediction_history.append(predicted_digit)
        if len(st.session_state.prediction_history) > 5:
            st.session_state.prediction_history.pop(0)

    # Handle canvas prediction
    if canvas_result.image_data is not None and predict_button:
        img_data = canvas_result.image_data
        if np.sum(img_data[:,:,3]) > 0:
            with st.spinner("Processing..."):
                gray_img = img_data[:, :, 0].astype('float32')
                pil_img = Image.fromarray(gray_img)
                final_img, input_tensor = preprocess_image(pil_img)
                prediction = model.predict(input_tensor)
                predicted_digit = np.argmax(prediction)
                display_prediction(final_img, prediction, predicted_digit)

    # Handle uploaded image
    if uploaded_file is not None and predict_button:
        with st.spinner("Processing uploaded image..."):
            img = Image.open(uploaded_file)
            final_img, input_tensor = preprocess_image(img)
            prediction = model.predict(input_tensor)
            predicted_digit = np.argmax(prediction)
            display_prediction(final_img, prediction, predicted_digit)

    # Display prediction history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("Recent Predictions")
        history_str = " → ".join(map(str, st.session_state.prediction_history))
        st.write(f"Last 5 predictions: {history_str}")

    # Save image option
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:,:,3]) > 0:
        if st.button("Save Drawn Image"):
            gray_img = canvas_result.image_data[:, :, 0].astype('uint8')
            pil_img = Image.fromarray(gray_img, mode='L')
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Image",
                data=byte_im,
                file_name="drawn_digit.png",
                mime="image/png"
            )

else:
    st.warning("Model could not be loaded. Cannot run classifier.")