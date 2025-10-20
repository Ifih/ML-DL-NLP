import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
from PIL import Image

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
st.title("🔢 Handwritten Digit Recognition")
st.markdown("Draw a single digit (0-9) on the canvas below and let the CNN predict it!")

if model is not None:
    # --- 3. Drawing Canvas Setup ---
    # Define parameters for the canvas
    canvas_size = 200
    stroke_width = 10 
    
    # Create the drawing canvas
    canvas_result = st_canvas(
        fill_color="#000000",  # Background color (black)
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",  # Drawing color (white)
        background_color="#000000",
        height=canvas_size,
        width=canvas_size,
        drawing_mode="freedraw",
        key="canvas",
    )

    # --- 4. Prediction Logic ---
    if canvas_result.image_data is not None:
        # Get the image data as a 4-channel numpy array (RGBA)
        img_data = canvas_result.image_data
        
        # Check if the user has drawn anything
        if np.sum(img_data[:,:,3]) > 0: # Check if alpha channel has any non-zero pixels
            
            # --- Preprocessing ---
            # 1. Convert to grayscale (use the R channel as drawing is white on black)
            # The canvas returns a white stroke (high R) on a black background (zero R)
            gray_img = img_data[:, :, 0].astype('float32')
            
            # 2. Resize to 28x28 (MNIST standard size)
            pil_img = Image.fromarray(gray_img)
            resized_img = pil_img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # 3. Normalize and reshape for the model (28, 28, 1)
            final_img = np.array(resized_img).astype('float32') / 255.0
            input_tensor = final_img.reshape(1, 28, 28, 1) # Add batch dimension
            
            # --- Prediction ---
            prediction = model.predict(input_tensor)
            predicted_digit = np.argmax(prediction)
            
            st.markdown("---")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Processed Image (28x28)")
                # Display the preprocessed 28x28 image
                st.image(final_img, width=100, clamp=True)
                
            with col2:
                st.subheader("🔮 Prediction:")
                st.metric(label="Predicted Digit", value=f"**{predicted_digit}**")
                
                # Show probability confidence
                st.write("Confidence:")
                
                # Create a small bar chart for probabilities
                prob_df = pd.DataFrame(
                    prediction[0], 
                    index=[str(i) for i in range(10)], 
                    columns=['Probability']
                )
                st.bar_chart(prob_df)

else:
    st.warning("Model could not be loaded. Cannot run classifier.")