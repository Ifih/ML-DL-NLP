# AI in Software Engineering - Week 3 Assignment

This repository contains implementations of various AI and machine learning tasks using Python, including classical machine learning, deep learning, and natural language processing. It also includes a Streamlit web app for interactive digit recognition.

## Files Overview

### app.ipynb
A Jupyter Notebook demonstrating three AI tasks:

1. **Classical ML with Scikit-learn (Iris Dataset)**:
   - Loads the Iris dataset.
   - Preprocesses data (handles missing values).
   - Trains a Decision Tree Classifier.
   - Evaluates the model with accuracy and classification report.

2. **Deep Learning with TensorFlow/Keras (MNIST)**:
   - Loads and preprocesses the MNIST dataset.
   - Builds a Convolutional Neural Network (CNN) for digit classification.
   - Trains the model and evaluates its performance.
   - Visualizes predictions on sample images.

3. **NLP with spaCy (NER and Sentiment Analysis)**:
   - Performs Named Entity Recognition (NER) on a sample product review.
   - Extracts product and brand names.
   - Conducts rule-based sentiment analysis using keyword matching.

#### Requirements
- Python 3.x
- Libraries: numpy, scikit-learn, matplotlib, tensorflow, spacy
- For spaCy: Download the model with `python -m spacy download en_core_web_sm`

#### How to Run
1. Install dependencies: `pip install numpy scikit-learn matplotlib tensorflow spacy`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Open `app.ipynb` in Jupyter Notebook or JupyterLab.
4. Run the cells sequentially.

### Streamlit_app.py
A Streamlit web application for handwritten digit recognition using the trained MNIST CNN model.

- **Features**:
  - Interactive drawing canvas for inputting digits.
  - Real-time prediction of drawn digits (0-9).
  - Displays processed image and confidence probabilities.
  - Loads the pre-trained model from `mnist_cnn_model.h5`.

#### Requirements
- Python 3.x
- Libraries: streamlit, numpy, pandas, tensorflow, streamlit-drawable-canvas, pillow
- Pre-trained model: `mnist_cnn_model.h5` (generated from `app.ipynb`)

#### How to Run
1. Ensure `mnist_cnn_model.h5` is in the same directory (run the MNIST section in `app.ipynb` first if needed).
2. Install dependencies: `pip install streamlit numpy pandas tensorflow streamlit-drawable-canvas pillow`
3. Run the app: `streamlit run Streamlit_app.py`
4. Open the provided local URL (e.g., http://localhost:8501) in your browser.
5. Draw a digit on the canvas and see the prediction.

## Setup Instructions
1. Clone or download this repository.
2. Install Python and required libraries as listed above.
3. For `app.ipynb`, ensure Jupyter is installed (`pip install jupyter`).
4. For `Streamlit_app.py`, run the MNIST training in `app.ipynb` to generate the model file.
5. Execute the respective files as described.

## Notes
- The MNIST model achieves >95% accuracy on test data.
- The Streamlit app preprocesses drawn images to match MNIST format (28x28 grayscale).
- Ensure all dependencies are installed to avoid import errors.

## Author
Falaye Ifeoluwa David
