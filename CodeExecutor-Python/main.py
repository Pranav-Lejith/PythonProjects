import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import zipfile
from io import BytesIO, StringIO
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from streamlit_ace import st_ace
import sys

# Set page config
st.set_page_config(page_title="Amphibiar Supergiant", page_icon="🐸", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        height: 60px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        font-size: 18px;
        margin-bottom: 15px;
        text-align: center;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .app-header {
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #333;
        text-align: center;
    }
    .project-description {
        font-size: 18px;
        color: #666;
        margin-bottom: 20px;
        text-align: center;
    }
    .back-button {
        position: fixed;
        top: 10px;
        left: 10px;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'labels' not in st.session_state:
    st.session_state.labels = {}
if 'num_classes' not in st.session_state:
    st.session_state.num_classes = 0
if 'label_mapping' not in st.session_state:
    st.session_state.label_mapping = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None

# Sidebar navigation
st.sidebar.title('Amphibiar Supergiant')
page = st.sidebar.radio('Navigation', ['Home', 'Creatus Model Creator', 'Crop Classifier', 'Python Executor'])

# Back button
if page != 'Home':
    if st.button('Back to Home', key='back_button'):
        st.session_state.page = 'Home'
        st.experimental_rerun()

# Home page
if page == 'Home':
    st.markdown('<p class="app-header">Welcome to Amphibiar Supergiant</p>', unsafe_allow_html=True)
    st.markdown('<p class="project-description">Explore our powerful AI tools for model creation, crop classification, and Python execution.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧠 Creatus Model Creator", key='home_creatus'):
            st.session_state.page = 'Creatus Model Creator'
            st.experimental_rerun()
        st.markdown("Create and train custom AI models with ease.")

    with col2:
        if st.button("🌾 Crop Classifier", key='home_crop'):
            st.session_state.page = 'Crop Classifier'
            st.experimental_rerun()
        st.markdown("Classify crops using state-of-the-art AI technology.")

    with col3:
        if st.button("💻 Python Executor", key='home_python'):
            st.session_state.page = 'Python Executor'
            st.experimental_rerun()
        st.markdown("Execute Python code in real-time within your browser.")

# Creatus Model Creator
elif page == 'Creatus Model Creator':
    st.title('Creatus Model Creator')
    
    # Sidebar for label input
    st.sidebar.title("Manage Labels")
    label_input = st.sidebar.text_input("Enter a new label:")
    if st.sidebar.button("Add Label"):
        if label_input and label_input not in st.session_state.labels:
            st.session_state.labels[label_input] = []
            st.session_state.num_classes += 1
            st.sidebar.success(f"Label '{label_input}' added!")
        else:
            st.sidebar.warning("Label already exists or is empty.")

    # Display labels with delete buttons
    st.sidebar.subheader("Existing Labels")
    for label in list(st.session_state.labels.keys()):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.write(label)
        if col2.button("Delete", key=f"delete_{label}"):
            del st.session_state.labels[label]
            st.session_state.num_classes -= 1

    # Display the existing labels and allow image upload
    if st.session_state.num_classes > 0:
        for label in st.session_state.labels:
            st.subheader(f"Upload images for label: {label}")
            uploaded_files = st.file_uploader(f"Upload images for {label}", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=label)
            
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    image_data = image.load_img(uploaded_file, target_size=(64, 64))
                    image_array = image.img_to_array(image_data)
                    st.session_state.labels[label].append(image_array)
                st.success(f"Uploaded {len(uploaded_files)} images for label '{label}'.")

    # Button to train the model
    if st.session_state.num_classes > 1:
        if st.button("Train Model"):
            all_images = []
            all_labels = []
            st.session_state.label_mapping = {label: idx for idx, label in enumerate(st.session_state.labels.keys())}
            
            for label, images in st.session_state.labels.items():
                all_images.extend(images)
                all_labels.extend([st.session_state.label_mapping[label]] * len(images))
            
            if len(all_images) > 0:
                st.write("Training the model...")
                progress_bar = st.progress(0)
                
                # Define the model architecture
                model = Sequential([
                    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(128, activation='relu'),
                    Dense(st.session_state.num_classes, activation='softmax')
                ])
                
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                
                X = np.array(all_images)
                y = np.array(all_labels)
                X = X / 255.0
                y = to_categorical(y, st.session_state.num_classes)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the model
                history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), 
                                    callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: progress_bar.progress((epoch + 1) / 10))])
                
                st.session_state.model = model
                
                # Calculate metrics
                y_pred = model.predict(X_test)
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_test, axis=1)
                
                st.session_state.metrics = {
                    'accuracy': accuracy_score(y_true_classes, y_pred_classes),
                    'precision': precision_score(y_true_classes, y_pred_classes, average='weighted'),
                    'recall': recall_score(y_true_classes, y_pred_classes, average='weighted'),
                    'f1_score': f1_score(y_true_classes, y_pred_classes, average='weighted')
                }
                
                st.success("Model trained!")
                
                # Display metrics
                st.subheader("Model Performance Metrics")
                for metric, value in st.session_state.metrics.items():
                    st.write(f"{metric.capitalize()}: {value:.4f}")
                
                # Plot metrics
                fig, ax = plt.subplots()
                ax.bar(st.session_state.metrics.keys(), st.session_state.metrics.values())
                ax.set_ylim(0, 1)
                ax.set_title("Model Performance Metrics")
                ax.set_ylabel("Score")
                for i, v in enumerate(st.session_state.metrics.values()):
                    ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')
                st.pyplot(fig)
            else:
                st.error("Please upload some images before training.")
    else:
        st.warning("At least two labels are required to train the model.")

    # Option to test the trained model
    if st.session_state.model is not None:
        st.subheader("Test the trained model")
        test_image = st.file_uploader("Upload an image to test", type=['jpg', 'jpeg', 'png'], key="test")
        
        if test_image:
            test_image_data = image.load_img(test_image, target_size=(64, 64))
            st.image(test_image_data, caption="Uploaded Image", use_column_width=True)
            
            test_image_array = image.img_to_array(test_image_data)
            test_image_array = np.expand_dims(test_image_array, axis=0) / 255.0
            
            prediction = st.session_state.model.predict(test_image_array)
            predicted_class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            
            labels_reverse_map = {v: k for k, v in st.session_state.label_mapping.items()}
            predicted_label = labels_reverse_map[predicted_class_index]
            
            st.write(f"Predicted Label: {predicted_label}")
            st.write(f"Confidence: {confidence:.2f}")

# Crop Classifier
elif page == 'Crop Classifier':
    st.title('Crop Classifier')
    
    # Model selection
    model_selection = st.selectbox(
        "Choose the model",
        ("Wheat and Maize", "Wheat, Maize, Cotton, and Gram")
    )

    if model_selection == "Wheat and Maize":
        model_path = "crop_classifier_model_wheat_maize.tflite"
        class_labels = {0: 'Wheat', 1: 'Maize'}
        image_size = (150, 150)
    else:
        model_path = "crop_classifier_model_wheat_maize_cotton_gram.tflite"
        class_labels = {0: 'Wheat', 1: 'Maize', 2: 'Cotton', 3: 'Gram'}
        image_size = (224, 224)

    # Load the selected model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # File uploader for image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Prepare the image
        image = image.resize(image_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], image_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_class_index = np.argmax(output_data)
        predicted_class = class_labels.get(predicted_class_index, "Unknown")

        st.write(f"🚀 The predicted class of crop is: **{predicted_class}**")

    # Model info
    with st.expander("Model Accuracy Information"):
        st.write("""
        | Model                               | Accuracy       | Crops |
        |-------------------------------------|----------------|-------|
        | Wheat and Maize                     | Medium         | 2     |
        | Wheat, Maize, Cotton, and Gram      | High           | 4     |
        """)

    st.sidebar.title("🌟 About the Project")
    st.sidebar.write("""
    This project uses a machine learning model to classify images of crops into the selected categories.

    Created by **Pranav Lejith (Amphibiar)** for AI Project.
    """)

    st.sidebar.title("💡 Note")
    st.sidebar.write("""
    This model is still in development and may not always be accurate. 

    For the best results, please ensure the wheat images include the stem to avoid confusion with maize. Also make sure that the images of cotton have leaves to avoid confusion with other crops.                     
    """)

    st.sidebar.title("🛠️ Functionality")
    st.sidebar.write("""
    This AI model works by using a convolutional neural network (CNN) to analyze images of crops. 
    The model has been trained on labeled images of the selected crops to learn the distinctive features of each crop. 
    When you upload an image, the model processes it and predicts the crop type based on the learned patterns.
    """)

# Python Executor
elif page == 'Python Executor':
    st.title('Python Executor')

    # Create Ace Editor for Python code
    code = st_ace(language='python', theme='monokai', height=300, key="ace_editor")

    # Button to execute the code
    if st.button("Run Code"):
        # Redirect stdout to capture the output
        old_stdout = sys.stdout
        redirected_output = StringIO()
        sys.stdout = redirected_output
        
        try:
            # Execute the user code
            exec(code)
            output = redirected_output.getvalue()  # Get the output of the code
        except Exception as e:
            output = f"Error: {str(e)}"  # Catch any errors during execution
        
        # Reset stdout
        sys.stdout = old_stdout
        
        # Display the output or error message
        st.subheader("Output:")
        st.code(output, language="python")  # Display output with syntax highlighting

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created by Pranav Lejith (Amphibiar)