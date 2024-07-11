import streamlit as st
import torch
import torch.nn as nn


# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the gesture labels
gesture_folders = ['01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
                   '06_index', '07_ok', '08_palm_moved', '09_c', '10_down']

# Load the trained model
model = CNNModel(num_classes=len(gesture_folders))
model.load_state_dict(torch.load('hand_gesture_model.pth'))  # Update with your model path
model.eval()

# Define the prediction function
def predict_gesture(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return gesture_folders[predicted.item()]

# Streamlit app
st.title("Hand Gesture Recognition")


# Option to choose between uploading an image or using the camera
option = st.radio("Choose an option:", ("Upload an image", "Use camera"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make prediction
        if st.button('Predict'):
            prediction = predict_gesture(image)
            st.write(f"Predicted Gesture: {prediction}")

elif option == "Use camera":
    # Use Streamlit's file uploader for camera input
    st.write("Use the camera to take a photo.")

    # File uploader for camera capture
    captured_image = st.camera_input("Take a picture")

    if captured_image is not None:
        # Read the captured image
        image = cv2.imdecode(np.frombuffer(captured_image.getvalue(), np.uint8), 1)

        # Display the captured image
        st.image(image, caption='Captured Image', use_column_width=True)

        # Make prediction
        prediction = predict_gesture(image)
        st.write(f"Predicted Gesture: {prediction}")
