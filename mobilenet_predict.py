import cv2
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# Load class names (same folder structure as training set)
class_names = os.listdir("dataset/train")

# Load trained model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 29)
model.load_state_dict(torch.load("mobilenet70breeds_thingwei.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_frame(frame):
    # Convert BGR to RGB and to PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# Open webcam
cap = cv2.VideoCapture(0)
print("📷 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for display (not model input)
    display_frame = cv2.resize(frame, (640, 480))

    # Predict breed
    try:
        breed = predict_frame(frame)
    except:
        breed = "Error predicting"

    # Display prediction
    cv2.putText(display_frame, f"Breed: {breed}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Dog Breed Detector", display_frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
