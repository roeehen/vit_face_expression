import os
from PIL import Image
import torch
import json
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from face_cropping import crop_face


# Load preprocessor_config.json from the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(script_dir, "preprocessor_config.json")
with open(config_file_path, "r") as f:
    preprocessor_config = json.load(f)

#model name
model_name = 'trpakov/vit-face-expression'

# Initialize ViTImageProcessor with the preprocessor_config
preprocessor = ViTImageProcessor.from_pretrained(model_name, do_rescale=False, config=preprocessor_config)

# Use the same preprocessor_config for processor initialization
processor = ViTImageProcessor.from_pretrained(model_name,do_rescale=False)

# Load ViTForImageClassification model
model = ViTForImageClassification.from_pretrained(model_name)

# # print the model architechture
# # print(model)

# Preprocess input image with face cropping
def preprocess_image(image_path, target_size):
    # Crop the face
    face_image = crop_face(image_path)

    # Convert the image to RGB format
    image = Image.fromarray(face_image)

    # Resize the image
    image = image.resize((target_size, target_size))

    # Convert image to tensor
    image = transforms.ToTensor()(image)
    inputs = preprocessor(images=image, return_tensors="pt")

    return inputs.pixel_values

def predict_emotion(image_path):
    input_image = preprocess_image(image_path, target_size=224)
    with torch.no_grad():
        outputs = model(input_image, output_hidden_states=True)
    predicted_class = torch.argmax(outputs.logits).item()
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    hidden_states = outputs.hidden_states
    print_hidden_states(hidden_states)
    return emotions[predicted_class]

def print_hidden_states(hidden_states):
    layer_names = [f"Layer_{i}" for i in range(1, len(hidden_states) + 1)]
    for i, layer_output in enumerate(hidden_states):
        print(f"{layer_names[i]}:")
        print(f"Sequence Length: {layer_output.shape[1]}, Hidden Size: {layer_output.shape[2]}")
        print(layer_output)
        print()

def get_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

folder_path = r"C:\\Users\\ROEE\\vit_data\\data"
all_images_paths = get_files_in_folder(folder_path)
for image_path in all_images_paths:
    predicted_emotion = predict_emotion(image_path)
    print("real emotion:", image_path[32:-4], ", Predicted Emotion:", predicted_emotion)