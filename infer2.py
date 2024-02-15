import os
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from torchsummary import summary


# Load the pre-trained ViT model and image processor
model_name = 'trpakov/vit-face-expression'
processor = ViTImageProcessor.from_pretrained(model_name,do_rescale=False)
model = ViTForImageClassification.from_pretrained(model_name)
print(model)

# Preprocess input image
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    # Convert the image to RGB format
    image = image.convert("RGB")
    image = image.resize((target_size, target_size))
    image = transforms.ToTensor()(image)
    inputs = processor(images=image, return_tensors="pt")
    return inputs.pixel_values

# Perform inference
def predict_emotion(image_path):
    input_image = preprocess_image(image_path, target_size=224)
    with torch.no_grad():
        outputs = model(input_image)
    predicted_class = torch.argmax(outputs.logits).item()
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    all_layers = get_all_children(model)

    #Iterate through the children modules
    for i, child in enumerate(all_layers):
        print(f"Layer {i + 1}:")
        try:
            tmp = child(input_image)
            layers_outputs = tmp # Pass input through the child module
        except RuntimeError as e:
            print(f"Error occurred for this image.\nExplanation: {str(e)}")
        except TypeError as e2:
            print(f"Error occurred for this image.\nExplanation: {str(e2)}")
        print(layers_outputs)  # Print the output of the child module


    return emotions[predicted_class]

def get_all_children(model):
    all_children = []
    for child in model.children():
        all_children.append(child)
        all_children.extend(get_all_children(child))
    return all_children

def get_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

folder_path = r"C:\Users\ROEE\OneDrive\מסמכים\חומרים לאוניברסיטה\סמסטר 2024א\סדנת מחקר\data"
all_images_paths = get_files_in_folder(folder_path)
for image_path in all_images_paths:
    predicted_emotion = predict_emotion(image_path)
    print("real emotion:", image_path[76:-4], ", Predicted Emotion:", predicted_emotion)
