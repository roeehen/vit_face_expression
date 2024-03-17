import os
from PIL import Image
import torch
import json
from torchvision import transforms                                             
from transformers import ViTForImageClassification, ViTImageProcessor
from face_cropping import crop_face
import cv2
from scipy.stats import pearsonr
from typing import List
import openpyxl
import csv
from collections import Counter



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

# print model architechture
def print_model_architechture(model):
    print(model)

def image_is_valid(image_path):
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, return True, else return False
    return len(faces) > 0




# Preprocess input image with face cropping
def preprocess_image(image_path, target_size):
    face_image = cv2.imread(image_path)
    # Crop the face
    #face_image = crop_face(image_path)
    

    # Convert the image to RGB format
    image = Image.fromarray(face_image)

    # Resize the image
    image = image.resize((target_size, target_size))

    # Convert image to tensor
    image = transforms.ToTensor()(image)
    inputs = preprocessor(images=image, return_tensors="pt")

    return inputs.pixel_values

# Run model on input image, return analyzed image
def run_model(image_path):
    input_image = preprocess_image(image_path, target_size=224)
    with torch.no_grad():
        outputs = model(input_image, output_hidden_states=True)
    return outputs

# Get the predicted emotion by the model
def predict_emotion(analyzed_image):
    predicted_class = torch.argmax(analyzed_image.logits).item()
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    return emotions[predicted_class]

# calculate the pearson correlation between all the pairs from 3 images over desired layer
def pearson_correlation(analyzed_images, layer_num=12):
    # Extract tensors from analyzed images for the desired layer
    tensors = [analyzed_image.hidden_states[layer_num] for analyzed_image in analyzed_images]
    
    # Initialize a list to store correlation coefficients for all pairs
    correlation_coefficients = []
    
    # Loop through all pairs of tensors
    for i in range(len(tensors)):
        for j in range(i + 1, len(tensors)):
            # Calculate the Pearson correlation coefficient between the flattened tensors
            correlation_coefficient = pearsonr(tensors[i].flatten().tolist(), tensors[j].flatten().tolist())
            
            # Append the pair index and correlation coefficient to the list
            correlation_coefficients.append((correlation_coefficient))
    
    # Return the list of correlation coefficients for all pairs (12, 13, 23)
    return correlation_coefficients

# Print the tensors output of each layer
def print_outputs_layers(analyzed_image):
    hidden_states = analyzed_image.hidden_states
    layer_names = [f"Layer_{i}" for i in range(1, len(hidden_states) + 1)]
    for i, layer_output in enumerate(hidden_states):
        print(f"{layer_names[i]}:")
        print(f"Sequence Length: {layer_output.shape[1]}, Hidden Size: {layer_output.shape[2]}")
        print(layer_output)
        print()

def print_correlation_coefficient(correlation_coefficient):
    print("Pearson correlation coefficient between the two images:", correlation_coefficient)

# Return all the files paths from the desired folder
def get_files_in_folder(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def get_folders_in_folder(folder_path: str) -> list:
    folder_paths = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_paths.append((int(item.split('_')[1]), item_path))
    
    # Sort the folder paths based on the numeric part of the folder names
    sorted_folder_paths = [path for _, path in sorted(folder_paths)]
    
    return sorted_folder_paths

def write_to_excel_triplets_result(image_folder_path: str, data: list[List[List[float]]], output_folder: str, csv_file_path: str):
    # Create a new workbook or load an existing one
    excel_file_path = os.path.join(output_folder, "triplets_result.xlsx")
    if os.path.exists(excel_file_path):
        workbook = openpyxl.load_workbook(excel_file_path)
    else:
        workbook = openpyxl.Workbook()

    # Create a new worksheet or use an existing one
    worksheet = workbook.active
    if "Triplets Result" not in workbook.sheetnames:
        worksheet.title = "Triplets Result"
    else:
        worksheet = workbook["Triplets Result"]

    # Read URLs of the images and human answers from the CSV file
    with open(csv_file_path, 'r') as file:
        max_correlation = 0
        max_column = 0
        reader = csv.reader(file)
        next(reader)
        # Write correlation coefficients to the next columns
        for triplet in (data):
            row = next(reader)
            # write the values only if the model detect face properly
            if (len(triplet[0]) != 3 or (triplet[0][0]==0 and triplet[0][1]==0 and triplet[0][2]==0)):
                continue
            next_row = worksheet.max_row + 1
            human_answers = human_answer(row[17:28:2])
            image_urls = [row[0], row[5], row[10],human_answers[0][0],human_answers[0][1]]
            for col_idx, url in enumerate(image_urls, start=1):
                worksheet.cell(row=next_row, column=col_idx, value=url)

            for layer_num, coefficients in enumerate(triplet):
                worksheet.cell(row=worksheet.max_row + 1, column=6, value=f"Layer {layer_num}")
                for col_idx, coefficient in enumerate(coefficients, start=7):
                    if (coefficient[0]>max_correlation): 
                        max_correlation = coefficient[0]
                        max_column = col_idx - 6
                    worksheet.cell(row=worksheet.max_row, column=col_idx, value=coefficient[0])
                    curr_idx = col_idx
    
                worksheet.cell(row=worksheet.max_row, column=curr_idx + 1, value=model_answer_success_rate(human_answers, max_column))
                max_column = 0
                max_correlation = 0

    workbook.save(excel_file_path)

def model_answer_success_rate (human_answers, max_column):
    # Map the answer if necessary
    if max_column == 1:
        mapped_max = 3
    elif max_column == 3:
        mapped_max = 1
    else:
        mapped_max = max_column
    for human_answer in human_answers:
        if (int(human_answer[0]) == mapped_max): 
            return human_answer[1]
    return 0

def human_answer(answers) -> list:
    # Count occurrences of each answer
    count_answers = Counter(answers)
    
    # Find the most common answers
    most_common_answers = count_answers.most_common(3)
    
    # Initialize a list to store the results
    results = []
    
    # Iterate over the most common answers
    for answer, count in most_common_answers:
        # Map the answer if necessary
        if answer == 1:
            mapped_answer = 3
        elif answer == 3:
            mapped_answer = 1
        else:
            mapped_answer = answer
        
        # Calculate the ratio of the count to the total number of answers
        ratio_count = count / len(answers)
        
        # Append the mapped answer and its ratio to the results list
        results.append((mapped_answer,ratio_count))
    # print (results)
    return results



# Usage
csv_file_path = r"C:\Users\ROEE\vit_data\fec\faceexp-comparison-data-test-public.csv"
excel_output_folder = r"C:\Users\ROEE\vit_face_expression"
image_folder_path = r"C:\\Users\\ROEE\\vit_data\\first_500_triplets"
all_triplets_paths = get_folders_in_folder(image_folder_path)
print ("should be 500 folders, is:", len(all_triplets_paths))
analyzed_images = []
correlation_coefficients = []
triplets_correlations = []

for triplet_folder in (all_triplets_paths):
    print (triplet_folder)
    # make a list of three analyzed images
    for image_path in get_files_in_folder(triplet_folder):
        if image_is_valid(image_path):
            analyzed_image = run_model(image_path)
            analyzed_images.append(analyzed_image)

    # make a list with 13 sets of 3 pearsons correlations, each triplets is the correlation between the pairs
    for layer_num in range (13):
        correlation_coefficient = pearson_correlation(analyzed_images, layer_num)
        correlation_coefficients.append(correlation_coefficient)
    triplets_correlations.append(correlation_coefficients)
    analyzed_images = []
    correlation_coefficients = []

write_to_excel_triplets_result(image_folder_path, triplets_correlations, excel_output_folder, csv_file_path)
