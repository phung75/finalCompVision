import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt



from pathlib import Path

# Function to get the data from XML Annotation
def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    
    # Initialise the info dict 
    info_dict = {}
    info_dict['bboxes'] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name 
        if elem.tag == "filename":
            info_dict['filename'] = elem.text
            
        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))
            
            info_dict['image_size'] = tuple(image_size)
        
        # Get details of the bounding box 
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    
                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)            
            info_dict['bboxes'].append(bbox)
    
    return info_dict


# Dictionary that maps class names to IDs
class_name_to_id_mapping = {'Chihuahua': 0, 'Japanese_spaniel': 1, 'Maltese_dog': 2, 'Pekinese': 3, 'Shih-Tzu': 4, 'Blenheim_spaniel': 5, 'papillon': 6, 'toy_terrier': 7, 'Rhodesian_ridgeback': 8,
 'Afghan_hound': 9, 'basset': 10, 'beagle': 11, 'bloodhound': 12, 'bluetick': 13, 'black-and-tan_coonhound': 14, 'Walker_hound': 15, 'English_foxhound': 16, 'redbone': 17, 'borzoi': 18, 'Irish_wolfhound': 19, 
 'Italian_greyhound': 20, 'whippet': 21, 'Ibizan_hound': 22, 'Norwegian_elkhound': 23, 'otterhound': 24, 'Saluki': 25, 'Scottish_deerhound': 26, 'Weimaraner': 27, 'Staffordshire_bullterrier': 28, 'American_Staffordshire_terrier': 29, 
 'Bedlington_terrier': 30, 'Border_terrier': 31, 'Kerry_blue_terrier': 32, 'Irish_terrier': 33, 'Norfolk_terrier': 34, 'Norwich_terrier': 35, 'Yorkshire_terrier': 36, 'wire-haired_fox_terrier': 37, 'Lakeland_terrier': 38, 'Sealyham_terrier': 39, 
 'Airedale': 40, 'cairn': 41, 'Australian_terrier': 42, 'Dandie_Dinmont': 43, 'Boston_bull': 44, 'miniature_schnauzer': 45, 'giant_schnauzer': 46, 'standard_schnauzer': 47, 'Scotch_terrier': 48, 'Tibetan_terrier': 49, 'silky_terrier': 50, 'soft-coated_wheaten_terrier': 51,
  'West_Highland_white_terrier': 52, 'Lhasa': 53, 'flat-coated_retriever': 54, 'curly-coated_retriever': 55, 'golden_retriever': 56, 'Labrador_retriever': 57, 'Chesapeake_Bay_retriever': 58, 'German_short-haired_pointer': 59, 'vizsla': 60, 'English_setter': 61, 'Irish_setter': 62, 
  'Gordon_setter': 63, 'Brittany_spaniel': 64, 'clumber': 65, 'English_springer': 66, 'Welsh_springer_spaniel': 67, 'cocker_spaniel': 68, 'Sussex_spaniel': 69, 'Irish_water_spaniel': 70, 'kuvasz': 71, 'schipperke': 72, 'groenendael': 73, 'malinois': 74, 'briard': 75, 'kelpie': 76, 
  'komondor': 77, 'Old_English_sheepdog': 78, 'Shetland_sheepdog': 79, 'collie': 80, 'Border_collie': 81, 'Bouvier_des_Flandres': 82, 'Rottweiler': 83, 'German_shepherd': 84, 'Doberman': 85, 'miniature_pinscher': 86, 'Greater_Swiss_Mountain_dog': 87, 'Bernese_mountain_dog': 88, 'Appenzeller': 89, 'EntleBucher': 90, 'boxer': 91, 'bull_mastiff': 92, 'Tibetan_mastiff': 93, 'French_bulldog': 94, 'Great_Dane': 95, 'Saint_Bernard': 96, 'Eskimo_dog': 97, 'malamute': 98, 'Siberian_husky': 99, 'affenpinscher': 100, 'basenji': 101, 'pug': 102, 'Leonberg': 103, 'Newfoundland': 104, 'Great_Pyrenees': 105, 'Samoyed': 106, 'Pomeranian': 107, 'chow': 108, 'keeshond': 109, 'Brabancon_griffon': 110, 'Pembroke': 111, 'Cardigan': 112, 'toy_poodle': 113, 'miniature_poodle': 114, 'standard_poodle': 115, 'Mexican_hairless': 116, 'dingo': 117, 'dhole': 118, 'African_hunting_dog': 119}

# Convert the info dict to the required yolo format and write it to disk
def convert_to_yolov5(info_dict):
    print_buffer = []
    
    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())
        
        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2 
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width    = (b["xmax"] - b["xmin"])
        b_height   = (b["ymax"] - b["ymin"])
        
        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]  
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        
        #Write the bbox details to the file 
        print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(class_id, b_center_x, b_center_y, b_width, b_height))
        
    # Name of the file which we have to save 
    save_file_name = os.path.join(r"dogoset120\labelsdump", info_dict["filename"])
    print(save_file_name)

    save_file_name = save_file_name + ".txt"
    
    # Save the annotation to disk
    print("\n".join(print_buffer), file= open(save_file_name, "w"))

    


class_id_to_name_mapping = dict(zip(class_name_to_id_mapping.values(), class_name_to_id_mapping.keys()))

def plot_bounding_box(image, annotation_list):
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = transformed_annotations[:,1] - (transformed_annotations[:,3] / 2)
    transformed_annotations[:,2] = transformed_annotations[:,2] - (transformed_annotations[:,4] / 2)
    transformed_annotations[:,3] = transformed_annotations[:,1] + transformed_annotations[:,3]
    transformed_annotations[:,4] = transformed_annotations[:,2] + transformed_annotations[:,4]
    
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        plotted_image.rectangle(((x0,y0), (x1,y1)))
        
        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])
    
    plt.imshow(np.array(image))
    plt.show()


#Utility function to move images 
def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False

    

def main():

    """
    filepath = Path(r"dogoset120\annotations\n02085620-Chihuahua\n02085620_949")
    print(extract_info_from_xml(filepath))

    extracted = extract_info_from_xml(filepath)

    convert_to_yolov5(extracted)
    """
    # Get the 

    #print(os.listdir(r'dogoset120\annotationsdump'))
    annotations = [os.path.join(r'dogoset120\annotationsdump', x) for x in os.listdir(r'dogoset120\annotationsdump')]

    #annotations = os.listdir(r'dogoset120\annotationsdump')

    annotations.sort()
    print(annotations)

    

    # Convert and save the annotations
    for ann in tqdm(annotations):
        info_dict = extract_info_from_xml(ann)
        print(ann)
        convert_to_yolov5(info_dict)
    annotations = [os.path.join(r'dogoset120\annotationsdump', x) for x in os.listdir(r'dogoset120\annotationsdump') if x[-3:] == "txt"]

    print(annotations)
    

    labels = [os.path.join(r'dogoset120\labelsdump', x) for x in os.listdir(r'dogoset120\labelsdump')]

    print(labels)
    # Get any random annotation file 
    annotation_file = random.choice(labels)

    annotation_file = annotation_file

    print(annotation_file)

    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace("labelsdump", "imagesdump").replace("txt", "jpg")
    print(image_file)
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_bounding_box(image, annotation_list)

    
    # Read images and annotations
    images = [os.path.join(r'dogoset120\imagesdump', x) for x in os.listdir(r'dogoset120\imagesdump')]
    annotations = [os.path.join(r'dogoset120\labelsdump', x) for x in os.listdir(r'dogoset120\labelsdump')]

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits 
    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    # Move the splits into their folders
    move_files_to_folder(train_images, r'dogoset120\train\images')
    move_files_to_folder(val_images, r'dogoset120\val\images')
    move_files_to_folder(test_images, r'dogoset120\test\images')
    move_files_to_folder(train_annotations, r'dogoset120\train\labels')
    move_files_to_folder(val_annotations, r'dogoset120\val\labels')
    move_files_to_folder(test_annotations, r'dogoset120\test\labels') 

    
    





main()




