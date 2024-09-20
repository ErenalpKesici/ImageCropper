import cv2
import pytesseract
from pytesseract import Output
from img2table.ocr import TesseractOCR
from img2table.document import Image
import webbrowser
import os

last_title = ''
def process_image(image_folder, image_name):
    global last_title, last_top
    current_title = ''
    current_top = 0
    try:
        # test
        img = cv2.imread(image_folder + '/' + image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        custom_config = r'-l tur --oem 3 --psm 6'
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)

        line_count_to_crop = 0
        for i in range(len(d['text'])):
            if (d['text'][i] == '' or d['text'][i].isupper() or not d['text'][i].isalnum()) and (current_top == 0 or d['top'][i] - current_top < 10):
                current_title += d['text'][i]
                current_top = d['top'][i]
                line_count_to_crop += 1
            else:
                break

        if last_title != '' and current_title == last_title:
            height_to_crop = d['top'][line_count_to_crop] - 10
            crop_img = img[height_to_crop:, :]
            img = cv2.resize(crop_img, (1920, 1080))
        
        last_title = current_title
        last_top = current_top
        cv2.imwrite('cropped/' + image_name, img)

    except Exception as e:
        print(f"Error processing image: {e}")

def process_all_images(image_folder): 
    try:
        image_files = os.listdir(image_folder)
        for image_name in image_files:
            process_image(image_folder, image_name)

    except Exception as e:
        print(f"Error processing images: {e}")

process_all_images('images')