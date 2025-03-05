import cv2
import numpy as np
import pytesseract
from pytesseract import Output
# from img2table.ocr import TesseractOCR
# from img2table.document import Image
import webbrowser
import os

last_title = ''


def remove_empty_rows_and_columns(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find all rows that are not empty
    non_empty_rows = np.where(gray.min(axis=1) < 255)[0]
    # Find all columns that are not empty
    non_empty_cols = np.where(gray.min(axis=0) < 255)[0]
    
    # Crop the image to remove empty rows and columns
    if non_empty_rows.size > 0 and non_empty_cols.size > 0:
        image = image[non_empty_rows[0]:non_empty_rows[-1] + 1, non_empty_cols[0]:non_empty_cols[-1] + 1]
    
    return image

def add_outer_border(image, border_size=10):
    outer_color = image[0, 0].tolist()
    image_with_border = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=outer_color
    )
    return image_with_border

def is_image_blank(image, background_threshold=99.0, min_contour_area=500):
    """
    Check if an image is effectively blank (contains only minimal content like page numbers)
    Works with any background color, not just white.
    
    Args:
        image: Input image in BGR format
        background_threshold: Percentage similarity to background color to consider blank (default 99.0%)
        min_contour_area: Minimum contour area to consider as significant content
        
    Returns:
        bool: True if image is considered blank
    """
    # Determine the most common color (likely the background)
    # Convert to RGB for easier analysis
    pixels = image.reshape(-1, 3)
    
    # Find the most common color using a histogram approach
    # This is more efficient than using Counter for large images
    pixel_counts = {}
    for pixel in pixels:
        # Convert BGR array to tuple for dict key
        pixel_tuple = tuple(pixel)
        if pixel_tuple in pixel_counts:
            pixel_counts[pixel_tuple] += 1
        else:
            pixel_counts[pixel_tuple] = 1
    
    # Find the most common color
    background_color = max(pixel_counts.items(), key=lambda x: x[1])[0]
    background_color = np.array(background_color)
    
    # Count pixels that closely match the background color
    # Using a small threshold to account for compression artifacts
    color_distance = np.sqrt(np.sum((pixels - background_color)**2, axis=1))
    background_pixels = np.sum(color_distance < 10)  # Allow some variance
    
    # Calculate percentage of background color
    background_percentage = (background_pixels / pixels.shape[0]) * 100
    
    # Method 1: Check percentage of background-colored pixels
    if background_percentage >= background_threshold:
        return True
    
    # Method 2: Check if there are only small elements (like page numbers)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a mask where background is white and content is black
    mask = np.zeros_like(gray)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.sqrt(np.sum((image[i, j] - background_color)**2)) < 10:  # Close to background color
                mask[i, j] = 255
    
    # Invert to get content as white and background as black
    mask = 255 - mask
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours, image is blank
    if not contours:
        return True
    
    # Check if all contours are small (like page numbers)
    total_contour_area = sum(cv2.contourArea(contour) for contour in contours)
    
    # If total area of all contours is small, consider image blank
    return total_contour_area < min_contour_area

def process_image(image_folder, image_name, operation, params):
    global last_title, last_top
    current_title = ''
    current_top = 0
    try:
        # test
        img = cv2.imread(image_folder + '/' + image_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        custom_config = r'-l tur --oem 3 --psm 6'
        d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)

        match operation:
            case 'title_cropper':
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

                if not os.path.exists('cropped'):
                    os.makedirs('cropped')
                cv2.imwrite('cropped/' + image_name, img)
            
            case 'splitter':
                height, width, _ = img.shape
                min_row_height = height // params[0]
                current_top = min_row_height
                previous_top = 0
                i = 0
                while i < params[0]:
                    for j in range(len(d['top'])):
                        if d['top'][j] > current_top:
                            break
                        if d['text'][j] == '':
                            continue
                        if d['top'][j] + d['height'][j] > current_top:
                            current_top = d['top'][j] + d['height'][j]
                    crop_img = img[previous_top:current_top, :]
                    
                    crop_img = remove_empty_rows_and_columns(crop_img)
                    
                    if is_image_blank(crop_img):
                        previous_top = current_top
                        current_top += min_row_height
                        i += 1
                        continue

                    crop_img = add_outer_border(crop_img)

                    crop_img = cv2.resize(crop_img, (1920, 1080))
                    cv2.imwrite('cropped/' + image_name + '_' + str(i) + '.png', crop_img)
                    previous_top = current_top
                    current_top += min_row_height
                    i += 1

    except Exception as e:
        print(f"Error processing image: {e}")

def process_all_images(image_folder, operation, params):
    try:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        image_files = os.listdir(image_folder)
        for image_name in image_files:
            process_image(image_folder, image_name, operation, params)

    except Exception as e:
        print(f"Error processing images: {e}")

process_all_images('images', 'splitter', [3])