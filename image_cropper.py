import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import numpy as np

last_title = ''

def remove_empty_rows_and_columns(image, empty_threshold=100):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use numpy's more efficient operations to find non-empty rows and columns
    row_sums = np.sum(gray < empty_threshold, axis=1)
    col_sums = np.sum(gray < empty_threshold, axis=0)
    
    non_empty_rows = np.where(row_sums > 0)[0]
    non_empty_cols = np.where(col_sums > 0)[0]
    
    # Crop the image to remove empty rows and columns
    if len(non_empty_rows) > 0 and len(non_empty_cols) > 0:
        image = image[non_empty_rows[0]:non_empty_rows[-1] + 1, non_empty_cols[0]:non_empty_cols[-1] + 1]
    
    return image

def add_outer_border(image, top_border=10, bottom_border=10, left_border=20, right_border=20):
    """
    Add an outer border to the image with specified border sizes and color.
    """
    outer_color = image[0, 0].tolist()
    bordered_image = cv2.copyMakeBorder(
        image,
        top_border,
        bottom_border,
        left_border,
        right_border,
        cv2.BORDER_CONSTANT,
        value=outer_color
    )
    return bordered_image

def is_image_blank(image, background_threshold=99.0, min_contour_area=500, std_dev_threshold=10):
    """
    Check if an image is effectively blank (contains only minimal content like page numbers)
    Works with any background color, not just white.
    """
    # Determine the most common color (likely the background) using histogram
    pixels = image.reshape(-1, 3)
    
    # Use numpy's bincount for faster counting
    # Convert to a unique value for each color
    pixel_values = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    unique_values, counts = np.unique(pixel_values, return_counts=True)
    background_value = unique_values[np.argmax(counts)]
    
    # Convert back to BGR
    background_color = np.array([
        (background_value >> 16) & 255,
        (background_value >> 8) & 255,
        background_value & 255
    ])
    
    # Calculate color distance using vectorized operations
    color_distance = np.sqrt(np.sum((pixels - background_color)**2, axis=1))
    background_pixels = np.sum(color_distance < 10)
    
    # Calculate percentage of background color
    background_percentage = (background_pixels / pixels.shape[0]) * 100
    
    # Method 1: Check percentage of background-colored pixels
    if background_percentage >= background_threshold:
        return True
    
       # Method 2: Check standard deviation of pixel values
    std_dev = np.std(pixels, axis=0)
    if np.all(std_dev < std_dev_threshold):
        return True
    
    # Method 3: Check if the image contains only numbers using OCR
    text = pytesseract.image_to_string(image)
    if re.fullmatch(r'\d+', text.strip()):
        return True
    
    return False

def crop_text_area(img, margin=40, page_size=(1920, 1080)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptif threshold ile yazı bölgelerini öne çıkar
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 25, 15)
    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img  # Hiç yazı yoksa orijinali döndür
    # Tüm konturları kapsayan dikdörtgeni bul
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    # Kenar boşluğu ekle
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(img.shape[1] - x, w + 2 * margin)
    h = min(img.shape[0] - y, h + 2 * margin)
    crop = img[y:y+h, x:x+w]
    # Yeni beyaz sayfa oluştur ve ortala
    page = np.full((page_size[1], page_size[0], 3), 255, dtype=np.uint8)
    ch, cw = crop.shape[:2]
    y_offset = (page_size[1] - ch) // 2
    x_offset = (page_size[0] - cw) // 2
    page[y_offset:y_offset+ch, x_offset:x_offset+cw] = crop
    return page

def split_and_center_text_blocks(img, d, lines_per_page=6, page_size=(1920, 1080), margin=80):
    lines = []
    for i in range(len(d['text'])):
        if d['text'][i].strip() != '':
            left = d['left'][i]
            top = d['top'][i]
            width = d['width'][i]
            height = d['height'][i]
            lines.append({'left': left, 'top': top, 'right': left+width, 'bottom': top+height})
    lines = sorted(lines, key=lambda x: x['top'])
    pages = [lines[i:i+lines_per_page] for i in range(0, len(lines), lines_per_page)]
    result_pages = []
    for page_lines in pages:
        if not page_lines:
            continue
        min_top = min(l['top'] for l in page_lines)
        max_bottom = max(l['bottom'] for l in page_lines)
        min_left = min(l['left'] for l in page_lines)
        max_right = max(l['right'] for l in page_lines)
        crop = img[min_top:max_bottom, min_left:max_right]
        # Yazı ile arka planı ayıran fonksiyonu uygula
        page = crop_text_area(crop, margin=margin, page_size=page_size)
        result_pages.append(page)
    return result_pages

def split_image_by_lines_no_ocr(img, lines_per_page=6, page_size=(1920, 1080), margin=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Yazı bölgelerini öne çıkar (beyaz zemin, koyu yazı için)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Gürültü temizliği ve satırları birleştirmek için yatay morfoloji
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1]//30, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Satır konturlarını bul
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Her kontur için dikdörtgen bul ve üstten alta sırala
    lines = [cv2.boundingRect(cnt) for cnt in contours]
    lines = sorted(lines, key=lambda x: x[1])
    # 6'şar gruplara ayır
    pages = [lines[i:i+lines_per_page] for i in range(0, len(lines), lines_per_page)]
    result_pages = []
    for page_lines in pages:
        if not page_lines:
            continue
        # Sayfa içindeki metin bloğunun sınırlarını bul
        min_left = min([x for x, y, w, h in page_lines])
        max_right = max([x+w for x, y, w, h in page_lines])
        min_top = min([y for x, y, w, h in page_lines])
        max_bottom = max([y+h for x, y, w, h in page_lines])
        # Kenar boşluğu ekle
        x1 = max(0, min_left - margin)
        y1 = max(0, min_top - margin)
        x2 = min(img.shape[1], max_right + margin)
        y2 = min(img.shape[0], max_bottom + margin)
        crop = img[y1:y2, x1:x2]
        # Yeni beyaz sayfa oluştur ve ortala
        page = np.full((page_size[1], page_size[0], 3), 255, dtype=np.uint8)
        ch, cw = crop.shape[:2]
        y_offset = (page_size[1] - ch) // 2
        x_offset = (page_size[0] - cw) // 2
        page[y_offset:y_offset+ch, x_offset:x_offset+cw] = crop
        result_pages.append(page)
    return result_pages

def process_image(image_path, operation, params):
    global last_title, last_top
    current_title = ''
    current_top = 0
    
    try:
        # Load image once
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return
            
        image_name = os.path.basename(image_path)
        
        # OCR configuration
        custom_config = r'-l tur --oem 3 --psm 6'
        
        # Only perform OCR when needed
        if operation == 'title_cropper' or operation == 'splitter':
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
                
                cv2.imwrite(os.path.join('cropped', image_name), img)
            
            case 'splitter':
                if not os.path.exists('cropped'):
                    os.makedirs('cropped')
                
                # OCR YOK: Satırları doğrudan görüntü işleme ile bul
                pages = split_image_by_lines_no_ocr(img, lines_per_page=params[0], page_size=(1920, 1080), margin=40)
                for i, page in enumerate(pages):
                    output_path = os.path.join('cropped', f"{image_name}_{i}.png")
                    cv2.imwrite(output_path, page)

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def process_all_images(image_folder, operation, params, max_workers=None):
    try:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        
        # Get list of full image paths
        image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                      if os.path.isfile(os.path.join(image_folder, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No image files found in {image_folder}")
            return
            
        # Set up the process_image function with fixed parameters
        process_fn = partial(process_image, operation=operation, params=params)
        
        # Process images in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_fn, image_files)

    except Exception as e:
        print(f"Error in process_all_images: {e}")

if __name__ == "__main__":
    # Sadece .png dosyalarını işle
    image_folder = 'images'
    if not os.path.exists('cropped'):
        os.makedirs('cropped')
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    max_workers = max(1, num_cores)
    # Sadece .png dosyalarını al
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                   if os.path.isfile(os.path.join(image_folder, f)) and f.lower().endswith('.png')]
    if image_files:
        process_fn = partial(process_image, operation='splitter', params=[6])
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process_fn, image_files)
    else:
        print("No .png image files found in images folder.")