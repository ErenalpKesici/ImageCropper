# ImageCropper with Document to PowerPoint Converter
# Enhanced version with improved PowerPoint formatting

import time
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import re
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import multiprocessing
import requests
import io
import urllib.parse
import random

<<<<<<< HEAD
# Initialize optional imports to None
Presentation = None
Inches = None
Pt = None
RGBColor = None
PP_ALIGN = None
docx = None
PyPDF2 = None
PPTX_AVAILABLE = False

# New imports for document to PowerPoint conversion
try:
    from pptx import Presentation as _Presentation
    from pptx.util import Inches as _Inches, Pt as _Pt
    from pptx.dml.color import RGBColor as _RGBColor
    from pptx.enum.text import PP_ALIGN as _PP_ALIGN
    import docx as _docx
    import PyPDF2 as _PyPDF2

    # Assign to module-level variables
    Presentation = _Presentation
    Inches = _Inches
    Pt = _Pt
    RGBColor = _RGBColor
    PP_ALIGN = _PP_ALIGN
    docx = _docx
    PyPDF2 = _PyPDF2
    PPTX_AVAILABLE = True
except ImportError:
    # PPTX_AVAILABLE remains False, and the variables remain None
    # The existing UI and logic checks for PPTX_AVAILABLE will handle this.
    pass
=======
def detect_row_counts(d):
    """
    Count the number of text rows in an OCR result
    Returns the actual number of lines of text
    """
    if not d['text'] or len(d['text']) == 0:
        return 0
        
    # Track unique row positions
    unique_row_positions = set()
    
    # Consider text items that aren't empty
    for i in range(len(d['text'])):
        if d['text'][i].strip() != '':
            # Add the top coordinate to our set of unique positions
            # We use a small tolerance (+-5 pixels) to account for slight vertical misalignments
            # by rounding to nearest 10
            row_position = (d['top'][i] // 10) * 10
            unique_row_positions.add(row_position)
    
    # The number of unique positions is the number of rows
    num_of_rows = len(unique_row_positions)
    
    # Return the number of rows, with a minimum of 1 if there's any text
    return max(1, num_of_rows) if num_of_rows > 0 else 0
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55

def contains_question_indicators(text):
    """Check if the text contains question indicators like a), A), 1), etc."""
    pattern = r'(?:^|\s)([a-zA-Z0-9])[.)]'
    matches = re.findall(pattern, text)
    unique_indicators = set(matches)
    return len(unique_indicators) >= 2

<<<<<<< HEAD
def remove_empty_rows_and_columns(image, empty_threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_sums = np.sum(gray < empty_threshold, axis=1)
    col_sums = np.sum(gray < empty_threshold, axis=0)
    
    non_empty_rows = np.where(row_sums > 0)[0]
    non_empty_cols = np.where(col_sums > 0)[0]
    
=======
def remove_empty_rows_and_columns(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Hem çok açık hem çok koyu pikselleri boş kabul et
    mask = ((gray > 30) & (gray < 220)).astype(np.uint8)
    row_sums = np.sum(mask, axis=1)
    col_sums = np.sum(mask, axis=0)
    non_empty_rows = np.where(row_sums > 0)[0]
    non_empty_cols = np.where(col_sums > 0)[0]
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
    if len(non_empty_rows) > 0 and len(non_empty_cols) > 0:
        image = image[non_empty_rows[0]:non_empty_rows[-1]+1, non_empty_cols[0]:non_empty_cols[-1]+1]
    return image

def add_outer_border(image, top_border=10, bottom_border=10, left_border=20, right_border=20):
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
    """Check if an image is effectively blank"""
    pixels = image.reshape(-1, 3)
    pixel_values = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    unique_values, counts = np.unique(pixel_values, return_counts=True)
    background_value = unique_values[np.argmax(counts)]
    
    background_color = np.array([
        (background_value >> 16) & 255,
        (background_value >> 8) & 255,
        background_value & 255
    ])
    
    color_distance = np.sqrt(np.sum((pixels - background_color)**2, axis=1))
    background_pixels = np.sum(color_distance < 10)
    background_percentage = (background_pixels / pixels.shape[0]) * 100
    
    if background_percentage >= background_threshold:
        return True
    
    std_dev = np.std(pixels, axis=0)
    if np.all(std_dev < std_dev_threshold):
        return True
    
    text = pytesseract.image_to_string(image)
    if text == '' or re.fullmatch(r'\d+', text.strip()):
        return True
    
    return False

<<<<<<< HEAD
class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper & Document Converter")
        self.root.geometry("800x700")
        self.root.minsize(800, 700)
=======
def remove_template_from_image(image, template, threshold=0.4, max_matches=20):
    """
    Enhanced logo removal function combining multiple techniques
    """
    # First resize large images to improve performance
    h_img, w_img = image.shape[:2]
    h_temp, w_temp = template.shape[:2]
    
    # Skip if template is too large compared to image
    if h_temp > h_img or w_temp > w_img:
        return image
        
    # Create a copy for the result
    result_image = image.copy()
    
    # 1. Feature-based matching (for rotated/transformed logos)
    try:
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) > 2 else template
        
        # Use SIFT for feature detection
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray_template, None)
        kp2, des2 = sift.detectAndCompute(gray_image, None)
        
        if des1 is not None and des2 is not None and len(kp1) > 2 and len(kp2) > 2:
            # Feature matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Only keep good matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            if len(good_matches) >= 4:
                # Extract good match points
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Find homography to detect logo even with perspective transformation
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Get the corners of the template
                    h, w = gray_template.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    
                    # Transform corners to image coordinates
                    dst = cv2.perspectiveTransform(pts, H)
                    
                    # Convert to integer
                    polygon = np.int32(dst)
                    
                    # Create a mask for the found region
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [polygon], 255)
                    
                    # Expand the mask slightly
                    kernel = np.ones((5, 5), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    # Inpaint the detected region
                    result_image = cv2.inpaint(result_image, mask, 5, cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Feature-based matching failed: {e}")
    
    # 2. Classic template matching with multiple methods and thresholds
    try:
        gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) > 2 else template
        
        # Try different matching methods
        methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED]
        
        for method in methods:
            # Perform template matching
            result = cv2.matchTemplate(gray_image, gray_template, method)
            
            # Find locations where match quality exceeds threshold
            locations = np.where(result >= threshold)
            points = list(zip(*locations[::-1]))  # Switch columns and rows
            
            # Sort points by match quality and limit to max_matches
            if len(points) > max_matches:
                match_values = [result[pt[1], pt[0]] for pt in points]
                sorted_indices = np.argsort(match_values)[::-1]  # Sort descending
                points = [points[i] for i in sorted_indices[:max_matches]]
            
            # Process all matching locations
            for pt in points:
                h, w = gray_template.shape
                
                # Use a larger mask for more aggressive removal
                padding = 5  # Add more padding for better removal
                y_start = max(0, pt[1] - padding)
                y_end = min(result_image.shape[0], pt[1] + h + padding)
                x_start = max(0, pt[0] - padding)
                x_end = min(result_image.shape[1], pt[0] + w + padding)
                
                # Create a mask for inpainting
                mask = np.zeros(result_image.shape[:2], np.uint8)
                mask[y_start:y_end, x_start:x_end] = 255
                
                # Inpaint the region with a larger radius
                result_image = cv2.inpaint(result_image, mask, 7, cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Template matching failed: {e}")
    
    # 3. Color-based logo detection (if the logo has distinctive colors)
    try:
        # Convert to HSV for better color matching
        hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2HSV)
        
        # Calculate the color histogram of the template
        hist_template = cv2.calcHist([hsv_template], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist_template, hist_template, 0, 255, cv2.NORM_MINMAX)
        
        # Use backprojection to find regions with similar color distribution
        dst = cv2.calcBackProject([hsv_image], [0, 1], hist_template, [0, 180, 0, 256], 1)
        
        # Apply a threshold to the backprojection
        _, mask = cv2.threshold(dst, 100, 255, cv2.THRESH_BINARY)
        mask = cv2.merge((mask, mask, mask))
        
        # Remove small noise with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Convert mask to single channel for inpainting
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size (similar to template size)
        template_area = h_temp * w_temp
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > template_area * 0.5 and area < template_area * 3:
                # Create a mask for this contour
                cont_mask = np.zeros(result_image.shape[:2], np.uint8)
                cv2.drawContours(cont_mask, [contour], 0, 255, -1)
                
                # Inpaint the contour region
                result_image = cv2.inpaint(result_image, cont_mask, 5, cv2.INPAINT_TELEA)
    except Exception as e:
        print(f"Color-based detection failed: {e}")
    
    return result_image

def ensure_dimensions_divisible_by_8(rect):
    """Adjust rectangle dimensions to ensure width and height are divisible by 8 for AI models"""
    x, y, w, h = rect
    
    # Calculate the closest dimensions divisible by 8
    adjusted_w = (w // 8) * 8
    adjusted_h = (h // 8) * 8
    
    # If rounding down makes the dimensions too small, round up instead
    if adjusted_w < w - 4:
        adjusted_w += 8
    if adjusted_h < h - 4:
        adjusted_h += 8
    
    # Center the adjusted rectangle in the original space
    x_offset = (w - adjusted_w) // 2
    y_offset = (h - adjusted_h) // 2
    
    adjusted_x = x + x_offset
    adjusted_y = y + y_offset
    
    return (adjusted_x, adjusted_y, adjusted_w, adjusted_h)

def search_google_images(self, query, min_width=300, min_height=300):
    """Use Google Custom Search API to find images"""
    try:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        cx = os.environ.get("GOOGLE_SEARCH_CX", "")
        
        if not api_key or not cx:
            return None
            
        self.status_var.set(f"Google kullanılarak görseller aranıyor: {query}")
        
        search_url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "searchType": "image",
            "num": 10,
            "imgSize": "large",
            "rights": "cc_publicdomain,cc_attribute,cc_sharealike"
        }
        print(params)
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        search_results = response.json()
        
        # Filter for minimum dimensions if metadata is available
        candidates = []
        for item in search_results.get("items", []):
            if "image" in item:
                width = int(item["image"].get("width", 0))
                height = int(item["image"].get("height", 0))
                if width >= min_width and height >= min_height:
                    candidates.append(item)
        
        # If no suitable candidates, use any results
        if not candidates and "items" in search_results:
            candidates = search_results["items"]
        
        if not candidates:
            return None
            
        # Pick a random image from the top results
        selected_image = random.choice(candidates[:5]) if len(candidates) >= 5 else candidates[0]
        
        # Download the image
        img_url = selected_image["link"]
        img_response = requests.get(img_url, timeout=5)
        img_response.raise_for_status()
        
        # Convert to OpenCV format
        img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        self.status_var.set("Google Özel Arama ile görsel bulundu")
        return img
            
    except Exception as e:
        print(f"Google görsel arama hatası: {e}")
        return None

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görsel Kırpıcı")
        self.root.geometry("800x700")  # Varsayılan yükseklik artırıldı
        self.root.minsize(800, 700)    # Minimum yükseklik artırıldı
        
        # Create default frame for all content
        self.main_content = ttk.Frame(self.root)
        self.main_content.pack(fill=tk.BOTH, expand=True)
        
        # Create fixed frame for buttons that stays at bottom
        self.button_area = ttk.Frame(self.root)
        self.button_area.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        
        self.input_files = []
        self.output_dir = os.path.join(os.getcwd(), "cropped")
        self.last_title = ''
        self.last_top = 0
        self.processing_cancelled = False
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame with scrolling
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input section
        input_frame = ttk.LabelFrame(scrollable_frame, text="Girdi", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_files_label = ttk.Label(input_frame, text="Dosya seçilmedi")
        self.input_files_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Button frame for file selection
        select_buttons_frame = ttk.Frame(input_frame)
        select_buttons_frame.pack(side=tk.RIGHT)
        
        select_folder_button = ttk.Button(select_buttons_frame, text="Klasör Seç", command=self.select_folder)
        select_folder_button.pack(side=tk.LEFT, padx=5)
        
<<<<<<< HEAD
        select_images_button = ttk.Button(select_buttons_frame, text="Select Images", command=self.select_files)
        select_images_button.pack(side=tk.LEFT, padx=5)
        
        select_docs_button = ttk.Button(select_buttons_frame, text="Select Documents", command=self.select_documents)
        select_docs_button.pack(side=tk.LEFT, padx=5)
=======
        select_button = ttk.Button(select_buttons_frame, text="Dosya Seç", command=self.select_files)
        select_button.pack(side=tk.LEFT, padx=5)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        
        # Output section
        output_frame = ttk.LabelFrame(scrollable_frame, text="Çıktı", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        self.output_dir_label = ttk.Label(output_frame, text=self.output_dir)
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_button = ttk.Button(output_frame, text="Klasör Seç", command=self.select_output_dir)
        output_button.pack(side=tk.RIGHT, padx=5)
        
<<<<<<< HEAD
        # Operation section
        operation_frame = ttk.LabelFrame(scrollable_frame, text="Operation", padding="10")
        operation_frame.pack(fill=tk.X, pady=5)
        
        self.operation_var = tk.StringVar(value="splitter")
        operations = [
            ("Split Images", "splitter"), 
            ("Crop Titles", "title_cropper"),
            ("Remove Blank Images", "blank_remover"),
            ("Convert Document to PowerPoint", "doc_to_pptx")
        ]
=======
        
        # Add a save settings button
        save_settings_button = ttk.Button(output_frame, text="Varsayılanı Kaydet", command=self.save_settings)
        save_settings_button.pack(anchor=tk.W, pady=5)
        
        # Create a frame to hold operation and parameters side by side
        op_param_container = ttk.Frame(scrollable_frame)
        op_param_container.pack(fill=tk.X, pady=5)
        
        # Operation section - now in left half
        operation_frame = ttk.LabelFrame(op_param_container, text="İşlem", padding="10")
        operation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.operation_var = tk.StringVar(value="splitter")
        operations = [("Görselleri Böl", "splitter"), 
             ("Başlıkları Kırp", "title_cropper"),
             ("Şablonları Kaldır", "template_remover"),
             ("Boş Görselleri Kaldır", "blank_remover"),
             ("Eksik Görsel Oluştur", "image_generator")]
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        
        for text, value in operations:
            ttk.Radiobutton(operation_frame, text=text, value=value, variable=self.operation_var).pack(anchor=tk.W)
        
<<<<<<< HEAD
        # Parameters section
        param_frame = ttk.LabelFrame(scrollable_frame, text="Parameters", padding="10")
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Lines per slide (for document conversion):").pack(anchor=tk.W)
        self.lines_per_slide_var = tk.StringVar(value="7")
        lines_entry = ttk.Entry(param_frame, textvariable=self.lines_per_slide_var, width=10)
        lines_entry.pack(anchor=tk.W, pady=5)
        
        # Add recommendation label
        rec_label = ttk.Label(param_frame, text="Recommended: 5-8 lines for no overflow, larger fonts", font=("Arial", 8))
        rec_label.pack(anchor=tk.W)
        
        ttk.Label(param_frame, text="Rows per page (for image splitting):").pack(anchor=tk.W, pady=(10, 0))
        self.rows_per_page_var = tk.StringVar(value="10")
        rows_entry = ttk.Entry(param_frame, textvariable=self.rows_per_page_var, width=10)
        rows_entry.pack(anchor=tk.W, pady=5)
=======
        # Parameters section - now in right half
        param_frame = ttk.LabelFrame(op_param_container, text="Parametreler", padding="10")
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(param_frame, text="Sayfa başına satır:").pack(anchor=tk.W)

        self.rows_per_page_var = tk.StringVar(value="10")
        self.splits_var = tk.StringVar(value="2")  # Default to 2 splits as fallback
        rows_per_page_entry = ttk.Entry(param_frame, textvariable=self.rows_per_page_var, width=5)
        rows_per_page_entry.pack(anchor=tk.W, pady=5)

        # Add a checkbox for using auto-detected rows
        self.use_row_detection_var = tk.BooleanVar(value=True)
        use_row_detection_checkbox = ttk.Checkbutton(
            param_frame, 
            text="Satırları otomatik algıla (önerilir)",
            variable=self.use_row_detection_var
        )
        use_row_detection_checkbox.pack(anchor=tk.W, pady=(0, 10))

        # Add a slider for template matching threshold
        ttk.Label(param_frame, text="Şablon Eşleşme Hassasiyeti:").pack(anchor=tk.W, pady=(10, 0))
        self.template_sensitivity_var = tk.DoubleVar(value=0.6)
        template_sensitivity_scale = ttk.Scale(
            param_frame, 
            from_=0.3, 
            to=0.8, 
            orient="horizontal", 
            variable=self.template_sensitivity_var, 
            length=150
        )
        template_sensitivity_scale.pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Label(param_frame, text="(Düşük = Daha agresif kaldırma)").pack(anchor=tk.W)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        
        self.detect_questions_var = tk.BooleanVar(value=True)
        question_checkbox = ttk.Checkbutton(
            param_frame, 
            text="Soruları algıla (bölme)",
            variable=self.detect_questions_var
        )
        question_checkbox.pack(anchor=tk.W, pady=(10, 0))
<<<<<<< HEAD
=======

        # Parameters for blank image detection
        ttk.Label(param_frame, text="Boş Görsel Algılama Hassasiyeti:").pack(anchor=tk.W, pady=(10, 0))
        self.blank_sensitivity_var = tk.DoubleVar(value=98.0)
        blank_sensitivity_scale = ttk.Scale(
            param_frame, 
            from_=85.0, 
            to=99.5, 
            orient="horizontal", 
            variable=self.blank_sensitivity_var, 
            length=150
        )
        blank_sensitivity_scale.pack(anchor=tk.W, pady=(0, 5), fill=tk.X)
        ttk.Label(param_frame, text="(Yüksek = Daha agresif boş algılama)").pack(anchor=tk.W)

        # Parameters for image generation
        ttk.Label(param_frame, text="Yapay Zeka Görsel Stili:").pack(anchor=tk.W, pady=(10, 0))
        self.image_style_var = tk.StringVar(value="educational")
        styles = [("Eğitsel", "educational"), 
                 ("Fotoğraf Gerçekçiliği", "photorealistic"),
                 ("Çizgi Film", "cartoon"),
                 ("Soyut", "abstract")]

        style_frame = ttk.Frame(param_frame)
        style_frame.pack(fill=tk.X, pady=5)

        # Create style options
        for text, value in styles:
            ttk.Radiobutton(style_frame, text=text, value=value, 
                           variable=self.image_style_var).pack(anchor=tk.W)

        # Option to set API key
        ttk.Label(param_frame, text="OpenAI API Anahtarı (isteğe bağlı):").pack(anchor=tk.W, pady=(10, 0))
        self.api_key_var = tk.StringVar()
        api_key_entry = ttk.Entry(param_frame, textvariable=self.api_key_var, show="*")
        api_key_entry.pack(anchor=tk.W, pady=5, fill=tk.X)

        # Add API key options for image search
        ttk.Label(param_frame, text="Görsel Arama API Anahtarları:").pack(anchor=tk.W, pady=(10, 0))
        
        # Bing Search API Key
        api_key_frame = ttk.Frame(param_frame)
        api_key_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(api_key_frame, text="Bing API Anahtarı:").pack(side=tk.LEFT)
        self.bing_api_key_var = tk.StringVar()
        bing_api_entry = ttk.Entry(api_key_frame, textvariable=self.bing_api_key_var, show="*", width=20)
        bing_api_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Add option to choose between web search and AI generation
        self.image_source_var = tk.StringVar(value="web_first")
        image_source_frame = ttk.Frame(param_frame)
        image_source_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(image_source_frame, text="Görsel Kaynağı:").pack(anchor=tk.W)
        
        sources = [
            ("Önce Web Arama, sonra Yapay Zeka (önerilir)", "web_first"),
            ("Sadece Yapay Zeka", "ai_only"),
            ("Sadece Web Arama", "web_only")
        ]
        
        for text, value in sources:
            ttk.Radiobutton(image_source_frame, text=text, value=value, 
                          variable=self.image_source_var).pack(anchor=tk.W)

        # Preview area - will show thumbnails of selected images, with reduced height
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Önizleme", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=1)

        self.preview_frame = ttk.Frame(preview_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_frame, height=150)  # Yükseklik biraz artırıldı
        scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)        # Add this in create_widgets method, right before the status_var definition
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        
        # Preview area
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_area = ttk.Frame(preview_frame)
        self.preview_area.pack(fill=tk.BOTH, expand=True)
        
        # Progress and status
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)
<<<<<<< HEAD
        
        self.status_var = tk.StringVar(value="Ready")
=======

        self.status_var = tk.StringVar(value="Hazır")
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(scrollable_frame)
<<<<<<< HEAD
        button_frame.pack(fill=tk.X, pady=10)
        
        process_button = ttk.Button(button_frame, text="Process", command=self.process_files)
=======
        button_frame.pack(fill=tk.X, pady=10)  # Added fill=tk.X

        process_button = ttk.Button(button_frame, text="Görselleri İşle", command=self.process_images)
        process_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="İptal", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Open output folder button - move to button_frame
        open_folder_button = ttk.Button(button_frame, text="Çıktı Klasörünü Aç", command=self.open_output_folder)
        open_folder_button.pack(side=tk.RIGHT, padx=5)        
        
        # Add a new section for template images that appears when "Remove Templates" is selected
        self.templates_frame = ttk.LabelFrame(scrollable_frame, text="Kaldırılacak Filigran/Logo Şablonları", padding="10")

        self.template_files = []
        self.template_files_label = ttk.Label(self.templates_frame, text="Şablon seçilmedi")
        self.template_files_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        select_templates_button = ttk.Button(self.templates_frame, text="Filigran/Logo Seç", 
                                             command=self.select_templates)
        select_templates_button.pack(side=tk.RIGHT)
        
        # Show/hide the templates frame based on operation selection
        self.operation_var.trace("w", self.toggle_templates_frame)

        # Add this button to your params frame
        ttk.Button(param_frame, text="Gelişmiş Görsel Ayarları", command=self.show_image_generation_options).pack(anchor=tk.W, pady=10)
        
        # Add a fixed-position frame at the bottom for action buttons that always stay visible
        button_container = ttk.Frame(self.root)
        button_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Move your buttons to this new container
        button_frame = ttk.Frame(button_container)
        button_frame.pack(fill=tk.X)
        
        process_button = ttk.Button(button_frame, text="Görselleri İşle", command=self.process_images)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        process_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="İptal", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
<<<<<<< HEAD
        open_folder_button = ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder)
=======
        # Open output folder button - move to button_frame
        open_folder_button = ttk.Button(button_frame, text="Çıktı Klasörünü Aç", command=self.open_output_folder)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        open_folder_button.pack(side=tk.RIGHT, padx=5)
    
    def select_files(self):
<<<<<<< HEAD
        """Select image files"""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
=======
        filetypes = [("Görsel dosyalar", "*.png *.jpg *.jpeg *.bmp *.tiff")]
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.input_files = list(files)
<<<<<<< HEAD
            self.input_files_label.config(text=f"{len(self.input_files)} image(s) selected")
=======
            self.input_files_label.config(text=f"{len(self.input_files)} dosya seçildi")
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
            self.update_preview()
    
    def select_documents(self):
        """Select Word or PDF documents for conversion to PowerPoint"""
        if not PPTX_AVAILABLE:
            messagebox.showerror("Missing Libraries", 
                "Document conversion requires python-pptx, python-docx, and PyPDF2.\n"
                "Install with: pip install python-pptx python-docx PyPDF2")
            return
            
        filetypes = [("Document files", "*.docx *.pdf"), ("Word files", "*.docx"), ("PDF files", "*.pdf")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.input_files = list(files)
            self.input_files_label.config(text=f"{len(self.input_files)} document(s) selected")
            self.update_document_preview()
    
    def select_folder(self):
        """Select a folder and add all images from it"""
        folder = filedialog.askdirectory()
        if folder:
            image_files = []
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(folder, file))
                    
            if image_files:
                self.input_files = image_files
                self.input_files_label.config(text=f"{len(self.input_files)} image(s) from folder")
                self.update_preview()
            else:
                messagebox.showinfo("No Images", "No image files found in the selected folder.")
    
    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.config(text=self.output_dir)
    
    def update_preview(self):
        """Show thumbnails of selected images"""
        for widget in self.preview_area.winfo_children():
            widget.destroy()
            
        max_previews = min(5, len(self.input_files))
        self.thumbnail_refs = []
        
        for i in range(max_previews):
            try:
                img = Image.open(self.input_files[i])
                img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img)
                
                self.thumbnail_refs.append(photo)
                
                frame = ttk.Frame(self.preview_area)
                frame.pack(side=tk.LEFT, padx=5)
                
                label = ttk.Label(frame, image=photo)
                label.pack()
                
                name_label = ttk.Label(frame, text=os.path.basename(self.input_files[i]), wraplength=150)
                name_label.pack()
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
<<<<<<< HEAD
    
    def update_document_preview(self):
        """Show document names for preview"""
        for widget in self.preview_area.winfo_children():
            widget.destroy()
            
        for i, doc_path in enumerate(self.input_files[:5]):
            frame = ttk.Frame(self.preview_area)
            frame.pack(side=tk.LEFT, padx=5)
            
            name_label = ttk.Label(frame, text=os.path.basename(doc_path), wraplength=150)
            name_label.pack()
            
            ext = os.path.splitext(doc_path)[1].upper()
            type_label = ttk.Label(frame, text=f"{ext} Document", font=("Arial", 8))
            type_label.pack()
    
    def convert_document_to_pptx(self, document_path, lines_per_slide):
        """Convert Word or PDF document to PowerPoint presentation with NO overflow"""
        if not PPTX_AVAILABLE:
            self.status_var.set("PowerPoint libraries not available")
            return False
            
        try:
            text_content = []
            file_ext = os.path.splitext(document_path)[1].lower()
            
            self.status_var.set(f"Reading {file_ext} file...")
            self.root.update()
            
            if file_ext == '.docx':
                doc = docx.Document(document_path)
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content.append(paragraph.text.strip())
            
            elif file_ext == '.pdf':
                with open(document_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        lines = [line.strip() for line in text.split('\n') if line.strip()]
                        text_content.extend(lines)
            
            if not text_content:
                self.status_var.set(f"No text found in {document_path}")
                return False
            
            # Create PowerPoint presentation
            prs = Presentation()
            prs.slide_width = Inches(16)
            prs.slide_height = Inches(9)

            slide_layout = prs.slide_layouts[6]  # Blank layout
            
            current_slide_lines = []
            slide_count = 0
            
            self.status_var.set("Creating slides with optimal spacing...")
            self.root.update()
            
            # Process lines with STRICT line limit to prevent overflow
            for line in text_content:
                current_slide_lines.append(line)
                
                # Create new slide when we reach the EXACT line limit
                if len(current_slide_lines) >= lines_per_slide:
                    slide_count += 1
                    self.create_optimized_slide(prs, slide_layout, current_slide_lines)
                    current_slide_lines = []
              # Add remaining lines to final slide
            if current_slide_lines:
                slide_count += 1
                self.create_optimized_slide(prs, slide_layout, current_slide_lines)
            
            # Save the presentation
            base_name = os.path.splitext(os.path.basename(document_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}.pptx")
            prs.save(output_path)
            
            self.status_var.set(f"Created PowerPoint: {base_name}.pptx with {slide_count} slides")
            return True
            
        except Exception as e:
            self.status_var.set(f"Error converting {document_path}: {str(e)}")
            print(f"Error converting document {document_path}: {e}")
            return False
    
    def calculate_text_height(self, lines, font_size, line_spacing=1.1):
        """Calculate approximate text height based on font size and line count"""
        line_height = font_size * line_spacing * 1.33  # Convert pt to approximate pixels
        total_height = len(lines) * line_height
        # Add some padding for margins and spacing
        return total_height + (len(lines) * 4)  # 4pt space_after per line
    
    def find_optimal_font_size(self, lines, available_height_inches, text_type='content'):
        """Find the maximum font size that fits without overflow"""
        available_height_pt = available_height_inches * 72  # Convert inches to points
        
        # Define size ranges based on text type
        if text_type == 'title':
            min_size, max_size = 24, 42
        elif text_type == 'subtitle':
            min_size, max_size = 22, 38
        elif text_type == 'bullet':
            min_size, max_size = 20, 34
        else:  # content
            min_size, max_size = 18, 30
          # Binary search for optimal font size
        optimal_size = min_size
        
        for size in range(min_size, max_size + 1, 2):  # Increment by 2pt
            estimated_height = self.calculate_text_height(lines, size)
            if estimated_height <= available_height_pt:
                optimal_size = size
            else:
                break
        
        return optimal_size
    
    def create_optimized_slide(self, prs, slide_layout, lines):
        """Create slide with dynamic font sizing for maximum content optimization"""
        slide = prs.slides.add_slide(slide_layout)
        
        # Use FULL slide dimensions for maximum space utilization
        left = Inches(0.3)      # Minimal left margin
        top = Inches(0.3)       # Minimal top margin  
        width = Inches(15.4)    # Almost full width (16" - 0.6" margins)
        height = Inches(8.4)    # Almost full height (9" - 0.6" margins)
        
        textbox = slide.shapes.add_textbox(left, top, width, height)
        text_frame = textbox.text_frame
        text_frame.word_wrap = True
        text_frame.auto_size = None  # Critical: prevent auto-sizing
        
        # Minimal margins for maximum space
        text_frame.margin_left = Inches(0.05)
        text_frame.margin_right = Inches(0.05)
        text_frame.margin_top = Inches(0.05)
        text_frame.margin_bottom = Inches(0.05)
        
        # Available height for text (accounting for minimal margins)
        available_height = height - Inches(0.1)  # Subtract tiny margins
        
        # Check if first line is a title
        is_title = len(lines) > 0 and (
            lines[0].isupper() or 
            len(lines[0]) < 50 or 
            any(keyword in lines[0].lower() for keyword in 
                ['chapter', 'section', 'introduction', 'conclusion', 'summary', 'overview'])
        )
        
        # Calculate optimal font sizes for different content types
        title_lines = [lines[0]] if is_title and len(lines) > 0 else []
        content_lines = lines[1:] if is_title else lines
        
        # Find optimal font sizes
        title_font_size = 24  # Default title font size
        if title_lines:
            title_font_size = self.find_optimal_font_size([title_lines[0]], available_height * 0.25, 'title')
        
        base_font_size = 16  # Default base font size
        if content_lines:
            # Reserve space for title if present
            content_height = available_height * 0.75 if title_lines else available_height
            
            # Calculate base font size for all content combined
            base_font_size = self.find_optimal_font_size(content_lines, content_height, 'content')
        
        print(f"🎯 Slide with {len(lines)} lines - Title: {title_font_size}pt, Content: {base_font_size}pt")
        
        for i, line in enumerate(lines):
            if i == 0:
                p = text_frame.paragraphs[0]
            else:
                p = text_frame.add_paragraph()
            
            p.text = line
            p.font.name = 'Calibri'
            p.word_wrap = True
            
            # Dynamic font sizing based on content type and available space
            if i == 0 and is_title:
                # Title: Dynamic sizing with emphasis
                p.font.size = Pt(max(title_font_size, 18))  # Minimum 18pt for titles
                p.font.bold = True
                p.alignment = PP_ALIGN.CENTER
                p.font.color.rgb = RGBColor(31, 73, 125)
                p.space_after = Pt(max(4, title_font_size // 4))
            elif line.startswith(('•', '-', '*', '▪', '○')) or any(line.startswith(f'{j}.') for j in range(1, 50)):
                # Bullet points: Slightly smaller than regular content
                p.font.size = Pt(max(base_font_size - 2, 12))
                p.font.bold = False
                p.alignment = PP_ALIGN.LEFT
                p.space_after = Pt(2)
            elif len(line) < 60 and not line.endswith(('.', '!', '?')) and i > 0:
                # Subtitles: Larger than regular content
                p.font.size = Pt(max(base_font_size + 2, 14))
                p.font.bold = True
                p.alignment = PP_ALIGN.LEFT
                p.font.color.rgb = RGBColor(68, 84, 106)
                p.space_after = Pt(3)
            else:
                # Regular content: Optimized dynamic size
                p.font.size = Pt(max(base_font_size, 12))  # Minimum 12pt for readability
                p.font.bold = False
                p.alignment = PP_ALIGN.LEFT
                p.space_after = Pt(1)
            
            # Very tight line spacing to maximize content
            p.line_spacing = 0.9
    
    def process_single_image(self, image_path, operation, params):
        """Process a single image based on the selected operation"""
=======
        
        self.preview_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def generate_image_for_text(self, text, size=(512, 512)):
        """Generate an image based on text content, prioritizing web search before AI generation"""
        try:
            self.status_var.set("Web'de görseller aranıyor...")
            
            # Clean the text for better search results
            clean_text = re.sub(r'[^\w\s]', '', text)
            clean_text = ' '.join(clean_text.split()[:10])  # Limit to 10 words for search
            
            # Try to get image from web search first
            web_image = self.search_web_image(clean_text)
            
            if web_image is not None:
                self.status_var.set("Web aramasından görsel bulundu!")
                return web_image
                
            # If web search fails, fall back to AI generation
            self.status_var.set("Çevrimiçi uygun görsel bulunamadı. Yapay Zeka ile oluşturuluyor...")
            
            # Set custom HF_HOME path before importing huggingface libraries
            import os
            
            # Specify a different drive location for cache
            custom_cache_dir = "D:/ml_cache/huggingface"  # Change this to your preferred location
            os.environ["HF_HOME"] = custom_cache_dir
            import torch
            from diffusers import StableDiffusionPipeline
            import gc
            
            # Set image style based on selection
            style = self.image_style_var.get()
            
            # Create style-specific prompt
            if style == "educational":
                prompt_prefix = "Basit, temiz eğitici grafik illüstrasyonu hakkında"
            elif style == "photorealistic":
                prompt_prefix = "Fotoğraf gerçekçiliğinde görüntü gösteren"
            elif style == "cartoon":
                prompt_prefix = "Renkli çizgi film illüstrasyonu tasvir eden"
            elif style == "abstract":
                prompt_prefix = "Soyut kavramsal sanat eseri temsil eden"
            else:
                prompt_prefix = "Basit illüstrasyon hakkında"
                
            # Clean up the text for better prompts - remove special characters and limit length
            clean_text = re.sub(r'[^\w\s]', '', text)
            clean_text = ' '.join(clean_text.split()[:20])  # Limit to 20 words
            
            prompt = f"{prompt_prefix}: {clean_text}"
            
            # For Intel GPU, use CPU with memory optimization
            device = "cpu"  # Start with CPU assumption
            dtype = torch.float32
            
            # Check if CUDA is available (NVIDIA GPU)
            if torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                self.status_var.set("NVIDIA GPU kullanılarak görsel oluşturuluyor")
            else:
                # Try to check if Intel's OneAPI is available
                try:
                    import intel_extension_for_pytorch as ipex
                    device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
                    if device == "xpu":
                        self.status_var.set("Intel GPU hızlandırma kullanılıyor")
                except ImportError:
                    self.status_var.set("CPU kullanılarak görsel oluşturuluyor (daha yavaş)")
            
            # Use a smaller, faster model for generation
            model_id = "runwayml/stable-diffusion-v1-5"  # More compatible model
            
            # Use low memory settings for slower devices
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype,
                safety_checker=None  # Disable safety checker for performance
            )
            
            # Apply memory optimizations
            pipe.enable_attention_slicing()
            
            # Try to use Intel optimizations if available
            if device == "xpu":
                pipe = pipe.to(device)
            elif device == "cuda":
                pipe = pipe.to(device)
            
            # Generate the image
            self.status_var.set("Görsel oluşturuluyor... (15-30 saniye sürebilir)")
            with torch.no_grad():
                image = pipe(
                    prompt, 
                    num_inference_steps=25,  # Reduced steps for speed
                    height=size[1], 
                    width=size[0]
                ).images[0]
            
            # Clear VRAM/memory
            del pipe
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "xpu":
                torch.xpu.empty_cache()
                
            # Convert to OpenCV format
            image_array = np.array(image)
            image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            return image_cv2
            
        except ImportError as e:
            messagebox.showwarning("Eksik Kütüphaneler", 
                f"Gerekli kütüphaneler eksik: {e}\n\nLütfen şu komutla yükleyin:\npip install torch diffusers transformers accelerate")
            return None
        except Exception as e:
            messagebox.showerror("Görsel Oluşturma Hatası", f"Görsel oluşturulurken hata oluştu: {str(e)}")
            return None

    def search_web_image(self, query, min_width=300, min_height=300):
        """Search for images on the web using available APIs"""
        try:
            self.status_var.set(f"İlgili görseller aranıyor: {query}")
            
            # Try different search engines based on what's configured
            if hasattr(self, 'bing_api_key_var') and self.bing_api_key_var.get():
                # If Bing API key is provided, try the Bing Image Search API
                result = self.search_bing_images(query, min_width, min_height)
                if result is not None:
                    return result
                    
            # Try Google Custom Search API if available
            google_result = search_google_images(self, query, min_width, min_height)
            if google_result is not None:
                return google_result
                
            # Fallback to a simple web scraper approach if no API keys worked
            return self.search_web_images_simple(query, min_width, min_height)
        except Exception as e:
            print(f"Web görsel arama hatası: {e}")
            return None

    def search_bing_images(self, query, min_width=300, min_height=300):
        """Use Bing Image Search API to find images"""
        try:
            subscription_key = self.api_key_var.get()
            if not subscription_key:
                return None
                
            search_url = "https://api.bing.microsoft.com/v7.0/images/search"
            
            headers = {"Ocp-Apim-Subscription-Key": subscription_key}
            params = {
                "q": query,
                "license": "public",  # Filter for public domain images
                "imageType": "photo",
                "count": 10
            }
            
            response = requests.get(search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()
            
            # Filter images by minimum dimensions
            valid_images = [img for img in search_results.get("value", []) 
                            if img.get("width", 0) >= min_width and 
                            img.get("height", 0) >= min_height]
            
            if not valid_images:
                return None
                
            # Pick a random image from the top results
            selected_image = random.choice(valid_images[:5]) if len(valid_images) >= 5 else valid_images[0]
            
            # Download the image
            img_url = selected_image["contentUrl"]
            img_response = requests.get(img_url, timeout=5)
            img_response.raise_for_status()
            
            # Convert to OpenCV format
            img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return None
                
            return img
            
        except Exception as e:
            print(f"Bing görsel arama hatası: {e}")
            return None

    def search_web_images_simple(self, query, min_width=300, min_height=300):
        """A simpler fallback method to get images without API requirements"""
        try:
            # Use Unsplash API for free images
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.unsplash.com/search/photos?query={encoded_query}&per_page=10"
            
            # Check if we have a client ID for Unsplash
            client_id = os.environ.get("UNSPLASH_CLIENT_ID", "")
            
            if client_id:
                headers = {"Authorization": f"Client-ID {client_id}"}
                response = requests.get(url, headers=headers)
            else:
                # If no Unsplash API, try to use Pixabay which has some free API access
                pixabay_key = os.environ.get("PIXABAY_API_KEY", "")
                if pixabay_key:
                    url = f"https://pixabay.com/api/?key={pixabay_key}&q={encoded_query}&image_type=photo&per_page=10"
                    response = requests.get(url)
                else:
                    # Last resort - try to use a search engine but this might not work reliably
                    self.status_var.set("Hiçbir görsel API anahtarı bulunamadı. Daha iyi sonuçlar için UNSPLASH_CLIENT_ID veya PIXABAY_API_KEY ortam değişkenlerini ayarlayın.")
                    return None
                    
            response.raise_for_status()
            data = response.json()
            
            # Extract image URLs
            if "results" in data:  # Unsplash response format
                images = data["results"]
                if not images:
                    return None
                    
                # Select a random image from top results
                selected = random.choice(images[:5]) if len(images) >= 5 else images[0]
                img_url = selected["urls"]["regular"]
                
            elif "hits" in data:  # Pixabay response format
                images = data["hits"]
                if not images:
                    return None
                    
                # Select a random image from top results
                selected = random.choice(images[:5]) if len(images) >= 5 else images[0]
                img_url = selected["webformatURL"]
            else:
                return None
                
            # Download the selected image
            img_response = requests.get(img_url, timeout=5)
            img_response.raise_for_status()
            
            # Convert to OpenCV format
            img_array = np.asarray(bytearray(img_response.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None or img.shape[0] < min_height or img.shape[1] < min_width:
                return None
                
            return img
            
        except Exception as e:
            print(f"Basit web görsel arama hatası: {e}")
            return None

    def process_single_image(self, image_path, operation, params, index, total):
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            img = cv2.imread(image_path)
            if img is None:
<<<<<<< HEAD
                print(f"Could not read image: {image_path}")
                return False
                
            image_name = os.path.basename(image_path)
            
            if operation == 'splitter':
                # Simple image splitting
                rows_per_page = params.get('rows_per_page', 10)
                detect_questions = params.get('detect_questions', True)
                
=======
                print(f"Görsel okunamadı: {image_path}")
                return
                
            image_name = os.path.basename(image_path)
            
            # OCR configuration
            custom_config = r'-l tur --oem 3 --psm 6'
            
            # Only perform OCR when needed
            if operation == 'title_cropper' or operation == 'splitter':
                d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)

            if operation == 'title_cropper':
                current_title = ''
                current_top = 0
                line_count_to_crop = 0
                for i in range(len(d['text'])):
                    if (d['text'][i] == '' or d['text'][i].isupper() or not d['text'][i].isalnum()) and (current_top == 0 or d['text'][i] - current_top < 10):
                        current_title += d['text'][i]
                        current_top = d['top'][i]
                        line_count_to_crop += 1
                    else:
                        break
                
                if self.last_title != '' and current_title == self.last_title:
                    height_to_crop = d['top'][line_count_to_crop] - 10
                    crop_img = img[height_to_crop:, :]
                    crop_img = remove_empty_rows_and_columns(crop_img)
                    img = cv2.resize(crop_img, (1920, 1080))
                else:
                    img = remove_empty_rows_and_columns(img)
                
                cv2.imwrite(os.path.join(self.output_dir, image_name), img)
            
            elif operation == 'splitter':
                detect_questions = params[1]
                rows_per_page = params[2]
                use_row_detection = params[3]
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                if detect_questions:
                    custom_config = r'-l tur --oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    has_questions = contains_question_indicators(ocr_text)
                else:
                    has_questions = False
                if has_questions:
<<<<<<< HEAD
                    # Don't split images with questions
                    cleaned_img = remove_empty_rows_and_columns(img)
                    cleaned_img = add_outer_border(cleaned_img, top_border=50, bottom_border=50)
=======
                    self.status_var.set(f"Görsel işleniyor {index+1}/{total} - Sorular içeriyor, bölünmüyor")
                    cleaned_img = remove_empty_rows_and_columns(img)
                    cleaned_img = add_outer_border(cleaned_img, top_border=100, bottom_border=100, left_border=40, right_border=40)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                    cleaned_img = cv2.resize(cleaned_img, (1920, 1080))
                    output_path = os.path.join(self.output_dir, image_name)
                    cv2.imwrite(output_path, cleaned_img)
                else:
<<<<<<< HEAD
                    # Split into 2 parts
                    height, width = img.shape[:2]
                    mid_height = height // 2
                    
                    for i in range(2):
                        if i == 0:
                            crop_img = img[0:mid_height, :]
                        else:
                            crop_img = img[mid_height:, :]
                        
=======
                    # --- Satırların üst ve alt sınırlarını bul ---
                    line_tops = []
                    line_bottoms = []
                    for i in range(len(d['text'])):
                        if d['text'][i].strip() != '':
                            top = d['top'][i]
                            height = d['height'][i]
                            bottom = top + height
                            line_tops.append(top)
                            line_bottoms.append(bottom)
                    if not line_tops:
                        # Hiç satır yoksa tüm resmi tek parça olarak işle
                        crop_img = remove_empty_rows_and_columns(img)
                        crop_img = add_outer_border(crop_img, top_border=100, bottom_border=100, left_border=40, right_border=40)
                        crop_img = cv2.resize(crop_img, (1920, 1080))
                        output_path = os.path.join(self.output_dir, f"{image_name}_0.png")
                        cv2.imwrite(output_path, crop_img)
                        return
                    # Satırları gruplara ayır
                    num_lines = len(line_tops)
                    splits = []
                    for i in range(0, num_lines, rows_per_page):
                        group_tops = line_tops[i:i+rows_per_page]
                        group_bottoms = line_bottoms[i:i+rows_per_page]
                        start = max(0, min(group_tops))
                        end = max(group_bottoms)
                        splits.append((start, end))
                    for i, (start, end) in enumerate(splits):
                        crop_img = img[start:end, :]
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                        crop_img = remove_empty_rows_and_columns(crop_img)
                        crop_img = add_outer_border(crop_img, top_border=100, bottom_border=100, left_border=40, right_border=40)
                        if not is_image_blank(crop_img):
<<<<<<< HEAD
                            crop_img = add_outer_border(crop_img, top_border=50, bottom_border=50)
                            crop_img = cv2.resize(crop_img, (1920, 1080))
                            output_path = os.path.join(self.output_dir, f"{image_name}_{i}.png")
                            cv2.imwrite(output_path, crop_img)
            
            elif operation == 'title_cropper':
                # Simple title cropping
                custom_config = r'-l tur --oem 3 --psm 6'
                d = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
                
                if len(d['text']) > 0:
                    # Find first non-empty text
                    first_text_top = None
                    for i, text in enumerate(d['text']):
                        if text.strip():
                            first_text_top = d['top'][i]
                            break
                    
                    if first_text_top and first_text_top > 50:
                        crop_img = img[first_text_top-20:, :]
                        img = cv2.resize(crop_img, (1920, 1080))
                
                cv2.imwrite(os.path.join(self.output_dir, image_name), img)
            
            elif operation == 'blank_remover':
                # Check if image is blank
                if is_image_blank(img, background_threshold=98.0):
=======
                            crop_img = cv2.resize(crop_img, (1920, 1080))
                            output_path = os.path.join(self.output_dir, f"{image_name}_{i}.png")
                            cv2.imwrite(output_path, crop_img)

            elif operation == 'template_remover':
                # New operation to remove templates from images
                result_img = img.copy()
                
                if not self.template_files:
                    messagebox.showwarning("Şablon Yok", "Lütfen önce şablon görselleri seçin.")
                    return
                
                # Add status updates
                self.status_var.set(f"Görsel işleniyor {index+1}/{total} - Şablonlar kaldırılıyor...")
                
                # Try with different scales in case the template size varies
                scales = [1.0, 0.75, 1.25]
                
                # Use the sensitivity value:
                threshold_base = self.template_sensitivity_var.get()
                thresholds = [threshold_base, threshold_base - 0.1, threshold_base - 0.2]
                
                # Process each template
                for template_path in self.template_files:
                    try:
                        # Use PIL to load the template
                        template_pil = Image.open(template_path)
                        template = cv2.cvtColor(np.array(template_pil), cv2.COLOR_RGB2BGR)
                        
                        if template is None:
                            continue
                            
                        # Process with different scales
                        for scale in scales:
                            if scale != 1.0:
                                h, w = template.shape[:2]
                                scaled_w, scaled_h = int(w * scale), int(h * scale)
                                if scaled_w > 0 and scaled_h > 0:
                                    scaled_template = cv2.resize(template, (scaled_w, scaled_h))
                                    # Use higher threshold for better performance
                                    result_img = remove_template_from_image(result_img, scaled_template, threshold=0.7, max_matches=5)
                            else:
                                # Use higher threshold for better performance
                                result_img = remove_template_from_image(result_img, template, threshold=0.7, max_matches=5)
                                
                    except Exception as e:
                        print(f"Şablon uygulanırken hata oluştu {template_path}: {e}")
                
                # Save the result using PIL to handle Unicode paths
                output_path = os.path.join(self.output_dir, f"cleaned_{image_name}")
                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                result_pil.save(output_path)
            
            elif operation == 'blank_remover':
                # Operation to identify and remove blank images
                self.status_var.set(f"Görsel işleniyor {index+1}/{total} - Boş olup olmadığı kontrol ediliyor...")
                
                # Get blank detection sensitivity from slider
                blank_threshold = self.blank_sensitivity_var.get()
                
                # Check if image is blank using our existing function
                if is_image_blank(img, background_threshold=blank_threshold, std_dev_threshold=15):
                    # If blank, move to a "blank_images" subfolder in output directory
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                    blank_dir = os.path.join(self.output_dir, "blank_images")
                    if not os.path.exists(blank_dir):
                        os.makedirs(blank_dir)
                    
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(blank_dir, image_name)
                    pil_img.save(output_path)
<<<<<<< HEAD
=======
                    
                    self.status_var.set(f"Görsel {image_name} boş olarak tanımlandı")
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(self.output_dir, image_name)
                    pil_img.save(output_path)
<<<<<<< HEAD
            
            return True
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return False
    
    def process_files_thread(self):
        """Process files in a separate thread"""
        try:
            if not self.input_files:
                messagebox.showwarning("No Input", "Please select files first.")
                return
                
            operation = self.operation_var.get()
=======

            elif operation == 'image_generator':
                # Operation to generate and insert images where appropriate
                self.status_var.set(f"Görsel işleniyor {index+1}/{total} - Görsel yerleştirme için analiz ediliyor...")
                
                try:
                    # Extract text from the slide
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    
                    # Skip if the text is very short (likely not enough content to generate from)
                    if len(ocr_text.strip()) < 20:
                        self.status_var.set(f"Görsel {index+1}/{total} - Görsel oluşturmak için yeterli metin yok")
                        # Just save the original image
                        output_path = os.path.join(self.output_dir, image_name)
                        cv2.imwrite(output_path, img)
                        return
                        
                    # Find space for an image
                    rect, corners = self.analyze_slide_for_image_placement(img)
                    
                    if rect:
                        # We found space for an image
                        x, y, w, h = rect
                        
                        # Ensure dimensions are compatible with AI models (divisible by 8)
                        x, y, w, h = ensure_dimensions_divisible_by_8((x, y, w, h))
                        
                        # Show message about space found
                        self.status_var.set(f"Görsel {index+1}/{total} - Görsel için alan bulundu ({w}x{h})")
                        self.root.update()  # Force UI update
                        
                        # Check image source preference
                        image_source = self.image_source_var.get()
                        generated_image = None
                        print(ocr_text)
                        if image_source == "web_only":
                            # Only use web search
                            generated_image = self.search_web_image(ocr_text, min_width=w//2, min_height=h//2)
                        elif image_source == "ai_only":
                            # Only use AI generation
                            generated_image = self.generate_image_for_text(ocr_text, size=(w, h))
                        else:  # web_first
                            # Try web search first, then fall back to AI
                            generated_image = self.search_web_image(ocr_text, min_width=w//2, min_height=h//2)
                            if generated_image is None:
                                self.status_var.set("Uygun web görselleri bulunamadı, Yapay Zeka kullanılıyor...")
                                generated_image = self.generate_image_for_text(ocr_text, size=(w, h))
                        
                        if generated_image is not None:
                            # Resize the generated image to fit the target space
                            print(w,h)
                            generated_image = cv2.resize(generated_image, (w, h))
                            
                            # Create a copy of the original image
                            result_img = img.copy()
                            
                            # Insert the generated image
                            result_img[y:y+h, x:x+w] = generated_image
                            
                            # Save the result
                            output_path = os.path.join(self.output_dir, f"enhanced_{image_name}")
                            cv2.imwrite(output_path, result_img)
                            
                            self.status_var.set(f"{image_name} görseline görsel eklendi")
                        else:
                            # Failed to find or generate image, save original
                            output_path = os.path.join(self.output_dir, image_name)
                            cv2.imwrite(output_path, img)
                            self.status_var.set(f"Uygun bir görsel bulunamadı veya oluşturulamadı")
                    else:
                        # No suitable space for an image
                        self.status_var.set(f"Görsel {index+1}/{total} - Görsel için uygun alan bulunamadı")
                        output_path = os.path.join(self.output_dir, image_name)
                        cv2.imwrite(output_path, img)
                except Exception as e:
                    print(f"Görsel oluşturma hatası: {e}")
                    output_path = os.path.join(self.output_dir, image_name)
                    cv2.imwrite(output_path, img)

            # Update progress
            progress = (index + 1) / total * 100
            self.progress_var.set(progress)
            self.status_var.set(f"{total} görselin {index + 1} tanesi işlendi")
            
        except Exception as e:
            print(f"Görsel işlenirken hata oluştu {image_path}: {e}")
    
    def process_images_thread(self):
        if not self.input_files:
            messagebox.showwarning("Girdi Yok", "Lütfen önce giriş dosyalarını seçin.")
            return
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
            
            # Get parameters
            try:
                lines_per_slide = int(self.lines_per_slide_var.get())
                if lines_per_slide < 1:
                    lines_per_slide = 7
            except ValueError:
                lines_per_slide = 7
            
            try:
                rows_per_page = int(self.rows_per_page_var.get())
                if rows_per_page < 1:
                    rows_per_page = 10
            except ValueError:
                rows_per_page = 10
<<<<<<< HEAD
            
            params = {
                'lines_per_slide': lines_per_slide,
                'rows_per_page': rows_per_page,
                'detect_questions': self.detect_questions_var.get()
            }
            
            # Reset progress
            self.progress_var.set(0)
            self.status_var.set("Processing...")
            
            total_files = len(self.input_files)
            success_count = 0
            
            if operation == 'doc_to_pptx':
                # Process documents
                for i, doc_path in enumerate(self.input_files):
                    if self.processing_cancelled:
                        break
                        
                    if self.convert_document_to_pptx(doc_path, lines_per_slide):
                        success_count += 1
                    
                    progress = (i + 1) / total_files * 100
                    self.progress_var.set(progress)
                    self.root.update()
=======
        except ValueError:
            rows_per_page = 10  # Default value
        
        # Build params list with all our settings
        params = [
            splits,  # Original splits value (as fallback)
            self.detect_questions_var.get(),  # Question detection
            rows_per_page,  # Rows per page
            self.use_row_detection_var.get()  # Whether to use row detection
        ]
        
        # Reset progress
        self.progress_var.set(0)
        self.status_var.set("İşleniyor...")
        
        # Add these variables before the ThreadPoolExecutor block
        blank_count = 0
        total_count = len(self.input_files)
        blank_results = []
        
        # Determine number of CPU cores for parallel processing
        num_cores = multiprocessing.cpu_count()
        max_workers = max(1, num_cores)
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, image_path in enumerate(self.input_files):
                future = executor.submit(
                    self.process_single_image, 
                    image_path, 
                    operation, 
                    params,
                    i,
                    len(self.input_files)
                )
                futures.append(future)
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
                
                self.status_var.set("Document conversion complete!")
                messagebox.showinfo("Complete", f"Successfully converted {success_count} of {total_files} documents to PowerPoint!")
            
            else:
                # Process images
                for i, image_path in enumerate(self.input_files):
                    if self.processing_cancelled:
                        break
                        
                    if self.process_single_image(image_path, operation, params):
                        success_count += 1
                    
                    progress = (i + 1) / total_files * 100
                    self.progress_var.set(progress)
                    self.status_var.set(f"Processed {i + 1} of {total_files} files")
                    self.root.update()
                
<<<<<<< HEAD
                if operation == 'blank_remover':
                    blank_dir = os.path.join(self.output_dir, "blank_images")
                    blank_count = len(os.listdir(blank_dir)) if os.path.exists(blank_dir) else 0
                    self.status_var.set(f"Processing complete! Found {blank_count} blank images")
                    messagebox.showinfo("Complete", f"Found {blank_count} blank images out of {total_files}")
                else:
                    self.status_var.set("Processing complete!")
                    messagebox.showinfo("Complete", f"Successfully processed {success_count} of {total_files} images!")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
        finally:
            self.cancel_button.config(state=tk.DISABLED)
            self.processing_cancelled = False
=======
            self.status_var.set(f"İşleme tamamlandı! {total_count} görselin {blank_count} tanesi boş olarak bulundu")
            messagebox.showinfo("Tamamlandı", f"Görsel işleme tamamlandı!\n\n{blank_count} boş görsel 'blank_images' klasörüne taşındı.\n{total_count - blank_count} boş olmayan görsel çıktı klasöründe tutuldu.")
        else:
            self.status_var.set("İşleme tamamlandı!")
            messagebox.showinfo("Tamamlandı", "Görsel işleme başarıyla tamamlandı!")
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
    
    def process_files(self):
        """Start processing files"""
        self.processing_cancelled = False
        self.cancel_button.config(state=tk.NORMAL)
        thread = threading.Thread(target=self.process_files_thread)
        thread.daemon = True
        thread.start()
    
    def cancel_processing(self):
        """Cancel the current processing"""
        self.processing_cancelled = True
<<<<<<< HEAD
        self.status_var.set("Cancelling...")
    
=======
        self.status_var.set("İptal ediliyor...")

>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        if os.path.exists(self.output_dir):
            os.startfile(self.output_dir)
        else:
<<<<<<< HEAD
            messagebox.showwarning("Folder Not Found", "Output folder does not exist yet.")
=======
            messagebox.showwarning("Klasör Bulunamadı", "Çıktı klasörü henüz mevcut değil.")
    
    def save_settings(self):
        """Save current settings as default"""
        try:
            settings = {
                'operation': self.operation_var.get(),
                'rows_per_page': self.rows_per_page_var.get(),
                'output_dir': self.output_dir,
                'detect_questions': str(self.detect_questions_var.get()),
                'use_row_detection': str(self.use_row_detection_var.get()),
                'template_sensitivity': str(self.template_sensitivity_var.get()),
                'blank_sensitivity': str(self.blank_sensitivity_var.get()),
                # Add API key storage
                'unsplash_api_key': os.environ.get("UNSPLASH_CLIENT_ID", ""),
                'pixabay_api_key': os.environ.get("PIXABAY_API_KEY", ""),
                'bing_api_key': self.bing_api_key_var.get() if hasattr(self, 'bing_api_key_var') else "",
                'google_api_key': os.environ.get("GOOGLE_API_KEY", ""),
                'google_cx': os.environ.get("GOOGLE_SEARCH_CX", "")
            }
            
            # Create config directory if it doesn't exist
            config_dir = os.path.join(os.path.expanduser("~"), ".imagecropper")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Save settings to file
            with open(os.path.join(config_dir, "settings.txt"), 'w') as f:
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
                    
            messagebox.showinfo("Ayarlar Kaydedildi", "Varsayılan ayarlar ve API anahtarları kaydedildi.")
        except Exception as e:
            messagebox.showerror("Hata", f"Ayarlar kaydedilemedi: {e}")
    
    def load_settings(self):
        """Load saved settings"""
        try:
            config_file = os.path.join(os.path.expanduser("~"), ".imagecropper", "settings.txt")
            if (os.path.exists(config_file)):
                with open(config_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key == 'operation':
                                self.operation_var.set(value)
                            elif key == 'rows_per_page':
                                self.rows_per_page_var.set(value)
                            elif key == 'detect_questions':
                                self.detect_questions_var.set(value.lower() == 'true')
                            elif key == 'output_dir':
                                if os.path.exists(value):
                                    self.output_dir = value
                                    self.output_dir_label.config(text=self.output_dir)
                            elif key == 'blank_sensitivity':
                                try:
                                    self.blank_sensitivity_var.set(float(value))
                                except ValueError:
                                    self.blank_sensitivity_var.set(98.0)  # Default value
                            # Load API keys
                            elif key == 'unsplash_api_key' and value.strip():
                                os.environ["UNSPLASH_CLIENT_ID"] = value
                            elif key == 'pixabay_api_key' and value.strip():
                                os.environ["PIXABAY_API_KEY"] = value
                            elif key == 'bing_api_key' and value.strip() and hasattr(self, 'bing_api_key_var'):
                                self.bing_api_key_var.set(value)
                            elif key == 'google_api_key' and value.strip():
                                os.environ["GOOGLE_API_KEY"] = value
                            elif key == 'google_cx' and value.strip():
                                os.environ["GOOGLE_SEARCH_CX"] = value
                                
                # Display a message about loaded API keys
                self.log_api_status()
        except Exception as e:
            print(f"Ayarlar yüklenirken hata oluştu: {e}")
            
    def select_folder(self):
        """Select a folder and add all images from it"""
        folder = filedialog.askdirectory()
        if folder:
            image_files = []
            # Get all image files from the selected folder
            for file in os.listdir(folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    full_path = os.path.join(folder, file)
                    image_files.append(full_path)
                    
            if image_files:
                self.input_files = image_files
                self.input_files_label.config(text=f"{len(self.input_files)} dosya klasörden seçildi")
                self.update_preview()
            else:
                messagebox.showwarning("Görsel Yok", "Seçilen klasörde görsel dosyası bulunamadı.")
    
    def toggle_templates_frame(self, *args):
        """Show or hide the templates frame based on the selected operation"""
        if self.operation_var.get() == "template_remover":
            # Make it appear right after the operation/parameters section
            self.templates_frame.pack(fill=tk.X, pady=5, after=self.op_param_container)
            
            # Show a message box to guide the user if no templates are selected yet
            # if not self.template_files:
            #     messagebox.showinfo("Şablon Seçimi", 
            #                        "Lütfen 'Şablon Seç' düğmesini kullanarak kaldırılacak filigran/logo görsellerini seçin.")
        else:
            self.templates_frame.pack_forget()

    def select_templates(self):
        """Select template images (logos or watermarks to remove)"""
        filetypes = [("Görsel dosyalar", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.template_files = list(files)
            self.template_files_label.config(text=f"{len(self.template_files)} şablon seçildi")
            self.update_template_preview()
            
            # Show message to explain the process
            messagebox.showinfo(
                "Şablonlar Seçildi", 
                "Seçilen şablonlar tüm giriş görsellerinden kaldırılacaktır.\n\n" +
                "İşleme, görsel sayısına ve boyutlarına bağlı olarak birkaç dakika sürebilir."
            )
            
    def update_template_preview(self):
        """Show thumbnails of selected template images"""
        # Clear current thumbnails in the templates frame
        for widget in self.templates_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                widget.destroy()
        
        # Create a new frame for thumbnails
        thumbnails_frame = ttk.Frame(self.templates_frame)
        thumbnails_frame.pack(fill=tk.X, expand=True, before=self.template_files_label)
        
        # Show thumbnails
        max_templates = min(5, len(self.template_files))
        self.template_thumbnails = []  # Keep references
        
        for i in range(max_templates):
            try:
                img = Image.open(self.template_files[i])
                img.thumbnail((50, 50))  # Smaller thumbnails for templates
                photo = ImageTk.PhotoImage(img)
                self.template_thumbnails.append(photo)
                
                frame = ttk.Frame(thumbnails_frame)
                frame.pack(side=tk.LEFT, padx=5)
                
                label = ttk.Label(frame, image=photo)
                label.pack()
            except Exception as e:
                print(f"Şablon küçük resim oluşturulurken hata oluştu: {e}")

    def use_ai_inpainting(self, image, mask):
        """Use an AI model to inpaint masked regions more naturally"""
        try:
            # This requires installing transformers and diffusers
            from diffusers import AutoPipelineForInpainting
            import torch
            
            # Load model - first time will download the model
            self.status_var.set("Yapay Zeka modeli yükleniyor (ilk kullanım biraz zaman alabilir)...")
            pipe = AutoPipelineForInpainting.from_pretrained(
                "runwayml/stable-diffusion-inpainting", 
                torch_dtype=torch.float16,
                variant="fp16"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Convert to PIL format
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            pil_mask = Image.fromarray(mask)
            
            # Run inpainting
            result = pipe(
                prompt="filigran olmadan temiz görsel",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=20
            ).images[0]
            
            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        except ImportError:
            messagebox.showwarning("Eksik Kütüphaneler", 
                "Yapay Zeka destekli kaldırma için şu komutu kullanarak yükleyin: pip install torch diffusers transformers")
            return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)  # Fallback

    def show_image_generation_options(self):    
        """Show a dialog with additional image generation options"""
        options_dialog = tk.Toplevel(self.root)
        options_dialog.title("Görsel Arama ve Oluşturma Ayarları")
        options_dialog.geometry("500x550")  # Increased for more options
        options_dialog.transient(self.root)
        options_dialog.grab_set()
        
        # Create a notebook with tabs for different settings
        notebook = ttk.Notebook(options_dialog)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create frames for each tab
        ai_frame = ttk.Frame(notebook, padding=10)
        api_frame = ttk.Frame(notebook, padding=10)
        notebook.add(ai_frame, text="Yapay Zeka Oluşturma")
        notebook.add(api_frame, text="Arama API'leri")
        
        # === AI Tab ===
        ttk.Label(ai_frame, text="Yapay Zeka Modeli Seçimi:").pack(anchor=tk.W, pady=(0, 5))
        
        model_var = tk.StringVar(value="runwayml/stable-diffusion-v1-5")
        models = [
            ("Stable Diffusion 1.5 (Daha Hızlı)", "runwayml/stable-diffusion-v1-5"),
            ("Stable Diffusion 2.1 (Daha İyi Kalite)", "stabilityai/stable-diffusion-2-1")
        ]
        
        for text, value in models:
            ttk.Radiobutton(ai_frame, text=text, value=value, variable=model_var).pack(anchor=tk.W, padx=10)
        
        ttk.Label(ai_frame, text="Görsel Kalitesi:").pack(anchor=tk.W, pady=(10, 5))
        
        quality_var = tk.IntVar(value=25)
        quality_scale = ttk.Scale(
            ai_frame, 
            from_=15, 
            to=50, 
            orient="horizontal", 
            variable=quality_var, 
            length=300
        )
        quality_scale.pack(pady=5, fill=tk.X)
        ttk.Label(ai_frame, text="Yüksek = Daha iyi kalite ancak daha yavaş").pack(anchor=tk.W)
        
        # === API Keys Tab ===
        ttk.Label(api_frame, text="Görsel Arama API'lerini Yapılandırın:").pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(api_frame, text="Hangi API'nin kullanıldığını görmek için günlükleri kontrol edin").pack(anchor=tk.W, pady=(0, 10))
        
        # Unsplash
        ttk.Label(api_frame, text="Unsplash Client ID:").pack(anchor=tk.W, pady=(10, 0))
        unsplash_var = tk.StringVar(value=os.environ.get("UNSPLASH_CLIENT_ID", ""))
        unsplash_entry = ttk.Entry(api_frame, textvariable=unsplash_var)
        unsplash_entry.pack(pady=5, fill=tk.X)
        ttk.Label(api_frame, text="Buradan alın: https://unsplash.com/developers", 
                 font=("", 8)).pack(anchor=tk.W)
        
        # Pixabay
        ttk.Label(api_frame, text="Pixabay API Anahtarı:").pack(anchor=tk.W, pady=(10, 0))
        pixabay_var = tk.StringVar(value=os.environ.get("PIXABAY_API_KEY", ""))
        pixabay_entry = ttk.Entry(api_frame, textvariable=pixabay_var)
        pixabay_entry.pack(pady=5, fill=tk.X)
        ttk.Label(api_frame, text="Buradan alın: https://pixabay.com/api/docs/", 
                 font=("", 8)).pack(anchor=tk.W)
        
        # Bing
        ttk.Label(api_frame, text="Bing Arama API Anahtarı:").pack(anchor=tk.W, pady=(10, 0))
        bing_var = tk.StringVar(value=self.bing_api_key_var.get() if hasattr(self, 'bing_api_key_var') else "")
        bing_entry = ttk.Entry(api_frame, textvariable=bing_var)
        bing_entry.pack(pady=5, fill=tk.X)
        ttk.Label(api_frame, text="Buradan alın: https://portal.azure.com/#create/Microsoft.CognitiveServicesBingSearch", 
                 font=("", 8)).pack(anchor=tk.W)
        
        # Google
        ttk.Label(api_frame, text="Google Özel Arama API Anahtarı:").pack(anchor=tk.W, pady=(10, 0))
        google_key_var = tk.StringVar(value=os.environ.get("GOOGLE_API_KEY", ""))
        google_key_entry = ttk.Entry(api_frame, textvariable=google_key_var)
        google_key_entry.pack(pady=5, fill=tk.X)
        
        ttk.Label(api_frame, text="Google Özel Arama Motoru ID (CX):").pack(anchor=tk.W, pady=(10, 0))
        google_cx_var = tk.StringVar(value=os.environ.get("GOOGLE_SEARCH_CX", ""))
        google_cx_entry = ttk.Entry(api_frame, textvariable=google_cx_var)
        google_cx_entry.pack(pady=5, fill=tk.X)
        ttk.Label(api_frame, text="Buradan alın: https://programmablesearchengine.google.com/", 
                 font=("", 8)).pack(anchor=tk.W)
        
        # Bottom buttons
        button_frame = ttk.Frame(options_dialog)
        button_frame.pack(fill=tk.X, pady=10)
        
        def save_options():
            self.image_model_var = model_var.get()
            self.image_quality_var = quality_var.get()
            
            # Save API keys to environment variables
            os.environ["UNSPLASH_CLIENT_ID"] = unsplash_var.get()
            os.environ["PIXABAY_API_KEY"] = pixabay_var.get()
            os.environ["GOOGLE_API_KEY"] = google_key_var.get()
            os.environ["GOOGLE_SEARCH_CX"] = google_cx_var.get()
            
            # Update Bing API key if available
            if hasattr(self, 'bing_api_key_var'):
                self.bing_api_key_var.set(bing_var.get())
            
            # Log what's currently active
            self.log_api_status()
            
            options_dialog.destroy()
        
        ttk.Button(button_frame, text="Ayarları Kaydet", command=save_options).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="İptal", command=options_dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def analyze_slide_for_image_placement(self, image):
        """Analyze a slide to find the best place for a smaller image in empty areas"""
        # Convert to grayscale for text detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        height, width = image.shape[:2]
        
        # Use OCR to find text areas
        custom_config = r'--oem 3 --psm 6'
        d = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
        
        # Create a mask for text areas with padding
        text_mask = np.zeros((height, width), dtype=np.uint8)
        padding = 20  # Padding around text
        
        for i in range(len(d['text'])):
            if d['text'][i].strip() != '':
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                # Add padding
                x_start = max(0, x - padding)
                y_start = max(0, y - padding)
                x_end = min(width, x + w + padding)
                y_end = min(height, y + h + padding)
                text_mask[y_start:y_end, x_start:x_end] = 255
        
        # Add a border mask to avoid placing images too close to the edge
        border_size = 10
        text_mask[:border_size, :] = 255  # Top border
        text_mask[-border_size:, :] = 255  # Bottom border
        text_mask[:, :border_size] = 255  # Left border
        text_mask[:, -border_size:] = 255  # Right border
        
        # Find the largest contiguous non-text area
        empty_mask = 255 - text_mask
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        empty_mask = cv2.morphologyEx(empty_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
        
        # First, check for empty corners that could hold a small image
        corner_candidates = []
        corner_size = min(width // 4, height // 4)  # Maximum 1/4 of slide dimension
        min_corner_size = 120  # Minimum size for a corner image
        
        # Check top-right corner
        if np.sum(text_mask[border_size:border_size+corner_size, width-corner_size-border_size:width-border_size]) == 0:
            corner_candidates.append({
                'rect': (width-corner_size-border_size, border_size, corner_size, corner_size),
                'score': 1.0,  # Highest priority
                'position': 'top-right'
            })
        
        # Check bottom-right corner
        if np.sum(text_mask[height-corner_size-border_size:height-border_size, width-corner_size-border_size:width-border_size]) == 0:
            corner_candidates.append({
                'rect': (width-corner_size-border_size, height-corner_size-border_size, corner_size, corner_size),
                'score': 0.9,
                'position': 'bottom-right'
            })
        
        # Check bottom-left corner
        if np.sum(text_mask[height-corner_size-border_size:height-border_size, border_size:border_size+corner_size]) == 0:
            corner_candidates.append({
                'rect': (border_size, height-corner_size-border_size, corner_size, corner_size),
                'score': 0.8,
                'position': 'bottom-left'
            })
            
        # If any corner is suitable, prefer that
        if corner_candidates:
            # Sort by score
            corner_candidates.sort(key=lambda x: x['score'], reverse=True)
            best_corner = corner_candidates[0]
            x, y, w, h = best_corner['rect']
            
            # Ensure minimum size
            if w >= min_corner_size and h >= min_corner_size:
                # Make it a bit smaller to avoid touching text
                safety_margin = 10
                x += safety_margin
                y += safety_margin
                w -= safety_margin * 2
                h -= safety_margin * 2
                
                # Further reduce size to make it look more natural in the corner
                w = int(w * 0.8)
                h = int(h * 0.8)
                
                # Ensure dimensions are reasonable
                w = max(min_corner_size, w)
                h = max(min_corner_size, h)
                
                return (x, y, w, h), (x, y, x + w, y + h)
        
        # If no suitable corner found, look for other empty areas
        valid_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            min_side = min(w, h)
            
            # Skip too small areas
            if min_side < min_corner_size:
                continue
            
            # Don't allow very large areas - we want a small decorative image
            max_allowed = min(width // 3, height // 3)  # Max 1/3 of slide dimension
            if w > max_allowed or h > max_allowed:
                # If area is too large, create a smaller rectangle within it
                new_w = min(w, max_allowed)
                new_h = min(h, max_allowed)
                
                # Position it in the lower right of the empty area if possible
                new_x = x + w - new_w - 10
                new_y = y + h - new_h - 10
                
                x, y, w, h = new_x, new_y, new_w, new_h
            
            # Calculate position score (prefer right side and bottom area)
            right_bias = x / width  # Higher if more to the right
            bottom_bias = y / height  # Higher if more to the bottom
            size_score = 1.0 - (w * h) / (width * height)  # Smaller areas preferred
            
            # Combined score with weights
            position_score = (0.4 * right_bias + 0.3 * bottom_bias + 0.3 * size_score)
            
            valid_contours.append({
                'contour': contour,
                'rect': (x, y, w, h),
                'area': area,
                'min_side': min_side,
                'score': position_score
            })
        
        if not valid_contours:
            return None, None
        
        # Sort by score, highest first
        valid_contours.sort(key=lambda x: x['score'], reverse=True)
        
        # Take the highest scoring contour
        best_match = valid_contours[0]
        x, y, w, h = best_match['rect']
        
        # Ensure the image isn't too large - limit to 30% of slide dimensions
        max_w = width // 3
        max_h = height // 3
        
        if w > max_w:
            w = max_w
        if h > max_h:
            h = max_h
        
        # Ensure we're not too close to text by reducing size slightly
        safety_margin = 5
        x += safety_margin
        y += safety_margin
        w -= safety_margin * 2
        h -= safety_margin * 2
        
        # Further reduce size to make images appear more proportional to the slide
        w = int(w * 0.8)
        h = int(h * 0.8)
        
        # Return the rectangle and corners
        return (x, y, w, h), (x, y, x + w, y + h)

    def log_api_status(self):
        """Log information about which API keys are available"""
        api_status = []
        
        # Check Unsplash
        if os.environ.get("UNSPLASH_CLIENT_ID", "").strip():
            api_status.append("✓ Unsplash API")
        
        # Check Pixabay
        if os.environ.get("PIXABAY_API_KEY", "").strip():
            api_status.append("✓ Pixabay API")
        
        # Check Bing
        if hasattr(self, 'bing_api_key_var') and self.bing_api_key_var.get().strip():
            api_status.append("✓ Bing Arama API")
        
        # Check Google
        if (os.environ.get("GOOGLE_API_KEY", "").strip() and 
            os.environ.get("GOOGLE_SEARCH_CX", "").strip()):
            api_status.append("✓ Google Özel Arama API")
        
        status_message = ""
        if not api_status:
            status_message = "Hiçbir görsel arama API'si yapılandırılmadı. Yedek yöntemler kullanılıyor."
        else:
            status_message = f"Aktif görsel hizmetleri: {', '.join(api_status)}"
        
        # Update the UI status
        self.status_var.set(status_message)
        
        # Also log to console
        print(f"[API Durumu] {status_message}")
>>>>>>> 4fa89278877989fca60fb26b586ba35ff2afca55

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
