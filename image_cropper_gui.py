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

def generate_image_for_text(self, text, size=(512, 512)):
    """Generate an image based on text content using Stable Diffusion locally"""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        import gc
        
        self.status_var.set("Loading image generation model (first time may take a while)...")
        
        # Set image style based on selection
        style = self.image_style_var.get()
        
        # Create style-specific prompt
        if style == "educational":
            prompt_prefix = "Simple, clean educational graphic illustration about"
        elif style == "photorealistic":
            prompt_prefix = "Photorealistic image showing"
        elif style == "cartoon":
            prompt_prefix = "Colorful cartoon illustration depicting"
        elif style == "abstract":
            prompt_prefix = "Abstract conceptual artwork representing"
        else:
            prompt_prefix = "Simple illustration about"
            
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
            self.status_var.set("Using NVIDIA GPU for image generation")
        else:
            # Try to check if Intel's OneAPI is available
            try:
                import intel_extension_for_pytorch as ipex
                device = "xpu" if hasattr(torch, 'xpu') and torch.xpu.is_available() else "cpu"
                if device == "xpu":
                    self.status_var.set("Using Intel GPU acceleration")
            except ImportError:
                self.status_var.set("Using CPU for image generation (slower)")
        
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
        self.status_var.set("Generating image... (this may take 15-30 seconds)")
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
        messagebox.showwarning("Libraries Missing", 
            f"Missing required libraries: {e}\n\nPlease install with:\npip install torch diffusers transformers accelerate")
        return None
    except Exception as e:
        messagebox.showerror("Image Generation Error", f"Error generating image: {str(e)}")
        return None

def detect_text_rows(d):
    num_of_rows = 0
    last_top = 0
    for i in range(len(d['text'])):
        # Skip empty results
        if d['text'][i].strip() == '':
            continue
        if d['top'][i] > last_top: 
            num_of_rows += 1
            
        last_top = d['top'][i]
    return last_top


def contains_question_indicators(text):
    """
    Check if the text contains question indicators like a), A), 1), etc.
    """
    # Look for patterns like "a)", "A)", "1)", etc.
    pattern = r'(?:^|\s)([a-zA-Z0-9])[.)]'
    matches = re.findall(pattern, text)
    
    # Count unique matches
    unique_indicators = set(matches)
    
    # If we have multiple indicators (like a, b, c or 1, 2, 3), it's likely a question
    return len(unique_indicators) >= 2

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
    if text == '' or re.fullmatch(r'\d+', text.strip()):
        return True
    
    return False

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

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper")
        self.root.geometry("800x700")  # Increase default height
        self.root.minsize(800, 700)    # Increase minimum height
        
        # Create default frame for all content
        self.main_content = ttk.Frame(self.root)
        self.main_content.pack(fill=tk.BOTH, expand=True)
        
        # Create fixed frame for buttons that stays at bottom
        self.button_area = ttk.Frame(self.root)
        self.button_area.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        self.input_files = []
        self.output_dir = os.path.join(os.getcwd(), "cropped")
        self.last_title = ''
        self.last_top = 0
        
        self.create_widgets()
        self.load_settings()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a scrollable area for the content
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
        
        # Move ALL your existing code that creates UI elements to use scrollable_frame instead of main_frame
        # Input section
        input_frame = ttk.LabelFrame(scrollable_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        self.input_files_label = ttk.Label(input_frame, text="No files selected")
        self.input_files_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Button frame to hold select buttons
        select_buttons_frame = ttk.Frame(input_frame)
        select_buttons_frame.pack(side=tk.RIGHT)
        
        select_folder_button = ttk.Button(select_buttons_frame, text="Select Folder", command=self.select_folder)
        select_folder_button.pack(side=tk.LEFT, padx=5)
        
        select_button = ttk.Button(select_buttons_frame, text="Select Files", command=self.select_files)
        select_button.pack(side=tk.LEFT, padx=5)
        
        # Output section
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        self.output_dir_label = ttk.Label(output_frame, text=self.output_dir)
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_button = ttk.Button(output_frame, text="Select Directory", command=self.select_output_dir)
        output_button.pack(side=tk.RIGHT, padx=5)
        
        
        # Add a save settings button
        save_settings_button = ttk.Button(output_frame, text="Save as Default", command=self.save_settings)
        save_settings_button.pack(anchor=tk.W, pady=5)
        
        # Create a frame to hold operation and parameters side by side
        op_param_container = ttk.Frame(scrollable_frame)
        op_param_container.pack(fill=tk.X, pady=5)
        
        # Operation section - now in left half
        operation_frame = ttk.LabelFrame(op_param_container, text="Operation", padding="10")
        operation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.operation_var = tk.StringVar(value="splitter")
        operations = [("Split Images", "splitter"), 
             ("Crop Titles", "title_cropper"),
             ("Remove Templates", "template_remover"),
             ("Remove Blank Images", "blank_remover"),
             ("Generate Missing Images", "image_generator")]
        
        for text, value in operations:
            ttk.Radiobutton(operation_frame, text=text, value=value, variable=self.operation_var).pack(anchor=tk.W)
        
        # Parameters section - now in right half
        param_frame = ttk.LabelFrame(op_param_container, text="Parameters", padding="10")
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(param_frame, text="Rows per page:").pack(anchor=tk.W)

        self.rows_per_page_var = tk.StringVar(value="10")
        self.splits_var = tk.StringVar(value="2")  # Default to 2 splits as fallback
        rows_per_page_entry = ttk.Entry(param_frame, textvariable=self.rows_per_page_var, width=5)
        rows_per_page_entry.pack(anchor=tk.W, pady=5)

        # Add a checkbox for using auto-detected rows
        self.use_row_detection_var = tk.BooleanVar(value=True)
        use_row_detection_checkbox = ttk.Checkbutton(
            param_frame, 
            text="Auto-detect rows (recommended)",
            variable=self.use_row_detection_var
        )
        use_row_detection_checkbox.pack(anchor=tk.W, pady=(0, 10))

        # Add a slider for template matching threshold
        ttk.Label(param_frame, text="Template Match Sensitivity:").pack(anchor=tk.W, pady=(10, 0))
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
        ttk.Label(param_frame, text="(Lower = More aggressive removal)").pack(anchor=tk.W)
        
        self.detect_questions_var = tk.BooleanVar(value=True)
        question_checkbox = ttk.Checkbutton(
            param_frame, 
            text="Detect questions (don't split)",
            variable=self.detect_questions_var
        )
        question_checkbox.pack(anchor=tk.W, pady=(10, 0))

        # Parameters for blank image detection
        ttk.Label(param_frame, text="Blank Detection Sensitivity:").pack(anchor=tk.W, pady=(10, 0))
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
        ttk.Label(param_frame, text="(Higher = More aggressive blank detection)").pack(anchor=tk.W)

        # Parameters for image generation
        ttk.Label(param_frame, text="AI Image Style:").pack(anchor=tk.W, pady=(10, 0))
        self.image_style_var = tk.StringVar(value="educational")
        styles = [("Educational", "educational"), 
                 ("Photorealistic", "photorealistic"),
                 ("Cartoon", "cartoon"),
                 ("Abstract", "abstract")]

        style_frame = ttk.Frame(param_frame)
        style_frame.pack(fill=tk.X, pady=5)

        # Create style options
        for text, value in styles:
            ttk.Radiobutton(style_frame, text=text, value=value, 
                           variable=self.image_style_var).pack(anchor=tk.W)

        # Option to set API key
        ttk.Label(param_frame, text="OpenAI API Key (optional):").pack(anchor=tk.W, pady=(10, 0))
        self.api_key_var = tk.StringVar()
        api_key_entry = ttk.Entry(param_frame, textvariable=self.api_key_var, show="*")
        api_key_entry.pack(anchor=tk.W, pady=5, fill=tk.X)

        # Preview area - will show thumbnails of selected images, with reduced height
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=1)

        self.preview_frame = ttk.Frame(preview_frame)
        self.preview_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_frame, height=150)  # Increased height slightly
        scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)        # Add this in create_widgets method, right before the status_var definition
        
        # Add a progress bar
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Add cancel button next to process button
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)  # Added fill=tk.X

        process_button = ttk.Button(button_frame, text="Process Images", command=self.process_images)
        process_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        # Open output folder button - move to button_frame
        open_folder_button = ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.pack(side=tk.RIGHT, padx=5)        
        
        # Add a new section for template images that appears when "Remove Templates" is selected
        self.templates_frame = ttk.LabelFrame(scrollable_frame, text="Watermark/Logo Templates to Remove", padding="10")

        self.template_files = []
        self.template_files_label = ttk.Label(self.templates_frame, text="No templates selected")
        self.template_files_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        select_templates_button = ttk.Button(self.templates_frame, text="Select Watermarks/Logos", 
                                             command=self.select_templates)
        select_templates_button.pack(side=tk.RIGHT)
        
        # Show/hide the templates frame based on operation selection
        self.operation_var.trace("w", self.toggle_templates_frame)

        # Add this button to your params frame
        ttk.Button(param_frame, text="Advanced Image Options", command=self.show_image_generation_options).pack(anchor=tk.W, pady=10)
        
        # Add a fixed-position frame at the bottom for action buttons that always stay visible
        button_container = ttk.Frame(self.root)
        button_container.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Move your buttons to this new container
        button_frame = ttk.Frame(button_container)
        button_frame.pack(fill=tk.X)
        
        process_button = ttk.Button(button_frame, text="Process Images", command=self.process_images)
        process_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Open output folder button - move to button_frame
        open_folder_button = ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.pack(side=tk.RIGHT, padx=5)

    def select_files(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.input_files = list(files)
            self.input_files_label.config(text=f"{len(self.input_files)} files selected")
            self.update_preview()
        
    def select_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir = directory
            self.output_dir_label.config(text=self.output_dir)
    
    def update_preview(self):
        # Clear current preview
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
            
        # Show thumbnails of selected images
        max_previews = min(5, len(self.input_files))
        self.thumbnail_refs = []  # Keep references to prevent garbage collection
        
        for i in range(max_previews):
            try:
                img = Image.open(self.input_files[i])
                img.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(img)
                self.thumbnail_refs.append(photo)
                
                frame = ttk.Frame(self.preview_frame)
                frame.pack(side=tk.LEFT, padx=5)
                
                label = ttk.Label(frame, image=photo)
                label.pack()
                
                name_label = ttk.Label(frame, text=os.path.basename(self.input_files[i]), wraplength=150)
                name_label.pack()
            except Exception as e:
                print(f"Error creating thumbnail: {e}")
        
        self.preview_frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def process_single_image(self, image_path, operation, params, index, total):
        try:
            # Make sure output directory exists
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # Modified process_image function to use selected output directory
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

            if operation == 'title_cropper':
                current_title = ''
                current_top = 0
                line_count_to_crop = 0
                for i in range(len(d['text'])):
                    if (d['text'][i] == '' or d['text'][i].isupper() or not d['text'][i].isalnum()) and (current_top == 0 or d['top'][i] - current_top < 10):
                        current_title += d['text'][i]
                        current_top = d['top'][i]
                        line_count_to_crop += 1
                    else:
                        break
                
                if self.last_title != '' and current_title == self.last_title:
                    height_to_crop = d['top'][line_count_to_crop] - 10
                    crop_img = img[height_to_crop:, :]
                    img = cv2.resize(crop_img, (1920, 1080))
                
                self.last_title = current_title
                self.last_top = current_top
                
                cv2.imwrite(os.path.join(self.output_dir, image_name), img)
            
            elif operation == 'splitter':
                detect_questions = params[1]  # Get the question detection flag
                rows_per_page = params[2]     # Get rows per page
                use_row_detection = params[3] # Whether to use auto row detection
                
                # Only check for questions if the feature is enabled
                if detect_questions:
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    has_questions = contains_question_indicators(ocr_text)
                else:
                    has_questions = False
                
                if has_questions:
                    # This image likely contains questions, so just resize without splitting
                    self.status_var.set(f"Processing image {index+1}/{total} - Contains questions, not splitting")
                    
                    # Clean up the image
                    cleaned_img = remove_empty_rows_and_columns(img)
                    cleaned_img = add_outer_border(cleaned_img, top_border=50, bottom_border=50, left_border=10, right_border=10)
                    
                    # Resize to desired dimensions
                    cleaned_img = cv2.resize(cleaned_img, (1920, 1080))
                    
                    output_path = os.path.join(self.output_dir, image_name)
                    cv2.imwrite(output_path, cleaned_img)
                else:
                    # No questions detected, proceed with splitting based on rows
                    
                    # Get the number of rows if auto-detection is enabled
                    if 0 and use_row_detection:
                        self.status_var.set(f"Processing image {index+1}/{total} - Detecting text rows...")
                        num_rows, row_positions = detect_text_rows(d)
                        
                        # Calculate number of splits based on rows per page
                        num_splits = max(1, (num_rows + rows_per_page - 1) // rows_per_page)
                        self.status_var.set(f"Processing image {index+1}/{total} - Detected {num_rows} rows, creating {num_splits} splits")
                    else:
                        # Fallback to manual splits
                        num_splits =4
                        
                    # Create splits based on height
                    height, width, _ = img.shape
                    
                    if 0 and use_row_detection and row_positions and len(row_positions) > 1:
                        print('no')
                        
                    else:
                        min_row_height = height // num_splits

                    current_top = min_row_height
                    previous_top = 0
                    i = 0
                    while i < num_splits:
                        for j in range(len(d['top'])):
                            if d['top'][j] > current_top:
                                break
                            if d['text'][j] == '':
                                continue
                            if d['top'][j] + d['height'][j] > current_top:
                                current_top = d['top'][j] + d['height'][j]
                        
                        crop_img = img[previous_top:current_top, :]
                        crop_img = remove_empty_rows_and_columns(crop_img)
                        
                        if not is_image_blank(crop_img):
                            crop_img = add_outer_border(crop_img, top_border=50, bottom_border=50, left_border=10, right_border=10)
                            crop_img = cv2.resize(crop_img, (1920, 1080))
                            output_path = os.path.join(self.output_dir, f"{image_name}_{i}.png")
                            cv2.imwrite(output_path, crop_img)
                        
                        previous_top = current_top
                        current_top += min_row_height
                        i += 1

            elif operation == 'template_remover':
                # New operation to remove templates from images
                result_img = img.copy()
                
                if not self.template_files:
                    messagebox.showwarning("No Templates", "Please select template images first.")
                    return
                
                # Add status updates
                self.status_var.set(f"Processing image {index+1}/{total} - Removing templates...")
                
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
                        print(f"Error applying template {template_path}: {e}")
                
                # Save the result using PIL to handle Unicode paths
                output_path = os.path.join(self.output_dir, f"cleaned_{image_name}")
                result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
                result_pil.save(output_path)
            
            elif operation == 'blank_remover':
                # Operation to identify and remove blank images
                self.status_var.set(f"Processing image {index+1}/{total} - Checking if blank...")
                
                # Get blank detection sensitivity from slider
                blank_threshold = self.blank_sensitivity_var.get()
                
                # Check if image is blank using our existing function
                if is_image_blank(img, background_threshold=blank_threshold, std_dev_threshold=15):
                    # If blank, move to a "blank_images" subfolder in output directory
                    blank_dir = os.path.join(self.output_dir, "blank_images")
                    if not os.path.exists(blank_dir):
                        os.makedirs(blank_dir)
                    
                    # Use PIL to save as it handles Unicode paths better
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(blank_dir, image_name)
                    pil_img.save(output_path)
                    
                    self.status_var.set(f"Image {image_name} identified as blank")
                else:
                    # If not blank, save to output directory
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(self.output_dir, image_name)
                    pil_img.save(output_path)

            elif operation == 'image_generator':
                # Operation to generate and insert images where appropriate
                self.status_var.set(f"Processing image {index+1}/{total} - Analyzing for image placement...")
                
                try:
                    # Extract text from the slide
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    
                    # Skip if the text is very short (likely not enough content to generate from)
                    if len(ocr_text.strip()) < 20:
                        self.status_var.set(f"Image {index+1}/{total} - Not enough text for generation")
                        # Just save the original image
                        output_path = os.path.join(self.output_dir, image_name)
                        cv2.imwrite(output_path, img)
                        return
                        
                    # Find space for an image
                    rect, corners = self.analyze_slide_for_image_placement(img)
                    
                    if rect:
                        # We found space for an image
                        x, y, w, h = rect
                        
                        # Show message about space found
                        self.status_var.set(f"Image {index+1}/{total} - Found space ({w}x{h}) for image generation")
                        self.root.update()  # Force UI update
                        
                        # Generate an image based on the text
                        generated_image = self.generate_image_for_text(ocr_text, size=(w, h))
                        
                        if generated_image is not None:
                            # Resize the generated image to fit the target space
                            generated_image = cv2.resize(generated_image, (w, h))
                            
                            # Create a copy of the original image
                            result_img = img.copy()
                            
                            # Insert the generated image
                            result_img[y:y+h, x:x+w] = generated_image
                            
                            # Save the result
                            output_path = os.path.join(self.output_dir, f"enhanced_{image_name}")
                            cv2.imwrite(output_path, result_img)
                            
                            self.status_var.set(f"Added AI-generated image to {image_name}")
                        else:
                            # Failed to generate image, save original
                            output_path = os.path.join(self.output_dir, image_name)
                            cv2.imwrite(output_path, img)
                    else:
                        # No suitable space for an image
                        self.status_var.set(f"Image {index+1}/{total} - No suitable space found for image")
                        output_path = os.path.join(self.output_dir, image_name)
                        cv2.imwrite(output_path, img)
                except Exception as e:
                    print(f"Error in image generation: {e}")
                    output_path = os.path.join(self.output_dir, image_name)
                    cv2.imwrite(output_path, img)

            # Update progress
            progress = (index + 1) / total * 100
            self.progress_var.set(progress)
            self.status_var.set(f"Processed {index + 1} of {total} images")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
    
    def process_images_thread(self):
        if not self.input_files:
            messagebox.showwarning("No Input", "Please select input files first.")
            return
            
        operation = self.operation_var.get()
        
        # Get splits value (as fallback)
        try:
            splits = int(self.splits_var.get())
            if splits < 1:
                splits = 1
        except ValueError:
            splits = 2  # Default value
        
        # Get rows per page value
        try:
            rows_per_page = int(self.rows_per_page_var.get())
            if rows_per_page < 1:
                rows_per_page = 10
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
        self.status_var.set("Processing...")
        
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
                
        # After the ThreadPoolExecutor block:
        if operation == 'blank_remover':
            blank_dir = os.path.join(self.output_dir, "blank_images")
            if os.path.exists(blank_dir):
                blank_count = len(os.listdir(blank_dir))
                
            self.status_var.set(f"Processing complete! Found {blank_count} blank images out of {total_count}")
            messagebox.showinfo("Complete", f"Image processing completed!\n\n{blank_count} blank images moved to 'blank_images' folder.\n{total_count - blank_count} non-blank images kept in output folder.")
        else:
            self.status_var.set("Processing complete!")
            messagebox.showinfo("Complete", "Image processing completed successfully!")
    
    def process_images(self):
        self.processing_cancelled = False
        self.cancel_button.config(state=tk.NORMAL)
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
    
    def cancel_processing(self):
        self.processing_cancelled = True
        self.status_var.set("Cancelling...")

    def open_output_folder(self):
        if os.path.exists(self.output_dir):
            # Open the folder in file explorer (works on Windows)
            os.startfile(self.output_dir)
        else:
            messagebox.showwarning("Folder Not Found", "Output folder does not exist yet.")
    
    def save_settings(self):
        """Save current settings as default"""
        try:
            settings = {
                'operation': self.operation_var.get(),
                'rows_per_page': self.rows_per_page_var.get(),
                # 'splits': self.splits_var.get(),  # Keep this for backward compatibility
                'output_dir': self.output_dir,
                'detect_questions': str(self.detect_questions_var.get()),
                'use_row_detection': str(self.use_row_detection_var.get()),
                'template_sensitivity': str(self.template_sensitivity_var.get()),
                'blank_sensitivity': str(self.blank_sensitivity_var.get())  # Add this line
            }
            
            # Create config directory if it doesn't exist
            config_dir = os.path.join(os.path.expanduser("~"), ".imagecropper")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Save settings to file
            with open(os.path.join(config_dir, "settings.txt"), 'w') as f:
                for key, value in settings.items():
                    f.write(f"{key}={value}\n")
                    
            messagebox.showinfo("Settings Saved", "Default settings have been saved.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
    
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
                            # elif key == 'splits':
                            #     self.splits_var.set(value)
                            elif key == 'rows_per_page':
                                self.rows_per_page_var.set(value)
                            elif key == 'detect_questions':
                                self.detect_questions_var.set(value.lower() == 'true')
                            # elif key == 'use_row_detection':
                            #     self.use_row_detection_var.set(value.lower() == 'true')
                            elif key == 'output_dir':
                                if os.path.exists(value):
                                    self.output_dir = value
                                    self.output_dir_label.config(text=self.output_dir)
                            elif key == 'blank_sensitivity':
                                try:
                                    self.blank_sensitivity_var.set(float(value))
                                except ValueError:
                                    self.blank_sensitivity_var.set(98.0)  # Default value
        except Exception as e:
            print(f"Error loading settings: {e}")
            
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
                self.input_files_label.config(text=f"{len(self.input_files)} files selected from folder")
                self.update_preview()
            else:
                messagebox.showwarning("No Images", "No image files found in the selected folder.")
    
    def toggle_templates_frame(self, *args):
        """Show or hide the templates frame based on the selected operation"""
        if self.operation_var.get() == "template_remover":
            # Make it appear right after the operation/parameters section
            self.templates_frame.pack(fill=tk.X, pady=5, after=self.op_param_container)
            
            # Show a message box to guide the user if no templates are selected yet
            # if not self.template_files:
            #     messagebox.showinfo("Template Selection", 
            #                        "Please select watermark/logo images to remove using the 'Select Templates' button.")
        else:
            self.templates_frame.pack_forget()

    def select_templates(self):
        """Select template images (logos or watermarks to remove)"""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.template_files = list(files)
            self.template_files_label.config(text=f"{len(self.template_files)} templates selected")
            self.update_template_preview()
            
            # Show message to explain the process
            messagebox.showinfo(
                "Templates Selected", 
                "Selected templates will be removed from all input images.\n\n" +
                "Processing may take several minutes depending on the number and size of images."
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
                print(f"Error creating template thumbnail: {e}")

    def use_ai_inpainting(self, image, mask):
        """Use an AI model to inpaint masked regions more naturally"""
        try:
            # This requires installing transformers and diffusers
            from diffusers import AutoPipelineForInpainting
            import torch
            
            # Load model - first time will download the model
            self.status_var.set("Loading AI model (first use may take a while)...")
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
                prompt="clean image without watermark",
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=20
            ).images[0]
            
            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        
        except ImportError:
            messagebox.showwarning("Libraries Missing", 
                "For AI-powered removal, install: pip install torch diffusers transformers")
            return cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)  # Fallback

    def show_image_generation_options(self):
        """Show a dialog with additional image generation options"""
        options_dialog = tk.Toplevel(self.root)
        options_dialog.title("Image Generation Options")
        options_dialog.geometry("400x300")
        options_dialog.transient(self.root)
        options_dialog.grab_set()
        
        ttk.Label(options_dialog, text="Model Selection:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        model_var = tk.StringVar(value="runwayml/stable-diffusion-v1-5")
        models = [
            ("Stable Diffusion 1.5 (Faster)", "runwayml/stable-diffusion-v1-5"),
            ("Stable Diffusion 2.1 (Better Quality)", "stabilityai/stable-diffusion-2-1")
        ]
        
        for text, value in models:
            ttk.Radiobutton(options_dialog, text=text, value=value, variable=model_var).pack(anchor=tk.W, padx=20)
        
        ttk.Label(options_dialog, text="Image Quality:").pack(anchor=tk.W, padx=10, pady=(10, 0))
        
        quality_var = tk.IntVar(value=25)
        quality_scale = ttk.Scale(
            options_dialog, 
            from_=15, 
            to=50, 
            orient="horizontal", 
            variable=quality_var, 
            length=300
        )
        quality_scale.pack(padx=10, pady=5, fill=tk.X)
        ttk.Label(options_dialog, text="Higher = Better quality but slower").pack(anchor=tk.W, padx=10)
        
        def save_options():
            self.image_model_var = model_var.get()
            self.image_quality_var = quality_var.get()
            options_dialog.destroy()
        
        ttk.Button(options_dialog, text="Save Options", command=save_options).pack(pady=20)
        
        # Add this button to your params frame
        # ttk.Button(param_frame, text="Advanced Image Options", command=self.show_image_generation_options).pack(anchor=tk.W, pady=10)

    def analyze_slide_for_image_placement(self, image):
        """Analyze a slide to find the best place to insert an AI-generated image"""
        # Convert to grayscale for text detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use OCR to find text areas
        custom_config = r'--oem 3 --psm 6'
        d = pytesseract.image_to_data(image, config=custom_config, output_type=Output.DICT)
        
        height, width = image.shape[:2]
        
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
        
        # Find the largest contiguous non-text area
        empty_mask = 255 - text_mask
        contours, _ = cv2.findContours(empty_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate the maximum square that fits in this rectangle
        size = min(w, h)
        
        # If the space is too small, return None
        if size < 200:  # Minimum size for a useful image
            return None, None
            
        # Center the square in the available space
        x_center = x + w // 2
        y_center = y + h // 2
        
        x_start = x_center - size // 2
        y_start = y_center - size // 2
        
        return (x_start, y_start, size, size), (x_start, y_start, x_start + size, y_start + size)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
