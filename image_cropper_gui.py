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
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
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
        
        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
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
        output_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        self.output_dir_label = ttk.Label(output_frame, text=self.output_dir)
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_button = ttk.Button(output_frame, text="Select Directory", command=self.select_output_dir)
        output_button.pack(side=tk.RIGHT, padx=5)
        
        
        # Add a save settings button
        save_settings_button = ttk.Button(output_frame, text="Save as Default", command=self.save_settings)
        save_settings_button.pack(anchor=tk.W, pady=5)
        
        # Create a frame to hold operation and parameters side by side
        self.op_param_container = ttk.Frame(main_frame)  # Store as instance variable
        self.op_param_container.pack(fill=tk.X, pady=5)
        
        # Operation section - now in left half
        operation_frame = ttk.LabelFrame(self.op_param_container, text="Operation", padding="10")
        operation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.operation_var = tk.StringVar(value="splitter")
        operations = [("Split Images", "splitter"), 
                     ("Crop Titles", "title_cropper"),
                     ("Remove Templates", "template_remover")]  # Add the new operation
        
        for text, value in operations:
            ttk.Radiobutton(operation_frame, text=text, value=value, variable=self.operation_var).pack(anchor=tk.W)
        
        # Parameters section - now in right half
        param_frame = ttk.LabelFrame(self.op_param_container, text="Parameters", padding="10")
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(param_frame, text="Splits (for splitter):").pack(anchor=tk.W)
        
        self.splits_var = tk.StringVar(value="6")
        splits_entry = ttk.Entry(param_frame, textvariable=self.splits_var, width=5)
        splits_entry.pack(anchor=tk.W, pady=5)

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
        
        # Preview area - will show thumbnails of selected images, with reduced height
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=1)
        
        self.canvas = tk.Canvas(preview_frame, height=150)  # Increased height slightly
        scrollbar = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.preview_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.preview_frame, anchor="nw")
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Add cancel button next to process button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        process_button = ttk.Button(button_frame, text="Process Images", command=self.process_images)
        process_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # Open output folder button
        open_folder_button = ttk.Button(main_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.pack(pady=5)
        
        # Add a new section for template images that appears when "Remove Templates" is selected
        self.templates_frame = ttk.LabelFrame(main_frame, text="Watermark/Logo Templates to Remove", padding="10")

        self.template_files = []
        self.template_files_label = ttk.Label(self.templates_frame, text="No templates selected")
        self.template_files_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        select_templates_button = ttk.Button(self.templates_frame, text="Select Watermarks/Logos", 
                                             command=self.select_templates)
        select_templates_button.pack(side=tk.RIGHT)
        
        # Show/hide the templates frame based on operation selection
        self.operation_var.trace("w", self.toggle_templates_frame)
        
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
            
            # Use PIL to read the image instead of OpenCV for better Unicode support
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
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
        try:
            splits = int(self.splits_var.get())
            if splits < 1:
                splits = 1
        except ValueError:
            splits = 6  # Default value
            
        params = [splits]
        
        # Reset progress
        self.progress_var.set(0)
        self.status_var.set("Processing...")
        
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
                'splits': self.splits_var.get(),
                'output_dir': self.output_dir
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
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            if key == 'operation':
                                self.operation_var.set(value)
                            elif key == 'splits':
                                self.splits_var.set(value)
                            elif key == 'output_dir':
                                if os.path.exists(value):
                                    self.output_dir = value
                                    self.output_dir_label.config(text(self.output_dir))
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
