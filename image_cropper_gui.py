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
        op_param_container = ttk.Frame(main_frame)
        op_param_container.pack(fill=tk.X, pady=5)
        
        # Operation section - now in left half
        operation_frame = ttk.LabelFrame(op_param_container, text="Operation", padding="10")
        operation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.operation_var = tk.StringVar(value="splitter")
        operations = [("Split Images", "splitter"), ("Crop Titles", "title_cropper")]
        
        for text, value in operations:
            ttk.Radiobutton(operation_frame, text=text, value=value, variable=self.operation_var).pack(anchor=tk.W)
        
        # Parameters section - now in right half
        param_frame = ttk.LabelFrame(op_param_container, text="Parameters", padding="10")
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        ttk.Label(param_frame, text="Splits (for splitter):").pack(anchor=tk.W)
        
        self.splits_var = tk.StringVar(value="6")
        splits_entry = ttk.Entry(param_frame, textvariable=self.splits_var, width=5)
        splits_entry.pack(anchor=tk.W, pady=5)
        
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
        
        # Process button
        process_button = ttk.Button(main_frame, text="Process Images", command=self.process_images)
        process_button.pack(pady=10)
        
        # Open output folder button
        open_folder_button = ttk.Button(main_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.pack(pady=5)
        
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
        thread = threading.Thread(target=self.process_images_thread)
        thread.daemon = True
        thread.start()
    
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
