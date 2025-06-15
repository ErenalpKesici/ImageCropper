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

def contains_question_indicators(text):
    """Check if the text contains question indicators like a), A), 1), etc."""
    pattern = r'(?:^|\s)([a-zA-Z0-9])[.)]'
    matches = re.findall(pattern, text)
    unique_indicators = set(matches)
    return len(unique_indicators) >= 2

def remove_empty_rows_and_columns(image, empty_threshold=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    row_sums = np.sum(gray < empty_threshold, axis=1)
    col_sums = np.sum(gray < empty_threshold, axis=0)
    
    non_empty_rows = np.where(row_sums > 0)[0]
    non_empty_cols = np.where(col_sums > 0)[0]
    
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

class ImageCropperApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Cropper & Document Converter")
        self.root.geometry("800x700")
        self.root.minsize(800, 700)
        
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
        
        select_images_button = ttk.Button(select_buttons_frame, text="Select Images", command=self.select_files)
        select_images_button.pack(side=tk.LEFT, padx=5)
        
        select_docs_button = ttk.Button(select_buttons_frame, text="Select Documents", command=self.select_documents)
        select_docs_button.pack(side=tk.LEFT, padx=5)
        
        # Output section
        output_frame = ttk.LabelFrame(scrollable_frame, text="Çıktı", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        self.output_dir_label = ttk.Label(output_frame, text=self.output_dir)
        self.output_dir_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        output_button = ttk.Button(output_frame, text="Klasör Seç", command=self.select_output_dir)
        output_button.pack(side=tk.RIGHT, padx=5)
        
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
        
        for text, value in operations:
            ttk.Radiobutton(operation_frame, text=text, value=value, variable=self.operation_var).pack(anchor=tk.W)
        
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
        
        self.detect_questions_var = tk.BooleanVar(value=True)
        question_checkbox = ttk.Checkbutton(
            param_frame, 
            text="Soruları algıla (bölme)",
            variable=self.detect_questions_var
        )
        question_checkbox.pack(anchor=tk.W, pady=(10, 0))
        
        # Preview area
        preview_frame = ttk.LabelFrame(scrollable_frame, text="Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.preview_area = ttk.Frame(preview_frame)
        self.preview_area.pack(fill=tk.BOTH, expand=True)
        
        # Progress and status
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(scrollable_frame, variable=self.progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(scrollable_frame, textvariable=self.status_var)
        status_label.pack(anchor=tk.W, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        process_button = ttk.Button(button_frame, text="Process", command=self.process_files)
        process_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(button_frame, text="İptal", command=self.cancel_processing, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        open_folder_button = ttk.Button(button_frame, text="Open Output Folder", command=self.open_output_folder)
        open_folder_button.pack(side=tk.RIGHT, padx=5)
    
    def select_files(self):
        """Select image files"""
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        
        if files:
            self.input_files = list(files)
            self.input_files_label.config(text=f"{len(self.input_files)} image(s) selected")
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
        """Convert Word or PDF document to PowerPoint presentation with table detection"""
        if not PPTX_AVAILABLE:
            self.status_var.set("PowerPoint libraries not available")
            return False
            
        try:
            content_items = []
            file_ext = os.path.splitext(document_path)[1].lower()
            
            self.status_var.set(f"Reading {file_ext} file and detecting tables...")
            self.root.update()
            
            if file_ext == '.docx':
                content_items = self.extract_docx_content_with_tables(document_path)
            
            elif file_ext == '.pdf':
                content_items = self.extract_pdf_content_with_tables(document_path)
            if not content_items:
                self.status_var.set(f"No content found in {document_path}")
                return False
            
            # Create PowerPoint presentation
            prs = Presentation()
            prs.slide_width = Inches(16)
            prs.slide_height = Inches(9)

            slide_layout = prs.slide_layouts[6]  # Blank layout
            
            current_slide_lines = []
            slide_count = 0
            table_count = 0
            self.status_var.set("Creating slides with tables and optimal spacing...")
            self.root.update()
            
            # Process content items (text and tables)
            for item in content_items:
                if item['type'] == 'table':
                    # Create a dedicated slide for the table
                    if current_slide_lines:
                        # Finish current text slide first
                        slide_count += 1
                        self.create_optimized_slide(prs, slide_layout, current_slide_lines)
                        current_slide_lines = []
                    
                    # Create table slide
                    slide_count += 1
                    table_count += 1
                    self.create_table_slide(prs, slide_layout, item['content'])
                    
                elif item['type'] == 'text':
                    # Add text to current slide
                    current_slide_lines.append(item['content'])
                    
                    # Create new slide when we reach the line limit
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
            
            status_msg = f"Created PowerPoint: {base_name}.pptx with {slide_count} slides"
            if table_count > 0:
                status_msg += f" (including {table_count} table(s))"
            self.status_var.set(status_msg)
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
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                return False
                
            image_name = os.path.basename(image_path)
            
            if operation == 'splitter':
                # Simple image splitting
                rows_per_page = params.get('rows_per_page', 10)
                detect_questions = params.get('detect_questions', True)
                
                if detect_questions:
                    custom_config = r'-l tur --oem 3 --psm 6'
                    ocr_text = pytesseract.image_to_string(img, config=custom_config)
                    has_questions = contains_question_indicators(ocr_text)
                else:
                    has_questions = False
                if has_questions:
                    # Don't split images with questions
                    cleaned_img = remove_empty_rows_and_columns(img)
                    cleaned_img = add_outer_border(cleaned_img, top_border=50, bottom_border=50)
                    cleaned_img = cv2.resize(cleaned_img, (1920, 1080))
                    output_path = os.path.join(self.output_dir, image_name)
                    cv2.imwrite(output_path, cleaned_img)
                else:
                    # Split into 2 parts
                    height, width = img.shape[:2]
                    mid_height = height // 2
                    
                    for i in range(2):
                        if i == 0:
                            crop_img = img[0:mid_height, :]
                        else:
                            crop_img = img[mid_height:, :]
                        
                        crop_img = remove_empty_rows_and_columns(crop_img)
                        crop_img = add_outer_border(crop_img, top_border=100, bottom_border=100, left_border=40, right_border=40)
                        if not is_image_blank(crop_img):
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
                    blank_dir = os.path.join(self.output_dir, "blank_images")
                    if not os.path.exists(blank_dir):
                        os.makedirs(blank_dir)
                    
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(blank_dir, image_name)
                    pil_img.save(output_path)
                else:
                    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    output_path = os.path.join(self.output_dir, image_name)
                    pil_img.save(output_path)
            
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
        self.status_var.set("Cancelling...")
    
    def open_output_folder(self):
        """Open the output folder in file explorer"""
        if os.path.exists(self.output_dir):
            os.startfile(self.output_dir)
        else:
            messagebox.showwarning("Folder Not Found", "Output folder does not exist yet.")
    
    def extract_docx_content_with_tables(self, document_path):
        """Extract content from Word document, preserving tables"""
        content_items = []
        
        try:
            doc = docx.Document(document_path)
            
            # Get all paragraphs and tables in document order
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
                    # Create paragraph object from element
                    for para in doc.paragraphs:
                        if para._element is element and para.text.strip():
                            content_items.append({
                                'type': 'text',
                                'content': para.text.strip()
                            })
                            break
                            
                elif element.tag.endswith('tbl'):  # Table
                    # Create table object from element
                    for table in doc.tables:
                        if table._element is element:
                            table_data = []
                            for row in table.rows:
                                row_data = [cell.text.strip() for cell in row.cells]
                                table_data.append(row_data)
                            
                            # Only add non-empty tables
                            if table_data and any(any(cell for cell in row) for row in table_data):
                                content_items.append({
                                    'type': 'table',
                                    'content': table_data,
                                    'rows': len(table_data),
                                    'cols': len(table_data[0]) if table_data else 0
                                })
                            break
                        
        except Exception as e:
            print(f"Error extracting Word content with tables: {e}")
            # Fallback to simple extraction
            try:
                doc = docx.Document(document_path)
                # Get all paragraphs first
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        content_items.append({
                            'type': 'text',
                            'content': paragraph.text.strip()
                        })
                
                # Then get all tables
                for table in doc.tables:
                    table_data = []
                    for row in table.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_data.append(row_data)
                    
                    if table_data and any(any(cell for cell in row) for row in table_data):
                        content_items.append({
                            'type': 'table',
                            'content': table_data,
                            'rows': len(table_data),
                            'cols': len(table_data[0]) if table_data else 0
                        })
            except Exception as fallback_error:
                print(f"Fallback extraction also failed: {fallback_error}")
        
        return content_items
    
    def extract_pdf_content_with_tables(self, document_path):
        """Extract content from PDF, attempting to detect tables"""
        content_items = []
        
        try:
            # Try to use pdfplumber for better table detection if available
            try:
                import pdfplumber
                use_pdfplumber = True
                print("Using pdfplumber for enhanced table detection")
            except ImportError:
                use_pdfplumber = False
                print("pdfplumber not available, using basic PDF extraction")
            
            if use_pdfplumber:
                with pdfplumber.open(document_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        # Extract tables first
                        tables = page.extract_tables()
                        if tables:
                            for table in tables:
                                # Filter empty rows and cells
                                cleaned_table = []
                                for row in table:
                                    if row and any(cell and str(cell).strip() for cell in row):
                                        cleaned_row = [str(cell).strip() if cell else '' for cell in row]
                                        cleaned_table.append(cleaned_row)
                                
                                if cleaned_table:
                                    content_items.append({
                                        'type': 'table',
                                        'content': cleaned_table,
                                        'rows': len(cleaned_table),
                                        'cols': len(cleaned_table[0]) if cleaned_table else 0,
                                        'page': page_num + 1
                                    })
                        
                        # Extract text
                        text = page.extract_text()
                        if text:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            for line in lines:
                                # Simple heuristic to avoid duplicating table content
                                if not self.is_likely_table_text(line):
                                    content_items.append({
                                        'type': 'text',
                                        'content': line,
                                        'page': page_num + 1
                                    })
            else:
                # Fallback to PyPDF2 with basic table detection
                with open(document_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            for line in lines:
                                # Simple table detection based on patterns
                                if self.is_likely_table_text(line):
                                    # Try to parse as table row
                                    table_row = self.parse_table_row(line)
                                    if table_row:
                                        content_items.append({
                                            'type': 'table',
                                            'content': [table_row],
                                            'rows': 1,
                                            'cols': len(table_row),
                                            'page': page_num + 1
                                        })
                                    else:
                                        content_items.append({
                                            'type': 'text',
                                            'content': line,
                                            'page': page_num + 1
                                        })
                                else:
                                    content_items.append({
                                        'type': 'text',
                                        'content': line,
                                        'page': page_num + 1
                                    })
                                    
        except Exception as e:
            print(f"Error extracting PDF content: {e}")
            # Ultimate fallback
            try:
                with open(document_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            for line in lines:
                                content_items.append({
                                    'type': 'text',
                                    'content': line,
                                    'page': page_num + 1
                                })
            except Exception as final_error:
                print(f"All PDF extraction methods failed: {final_error}")
        
        return content_items
    
    def is_likely_table_text(self, text):
        """Heuristic to detect if text line is likely from a table"""
        # Look for common table patterns
        patterns = [
            r'\s+\|\s+',              # Pipe separators
            r'\t{2,}',                # Multiple tabs
            r'\s{4,}',                # Multiple spaces (4 or more)
            r'^\s*\d+\.\s+.*\s+\d+',  # Number, text, number pattern
            r'.*\s+\$\d+',            # Text followed by dollar amount
            r'.*\s+\d+%',             # Text followed by percentage
        ]
        
        for pattern in patterns:
            if re.search(pattern, text):
                return True
                
        # Check if line has structure like "Item1    Item2    Item3"
        parts = text.split()
        if len(parts) >= 3:
            # Check for significant spacing between words
            original_spaces = len(text) - len(text.replace(' ', ''))
            if original_spaces > len(parts) * 3:  # More than 3 spaces per word on average
                return True
                
        return False
    
    def parse_table_row(self, text):
        """Attempt to parse a text line as a table row"""
        # Try different separators
        separators = ['|', '\t', '    ', '   ']  # Tab, multiple spaces
        
        for sep in separators:
            if sep in text:
                cells = [cell.strip() for cell in text.split(sep)]
                # Filter out empty cells and ensure we have at least 2 columns
                cells = [cell for cell in cells if cell]
                if len(cells) >= 2:
                    return cells
        
        return None

    def create_table_slide(self, prs, slide_layout, table_data):
        """Create a slide with a table"""
        slide = prs.slides.add_slide(slide_layout)
        
        # Calculate table dimensions
        rows = len(table_data)
        cols = max(len(row) for row in table_data) if table_data else 1
        
        # Position table in center of slide with appropriate sizing
        left = Inches(0.5)
        top = Inches(1.5)
        width = Inches(15)    # Almost full width
        height = Inches(6.5)  # Leave space for potential title
        
        # Add table shape
        table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
        table = table_shape.table
        
        # Populate table
        for row_idx, row_data in enumerate(table_data):
            for col_idx, cell_data in enumerate(row_data):
                if col_idx < cols:  # Ensure we don't exceed table columns
                    cell = table.cell(row_idx, col_idx)
                    cell.text = str(cell_data) if cell_data else ''
                    
                    # Style the cells
                    for paragraph in cell.text_frame.paragraphs:
                        paragraph.font.name = 'Calibri'
                        paragraph.font.size = Pt(11)  # Slightly smaller for tables
                        
                        # Make header row bold and centered
                        if row_idx == 0:
                            paragraph.font.bold = True
                            paragraph.alignment = PP_ALIGN.CENTER
                            cell.fill.solid()
                            cell.fill.fore_color.rgb = RGBColor(68, 114, 196)  # Blue header
                            paragraph.font.color.rgb = RGBColor(255, 255, 255)  # White text
                        else:
                            paragraph.font.bold = False
                            paragraph.alignment = PP_ALIGN.LEFT
        
        # Apply table styling
        try:
            table.style = 'LightShading-Accent1'  # Built-in table style
        except:
            pass  # Style might not be available
        
        return slide

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()
