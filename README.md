# Image Cropper & Document Converter

A comprehensive tool for image processing and document-to-PowerPoint conversion.

## ✨ Key Improvements in Latest Version

### 🔧 **Fixed Issues (Latest Update)**
1. **Multiple Lines Per Slide**: Fixed bug where only 1 line appeared per slide regardless of setting
2. **Full Slide Utilization**: Textbox now uses almost entire slide (15.4" × 8.4" vs previous 8.8" × 7.0")
3. **Better Space Management**: Minimal margins (0.05") and tighter spacing for maximum content
4. **Enhanced Font Sizing**: Dynamic font sizing now works with the larger text area

### 🎯 **Dynamic Font Sizing Features**
- **Adaptive Sizing**: Font sizes automatically adjust based on content density
- **Content-Aware**: Different font ranges for titles (18-36pt), subtitles (16-32pt), bullets (14-28pt), and content (12-24pt)  
- **Overflow Prevention**: Guarantees no text overflow beyond slide boundaries
- **Space Optimization**: Uses largest possible fonts while fitting all content

## Features

### Image Processing
- **Split Images**: Automatically split images into multiple parts based on content
- **Crop Titles**: Remove title sections from images  
- **Remove Blank Images**: Identify and separate blank or nearly-blank images
- **Question Detection**: Automatically detect images containing questions and avoid splitting them

### Document Conversion (NEW!)
- **Word to PowerPoint**: Convert .docx files to PowerPoint presentations  
- **PDF to PowerPoint**: Convert .pdf files to PowerPoint presentations
- **Table Detection & Preservation**: Automatically detects tables in documents and preserves them
- **Professional Table Formatting**: Tables get dedicated slides with blue headers and proper styling
- **Mixed Content Support**: Handles documents with both text and tables seamlessly
- **Document Order Preservation**: Maintains original sequence (text → table → text → table)
- **Configurable Layout**: Set how many lines per slide for optimal spacing
- **Smart Formatting**: Automatic title detection and formatting

## Installation

1. Install Python 3.8 or higher
2. Install required packages:
```bash
pip install -r requirements.txt
```

## Table Detection Features

### Word Documents (.docx)
- **Native Table Detection**: Recognizes Word tables and preserves structure
- **Cell Content Extraction**: Maintains text formatting and cell relationships  
- **Document Order**: Preserves sequence of tables and text as they appear

### PDF Documents (.pdf)
- **Enhanced Detection**: Uses `pdfplumber` library for superior table extraction
- **Pattern Recognition**: Detects table-like text patterns as fallback
- **Multi-format Support**: Handles pipe-separated, tab-separated, and space-aligned tables

### PowerPoint Output
- **Dedicated Table Slides**: Each table gets its own professionally formatted slide
- **Professional Styling**: Blue headers (#4472C4) with white text for contrast
- **Optimal Sizing**: Tables sized at 15" × 6.5" for maximum visibility
- **Calibri Font**: 11pt font size optimized for readability

## Required Libraries

### Core Libraries
- opencv-python
- pytesseract  
- pillow
- numpy

### Document Processing Libraries (for new features)
- python-pptx
- python-docx  
- PyPDF2
- pdfplumber (for enhanced PDF table detection)

## Usage

### Running the Application
```bash
python image_cropper_gui.py
```

Or use the bootstrap launcher:
```bash
python bootstrap.py
```

### Image Processing
1. Click "Select Images" or "Select Folder" to choose image files
2. Choose your output directory
3. Select an operation (Split Images, Crop Titles, etc.)
4. Adjust parameters as needed
5. Click "Process" to start

### Document to PowerPoint Conversion
1. Click "Select Documents" to choose .docx or .pdf files
2. Choose your output directory  
3. Select "Convert Document to PowerPoint" operation
4. Set "Lines per slide" parameter (default: 10)
5. Click "Process" to start conversion

## Parameters

### Lines per Slide
- Controls how many text lines are placed on each PowerPoint slide
- **Recommended: 5-8 lines** for optimal formatting with dynamic font sizing
- Lower values = larger fonts, more readable content
- Higher values = smaller fonts to fit more content
- **NEW**: Dynamic font sizing automatically optimizes font sizes based on content

### Font Sizes (Enhanced for Readability)
- **Title slides**: 26pt, bold, centered, professional blue
- **Subtitles**: 24pt, bold, left-aligned, darker blue
- **Bullet points**: 22pt, normal weight
- **Regular content**: 20pt, normal weight
- **All fonts**: Calibri for modern, professional appearance

### Rows per Page (Image Splitting)
- Controls how images are split based on text rows
- Only applies to image splitting operations

### Question Detection
- When enabled, automatically detects images containing questions (a), b), c) patterns
- Prevents splitting of question images to maintain readability

## Output

### Image Processing
- Processed images saved to output directory
- Blank images moved to "blank_images" subfolder (if using blank removal)
- Split images numbered with suffixes (_0, _1, etc.)

### Document Conversion
- PowerPoint files (.pptx) created in output directory
- Maintains original document name with .pptx extension
- Professional formatting with automatic title detection

## Supported File Types

### Input Images
- PNG, JPG, JPEG, BMP, TIFF

### Input Documents  
- Microsoft Word (.docx)
- PDF (.pdf)

### Output
- PNG images (processed images)
- PowerPoint presentations (.pptx)

## Tips for Best Results

### Document Conversion
- Use well-formatted source documents for best results
- Ensure text is selectable in PDF files (not scanned images)
- **Set "Lines per slide" to 5-8** to prevent text overflow
- Use larger fonts (20pt-26pt) for better readability and minimal gaps
- Review generated slides and adjust formatting as needed

### Image Processing
- Use high-quality source images for better OCR results
- Ensure text is clear and readable for question detection
- Adjust blank detection sensitivity if needed

## Troubleshooting

### Missing Libraries Error
If you see errors about missing libraries when trying document conversion:
```bash
pip install python-pptx python-docx PyPDF2
```

### Tesseract Not Found
- Install Tesseract OCR on your system
- Ensure it's in your system PATH
- On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

### PDF Text Extraction Issues
- Ensure PDF contains selectable text (not scanned images)
- For scanned PDFs, use OCR software first to make text selectable

## Building Executable

Use PyInstaller to create a standalone executable:
```bash
python build_exe.py
```

## Project Structure

```
ImageCropper/
├── image_cropper_gui.py    # Main application
├── bootstrap.py            # Launcher with Tesseract setup
├── build_exe.py           # PyInstaller build script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── cropped/              # Default output directory
└── images/               # Sample images
```

## Recent Changes

### Version 2.0 Features
- ✅ Added Word document to PowerPoint conversion
- ✅ Added PDF to PowerPoint conversion  
- ✅ Configurable lines per slide
- ✅ Smart title detection and formatting
- ✅ Improved user interface with separate document selection
- ✅ Better error handling and user feedback
- ✅ Cleaned up redundant files

### Removed
- ❌ Removed unused non-GUI image_cropper.py
- ❌ Removed redundant .spec files
- ❌ Cleaned up project structure

## License

Open source - feel free to modify and distribute.

## Contributing

Submit issues and pull requests on the project repository.
