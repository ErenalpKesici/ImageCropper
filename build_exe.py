import PyInstaller.__main__
import os

# Get the path to Tesseract executable
# You need to have Tesseract-OCR installed on your development machine
tesseract_path = r'C:\Program Files\Tesseract-OCR'  # Adjust this path if needed

# Run PyInstaller
PyInstaller.__main__.run([
    'image_cropper_gui.py',
    '--name=ImageCropper',
    '--onefile',
    '--windowed',
    f'--add-binary={tesseract_path};Tesseract-OCR',
    '--icon=app_icon.ico',  # Add this line if you have an icon file
    '--add-data=tessdata;tessdata',
    '--hidden-import=PIL',
    '--hidden-import=PIL._tkinter_finder',
    '--hidden-import=pptx',
    '--hidden-import=docx',
    '--hidden-import=PyPDF2',
])

print("Build complete. Executable is in the dist folder.")