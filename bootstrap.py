import os
import sys
import pytesseract

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

# Configure Tesseract path
tessdata_dir = resource_path("tessdata")
os.environ["TESSDATA_PREFIX"] = tessdata_dir

# Import and run the main application
from image_cropper_gui import *

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCropperApp(root)
    root.mainloop()