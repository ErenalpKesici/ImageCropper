# Intelligent Image Splitting Feature

## Problem Solved
The original image splitting function used a simple approach that cut images exactly in the middle (`mid_height = height // 2`), which frequently resulted in cutting through text lines, making the split images difficult to read.

## Solution Implemented

### 1. Smart Text Detection
- **OCR Analysis**: Uses Tesseract OCR to detect text positions and bounding boxes
- **Visual Analysis**: Analyzes image content using computer vision to find areas with minimal content
- **Combined Scoring**: Merges OCR text density with visual variance analysis for optimal results

### 2. Intelligent Split Point Selection
- **Gap Detection**: Finds horizontal bands with minimal text content
- **Proximity Optimization**: Selects split points closest to the target middle while avoiding text
- **Fallback Mechanism**: Falls back to simple middle split if no good gaps are found

### 3. Advanced Multi-Split Support
- **Configurable Splits**: User can specify 1-5 split parts via new UI field "Maximum number of splits"
- **Minimum Distance**: Ensures splits are well-spaced (at least 25% of image height apart)
- **Content-Aware**: Prioritizes areas with least content for splitting

## New Features Added

### UI Enhancements
1. **Max Splits Field**: New input field allowing users to specify how many parts to split into
   - Range: 1-5 splits
   - Default: 2 splits (maintains backward compatibility)
   - Includes helpful tooltip text

2. **Multilingual Support**: Added translations for new UI elements in both Turkish and English

### Core Functions Added

#### `find_optimal_split_points(img)`
- Analyzes image using OCR to detect text positions
- Finds gaps between text lines
- Returns optimal 2-way split coordinates
- Falls back to middle split if no suitable gaps found

#### `find_text_free_zones(img)`
- Advanced content analysis using both OCR and visual methods
- Creates density map of text content per row
- Uses Gaussian smoothing to reduce noise
- Returns content scores for the entire image height

#### `find_multiple_split_points(img, max_splits=3)`
- Supports splitting into multiple parts (up to 5)
- Uses local minima detection to find best split zones
- Ensures minimum distance between split points
- Optimizes for content distribution

## Technical Improvements

### Error Handling
- Graceful fallback to simple splits if OCR fails
- Exception handling for all edge cases
- Informative logging for debugging

### Performance Optimization
- Efficient OCR configuration for Turkish/English text
- Adaptive kernel sizing for image smoothing
- Binary search for optimal split point selection

### Code Quality
- Well-documented functions with clear docstrings
- Modular design for easy maintenance
- Proper parameter validation and bounds checking

## Usage

### Basic Usage (2-way split)
1. Select "Split Images" operation
2. Set "Maximum number of splits" to 2 (default)
3. Process images - they will be split intelligently avoiding text

### Advanced Usage (multi-way split)
1. Select "Split Images" operation
2. Set "Maximum number of splits" to 3-5
3. Process images - they will be split into multiple parts optimally

## Benefits

1. **No More Text Cutting**: OCR analysis ensures splits occur between text lines
2. **Flexible Splitting**: Support for 1-5 way splits based on user needs
3. **Intelligent Placement**: Splits are placed in areas with minimal content disruption
4. **Backward Compatible**: Existing workflows continue to work unchanged
5. **Robust Fallback**: If intelligent analysis fails, falls back to simple splitting

## Example Output
```
📏 Found optimal split point at y=215 (target was y=300)
✂️ Splitting image: [0:215] and [215:600]
🎯 Applying 2 intelligent splits
```

The intelligent splitting now ensures that images are split in empty areas between text blocks rather than cutting through the middle of sentences or words, significantly improving readability of the output images.
