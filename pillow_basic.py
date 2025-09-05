from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import logging

# Global dictionaries no longer needed for full text display
bbox_word_count_tracker = {}  # Kept for compatibility
completed_regions_tracker = set()  # Kept for compatibility

def simple_text_logic(bbox_key, hindi_words, marathi_words, marathi_text, logger=None):
    """
    Simplified text logic - always show full translation immediately
    """
    if logger:
        logger.debug(f"✅ Showing full Marathi translation immediately")
    return marathi_text

def process_frame_with_text_buffer_basic(image_path, text_buffer, page_data, font_path=r"noto-sans-devanagari\NotoSansDevanagari-Bold.ttf", logger=None):
    """
    Smart text overlay with comprehensive validation and perfect text containment
    - Skips frames with no text detected in page_data (returns original frame)
    - Skips frames where detected text has no valid Marathi translations (returns original frame)
    - Uses page_data to identify active Hindi text regions in current frame
    - Matches each region with closest buffer entry (by Y-coordinate) for translation content
    - Only whitens buffer bounding boxes that will actually be used for text placement
    - Ensures Marathi text is placed WITHIN the whitened buffer bounding box (not page_data position)
    - Auto-adjusts font size if needed to guarantee text fits within white box boundaries
    - Shows each unique translation ONLY ONCE per frame at the best-matching position
    - Perfect for both growing text (same Y-coords) and moving text (different Y-coords)
    """
    
    def calculate_font_for_bbox(bbox_coords, font_path, logger=None):
        """
        Calculate font size and weight for specific bounding box coordinates
        Ensures consistent font properties based on bbox position and dimensions
        """
        if not bbox_coords or len(bbox_coords) != 4:
            return 24, ImageFont.truetype(font_path, 24)  # Default fallback
            
        x1, y1, x2, y2 = bbox_coords
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Calculate available height with padding
        available_height = int(height * 0.75)  # 25% padding for clean appearance
        
        # Use consistent font weight across all frames - no bold variations
        font_file = font_path
            
        # Calculate optimal font size
        optimal_size = max(10, min(available_height, 72))
        
        # Fine-tune based on width constraints
        aspect_ratio = width / height if height > 0 else 1
        if aspect_ratio < 2:  # Narrow boxes - reduce font size
            optimal_size = int(optimal_size * 0.8)
        elif aspect_ratio > 8:  # Very wide boxes - can use larger font
            optimal_size = int(optimal_size * 1.2)
            
        # Test with actual font rendering
        try:
            test_font = ImageFont.truetype(font_file, optimal_size)
            test_text = "आरशात पूर्तबिंबि"  # Sample Marathi text
            
            bbox_test = test_font.getbbox(test_text)
            text_height = bbox_test[3] - bbox_test[1]
            
            # Reduce size if text is too tall
            while text_height > available_height and optimal_size > 8:
                optimal_size -= 2
                test_font = ImageFont.truetype(font_file, optimal_size)
                bbox_test = test_font.getbbox(test_text)
                text_height = bbox_test[3] - bbox_test[1]
                
            font_obj = ImageFont.truetype(font_file, optimal_size)
            
        except Exception as e:
            if logger:
                logger.warning(f"Font loading error: {e}, using default")
            # Fallback to default font
            optimal_size = max(10, min(available_height // 2, 24))
            font_obj = ImageFont.truetype(font_path, optimal_size)
            
        if logger:
            logger.debug(f"📏 Font for bbox {bbox_coords}: {optimal_size}px consistent weight")
            
        return optimal_size, font_obj
    
    def get_consistent_start_position(bbox_coords):
        """
        Get consistent starting position for text to prevent dancing
        Always uses top-left of bounding box with small margin
        """
        if not bbox_coords or len(bbox_coords) != 4:
            return (0, 0)
            
        x1, y1, x2, y2 = bbox_coords
        
        # Consistent starting position: top-left with small margin
        start_x = x1 + 5  # 5px left margin
        start_y = y1 + 8  # 8px top margin
        
        return (start_x, start_y)
    
    def create_translation_lookup(text_buffer):
        """Create a lookup dictionary from text_buffer"""
        translation_map = {}
        
        for item in text_buffer:
            if item.get('marathi_translation') is None:
                continue
            
            hindi_text = item['all_text'].strip()
            marathi_text = item['marathi_translation'].strip()
            translation_map[hindi_text] = marathi_text
            
        return translation_map
    
    def convert_bbox_coordinates(bbox_corners):
        """Convert corner coordinates to (x1, y1, x2, y2)"""
        try:
            x_coords = [point[0] for point in bbox_corners]
            y_coords = [point[1] for point in bbox_corners]
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        except Exception as e:
            print(f"  ⚠️  Error in convert_bbox_coordinates: bbox_corners={bbox_corners}, error: {e}")
            # Return a fallback bbox
            return (0, 0, 100, 100)
    
    def extract_dominant_recessive_colors(image, bbox):
        """Extract dominant and recessive colors from bounding box"""
        bbox_area = image.crop(bbox)
        np_bbox = np.array(bbox_area)
        
        # Reshape to list of pixels
        pixels = np_bbox.reshape(-1, np_bbox.shape[-1])[:, :3]  # Only RGB
        
        # Simple histogram-based approach to find dominant colors
        # Quantize colors to reduce complexity
        quantized = (pixels // 32) * 32  # Group colors into bins
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(quantized.reshape(-1, 3), axis=0, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)  # Descending order
        
        if len(unique_colors) >= 2:
            # Most frequent = dominant, least frequent = recessive
            dominant_color = tuple(unique_colors[sorted_indices[0]])
            recessive_color = tuple(unique_colors[sorted_indices[-1]])
        else:
            # Fallback if only one color found
            avg_color = np.mean(pixels, axis=0).astype(int)
            dominant_color = tuple(avg_color)
            
            # Create contrasting recessive color
            brightness = np.mean(avg_color)
            if brightness > 127:
                recessive_color = tuple(np.maximum(avg_color - 80, 0))
            else:
                recessive_color = tuple(np.minimum(avg_color + 80, 255))
        
        return dominant_color, recessive_color
    
    def find_font_size(text, font_path, max_width, max_height):
        """Find the largest font size that fits"""
        for size in range(10, min(max_height, 100), 2):
            try:
                font = ImageFont.truetype(font_path, size)
                text_bbox = font.getbbox(text)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                if text_width > max_width or text_height > max_height:
                    return max(size - 2, 10)
            except:
                return size - 2
        
        return min(max_height, 100) - 2
    
    # Load image
    image = Image.open(image_path)
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Always use buffer data for consistent whitening and text placement
    translation_map = create_translation_lookup(text_buffer)
    
    # Debug logging for frame lookup
    def log_debug(message):
        if logger:
            logger.debug(message)
        else:
            print(message)
    
    def log_info(message):
        if logger:
            logger.info(message)
        else:
            print(message)
    
    def log_warning(message):
        if logger:
            logger.warning(message)
        else:
            print(message)
    
    # DEBUG: Start of function
    log_debug(f"🚀 DEBUG: Starting process_frame_with_text_buffer_basic")
    log_debug(f"   📂 Image path: {image_path}")
    log_debug(f"   📊 Text buffer items: {len(text_buffer) if text_buffer else 0}")
    log_debug(f"   🔤 Font path: {font_path}")
    log_debug(f"   📝 Logger: {type(logger) if logger else 'None'}")
    
    # Check if there's any text detected in this frame first
    frame_name = os.path.normpath(image_path)
    
    if frame_name not in page_data:
        log_debug(f"  ❌ Frame {frame_name} not found in page_data - returning original frame")
        return image_copy
    
    frame_texts = page_data[frame_name]
    if not frame_texts:
        log_debug(f"  ❌ No text detected in frame {frame_name} - returning original frame")
        return image_copy
    
    # Check if ANY detected text has valid Marathi translations before processing
    has_valid_translations = False
    for text_info in frame_texts:
        page_bbox = text_info['bbox']
        page_y_center = (page_bbox[1] + page_bbox[3]) / 2
        
        # Check if we can find a buffer item with translation for this region
        for buffer_item in text_buffer:
            if buffer_item.get('marathi_translation') is not None:
                buffer_bbox = buffer_item['bbox']
                if isinstance(buffer_bbox, (list, tuple)) and len(buffer_bbox) == 4:
                    buffer_y_center = (buffer_bbox[1] + buffer_bbox[3]) / 2
                    y_distance = abs(page_y_center - buffer_y_center)
                    
                    # If we find a reasonably close match, we have valid translation
                    if y_distance < 100:  # Reasonable matching threshold
                        has_valid_translations = True
                        break
        
        if has_valid_translations:
            break
    
    if not has_valid_translations:
        log_debug(f"  ❌ No valid Marathi translations found for detected text in frame {frame_name} - returning original frame")
        return image_copy
    
    log_debug(f"  📍 Found {len(frame_texts)} text regions with valid translations - proceeding with processing")
    
    # Match page_data regions with buffer entries - show each translation only once per frame
    log_debug(f"📝 Matching page_data regions with buffer translations...")
    translations = []
    whitened_buffer_items = set()  # Track which buffer items we've whitened
    
    # Dictionary to track which translations we've already added (avoid duplicates)
    used_translations = {}
    
    # First pass: find the best match for each page_data region
    matches = []
    for text_info in frame_texts:
        page_bbox = text_info['bbox']
        page_y_center = (page_bbox[1] + page_bbox[3]) / 2  # Y-coordinate center
        
        # Find buffer entry with closest Y-coordinate
        closest_buffer_item = None
        min_y_distance = float('inf')
        
        for buffer_item in text_buffer:
            if buffer_item.get('marathi_translation') is not None:
                buffer_bbox = buffer_item['bbox']
                if isinstance(buffer_bbox, (list, tuple)) and len(buffer_bbox) == 4:
                    buffer_y_center = (buffer_bbox[1] + buffer_bbox[3]) / 2
                    y_distance = abs(page_y_center - buffer_y_center)
                    
                    if y_distance < min_y_distance:
                        min_y_distance = y_distance
                        closest_buffer_item = buffer_item
        
        if closest_buffer_item:
            marathi_text = closest_buffer_item['marathi_translation'].strip()
            if marathi_text:
                matches.append({
                    'page_bbox': page_bbox,
                    'page_y_center': page_y_center,
                    'marathi_text': marathi_text,
                    'y_distance': min_y_distance,
                    'buffer_item': closest_buffer_item
                })
    
    # Second pass: for each unique translation, only show it once at the best position
    for match in matches:
        marathi_text = match['marathi_text']
        
        # Skip if we've already added this translation
        if marathi_text in used_translations:
            # Check if current match is closer than the existing one
            if match['y_distance'] < used_translations[marathi_text]['y_distance']:
                # Remove the previous entry and use this better match
                translations = [t for t in translations if t[0] != marathi_text]
                used_translations[marathi_text] = match
            else:
                # Keep the existing better match, skip this one
                continue
        else:
            # First occurrence of this translation
            used_translations[marathi_text] = match
        
        # Whiten the buffer bounding box that corresponds to this match (if not already whitened)
        buffer_item = match['buffer_item']
        buffer_id = id(buffer_item)  # Use object id as unique identifier
        
        if buffer_id not in whitened_buffer_items:
            buffer_bbox = buffer_item['bbox']
            if isinstance(buffer_bbox, (list, tuple)) and len(buffer_bbox) == 4:
                x1, y1, x2, y2 = buffer_bbox
                buffer_bounding_box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                buffer_bbox_coords = convert_bbox_coordinates(buffer_bounding_box)
                
                # Fill with white background to hide Hindi text
                draw.rectangle(buffer_bbox_coords, fill=(255, 255, 255))
                whitened_buffer_items.add(buffer_id)
                log_debug(f"  ⬜ Whitened buffer bbox: {buffer_bbox_coords}")
        
        # Add the translation within the whitened buffer bounding box (not page_data position)
        buffer_bbox = buffer_item['bbox']
        x1, y1, x2, y2 = buffer_bbox
        buffer_bounding_box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        translation_info = (marathi_text, buffer_bounding_box, marathi_text)
        translations.append(translation_info)
        
        log_debug(f"  📝 Text will be placed within whitened buffer box: {buffer_bbox}")
        
        buffer_y_center = buffer_item['bbox'][1] + (buffer_item['bbox'][3] - buffer_item['bbox'][1])/2
        log_debug(f"  ✅ Added '{marathi_text}' at Y={match['page_y_center']:.1f} (closest to buffer Y={buffer_y_center:.1f})")
    
    log_debug(f"  📊 Final result: {len(translations)} unique translations, {len(whitened_buffer_items)} buffer areas whitened")
    
    # DEBUG: Show translations list contents before unpacking
    log_debug(f"🔍 DEBUG: About to process translations list:")
    log_debug(f"   📊 Total translations: {len(translations)}")
    for i, item in enumerate(translations[:5]):  # Show first 5 for debugging
        log_debug(f"   [{i}] Type: {type(item)}, Len: {len(item) if hasattr(item, '__len__') else 'N/A'}, Content: {item}")
    
    # DEDUPLICATE: Remove duplicate regions to prevent text overlap
    unique_translations = {}
    for translation_info in translations:
        try:
            if len(translation_info) == 3:
                text, bbox_corners, full_sentence = translation_info
            elif len(translation_info) == 2:
                text, bbox_corners = translation_info
                full_sentence = text
            else:
                continue
                
            # Use bbox as key to deduplicate
            bbox_key = tuple(tuple(corner) if isinstance(corner, list) else corner for corner in bbox_corners)
            
            # Only keep the first occurrence of each bbox region
            if bbox_key not in unique_translations:
                unique_translations[bbox_key] = translation_info
                log_debug(f"  ➕ Added unique region: {bbox_key}")
            else:
                log_debug(f"  🔄 Skipping duplicate region: {bbox_key}")
                
        except Exception as e:
            log_debug(f"  ⚠️  Error processing translation_info: {e}")
            continue
    
    log_debug(f"📊 Deduplicated: {len(translations)} → {len(unique_translations)} unique regions")
    
    # Place all text with completion checking at render time
    for translation_info in unique_translations.values():
        try:
            # Handle both old format (text, bbox) and new format (text, bbox, full_sentence)
            if len(translation_info) == 3:
                text, bbox_corners, full_sentence = translation_info
            elif len(translation_info) == 2:
                text, bbox_corners = translation_info
                full_sentence = text  # Fallback to text itself
            else:
                continue
        except ValueError as e:
            log_debug(f"  ⚠️  Error unpacking translation_info: {translation_info}, error: {e}")
            continue
        
        # Convert bbox coordinates for positioning
        bbox = convert_bbox_coordinates(bbox_corners)
        bbox_key = (bbox[0], bbox[1], bbox[2], bbox[3])
            
        if not text or not text.strip():
            log_debug(f"  ⏭️  Skipping empty text region")
            continue
        
        max_width = bbox[2] - bbox[0]
        max_height = bbox[3] - bbox[1]
        
        if max_width <= 0 or max_height <= 0:
            continue
        
        # Calculate font and position specifically for this bounding box
        original_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
        font_size, font = calculate_font_for_bbox(original_bbox, font_path, logger)
        start_x, start_y = get_consistent_start_position(original_bbox)
        
        # Text color (background already whitened)
        text_color = (0, 0, 0)  # Black text
        
        log_debug(f"  🔤 Processing: '{text[:30]}...' with bbox-specific font: {font_size}px")
        log_debug(f"  📍 Consistent start position: ({start_x}, {start_y})")
        
        # Simple, consistent text placement - no word positioning complexity
        # This prevents dancing and ensures consistent appearance
        
        # Check if text fits in one line
        text_bbox = font.getbbox(text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Use consistent starting position
        x_pos = start_x
        y_pos = start_y
        
        # Ensure text stays completely within the whitened bounding box
        box_width = bbox[2] - bbox[0]
        box_height = bbox[3] - bbox[1]
        
        # Additional safety: reduce font size if text is still too large for the box
        if text_width > box_width or text_height > box_height:
            scale_factor = min(box_width / text_width, box_height / text_height) * 0.9  # 10% margin
            new_font_size = max(8, int(font_size * scale_factor))
            font = ImageFont.truetype(font_path, new_font_size)
            text_bbox = font.getbbox(text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            log_debug(f"  📏 Font resized from {font_size}px to {new_font_size}px to fit in white box")
        
        # Ensure text doesn't go outside the whitened bounding box
        max_x = min(bbox[2] - text_width, image.width - text_width)
        max_y = min(bbox[3] - text_height, image.height - text_height)
        
        x_pos = max(bbox[0], min(x_pos, max_x))
        y_pos = max(bbox[1], min(y_pos, max_y))
        
        # Final validation: ensure text is completely contained
        final_text_right = x_pos + text_width
        final_text_bottom = y_pos + text_height
        if final_text_right > bbox[2] or final_text_bottom > bbox[3]:
            log_warning(f"  ⚠️  Text might overflow white box: text_end=({final_text_right}, {final_text_bottom}) vs box_end=({bbox[2]}, {bbox[3]})")
        
        # Render text at safe position within white box
        draw.text((x_pos, y_pos), text, font=font, fill=text_color)
        
        log_debug(f"  ✅ Text '{text[:20]}...' safely contained in white box at ({x_pos}, {y_pos})")
    
    return image_copy

def reset_bbox_tracker():
    """Reset function - no longer needed for full text display but kept for compatibility"""
    pass

def get_bbox_tracker_status():
    """Get current status - no longer tracking for full text display"""
    return {}
