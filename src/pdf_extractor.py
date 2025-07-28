import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any, Tuple, Optional
import re
import numpy as np
from collections import Counter, defaultdict
from functools import lru_cache

class PDFExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._font_cache = {}
        self._block_cache = {}
    
    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text with comprehensive formatting information from PDF"""
        try:
            doc = fitz.open(pdf_path)
            structured_text = []
            page_layouts = {}
            
            # First pass: analyze page layouts
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_layouts[page_num] = self._analyze_page_layout(page)
            
            # Second pass: extract text with enhanced context
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_layout = page_layouts[page_num]
                
                # Get text with detailed formatting
                blocks = page.get_text("dict")
                page_text_elements = []
                
                for block in blocks["blocks"]:
                    if "lines" not in block:  # Skip image blocks
                        continue
                    
                    block_elements = self._process_text_block(
                        block, page_num + 1, page_layout, page.rect
                    )
                    page_text_elements.extend(block_elements)
                
                # Post-process page elements
                processed_elements = self._post_process_page_elements(
                    page_text_elements, page_layout
                )
                structured_text.extend(processed_elements)
            
            doc.close()
            
            # Final processing: reconstruct logical text blocks
            structured_text = self._reconstruct_logical_blocks(structured_text)
            
            return structured_text
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF: {e}")
            return []
    
    def _analyze_page_layout(self, page) -> Dict:
        """Analyze page layout characteristics"""
        page_rect = page.rect
        blocks = page.get_text("dict")
        
        # Collect all text elements for analysis
        all_elements = []
        for block in blocks["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        all_elements.append({
                            "bbox": span["bbox"],
                            "font_size": span["size"],
                            "text": span["text"],
                            "font_name": span["font"],
                            "flags": span["flags"]
                        })
        
        if not all_elements:
            return {"margins": {}, "columns": 1, "text_density": 0, "dominant_fonts": []}
        
        # Analyze margins
        x_positions = [elem["bbox"][0] for elem in all_elements]
        y_positions = [elem["bbox"][1] for elem in all_elements]
        
        margins = {
            "left": min(x_positions) if x_positions else 0,
            "right": page_rect.width - max([elem["bbox"][2] for elem in all_elements]),
            "top": min(y_positions) if y_positions else 0,
            "bottom": page_rect.height - max([elem["bbox"][3] for elem in all_elements])
        }
        
        # Detect column layout
        columns = self._detect_columns(all_elements, page_rect.width)
        
        # Calculate text density
        total_text_area = sum(
            (elem["bbox"][2] - elem["bbox"][0]) * (elem["bbox"][3] - elem["bbox"][1])
            for elem in all_elements
        )
        page_area = page_rect.width * page_rect.height
        text_density = total_text_area / page_area if page_area > 0 else 0
        
        # Find dominant fonts
        font_counter = Counter()
        for elem in all_elements:
            font_key = (elem["font_name"], round(elem["font_size"], 1))
            font_counter[font_key] += 1
        
        dominant_fonts = font_counter.most_common(5)
        
        return {
            "margins": margins,
            "columns": columns,
            "text_density": text_density,
            "dominant_fonts": dominant_fonts,
            "page_width": page_rect.width,
            "page_height": page_rect.height,
            "total_elements": len(all_elements)
        }
    
    def _detect_columns(self, elements: List[Dict], page_width: float) -> int:
        """Detect number of columns in page layout"""
        if not elements:
            return 1
        
        # Group elements by approximate X position
        x_positions = [elem["bbox"][0] for elem in elements]
        x_clusters = self._cluster_positions(x_positions, tolerance=page_width * 0.1)
        
        # Simple heuristic: number of major x-position clusters
        return min(len(x_clusters), 3)  # Cap at 3 columns
    
    def _cluster_positions(self, positions: List[float], tolerance: float) -> List[List[float]]:
        """Cluster positions within tolerance"""
        if not positions:
            return []
        
        sorted_positions = sorted(positions)
        clusters = [[sorted_positions[0]]]
        
        for pos in sorted_positions[1:]:
            if pos - clusters[-1][-1] <= tolerance:
                clusters[-1].append(pos)
            else:
                clusters.append([pos])
        
        # Only return clusters with significant number of elements
        significant_clusters = [cluster for cluster in clusters if len(cluster) >= 3]
        return significant_clusters if significant_clusters else clusters
    
    def _process_text_block(self, block: Dict, page_num: int, 
                           page_layout: Dict, page_rect) -> List[Dict]:
        """Process a text block with enhanced analysis"""
        elements = []
        
        for line_idx, line in enumerate(block["lines"]):
            line_elements = []
            line_bbox = None
            
            # First, collect all spans in the line
            for span_idx, span in enumerate(line["spans"]):
                text = span["text"].strip()
                
                if len(text) >= 1:  # Keep even single characters for analysis
                    bbox = span["bbox"]
                    
                    # Update line bounding box
                    if line_bbox is None:
                        line_bbox = list(bbox)
                    else:
                        line_bbox[0] = min(line_bbox[0], bbox[0])  # left
                        line_bbox[1] = min(line_bbox[1], bbox[1])  # top
                        line_bbox[2] = max(line_bbox[2], bbox[2])  # right
                        line_bbox[3] = max(line_bbox[3], bbox[3])  # bottom
                    
                    element = self._create_text_element(
                        span, text, page_num, page_layout, page_rect, line_idx, span_idx
                    )
                    line_elements.append(element)
            
            # Post-process line elements
            if line_elements and line_bbox:
                processed_line_elements = self._post_process_line_elements(
                    line_elements, line_bbox, page_layout
                )
                elements.extend(processed_line_elements)
        
        return elements
    
    def _create_text_element(self, span: Dict, text: str, page_num: int,
                           page_layout: Dict, page_rect, line_idx: int, span_idx: int) -> Dict:
        """Create enhanced text element with comprehensive metadata"""
        bbox = span["bbox"]
        font_size = span["size"]
        font_flags = span["flags"]
        font_name = span["font"]
        
        # Calculate formatting properties
        is_bold = bool(font_flags & 2**4)
        is_italic = bool(font_flags & 2**1)
        is_superscript = bool(font_flags & 2**0)
        
        # Position and size analysis
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        
        # Calculate relative positions
        page_width = page_rect.width
        page_height = page_rect.height
        
        relative_x = x / page_width if page_width > 0 else 0
        relative_y = y / page_height if page_height > 0 else 0
        relative_width = width / page_width if page_width > 0 else 0
        
        # Calculate alignment metrics
        left_margin = x - page_layout["margins"]["left"]
        right_margin = page_layout["margins"]["right"] - x2
        
        # Center alignment calculation
        page_center = page_width / 2
        element_center = (x + x2) / 2
        center_deviation = abs(element_center - page_center) / page_width if page_width > 0 else 1
        
        # Isolation analysis
        is_isolated = self._check_text_isolation_enhanced(bbox, page_layout, font_size)
        
        # Font characteristics
        char_density = len(text) / width if width > 0 else 0
        
        # Reading order hint
        reading_order = line_idx * 1000 + span_idx
        
        return {
            "text": text,
            "font_size": font_size,
            "font_flags": font_flags,
            "font_name": font_name,
            "page": page_num,
            "bbox": bbox,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "is_superscript": is_superscript,
            "width": width,
            "height": height,
            "x": x,
            "y": y,
            "relative_x": relative_x,
            "relative_y": relative_y,
            "relative_width": relative_width,
            "char_density": char_density,
            "is_isolated": is_isolated,
            "left_margin": left_margin,
            "right_margin": right_margin,
            "center_deviation": center_deviation,
            "reading_order": reading_order,
            "line_height": height,
            "estimated_column": self._estimate_column(x, page_layout),
        }
    
    def _check_text_isolation_enhanced(self, bbox: Tuple, page_layout: Dict, font_size: float) -> bool:
        """Enhanced isolation checking with font size consideration"""
        # Larger elements need more space to be considered isolated
        threshold = max(15, font_size * 1.5)
        
        # This is a simplified version - in practice, you'd check against other elements
        # For now, we'll use layout-based heuristics
        
        margins = page_layout["margins"]
        x, y, x2, y2 = bbox
        
        # Check if element is close to page edges (might be isolated)
        close_to_left = x - margins["left"] < threshold
        close_to_right = margins["right"] - x2 < threshold
        close_to_top = y - margins["top"] < threshold * 2  # Headers often at top
        
        # Simple isolation heuristic based on positioning
        if close_to_top and (close_to_left or close_to_right):
            return True
        
        # Check if font size is significantly larger (likely heading)
        dominant_fonts = page_layout.get("dominant_fonts", [])
        if dominant_fonts:
            common_size = dominant_fonts[0][0][1]  # Most common font size
            if font_size > common_size * 1.3:
                return True
        
        return False
    
    def _estimate_column(self, x_position: float, page_layout: Dict) -> int:
        """Estimate which column the text element belongs to"""
        page_width = page_layout.get("page_width", 600)
        num_columns = page_layout.get("columns", 1)
        
        if num_columns == 1:
            return 1
        
        column_width = page_width / num_columns
        estimated_column = int(x_position / column_width) + 1
        
        return min(estimated_column, num_columns)
    
    def _post_process_line_elements(self, line_elements: List[Dict], 
                                   line_bbox: List[float], page_layout: Dict) -> List[Dict]:
        """Post-process elements within a line"""
        if not line_elements:
            return []
        
        # Merge consecutive elements with same formatting if they're close
        merged_elements = []
        current_element = line_elements[0].copy()
        
        for next_element in line_elements[1:]:
            # Check if elements should be merged
            if self._should_merge_elements(current_element, next_element):
                current_element = self._merge_text_elements(current_element, next_element)
            else:
                merged_elements.append(current_element)
                current_element = next_element.copy()
        
        merged_elements.append(current_element)
        
        # Add line-level context
        for element in merged_elements:
            element["line_bbox"] = line_bbox
            element["elements_in_line"] = len(merged_elements)
            element["is_line_start"] = element == merged_elements[0]
            element["is_line_end"] = element == merged_elements[-1]
        
        return merged_elements
    
    def _should_merge_elements(self, elem1: Dict, elem2: Dict) -> bool:
        """Determine if two text elements should be merged"""
        # Same formatting characteristics
        same_font = (elem1["font_name"] == elem2["font_name"] and
                    abs(elem1["font_size"] - elem2["font_size"]) < 0.5 and
                    elem1["is_bold"] == elem2["is_bold"] and
                    elem1["is_italic"] == elem2["is_italic"])
        
        if not same_font:
            return False
        
        # Close proximity (horizontal gap less than average character width)
        gap = elem2["x"] - (elem1["x"] + elem1["width"])
        avg_char_width = elem1["width"] / len(elem1["text"]) if elem1["text"] else 5
        close_proximity = gap < avg_char_width * 2
        
        # Same vertical level
        same_level = abs(elem1["y"] - elem2["y"]) < 2
        
        # Neither element is too long (avoid merging full sentences)
        reasonable_length = len(elem1["text"]) < 50 and len(elem2["text"]) < 50
        
        return same_font and close_proximity and same_level and reasonable_length
    
    def _merge_text_elements(self, elem1: Dict, elem2: Dict) -> Dict:
        """Merge two text elements into one"""
        merged = elem1.copy()
        
        # Combine text with appropriate spacing
        gap = elem2["x"] - (elem1["x"] + elem1["width"])
        avg_char_width = elem1["width"] / len(elem1["text"]) if elem1["text"] else 5
        
        if gap > avg_char_width * 0.8:
            merged["text"] = elem1["text"] + " " + elem2["text"]
        else:
            merged["text"] = elem1["text"] + elem2["text"]
        
        # Update bounding box
        merged["bbox"] = (
            min(elem1["bbox"][0], elem2["bbox"][0]),  # left
            min(elem1["bbox"][1], elem2["bbox"][1]),  # top
            max(elem1["bbox"][2], elem2["bbox"][2]),  # right
            max(elem1["bbox"][3], elem2["bbox"][3])   # bottom
        )
        
        # Update dimensions
        merged["width"] = merged["bbox"][2] - merged["bbox"][0]
        merged["height"] = merged["bbox"][3] - merged["bbox"][1]
        
        # Update character density
        merged["char_density"] = len(merged["text"]) / merged["width"] if merged["width"] > 0 else 0
        
        return merged
    
    def _post_process_page_elements(self, page_elements: List[Dict], 
                                   page_layout: Dict) -> List[Dict]:
        """Post-process all elements on a page"""
        if not page_elements:
            return []
        
        # Sort elements by reading order (top to bottom, left to right)
        sorted_elements = sorted(page_elements, key=lambda x: (x["y"], x["x"]))
        
        # Add sequential context
        for i, element in enumerate(sorted_elements):
            element["page_element_index"] = i
            element["is_page_start"] = i == 0
            element["is_page_end"] = i == len(sorted_elements) - 1
            
            # Add context about surrounding elements
            if i > 0:
                prev_element = sorted_elements[i - 1]
                element["vertical_gap_above"] = element["y"] - (prev_element["y"] + prev_element["height"])
            else:
                element["vertical_gap_above"] = 0
            
            if i < len(sorted_elements) - 1:
                next_element = sorted_elements[i + 1]
                element["vertical_gap_below"] = next_element["y"] - (element["y"] + element["height"])
            else:
                element["vertical_gap_below"] = 0
        
        return sorted_elements
    
    def _reconstruct_logical_blocks(self, structured_text: List[Dict]) -> List[Dict]:
        """Reconstruct logical text blocks from individual elements"""
        if not structured_text:
            return []
        
        # Group elements by page
        pages = defaultdict(list)
        for element in structured_text:
            pages[element["page"]].append(element)
        
        reconstructed = []
        
        for page_num, page_elements in pages.items():
            # Sort elements by reading order
            sorted_elements = sorted(page_elements, key=lambda x: (x["y"], x["x"]))
            
            # Group into logical blocks
            logical_blocks = self._group_into_logical_blocks(sorted_elements)
            
            # Process each logical block
            for block in logical_blocks:
                if len(block) == 1:
                    # Single element block
                    reconstructed.append(block[0])
                else:
                    # Multi-element block - create merged representation
                    merged_block = self._merge_logical_block(block)
                    reconstructed.append(merged_block)
        
        return reconstructed
    
    def _group_into_logical_blocks(self, elements: List[Dict]) -> List[List[Dict]]:
        """Group elements into logical blocks based on proximity and formatting"""
        if not elements:
            return []
        
        blocks = []
        current_block = [elements[0]]
        
        for i in range(1, len(elements)):
            current_elem = elements[i]
            prev_elem = current_block[-1]
            
            # Check if elements should be in the same block
            should_group = self._should_group_into_block(prev_elem, current_elem)
            
            if should_group:
                current_block.append(current_elem)
            else:
                blocks.append(current_block)
                current_block = [current_elem]
        
        blocks.append(current_block)
        return blocks
    
    def _should_group_into_block(self, elem1: Dict, elem2: Dict) -> bool:
        """Determine if two elements should be grouped into the same logical block"""
        # Vertical proximity
        vertical_gap = elem2["y"] - (elem1["y"] + elem1["height"])
        line_height = max(elem1["height"], elem2["height"])
        
        # Elements are close vertically (within 1.5 line heights)
        close_vertically = vertical_gap < line_height * 1.5
        
        # Similar horizontal positioning (same column or close alignment)
        horizontal_alignment = abs(elem1["x"] - elem2["x"]) < 20
        
        # Similar formatting (but not required for grouping)
        similar_formatting = (
            abs(elem1["font_size"] - elem2["font_size"]) < 2 and
            elem1["font_name"] == elem2["font_name"]
        )
        
        # Don't group if there's a significant formatting difference suggesting hierarchy
        significant_size_diff = abs(elem1["font_size"] - elem2["font_size"]) > 3
        different_bold_status = elem1["is_bold"] != elem2["is_bold"]
        
        if significant_size_diff or different_bold_status:
            return False
        
        # Group if elements are close and reasonably aligned
        return close_vertically and (horizontal_alignment or similar_formatting)
    
    def _merge_logical_block(self, block: List[Dict]) -> Dict:
        """Merge multiple elements into a single logical block"""
        if not block:
            return {}
        
        if len(block) == 1:
            return block[0]
        
        # Use the first element as base
        merged = block[0].copy()
        
        # Combine text from all elements
        texts = []
        for elem in block:
            text = elem["text"].strip()
            if text:
                texts.append(text)
        
        # Join with spaces, but be smart about punctuation
        combined_text = ""
        for i, text in enumerate(texts):
            if i == 0:
                combined_text = text
            else:
                # Add space unless previous text ends with hyphen or current starts with punctuation
                prev_text = texts[i - 1]
                if prev_text.endswith('-') or text.startswith(('.', ',', ':', ';', '!', '?')):
                    combined_text += text
                else:
                    combined_text += " " + text
        
        merged["text"] = combined_text
        
        # Update bounding box to encompass all elements
        all_bboxes = [elem["bbox"] for elem in block]
        merged["bbox"] = (
            min(bbox[0] for bbox in all_bboxes),  # left
            min(bbox[1] for bbox in all_bboxes),  # top
            max(bbox[2] for bbox in all_bboxes),  # right
            max(bbox[3] for bbox in all_bboxes)   # bottom
        )
        
        # Update derived properties
        merged["width"] = merged["bbox"][2] - merged["bbox"][0]
        merged["height"] = merged["bbox"][3] - merged["bbox"][1]
        merged["char_density"] = len(merged["text"]) / merged["width"] if merged["width"] > 0 else 0
        
        # Use dominant formatting properties
        font_sizes = [elem["font_size"] for elem in block]
        merged["font_size"] = max(font_sizes)  # Use largest font size in block
        
        # Bold if any element is bold
        merged["is_bold"] = any(elem["is_bold"] for elem in block)
        
        # Use most common font name
        font_names = [elem["font_name"] for elem in block]
        merged["font_name"] = Counter(font_names).most_common(1)[0][0]
        
        # Mark as merged block
        merged["is_merged_block"] = True
        merged["original_element_count"] = len(block)
        
        return merged
    
    def get_font_statistics(self, structured_text: List[Dict]) -> Dict:
        """Enhanced font statistics analysis with better insights"""
        if not structured_text:
            return {
                "avg_font_size": 12,
                "max_font_size": 12,
                "min_font_size": 12,
                "font_size_std": 0,
                "common_fonts": [],
                "font_size_distribution": {},
                "bold_ratio": 0,
                "font_diversity": 0,
                "font_hierarchy": {}
            }
        
        # Collect font information
        font_sizes = [item["font_size"] for item in structured_text if item.get("font_size", 0) > 0]
        font_names = [item["font_name"] for item in structured_text if item.get("font_name")]
        bold_count = sum(1 for item in structured_text if item.get("is_bold", False))
        italic_count = sum(1 for item in structured_text if item.get("is_italic", False))
        
        if not font_sizes:
            return {
                "avg_font_size": 12,
                "max_font_size": 12,
                "min_font_size": 12,
                "font_size_std": 0,
                "common_fonts": [],
                "font_size_distribution": {},
                "bold_ratio": 0,
                "font_diversity": 0,
                "font_hierarchy": {}
            }
        
        # Calculate basic statistics
        avg_font_size = np.mean(font_sizes)
        max_font_size = max(font_sizes)
        min_font_size = min(font_sizes)
        font_size_std = np.std(font_sizes) if len(font_sizes) > 1 else 0
        
        # Font size distribution
        font_size_counter = Counter([round(size, 1) for size in font_sizes])
        
        # Font family analysis
        font_name_counter = Counter(font_names)
        
        # Calculate ratios
        total_elements = len(structured_text)
        bold_ratio = bold_count / total_elements if total_elements > 0 else 0
        italic_ratio = italic_count / total_elements if total_elements > 0 else 0
        font_diversity = len(set(font_names)) / len(font_names) if font_names else 0
        
        # Build font hierarchy
        font_hierarchy = self._build_font_hierarchy_from_stats(
            font_size_counter, bold_count, total_elements
        )
        
        return {
            "avg_font_size": avg_font_size,
            "max_font_size": max_font_size,
            "min_font_size": min_font_size,
            "font_size_std": font_size_std,
            "common_fonts": font_name_counter.most_common(5),
            "font_size_distribution": dict(font_size_counter.most_common(10)),
            "bold_ratio": bold_ratio,
            "italic_ratio": italic_ratio,
            "font_diversity": font_diversity,
            "total_unique_sizes": len(set(font_sizes)),
            "font_hierarchy": font_hierarchy,
            "size_range": max_font_size - min_font_size
        }
    
    def _build_font_hierarchy_from_stats(self, font_size_counter: Counter, 
                                       bold_count: int, total_elements: int) -> Dict:
        """Build font hierarchy information from statistics"""
        # Sort font sizes by frequency (ascending) and size (descending)
        # Less frequent, larger fonts are more likely to be headings
        sorted_sizes = sorted(
            font_size_counter.items(),
            key=lambda x: (x[1], -x[0])  # Sort by frequency (asc), then size (desc)
        )
        
        hierarchy = {}
        
        if len(sorted_sizes) >= 3:
            # Assign hierarchy levels based on size and frequency
            largest_sizes = sorted(font_size_counter.keys(), reverse=True)
            
            if len(largest_sizes) >= 1:
                hierarchy["H1"] = {
                    "font_size": largest_sizes[0],
                    "frequency": font_size_counter[largest_sizes[0]],
                    "relative_frequency": font_size_counter[largest_sizes[0]] / total_elements
                }
            
            if len(largest_sizes) >= 2:
                hierarchy["H2"] = {
                    "font_size": largest_sizes[1],
                    "frequency": font_size_counter[largest_sizes[1]],
                    "relative_frequency": font_size_counter[largest_sizes[1]] / total_elements
                }
            
            if len(largest_sizes) >= 3:
                hierarchy["H3"] = {
                    "font_size": largest_sizes[2],
                    "frequency": font_size_counter[largest_sizes[2]],
                    "relative_frequency": font_size_counter[largest_sizes[2]] / total_elements
                }
            
            # Body text is the most frequent size
            most_common_size = font_size_counter.most_common(1)[0]
            hierarchy["body"] = {
                "font_size": most_common_size[0],
                "frequency": most_common_size[1],
                "relative_frequency": most_common_size[1] / total_elements
            }
        
        return hierarchy
    
    def analyze_document_structure(self, structured_text: List[Dict]) -> Dict:
        """Enhanced document structure analysis"""
        if not structured_text:
            return {"total_pages": 0, "total_items": 0}
        
        # Group by pages
        pages = defaultdict(list)
        for item in structured_text:
            pages[item["page"]].append(item)
        
        # Analyze each page
        page_analysis = {}
        overall_stats = {
            "total_pages": len(pages),
            "total_items": len(structured_text),
            "avg_items_per_page": len(structured_text) / len(pages) if pages else 0
        }
        
        for page_num, page_items in pages.items():
            font_sizes = [item["font_size"] for item in page_items if item.get("font_size")]
            
            # Spatial analysis
            y_positions = [item["y"] for item in page_items if "y" in item]
            x_positions = [item["x"] for item in page_items if "x" in item]
            
            # Text characteristics
            total_text_length = sum(len(item.get("text", "")) for item in page_items)
            avg_text_length = total_text_length / len(page_items) if page_items else 0
            
            # Formatting analysis
            bold_items = [item for item in page_items if item.get("is_bold", False)]
            large_items = [item for item in page_items 
                          if item.get("font_size", 12) > np.mean(font_sizes) * 1.2] if font_sizes else []
            
            # Isolation analysis
            isolated_items = [item for item in page_items if item.get("is_isolated", False)]
            
            # Layout analysis
            if y_positions:
                top_threshold = min(y_positions) + 100
                top_items = [item for item in page_items if item.get("y", 0) <= top_threshold]
            else:
                top_items = []
            
            page_analysis[page_num] = {
                "item_count": len(page_items),
                "avg_font_size": np.mean(font_sizes) if font_sizes else 12,
                "max_font_size": max(font_sizes) if font_sizes else 12,
                "min_font_size": min(font_sizes) if font_sizes else 12,
                "font_size_range": max(font_sizes) - min(font_sizes) if font_sizes else 0,
                "has_large_text": len(large_items) > 0,
                "has_bold_text": len(bold_items) > 0,
                "bold_ratio": len(bold_items) / len(page_items) if page_items else 0,
                "avg_text_length": avg_text_length,
                "top_elements": len(top_items),
                "isolated_elements": len(isolated_items),
                "potential_headings": len([item for item in page_items 
                                         if self._is_potential_heading_element(item, font_sizes)]),
                "layout_complexity": self._calculate_layout_complexity(page_items)
            }
        
        overall_stats["page_analysis"] = page_analysis
        
        # Overall document characteristics
        all_font_sizes = [item["font_size"] for item in structured_text if item.get("font_size")]
        overall_stats.update({
            "document_font_diversity": len(set(item.get("font_name", "") for item in structured_text)),
            "document_avg_font_size": np.mean(all_font_sizes) if all_font_sizes else 12,
            "document_has_hierarchy": len(set(all_font_sizes)) > 2 if all_font_sizes else False,
            "document_bold_ratio": sum(1 for item in structured_text if item.get("is_bold", False)) / len(structured_text),
            "document_complexity_score": self._calculate_document_complexity(page_analysis)
        })
        
        return overall_stats
    
    def _is_potential_heading_element(self, item: Dict, page_font_sizes: List[float]) -> bool:
        """Quick assessment if an element could be a heading"""
        if not page_font_sizes:
            return False
        
        text = item.get("text", "").strip()
        if len(text) < 3 or len(text) > 150:
            return False
        
        font_size = item.get("font_size", 12)
        avg_size = np.mean(page_font_sizes)
        
        # Larger than average font
        size_boost = font_size > avg_size * 1.1
        
        # Bold formatting
        bold_boost = item.get("is_bold", False)
        
        # Isolation
        isolation_boost = item.get("is_isolated", False)
        
        # Pattern indicators
        pattern_boost = bool(re.match(r'^\d+\.?\s+[A-Z]', text))
        
        return size_boost or bold_boost or isolation_boost or pattern_boost
    
    def _calculate_layout_complexity(self, page_items: List[Dict]) -> float:
        """Calculate layout complexity score for a page"""
        if not page_items:
            return 0.0
        
        complexity_factors = []
        
        # Font diversity
        unique_fonts = len(set(item.get("font_name", "") for item in page_items))
        font_diversity = min(unique_fonts / 5, 1.0)  # Normalize to 0-1
        complexity_factors.append(font_diversity)
        
        # Size diversity
        font_sizes = [item.get("font_size", 12) for item in page_items]
        unique_sizes = len(set(font_sizes))
        size_diversity = min(unique_sizes / 6, 1.0)  # Normalize to 0-1
        complexity_factors.append(size_diversity)
        
        # Formatting diversity
        formatting_types = set()
        for item in page_items:
            formatting = (item.get("is_bold", False), item.get("is_italic", False))
            formatting_types.add(formatting)
        format_diversity = min(len(formatting_types) / 4, 1.0)
        complexity_factors.append(format_diversity)
        
        # Spatial distribution
        if len(page_items) > 1:
            y_positions = [item.get("y", 0) for item in page_items]
            y_std = np.std(y_positions) if y_positions else 0
            spatial_complexity = min(y_std / 200, 1.0)  # Normalize
            complexity_factors.append(spatial_complexity)
        
        return np.mean(complexity_factors)
    
    def _calculate_document_complexity(self, page_analysis: Dict) -> float:
        """Calculate overall document complexity score"""
        if not page_analysis:
            return 0.0
        
        page_complexities = [analysis.get("layout_complexity", 0) 
                           for analysis in page_analysis.values()]
        
        return np.mean(page_complexities) if page_complexities else 0.0
    
    @lru_cache(maxsize=100)
    def get_heading_candidates_by_position(self, pdf_path: str) -> List[Dict]:
        """Get heading candidates based on spatial positioning with caching"""
        structured_text = self.extract_text_with_formatting(pdf_path)
        return self._get_position_based_candidates(structured_text)
    
    def _get_position_based_candidates(self, structured_text: List[Dict]) -> List[Dict]:
        """Get heading candidates based on spatial positioning"""
        candidates = []
        
        # Group by page for position analysis
        pages = defaultdict(list)
        for item in structured_text:
            pages[item["page"]].append(item)
        
        for page_num, page_items in pages.items():
            # Sort by Y position (top to bottom)
            page_items.sort(key=lambda x: x.get("y", 0))
            
            # Identify potential headings by spatial characteristics
            for i, item in enumerate(page_items):
                position_score = self._calculate_enhanced_position_score(item, page_items, i)
                
                if position_score > 0.3:  # Threshold for position-based candidacy
                    candidate = item.copy()
                    candidate["position_score"] = position_score
                    candidates.append(candidate)
        
        return candidates
    
    def _calculate_enhanced_position_score(self, item: Dict, page_items: List[Dict], 
                                         item_index: int) -> float:
        """Calculate enhanced position score for heading likelihood"""
        score = 0.0
        
        # Isolation analysis
        if item.get("is_isolated", False):
            score += 0.4
        
        # Vertical spacing analysis
        vertical_gap_above = item.get("vertical_gap_above", 0)
        vertical_gap_below = item.get("vertical_gap_below", 0)
        
        # Large gaps suggest section breaks
        if vertical_gap_above > 20:
            score += 0.3
        elif vertical_gap_above > 10:
            score += 0.2
        
        if vertical_gap_below > 15:
            score += 0.2
        
        # Position on page
        page_position = item_index / len(page_items) if page_items else 0
        
        # Top of page bonus
        if page_position < 0.1:
            score += 0.3
        elif page_position < 0.2:
            score += 0.2
        
        # Not at very bottom (footers)
        if page_position > 0.9:
            score -= 0.2
        
        # Center alignment
        center_deviation = item.get("center_deviation", 1.0)
        if center_deviation < 0.1:  # Very centered
            score += 0.3
        elif center_deviation < 0.2:
            score += 0.2
        
        # Left margin consistency (structured documents)
        left_margin = item.get("left_margin", 0)
        if 10 < left_margin < 100:  # Reasonable indentation
            score += 0.1
        
        # Font size relative to page
        if len(page_items) > 1:
            page_font_sizes = [elem.get("font_size", 12) for elem in page_items]
            avg_page_size = np.mean(page_font_sizes)
            item_size = item.get("font_size", 12)
            
            if item_size > avg_page_size * 1.3:
                score += 0.3
            elif item_size > avg_page_size * 1.1:
                score += 0.2
        
        return min(score, 1.0)
    
    def extract_document_metadata(self, pdf_path: str) -> Dict:
        """Extract document metadata and structure information"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            # Basic document info
            doc_info = {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "page_count": len(doc),
                "encrypted": doc.needs_pass,
            }
            
            # Analyze document structure
            structured_text = self.extract_text_with_formatting(pdf_path)
            structure_info = self.analyze_document_structure(structured_text)
            font_stats = self.get_font_statistics(structured_text)
            
            doc.close()
            
            return {
                "metadata": doc_info,
                "structure": structure_info,
                "fonts": font_stats,
                "extraction_quality": self._assess_extraction_quality(structured_text)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting document metadata: {e}")
            return {}
    
    def _assess_extraction_quality(self, structured_text: List[Dict]) -> Dict:
        """Assess the quality of text extraction"""
        if not structured_text:
            return {"quality_score": 0.0, "issues": ["No text extracted"]}
        
        total_chars = sum(len(item.get("text", "")) for item in structured_text)
        total_elements = len(structured_text)
        
        # Quality indicators
        avg_text_length = total_chars / total_elements if total_elements > 0 else 0
        
        # Check for extraction issues
        issues = []
        quality_score = 100.0
        
        # Very short average text length suggests fragmentation
        if avg_text_length < 5:
            issues.append("Text appears highly fragmented")
            quality_score -= 30
        
        # Check for encoding issues
        corrupted_count = sum(1 for item in structured_text 
                            if any(char in item.get("text", "") for char in ['�', '']))
        if corrupted_count > total_elements * 0.1:
            issues.append("Potential encoding/corruption issues detected")
            quality_score -= 20
        
        # Check font diversity (very low might indicate issues)
        unique_fonts = len(set(item.get("font_name", "") for item in structured_text))
        if unique_fonts < 2 and total_elements > 50:
            issues.append("Very low font diversity - possible extraction limitations")
            quality_score -= 10
        
        # Check for reasonable size distribution
        font_sizes = [item.get("font_size", 12) for item in structured_text]
        if font_sizes and len(set(font_sizes)) < 2:
            issues.append("No font size variation detected")
            quality_score -= 15
        
        return {
            "quality_score": max(quality_score, 0.0),
            "total_elements": total_elements,
            "total_characters": total_chars,
            "avg_text_length": avg_text_length,
            "unique_fonts": unique_fonts,
            "issues": issues
        }
    
    def clear_cache(self):
        """Clear internal caches to free memory"""
        self._font_cache.clear()
        self._block_cache.clear()
        # Clear LRU cache
        self.get_heading_candidates_by_position.cache_clear()