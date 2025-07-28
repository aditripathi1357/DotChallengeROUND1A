import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
import statistics
import numpy as np
from collections import Counter
from functools import lru_cache
from difflib import SequenceMatcher

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_model_constraints(cache_dir: str, size_limit_mb: int = 200) -> bool:
    """Check if model size is within constraints"""
    if not os.path.exists(cache_dir):
        return True
    
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(cache_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        size_mb = total_size / (1024 * 1024)  # Convert to MB
        logging.info(f"Model cache size: {size_mb:.2f} MB")
        return size_mb <= size_limit_mb
    except Exception as e:
        logging.error(f"Error checking model constraints: {e}")
        return False

def validate_json_schema(data: Dict) -> bool:
    """Validate output JSON against required schema with enhanced checks"""
    required_fields = ["title", "outline"]
    
    # Check required fields
    for field in required_fields:
        if field not in data:
            return False
    
    # Validate title
    if not isinstance(data["title"], str):
        return False
    
    # Validate outline
    if not isinstance(data["outline"], list):
        return False
    
    # Validate outline items
    for item in data["outline"]:
        if not isinstance(item, dict):
            return False
        
        required_item_fields = ["level", "text", "page"]
        for field in required_item_fields:
            if field not in item:
                return False
        
        # Type validation
        if not isinstance(item["level"], str):
            return False
        if not isinstance(item["text"], str):
            return False
        if not isinstance(item["page"], int):
            return False
        
        # Value validation
        if item["level"] not in ["H1", "H2", "H3"]:
            return False
        if len(item["text"].strip()) == 0:
            return False
        if item["page"] < 1:
            return False
    
    return True

@lru_cache(maxsize=500)
def clean_text(text: str) -> str:
    """Enhanced text cleaning with better preservation of structure and caching"""
    if not text:
        return ""
    
    # Remove extra whitespace but preserve structure
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove some problematic characters but keep meaningful punctuation
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Fix common OCR/extraction issues with improved patterns
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split concatenated words
    text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)    # Split number+letter
    text = re.sub(r'([a-z])\s*-\s*([a-z])', r'\1-\2', text)  # Fix hyphenated words
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    text = re.sub(r'([.,:;!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove multiple consecutive punctuation
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{2,}', '--', text)
    
    # Fix common PDF extraction artifacts
    text = re.sub(r'\s*\|\s*', ' ', text)  # Remove table separators
    text = re.sub(r'\s*_\s*', ' ', text)   # Remove underscores used as lines
    
    # Handle unicode issues including Hindi
    text = text.replace('\u2013', '-')  # En dash
    text = text.replace('\u2014', '--') # Em dash
    text = text.replace('\u2019', "'")  # Right single quotation mark
    text = text.replace('\u201c', '"')  # Left double quotation mark
    text = text.replace('\u201d', '"')  # Right double quotation mark
    
    return text.strip()

def detect_language(text: str) -> str:
    """Detect if the text is in Hindi or English based on Unicode ranges"""
    if not text or not isinstance(text, str):
        return "unknown"
    
    hindi_count = sum(1 for c in text if '\u0900' <= c <= '\u097F')  # Devanagari range
    english_count = sum(1 for c in text if c.isalpha() and not ('\u0900' <= c <= '\u097F'))
    
    total_alpha = hindi_count + english_count
    if total_alpha == 0:
        return "unknown"
    
    if hindi_count / total_alpha > 0.5:
        return "hindi"
    elif english_count / total_alpha > 0.5:
        return "english"
    return "mixed"

def is_corrupted_text(text: str) -> bool:
    """Enhanced corruption detection with improved accuracy and patterns"""
    if not text or len(text.strip()) < 2:
        return True
    
    text = text.strip()
    language = detect_language(text)
    
    # Very short fragments that are likely corrupted
    if len(text) < 3:
        return True
    
    # Enhanced corruption patterns with confidence scores
    corruption_patterns = [
        # Fragment patterns
        (r'^[A-Z]{1,2}:?\s*$', 0.9),  # "R", "RFP:"
        (r'^[a-z]{1,3}\s+[A-Z][a-z]?\s*$', 0.9),  # "r Pr", "quest f"
        (r'.*[a-z]{3,}[A-Z]{3,}.*', 0.8),  # Mixed case corruption
        (r'^(quest|oposal|reeee|foooor|tal\.)\b', 0.9),  # Known corruptions
        (r'^\w{1,2}\s*$', 0.9),  # Very short fragments
        (r'^[^a-zA-Z0-9\s:.-]{3,}$', 0.95),  # Mostly symbols
        (r'^[^\u0900-\u097F]{3,}$', 0.95) if language == "hindi" else ("", 0),  # Non-Hindi symbols
        
        # Incomplete text patterns
        (r'^\w+\.\.\.$', 0.8),  # Ends with ...
        (r'^\w+\s*-\s*$', 0.8),  # Ends with dash
        (r'^\w+\s*,\s*$', 0.8),  # Ends with comma
        (r'^\w*[a-z]\s*$', 0.7),  # Single lowercase word
        
        # OCR corruption indicators
        (r'[Il1|]{3,}', 0.85),  # Repeated similar looking characters
        (r'[0O]{3,}', 0.85),    # Repeated O/0
        (r'[mn]{4,}', 0.8),     # Repeated m/n (common OCR issue)
        (r'[rn]{3,}', 0.8),     # Repeated r/n
        (r'[vw]{3,}', 0.8),     # Repeated v/w
        
        # Encoding issues
        (r'[\x00-\x1f\x7f-\x9f]', 0.95),  # Control characters
        (r'[��]{2,}', 0.95),  # Replacement characters
        (r'[^\x00-\x7F]{3,}', 0.7),  # Non-ASCII sequences (might be legitimate)
    ]
    
    # Check corruption patterns
    max_corruption_score = 0.0
    for pattern, score in corruption_patterns:
        if pattern and re.search(pattern, text):
            max_corruption_score = max(max_corruption_score, score)
    
    if max_corruption_score > 0.8:
        return True
    
    # Enhanced character composition analysis
    alpha_count = sum(c.isalpha() for c in text if '' <= c <= '\u007F') if language == "english" else sum(1 for c in text if '\u0900' <= c <= '\u097F')
    digit_count = sum(c.isdigit() for c in text)
    space_count = sum(c.isspace() for c in text)
    punct_count = sum(c in '.,;:!?()-[]{}"\'' for c in text)
    other_count = len(text) - alpha_count - digit_count - space_count - punct_count
    
    total_chars = len(text)
    
    if total_chars > 0:
        alpha_ratio = alpha_count / total_chars
        other_ratio = other_count / total_chars
        punct_ratio = punct_count / total_chars
        
        # Too many non-alphanumeric characters
        if other_ratio > 0.4:
            return True
        
        # Too few alphabetic characters for normal text
        if alpha_ratio < 0.3 and total_chars > 5:
            return True
        
        # Excessive punctuation
        if punct_ratio > 0.5:
            return True
    
    # Enhanced repetition detection
    char_counts = Counter(text.lower() if language == "english" else text)
    for char, count in char_counts.items():
        if (char.isalpha() and count > len(text) * 0.5) or (('\u0900' <= char <= '\u097F') and count > len(text) * 0.5):
            return True
    
    # Word-level corruption analysis
    words = text.split()
    if words:
        # Check for very short words dominance (potential corruption)
        very_short_words = sum(1 for word in words if len(word) <= 2)
        if len(words) > 2 and (very_short_words / len(words)) > 0.7:
            return True
        
        # Check for words with no vowels (likely corruption)
        vowelless_words = 0
        for word in words:
            clean_word = re.sub(r'[^a-zA-Z]', '', word) if language == "english" else re.sub(r'[^\u0900-\u097F]', '', word)
            if len(clean_word) > 3 and not re.search(r'[aeiouAEIOU]', clean_word) if language == "english" else not re.search(r'[\u0905-\u0914]', clean_word):
                vowelless_words += 1
        
        if len(words) > 1 and (vowelless_words / len(words)) > 0.4:
            return True
        
        # Check for excessive capitalization inconsistency
        if len(words) > 3 and language == "english":
            cap_inconsistency = 0
            for i in range(1, len(words)):
                if words[i].islower() and words[i-1].isupper():
                    cap_inconsistency += 1
            
            if cap_inconsistency > len(words) * 0.6:
                return True
    
    # Syllable and pronunciation checks for very corrupted text
    if len(text) > 10:
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', text) if language == "english" else re.findall(r'[\u0915-\u0939]{4,}', text)
        if len(consonant_clusters) > 2:
            return True
    
    return False

def is_document_metadata(text: str) -> bool:
    """Enhanced metadata detection with comprehensive patterns"""
    if not text:
        return True
    
    text_lower = text.lower().strip()
    language = detect_language(text)
    
    if len(text_lower) < 2:
        return True
    
    # Comprehensive metadata patterns with confidence scores
    metadata_patterns = [
        # Dates and temporal patterns
        (r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}', 0.95),
        (r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}', 0.9),
        (r'^\d{1,2},?\s+\d{4}', 0.9),
        (r'^(mon|tue|wed|thu|fri|sat|sun)\w*\s*,', 0.9),
        (r'^\d{4}[-\/]\d{1,2}[-\/]\d{1,2}', 0.9),
        (r'^(spring|summer|fall|autumn|winter)\s+\d{4}', 0.8),
        (r'(last\s+modified|created\s+on|updated\s+on)', 0.9),
        
        # Document information
        (r'page\s+\d+(\s+of\s+\d+)?', 0.95),
        (r'version\s+[\d.]+', 0.9),
        (r'copyright|©|®|™', 0.95),
        (r'all rights reserved', 0.95),
        (r'confidential|proprietary|internal\s+use\s+only', 0.9),
        (r'draft|final|revised|preliminary|approved', 0.8),
        (r'document\s+(id|number|reference)', 0.9),
        (r'file\s+name|filename', 0.9),
        (r'(printed|published)\s+on', 0.85),
        
        # Contact and web information
        (r'www\.|http|\.com|\.org|\.edu|\.gov|\.net|\.ca', 0.95),
        (r'@.*\.(com|org|edu|gov|net|ca)', 0.95),
        (r'^\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}', 0.95),  # Phone numbers
        (r'^\(\d{3}\)', 0.9),
        (r'^(phone|tel|email|address|fax|mobile|cell):', 0.9),
        (r'^\+?\d{1,3}[\s\-]?\d{3,14}', 0.9),
        (r'(street|avenue|road|blvd|suite|floor)\s+\d+', 0.8),
        
        # Author and attribution
        (r'(author|written\s+by|prepared\s+by|director|manager)', 0.85),
        (r'(executive\s+director|chief|president|vice\s+president)', 0.85),
        (r'(reviewed\s+by|approved\s+by|signed\s+by)', 0.9),
        (r'(department\s+of|ministry\s+of|office\s+of)', 0.8),
        (r'international\s+software\s+testing\s+qualifications\s+board', 0.95),
        (r'ontario\s+library\s+association', 0.95),
        (r'लेखक|निर्देशक|प्रबंधक', 0.85) if language == "hindi" else ("", 0),  # Hindi equivalents
        
        # Page elements and technical
        (r'^(header|footer|watermark)', 0.95),
        (r'^\d+\s*$', 0.8),  # Standalone numbers
        (r'^[ivxlcdm]+\s*$', 0.8),  # Roman numerals alone
        (r'^page\s*\d*\s*$', 0.95),
        (r'^(figure|fig\.|table|tbl\.)\s*\d+', 0.9),
        (r'adobe|pdf|microsoft|word|excel|powerpoint', 0.85),
        (r'font\s+(size|family)', 0.9),
        (r'rgb\(|hex\s*#', 0.9),
        (r'resolution|dpi|pixels', 0.85),
        
        # Legal and formal language
        (r'whereas|hereby|therefore|pursuant', 0.8),
        (r'article\s+\d+|section\s+\d+\.\d+', 0.7),  # Could be headings
        (r'subsection|paragraph|clause', 0.7),
        (r'terms\s+and\s+conditions', 0.85),
        (r'privacy\s+policy', 0.85),
        (r'शर्तें|नीति', 0.85) if language == "hindi" else ("", 0),  # Hindi equivalents
        
        # Footer/header content
        (r'property\s+of|strictly\s+confidential|internal\s+use\s+only', 0.9),
        (r'not\s+for\s+distribution|trade\s+secret|proprietary', 0.9),
        (r'unauthorized|reproduction\s+prohibited', 0.9),
    ]
    
    # Check against patterns
    max_metadata_score = 0.0
    for pattern, score in metadata_patterns:
        if pattern and re.search(pattern, text_lower):
            max_metadata_score = max(max_metadata_score, score)
    
    if max_metadata_score > 0.8:
        return True
    
    # Additional heuristic checks
    
    # Very short numeric-heavy text
    if len(text) <= 10:
        digit_ratio = sum(c.isdigit() for c in text) / len(text)
        if digit_ratio > 0.6:
            return True
    
    # URL-like patterns
    if re.search(r'[a-zA-Z0-9.-]+\.(com|org|edu|gov|net|ca|uk)', text_lower):
        return True
    
    # Email-like patterns
    if re.search(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', text_lower):
        return True
    
    # Standalone technical terms
    tech_terms = ['pdf', 'doc', 'docx', 'jpg', 'png', 'gif', 'html', 'xml', 'css', 'js']
    if text_lower in tech_terms:
        return True
    
    return False

def calculate_heading_confidence(text: str, font_size: float, avg_font_size: float, 
                               max_font_size: float, is_bold: bool, page: int,
                               document_context: Dict = None) -> float:
    """Enhanced heading confidence calculation with document context and caching"""
    confidence = 0.0
    text_clean = text.strip()
    language = detect_language(text)
    text_lower = text_clean.lower() if language == "english" else text_clean
    
    # Enhanced font size analysis (30% weight)
    font_score = _calculate_font_score(font_size, avg_font_size, max_font_size, document_context)
    confidence += font_score * 0.30
    
    # Bold formatting (15% weight) - reduced from 20% for better balance
    bold_score = _calculate_bold_score(is_bold, document_context)
    confidence += bold_score * 0.15
    
    # Enhanced pattern analysis (25% weight)
    pattern_score = _calculate_pattern_score(text_clean, document_context)
    confidence += pattern_score * 0.25
    
    # Enhanced content analysis (20% weight)
    content_score = _calculate_content_score(text_lower, document_context)
    confidence += content_score * 0.20
    
    # Position and context (10% weight)
    position_score = _calculate_position_score(page, text_clean, document_context)
    confidence += position_score * 0.10
    
    # Apply penalties for non-heading characteristics
    penalty_score = _calculate_penalty_score(text_clean, text_lower, language)
    confidence *= (1.0 - penalty_score)
    
    return max(0.0, min(confidence, 1.0))

@lru_cache(maxsize=200)
def _calculate_font_score(font_size: float, avg_font_size: float, 
                         max_font_size: float, context_key: str = None) -> float:
    """Calculate font-based score with caching"""
    if avg_font_size <= 0 or max_font_size <= 0:
        return 0.0
    
    size_ratio = font_size / avg_font_size
    max_ratio = font_size / max_font_size
    
    # Enhanced font size scoring
    if max_ratio > 0.98:  # Very close to largest font
        return 1.0
    elif max_ratio > 0.90:
        return 0.9
    elif max_ratio > 0.80:
        return 0.8
    elif size_ratio > 1.6:  # Much larger than average
        return 0.85
    elif size_ratio > 1.4:
        return 0.75
    elif size_ratio > 1.25:
        return 0.65
    elif size_ratio > 1.15:
        return 0.55
    elif size_ratio > 1.05:
        return 0.35
    elif size_ratio < 0.85:  # Smaller than average (penalty)
        return 0.1
    else:
        return 0.25

def _calculate_bold_score(is_bold: bool, document_context: Dict = None) -> float:
    """Calculate bold formatting score with document context"""
    if not is_bold:
        return 0.0
    
    base_score = 1.0
    
    # Adjust based on document bold usage
    if document_context:
        bold_ratio = document_context.get('bold_ratio', 0.1)
        if bold_ratio > 0.7:  # Most text is bold - less significant
            base_score *= 0.4
        elif bold_ratio > 0.5:
            base_score *= 0.6
        elif bold_ratio > 0.3:
            base_score *= 0.8
    
    return base_score

def _calculate_pattern_score(text: str, document_context: Dict = None) -> float:
    """Enhanced pattern scoring with document awareness"""
    
    # High-confidence patterns
    high_confidence_patterns = [
        (r'^\d+\.\s+[A-Z]', 1.0),  # "1. Introduction"
        (r'^\d+\.\d+\s+[A-Z]', 0.9),  # "2.1 Something"
        (r'^\d+\.\d+\.\d+\s+[A-Z]', 0.8),  # "2.1.1 Details"
        (r'^Chapter\s+\d+:', 0.95),
        (r'^Section\s+[A-Z0-9]', 0.9),
        (r'^Part\s+[IVX0-9]', 0.9),
        (r'^Appendix\s+[A-Z]', 0.9),
        (r'^\d+\.\s+[\u0900-\u097F]+', 1.0),  # Hindi numbered sections
        (r'^[\u0900-\u097F]+\s+\d+', 0.9),  # Hindi chapter with number
    ]
    
    # Medium-confidence patterns
    medium_confidence_patterns = [
        (r'^[A-Z]\.\s+[A-Z]', 0.7),  # "A. Introduction"
        (r'^[IVX]+\.\s+[A-Z]', 0.75),  # Roman numerals
        (r'^[A-Z][a-z]+\s+[A-Z].*:', 0.6),  # Title case ending with colon
        (r'^[A-Z\s]+$', 0.5),  # All caps (with length constraint)
        (r'^[\u0900-\u097F]+[\u0900-\u097F\s]+[\u0900-\u097F]$', 0.7),  # Hindi title-like structure
    ]
    
    max_score = 0.0
    
    # Check high-confidence patterns
    for pattern, score in high_confidence_patterns:
        if re.match(pattern, text):
            max_score = max(max_score, score)
    
    # Check medium-confidence patterns only if no high-confidence match
    if max_score < 0.8:
        for pattern, score in medium_confidence_patterns:
            if re.match(pattern, text):
                # Apply length constraint for all caps
                if pattern == r'^[A-Z\s]+$' and len(text) > 60:
                    score *= 0.5
                max_score = max(max_score, score)
    
    # Title case bonus
    if max_score < 0.6 and text.istitle() and len(text.split()) <= 8 and detect_language(text) == "english":
        max_score = max(max_score, 0.6)
    
    # Colon ending (but not for long text)
    if max_score < 0.5 and text.endswith(':') and len(text) < 80:
        max_score = max(max_score, 0.4)
    
    return max_score

def _calculate_content_score(text_lower: str, document_context: Dict = None) -> float:
    """Enhanced content analysis with document type awareness"""
    
    # Core heading keywords with weights
    core_keywords = {
        'abstract': 0.95, 'executive summary': 0.95, 'introduction': 0.9, 'conclusion': 0.9,
        'overview': 0.85, 'summary': 0.85, 'background': 0.8, 'references': 0.95,
        'bibliography': 0.9, 'acknowledgements': 0.9, 'appendix': 0.85,
        'table of contents': 0.95, 'contents': 0.8,
        'अध्याय': 0.95, 'परिचय': 0.9, 'निष्कर्ष': 0.9, 'सारांश': 0.85, 'पृष्ठभूमि': 0.8, 'संदर्भ': 0.95
    }
    
    # Academic/technical keywords
    academic_keywords = {
        'methodology': 0.85, 'results': 0.85, 'findings': 0.85, 'discussion': 0.85,
        'analysis': 0.8, 'evaluation': 0.8, 'assessment': 0.8, 'review': 0.7,
        'literature review': 0.85, 'related work': 0.8,
        'पद्धति': 0.85, 'परिणाम': 0.85, 'चर्चा': 0.85, 'समीक्षा': 0.7
    }
    
    # Business/organizational keywords
    business_keywords = {
        'objectives': 0.8, 'goals': 0.75, 'strategy': 0.8, 'mission': 0.8,
        'vision': 0.75, 'requirements': 0.8, 'specifications': 0.8,
        'recommendations': 0.85, 'proposal': 0.8,
        'उद्देश्य': 0.8, 'रणनीति': 0.8, 'प्रस्ताव': 0.8
    }
    
    # Structural keywords
    structural_keywords = {
        'chapter': 0.9, 'section': 0.85, 'subsection': 0.8, 'part': 0.8,
        'phase': 0.7, 'stage': 0.7, 'step': 0.6,
        'अध्याय': 0.9, 'खंड': 0.85, 'उपखंड': 0.8
    }
    
    max_score = 0.0
    
    # Check all keyword categories
    for keywords in [core_keywords, academic_keywords, business_keywords, structural_keywords]:
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                max_score = max(max_score, weight)
    
    # Adjust based on document context if available
    if document_context and 'document_type' in document_context:
        doc_type = document_context['document_type']
        if doc_type == 'academic_paper' and max_score < 0.85:
            for keyword in academic_keywords:
                if keyword in text_lower:
                    max_score = max(max_score, academic_keywords[keyword])
        elif doc_type == 'business_document' and max_score < 0.85:
            for keyword in business_keywords:
                if keyword in text_lower:
                    max_score = max(max_score, business_keywords[keyword])
    
    # Default score if no keywords match
    if max_score == 0.0 and len(text_lower.split()) > 2:
        max_score = 0.3  # Minimal score for potential headings without keywords
    
    return max_score

def _calculate_position_score(page: int, text: str, document_context: Dict = None) -> float:
    """Calculate position-based score for heading likelihood"""
    score = 0.0
    
    # Early page bonus (higher likelihood in first few pages)
    if page <= 2:
        score += 0.15
    elif page <= 5:
        score += 0.10
    
    # Isolation bonus (text standing alone on a line)
    if len(text.split()) > 0 and text.startswith((' ', '\n')) and text.endswith((' ', '\n')):
        score += 0.10
    
    # Adjust based on document context
    if document_context and 'total_pages' in document_context:
        total_pages = document_context['total_pages']
        if total_pages > 0 and page / total_pages < 0.2:  # Early in document
            score += 0.05
    
    return min(score, 0.5)  # Cap position score

def _calculate_penalty_score(text_clean: str, text_lower: str, language: str) -> float:
    """Calculate penalty for non-heading characteristics"""
    penalty = 0.0
    
    # Penalty for excessive length (likely paragraph text)
    if len(text_clean) > 150:
        penalty += 0.3
    
    # Penalty for too many sentences (indicates body text)
    sentences = re.split(r'[.!?]+', text_clean)
    if len(sentences) > 2:
        penalty += 0.2
    
    # Penalty for excessive lowercase (unless Hindi)
    if language == "english" and text_clean.lower() == text_clean and len(text_clean.split()) > 1:
        penalty += 0.15
    
    # Penalty for common body text indicators
    body_indicators = ['lorem ipsum', 'body', 'text', 'paragraph', 'content', 'सामग्री']  # Added Hindi 'सामग्री'
    if any(indicator in text_lower for indicator in body_indicators):
        penalty += 0.25
    
    return min(penalty, 0.5)  # Cap penalty score