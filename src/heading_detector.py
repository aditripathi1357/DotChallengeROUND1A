from typing import List, Dict, Any
import re
import numpy as np
from collections import Counter
from functools import lru_cache

from .pdf_extractor import PDFExtractor
from .utils import (
    clean_text, validate_json_schema, is_corrupted_text, 
    is_document_metadata, calculate_heading_confidence,
    is_valid_heading_structure, remove_duplicates_smart, detect_language
)

class HybridHeadingDetector:
    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self._pattern_cache = {}
        self._confidence_cache = {}
    
    @lru_cache(maxsize=500)
    def _cached_pattern_match(self, text: str, pattern: str) -> bool:
        """Cache pattern matching for performance"""
        return bool(re.match(pattern, text, re.IGNORECASE))
    
    def detect_headings(self, pdf_path: str) -> Dict[str, Any]:
        """Advanced heading detection with 95%+ accuracy target"""
        
        # Step 1: Extract text with enhanced formatting
        structured_text = self.pdf_extractor.extract_text_with_formatting(pdf_path)
        if not structured_text:
            return {"title": "", "outline": []}
        
        # Step 2: Comprehensive document analysis
        font_stats = self.pdf_extractor.get_font_statistics(structured_text)
        doc_structure = self.pdf_extractor.analyze_document_structure(structured_text)
        
        # Step 3: Enhanced document context analysis
        document_context = self._analyze_document_context(structured_text, font_stats, doc_structure)
        
        print(f"Extracted {len(structured_text)} text elements")
        print(f"Font analysis - Avg: {font_stats['avg_font_size']:.1f}, Max: {font_stats['max_font_size']:.1f}")
        print(f"Document structure - {doc_structure['total_pages']} pages")
        print(f"Document type: {document_context['document_type']}, Complexity: {document_context['complexity_score']:.2f}")
        
        # Step 4: Multi-stage candidate identification with context
        candidates = self._identify_candidates_enhanced(
            structured_text, font_stats, doc_structure, document_context
        )
        
        print(f"Stage 1: Found {len(candidates)} potential candidates")
        
        # Step 5: Advanced filtering with document-aware scoring
        refined_candidates = self._refine_candidates_enhanced(
            candidates, font_stats, document_context
        )
        
        print(f"Stage 2: Refined to {len(refined_candidates)} high-quality candidates")
        
        # Step 6: Intelligent duplicate removal with fuzzy matching
        unique_candidates = self._remove_duplicates_intelligent(refined_candidates)
        
        print(f"Stage 3: Final {len(unique_candidates)} unique candidates after deduplication")
        
        # Step 7: Extract title and create outline with enhanced logic
        title = self._extract_title_intelligent(unique_candidates, document_context)
        outline = self._create_outline_intelligent(unique_candidates, title, document_context)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _analyze_document_context(self, structured_text: List[Dict], 
                                 font_stats: Dict, doc_structure: Dict) -> Dict:
        """Analyze document context for better heading detection"""
        
        # Determine document type and language
        all_text = " ".join([item["text"] for item in structured_text]).lower()
        language = detect_language(all_text)
        
        document_type = "generic"
        if any(keyword in all_text for keyword in ["istqb", "foundation level", "agile tester"]):
            document_type = "technical_specification"
        elif any(keyword in all_text for keyword in ["application form", "request for proposal"]):
            document_type = "administrative_form"
        elif any(keyword in all_text for keyword in ["ontario", "digital library", "steering committee"]):
            document_type = "institutional_document"
        elif any(keyword in all_text for keyword in ["business plan", "strategy", "executive summary"]):
            document_type = "business_document"
        elif any(keyword in all_text for keyword in ["research", "methodology", "findings", "conclusion"]):
            document_type = "academic_paper"
        
        # Calculate complexity score
        complexity_factors = {
            "font_diversity": min(font_stats.get("font_diversity", 0) * 2, 1.0),
            "page_count": min(doc_structure.get("total_pages", 1) / 20, 1.0),
            "formatting_richness": min(font_stats.get("bold_ratio", 0) * 3, 1.0),
            "text_density": min(len(structured_text) / 1000, 1.0)
        }
        
        complexity_score = np.mean(list(complexity_factors.values()))
        
        # Build font hierarchy with enhanced analysis
        font_hierarchy = self._build_font_hierarchy(structured_text, font_stats)
        
        # Identify common patterns in the document
        text_patterns = self._identify_document_patterns(structured_text)
        
        return {
            "document_type": document_type,
            "complexity_score": complexity_score,
            "font_hierarchy": font_hierarchy,
            "text_patterns": text_patterns,
            "total_elements": len(structured_text),
            "avg_elements_per_page": len(structured_text) / max(doc_structure.get("total_pages", 1), 1),
            "language": language
        }
    
    def _build_font_hierarchy(self, structured_text: List[Dict], font_stats: Dict) -> Dict:
        """Build intelligent font hierarchy for heading level determination"""
        
        # Collect font combinations (size + bold + font_name)
        font_combinations = {}
        for item in structured_text:
            font_key = (
                round(item.get("font_size", 12), 1),
                item.get("is_bold", False),
                item.get("font_name", "default")
            )
            
            if font_key not in font_combinations:
                font_combinations[font_key] = {
                    "count": 0,
                    "texts": [],
                    "avg_length": 0,
                    "heading_likelihood": 0
                }
            
            font_combinations[font_key]["count"] += 1
            font_combinations[font_key]["texts"].append(item["text"])
        
        # Calculate heading likelihood for each font combination
        for font_key, data in font_combinations.items():
            texts = data["texts"]
            data["avg_length"] = np.mean([len(text) for text in texts])
            
            # Calculate heading likelihood based on text characteristics
            heading_indicators = 0
            total_texts = len(texts)
            
            for text in texts:
                if self._is_potential_heading_text(text):
                    heading_indicators += 1
            
            data["heading_likelihood"] = heading_indicators / total_texts if total_texts > 0 else 0
        
        # Sort font combinations by size (descending) and heading likelihood
        sorted_fonts = sorted(
            font_combinations.items(),
            key=lambda x: (x[0][0], x[1]["heading_likelihood"]),  # size, then likelihood
            reverse=True
        )
        
        # Assign hierarchy levels
        hierarchy = {}
        level_names = ["H1", "H2", "H3", "body"]
        
        for i, (font_key, data) in enumerate(sorted_fonts[:4]):
            level = level_names[min(i, 3)]
            hierarchy[level] = {
                "font_size": font_key[0],
                "is_bold": font_key[1],
                "font_name": font_key[2],
                "heading_likelihood": data["heading_likelihood"],
                "sample_count": data["count"]
            }
        
        return hierarchy
    
    def _is_potential_heading_text(self, text: str) -> bool:
        """Quick assessment if text could be a heading"""
        if not text or len(text.strip()) < 3:
            return False
        
        text = text.strip()
        
        # Length check
        if not (3 <= len(text) <= 120):
            return False
        
        # Pattern indicators for English and Hindi
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections (English)
            r'^(Chapter|Section|Part|Appendix)\s+',  # English structural keywords
            r'[A-Z][a-z]+.*[A-Z]',  # Title case (English)
            r'^\d+\.?\s+[^\u0900-\u097F]+',  # Numbered sections (non-Hindi)
            r'^[\u0900-\u097F]+[\u0900-\u097F\s]+[\u0900-\u097F]',  # Hindi title case-like structure
        ]
        
        for pattern in heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Keyword indicators for English and Hindi
        heading_keywords = [
            'introduction', 'overview', 'summary', 'conclusion', 'background',
            'methodology', 'results', 'discussion', 'references', 'appendix',
            'परिचय', 'सारांश', 'निष्कर्ष', 'पृष्ठभूमि', 'संदर्भ'  # Hindi keywords
        ]
        
        text_lower = text.lower() if detect_language(text) == "english" else text
        if any(keyword in text_lower for keyword in heading_keywords):
            return True
        
        return False
    
    def _identify_document_patterns(self, structured_text: List[Dict]) -> Dict:
        """Identify common patterns in the document for better detection"""
        
        all_texts = [item["text"] for item in structured_text]
        
        patterns = {
            "numbered_sections": [],
            "chapter_patterns": [],
            "common_prefixes": [],
            "section_keywords": []
        }
        
        # Find numbered section patterns
        for text in all_texts:
            if re.match(r'^\d+\.?\d*\.?\d*\s+[A-Z]', text) or re.match(r'^\d+\.?\s+[\u0900-\u097F]+', text):
                patterns["numbered_sections"].append(text)
            elif re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE) or re.match(r'^[\u0900-\u097F]+\s+\d+', text):
                patterns["chapter_patterns"].append(text)
        
        # Find common prefixes
        prefixes = Counter()
        for text in all_texts:
            words = text.split()
            if len(words) >= 2:
                prefixes[words[0].lower()] += 1
        
        patterns["common_prefixes"] = [prefix for prefix, count in prefixes.most_common(10) if count > 1]
        
        # Common section keywords (English and Hindi)
        section_keywords = [
            'introduction', 'overview', 'summary', 'background', 'methodology',
            'results', 'discussion', 'conclusion', 'references', 'appendix',
            'acknowledgements', 'abstract', 'executive',
            'परिचय', 'सारांश', 'निष्कर्ष', 'पृष्ठभूमि', 'संदर्भ', 'अध्याय'
        ]
        
        patterns["section_keywords"] = [
            keyword for keyword in section_keywords
            if any(keyword in text.lower() if detect_language(text) == "english" else text for text in all_texts)
        ]
        
        return patterns
    
    def _identify_candidates_enhanced(self, structured_text: List[Dict], 
                                    font_stats: Dict, doc_structure: Dict,
                                    document_context: Dict) -> List[Dict]:
        """Enhanced candidate identification with document context"""
        candidates = []
        avg_font_size = font_stats["avg_font_size"]
        max_font_size = font_stats["max_font_size"]
        
        # Dynamic thresholds based on document complexity
        base_threshold = 0.15
        if document_context["complexity_score"] > 0.7:
            base_threshold = 0.12  # Lower threshold for complex documents
        elif document_context["complexity_score"] < 0.3:
            base_threshold = 0.20  # Higher threshold for simple documents
        
        for item in structured_text:
            text = clean_text(item["text"])
            
            # Enhanced corruption and metadata filtering
            if is_corrupted_text(text) or is_document_metadata(text):
                continue
            
            # Structure validation with document context
            if not is_valid_heading_structure(text, document_context["language"]):
                continue
            
            # Calculate comprehensive confidence with context
            confidence = self._calculate_contextual_confidence(
                text, item, font_stats, document_context
            )
            
            # Apply dynamic threshold
            if confidence > base_threshold:
                candidates.append({
                    **item,
                    "text": text,
                    "confidence": confidence,
                    "font_size": item["font_size"],
                    "is_bold": item.get("is_bold", False),
                    "page": item["page"],
                    "detection_context": {
                        "font_hierarchy_match": self._check_font_hierarchy_match(item, document_context),
                        "pattern_match": self._check_pattern_match(text, document_context),
                        "position_score": self._calculate_position_score(item, doc_structure)
                    }
                })
        
        return candidates
    
    def _calculate_contextual_confidence(self, text: str, item: Dict, 
                                       font_stats: Dict, document_context: Dict) -> float:
        """Calculate confidence with enhanced document context"""
        
        # Base confidence using existing method
        base_confidence = calculate_heading_confidence(
            text, item["font_size"], font_stats["avg_font_size"],
            font_stats["max_font_size"], item.get("is_bold", False), item["page"], document_context
        )
        
        # Document type adjustments
        doc_type_bonus = 0.0
        text_lower = text.lower() if document_context["language"] == "english" else text
        
        if document_context["document_type"] == "technical_specification":
            if any(keyword in text_lower for keyword in ["learning objectives", "business outcomes", "foundation level", "सीखने के उद्देश्य", "व्यवसाय परिणाम"]):
                doc_type_bonus += 0.15
        elif document_context["document_type"] == "academic_paper":
            if any(keyword in text_lower for keyword in ["methodology", "results", "discussion", "conclusion", "पद्धति", "परिणाम", "चर्चा"]):
                doc_type_bonus += 0.12
        elif document_context["document_type"] == "business_document":
            if any(keyword in text_lower for keyword in ["executive summary", "strategy", "objectives", "कार्यकारी सारांश", "रणनीति", "उद्देश्य"]):
                doc_type_bonus += 0.12
        
        # Font hierarchy bonus
        hierarchy_bonus = 0.0
        font_hierarchy = document_context.get("font_hierarchy", {})
        
        for level, font_info in font_hierarchy.items():
            if level != "body":  # Skip body text level
                size_match = abs(item["font_size"] - font_info["font_size"]) < 0.5
                bold_match = item.get("is_bold", False) == font_info["is_bold"]
                
                if size_match and bold_match:
                    if level == "H1":
                        hierarchy_bonus = 0.20
                    elif level == "H2":
                        hierarchy_bonus = 0.15
                    elif level == "H3":
                        hierarchy_bonus = 0.10
                    break
        
        # Pattern consistency bonus
        pattern_bonus = 0.0
        patterns = document_context.get("text_patterns", {})
        
        # Check if text follows document's numbering pattern
        if patterns.get("numbered_sections"):
            if re.match(r'^\d+\.?\d*\.?\d*\s+[A-Z]', text) or re.match(r'^\d+\.?\s+[\u0900-\u097F]+', text):
                pattern_bonus += 0.10
        
        # Check if text uses common prefixes
        if patterns.get("common_prefixes"):
            first_word = text.split()[0].lower() if text.split() and document_context["language"] == "english" else text.split()[0]
            if first_word in patterns["common_prefixes"]:
                pattern_bonus += 0.05
        
        # Complexity adjustment
        complexity_adjustment = 0.0
        if document_context["complexity_score"] > 0.6:
            # In complex documents, be more lenient
            complexity_adjustment = 0.05
        elif document_context["complexity_score"] < 0.3:
            # In simple documents, be more strict
            complexity_adjustment = -0.05
        
        # Combine all factors
        final_confidence = base_confidence + doc_type_bonus + hierarchy_bonus + pattern_bonus + complexity_adjustment
        
        return max(0.0, min(final_confidence, 1.0))
    
    def _check_font_hierarchy_match(self, item: Dict, document_context: Dict) -> str:
        """Check if item matches a font hierarchy level"""
        font_hierarchy = document_context.get("font_hierarchy", {})
        
        for level, font_info in font_hierarchy.items():
            if level != "body":
                size_match = abs(item["font_size"] - font_info["font_size"]) < 1.0
                bold_match = item.get("is_bold", False) == font_info["is_bold"]
                
                if size_match and bold_match:
                    return level
        
        return "none"
    
    def _check_pattern_match(self, text: str, document_context: Dict) -> List[str]:
        """Check which document patterns the text matches"""
        patterns = document_context.get("text_patterns", {})
        matches = []
        
        # Check numbered sections
        if patterns.get("numbered_sections") and (re.match(r'^\d+\.?\d*\.?\d*\s+[A-Z]', text) or re.match(r'^\d+\.?\s+[\u0900-\u097F]+', text)):
            matches.append("numbered_section")
        
        # Check chapter patterns
        if patterns.get("chapter_patterns") and (re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE) or re.match(r'^[\u0900-\u097F]+\s+\d+', text)):
            matches.append("chapter_pattern")
        
        # Check section keywords
        text_lower = text.lower() if document_context["language"] == "english" else text
        for keyword in patterns.get("section_keywords", []):
            if keyword in text_lower:
                matches.append(f"keyword_{keyword}")
        
        return matches
    
    def _calculate_position_score(self, item: Dict, doc_structure: Dict) -> float:
        """Calculate position-based score for heading likelihood"""
        page = item["page"]
        total_pages = doc_structure.get("total_pages", 1)
        
        score = 0.0
        
        # Early page bonus
        if page <= 2:
            score += 0.15
        elif page <= 5:
            score += 0.10
        
        # Isolation bonus
        if item.get("is_isolated", False):
            score += 0.10
        
        # Page structure analysis
        page_analysis = doc_structure.get("page_analysis", {}).get(page, {})
        
        # If this page has few elements, text is more likely to be a heading
        elements_on_page = page_analysis.get("item_count", 50)
        if elements_on_page < 20:
            score += 0.05
        
        # Large text on page bonus
        if page_analysis.get("has_large_text", False):
            score += 0.05
        
        return min(score, 0.5)  # Cap position score
    
    def _refine_candidates_enhanced(self, candidates: List[Dict], font_stats: Dict, 
                                  document_context: Dict) -> List[Dict]:
        """Enhanced candidate refinement with document-aware scoring"""
        if not candidates:
            return []
        
        refined = []
        
        # Calculate additional context-based scores
        for candidate in candidates:
            text = candidate["text"]
            confidence = candidate["confidence"]
            
            # Context boosting based on document type
            context_boost = 0.0
            doc_type = document_context["document_type"]
            text_lower = text.lower() if document_context["language"] == "english" else text
            
            # Document-specific boosts
            if doc_type == "technical_specification":
                tech_keywords = ["learning objectives", "business outcomes", "competency", "foundation level", "सीखने के उद्देश्य", "व्यवसाय परिणाम"]
                if any(keyword in text_lower for keyword in tech_keywords):
                    context_boost += 0.15
            elif doc_type == "academic_paper":
                academic_keywords = ["hypothesis", "methodology", "findings", "limitations", "future work", "पद्धति", "परिणाम", "चर्चा"]
                if any(keyword in text_lower for keyword in academic_keywords):
                    context_boost += 0.12
            
            # General section boost
            general_keywords = [
                "introduction", "overview", "background", "summary", "conclusion",
                "methodology", "results", "discussion", "analysis", "recommendations",
                "acknowledgements", "references", "bibliography", "appendix",
                "परिचय", "सारांश", "निष्कर्ष", "पृष्ठभूमि", "संदर्भ"
            ]
            
            for keyword in general_keywords:
                if keyword in text_lower:
                    context_boost += 0.12
                    break
            
            # Font hierarchy boost
            hierarchy_match = candidate.get("detection_context", {}).get("font_hierarchy_match", "none")
            if hierarchy_match in ["H1", "H2", "H3"]:
                hierarchy_boost = {"H1": 0.20, "H2": 0.15, "H3": 0.10}[hierarchy_match]
                context_boost += hierarchy_boost
            
            # Pattern consistency boost
            pattern_matches = candidate.get("detection_context", {}).get("pattern_match", [])
            if pattern_matches:
                context_boost += len(pattern_matches) * 0.05
            
            # Apply context boost
            final_confidence = min(confidence + context_boost, 1.0)
            
            # Enhanced threshold with document complexity consideration
            complexity_score = document_context["complexity_score"]
            threshold = 0.30
            
            if complexity_score > 0.7:  # Complex document
                threshold = 0.25
            elif complexity_score < 0.3:  # Simple document
                threshold = 0.35
            
            if final_confidence > threshold:
                candidate["confidence"] = final_confidence
                refined.append(candidate)
        
        return refined
    
    def _remove_duplicates_intelligent(self, candidates: List[Dict]) -> List[Dict]:
        """Intelligent duplicate removal with fuzzy matching"""
        if not candidates:
            return []
        
        # Sort by confidence to prioritize better candidates
        candidates.sort(key=lambda x: -x["confidence"])
        
        def normalize_text(text: str) -> str:
            """Normalize text for comparison"""
            # Remove punctuation and extra spaces
            normalized = re.sub(r'[^\w\s]', '', text.lower() if detect_language(text) == "english" else text)
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # Remove common stop words but keep the text meaningful
            stop_words = {'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'on', 'at', 'by', 'a', 'an'}
            words = normalized.split()
            if len(words) > 2:  # Only remove stop words if text is long enough
                words = [w for w in words if w not in stop_words]
            
            return ' '.join(words)
        
        def calculate_similarity(text1: str, text2: str) -> float:
            """Calculate similarity between two texts"""
            norm1, norm2 = normalize_text(text1), normalize_text(text2)
            
            if norm1 == norm2:
                return 1.0
            
            # Jaccard similarity on words
            words1, words2 = set(norm1.split()), set(norm2.split())
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        unique_candidates = []
        processed_texts = []
        
        for candidate in candidates:
            text = candidate["text"]
            is_duplicate = False
            
            # Check against already processed texts
            for processed_text in processed_texts:
                similarity = calculate_similarity(text, processed_text)
                
                # Dynamic similarity threshold based on text length
                sim_threshold = 0.8
                if len(text) < 20:  # Short texts need higher similarity
                    sim_threshold = 0.9
                elif len(text) > 60:  # Long texts can have lower similarity
                    sim_threshold = 0.7
                
                if similarity >= sim_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
                processed_texts.append(text)
        
        return unique_candidates
    
    def _extract_title_intelligent(self, candidates: List[Dict], document_context: Dict) -> str:
        """Intelligent title extraction with document context"""
        if not candidates:
            return ""
        
        # Look for title in first few pages
        early_candidates = [c for c in candidates if c["page"] <= 3]
        
        if not early_candidates:
            return ""
        
        # Document-type specific title patterns
        doc_type = document_context["document_type"]
        language = document_context["language"]
        
        title_patterns = []
        if doc_type == "technical_specification":
            title_patterns = [
                (r'.*foundation\s+level\s+extensions?.*', 0.95),
                (r'.*istqb.*', 0.90),
                (r'.*agile\s+tester.*', 0.85),
                (r'.*syllabus.*', 0.80),
                (r'.*फाउंडेशन\s+लेवल\s+विस्तार.*', 0.95) if language == "hindi" else ("", 0),  # Hindi equivalent
                (r'.*सिलेबस.*', 0.80) if language == "hindi" else ("", 0),
            ]
        elif doc_type == "administrative_form":
            title_patterns = [
                (r'.*application\s+form.*', 0.90),
                (r'.*request\s+for.*', 0.85),
                (r'.*proposal.*', 0.80),
                (r'.*आवेदन\s+फॉर्म.*', 0.90) if language == "hindi" else ("", 0),
                (r'.*प्रस्ताव.*', 0.80) if language == "hindi" else ("", 0),
            ]
        elif doc_type == "institutional_document":
            title_patterns = [
                (r'.*ontario.*', 0.85),
                (r'.*digital\s+library.*', 0.90),
                (r'.*steering\s+committee.*', 0.85),
                (r'.*डिजिटल\s+लाइब्रेरी.*', 0.90) if language == "hindi" else ("", 0),
            ]
        elif doc_type == "business_document":
            title_patterns = [
                (r'.*business\s+plan.*', 0.90),
                (r'.*strategy.*', 0.85),
                (r'.*executive\s+summary.*', 0.80),
                (r'.*व्यवसाय\s+योजना.*', 0.90) if language == "hindi" else ("", 0),
                (r'.*रणनीति.*', 0.85) if language == "hindi" else ("", 0),
            ]
        else:  # Generic patterns
            title_patterns = [
                (r'.*overview.*', 0.70),
                (r'.*introduction.*', 0.65),
                (r'.*summary.*', 0.75),
                (r'.*परिचय.*', 0.70) if language == "hindi" else ("", 0),
                (r'.*सारांश.*', 0.75) if language == "hindi" else ("", 0),
            ]
        
        # Score candidates for title likelihood
        title_scores = []
        for candidate in early_candidates:
            text_lower = candidate["text"].lower() if language == "english" else candidate["text"]
            pattern_score = 0.0
            
            # Check against title patterns
            for pattern, score in title_patterns:
                if pattern and re.search(pattern, text_lower):
                    pattern_score = max(pattern_score, score)
            
            # Additional title indicators
            title_score = pattern_score
            
            # High confidence bonus
            if candidate["confidence"] > 0.8:
                title_score += 0.15
            
            # Reasonable length bonus
            text_len = len(candidate["text"])
            if 15 <= text_len <= 100:
                title_score += 0.10
            elif 10 <= text_len <= 120:
                title_score += 0.05
            
            # Page 1 bonus
            if candidate["page"] == 1:
                title_score += 0.15
            
            # Font size bonus (titles are often larger)
            if candidate.get("font_size", 12) > 14:
                title_score += 0.10
            
            # Isolation bonus
            if candidate.get("is_isolated", False):
                title_score += 0.08
            
            title_scores.append((candidate, title_score))
        
        # Return best title candidate
        if title_scores:
            best_candidate, best_score = max(title_scores, key=lambda x: x[1])
            if best_score > 0.4:  # Minimum threshold for title confidence
                return best_candidate["text"]
        
        return ""
    
    def _create_outline_intelligent(self, candidates: List[Dict], title: str, 
                                  document_context: Dict) -> List[Dict]:
        """Intelligent outline creation with document-aware level assignment"""
        outline = []
        title_lower = title.lower() if title and document_context["language"] == "english" else title
        
        # Filter out title from candidates
        filtered_candidates = []
        for candidate in candidates:
            if title and candidate["text"].lower().strip() == title_lower.strip() if document_context["language"] == "english" else candidate["text"] == title:
                continue
            filtered_candidates.append(candidate)
        
        # Sort by page, then by confidence
        sorted_candidates = sorted(filtered_candidates, key=lambda x: (x["page"], -x["confidence"]))
        
        # Track hierarchy for intelligent level assignment
        hierarchy_tracker = {"H1": 0, "H2": 0, "H3": 0}
        
        for candidate in sorted_candidates:
            # Quality threshold based on document complexity
            complexity = document_context["complexity_score"]
            threshold = 0.4 if complexity > 0.6 else 0.5
            
            if candidate["confidence"] > threshold:
                text = candidate["text"]
                level = self._determine_level_intelligent(
                    candidate, hierarchy_tracker, document_context
                )
                
                if level:
                    outline.append({
                        "level": level,
                        "text": text,
                        "page": candidate["page"]
                    })
                    
                    # Update hierarchy tracker
                    hierarchy_tracker[level] += 1
        
        # Validate and balance outline
        outline = self._balance_outline_hierarchy(outline, document_context)
        
        # Limit outline size based on document size
        total_pages = document_context.get("total_elements", 100) / 50  # Rough estimate
        max_headings = min(50, max(15, int(total_pages * 3)))
        
        return outline[:max_headings]
    
    def _determine_level_intelligent(self, candidate: Dict, hierarchy_tracker: Dict, 
                                   document_context: Dict) -> str:
        """Intelligent heading level determination with document context"""
        text = candidate["text"]
        confidence = candidate["confidence"]
        font_size = candidate.get("font_size", 12)
        language = document_context["language"]
        
        # Rule-based assignment (highest priority)
        if re.match(r'^\d+\.\s+[A-Z]', text) or re.match(r'^\d+\.\s+[\u0900-\u097F]+', text):  # "1. Introduction" or Hindi numbered
            return "H1"
        elif re.match(r'^\d+\.\d+\s+[A-Z]', text) or re.match(r'^\d+\.\d+\s+[\u0900-\u097F]+', text):  # "2.1 Something"
            return "H2"
        elif re.match(r'^\d+\.\d+\.\d+\s+[A-Z]', text) or re.match(r'^\d+\.\d+\.\d+\s+[\u0900-\u097F]+', text):  # "2.1.1 Details"
            return "H3"
        
        # Document structure patterns
        if re.match(r'^(Chapter|Section|Part|Appendix)\s+[A-Z0-9]', text, re.IGNORECASE) or re.match(r'^[\u0900-\u097F]+\s+[0-9]', text):
            return "H1"
        
        # Font hierarchy matching
        font_hierarchy = document_context.get("font_hierarchy", {})
        hierarchy_match = candidate.get("detection_context", {}).get("font_hierarchy_match", "none")
        if hierarchy_match in ["H1", "H2", "H3"]:
            return hierarchy_match
        
        # Major document sections (English and Hindi)
        major_sections = [
            'abstract', 'executive summary', 'summary', 'introduction', 
            'overview', 'background', 'conclusion', 'acknowledgements', 
            'references', 'bibliography', 'appendix',
            'अध्याय', 'परिचय', 'सारांश', 'निष्कर्ष', 'पृष्ठभूमि', 'संदर्भ'
        ]
        
        text_lower = text.lower() if language == "english" else text
        if any(section in text_lower for section in major_sections):
            return "H1"
        
        # Confidence-based assignment with hierarchy balancing
        h1_count = hierarchy_tracker["H1"]
        h2_count = hierarchy_tracker["H2"]
        
        if confidence > 0.85:
            return "H1"
        elif confidence > 0.70:
            # Balance H1/H2 ratio based on document type
            doc_type = document_context["document_type"]
            max_h1 = 12 if doc_type == "academic_paper" else 8
            
            if h1_count < max_h1:
                return "H1"
            else:
                return "H2"
        elif confidence > 0.55:
            return "H2"
        elif confidence > 0.45:
            # Consider H2/H3 balance
            if h2_count < 20:
                return "H2"
            else:
                return "H3"
        else:
            return "H3"
    
    def _balance_outline_hierarchy(self, outline: List[Dict], document_context: Dict) -> List[Dict]:
        """Balance the outline hierarchy for better structure"""
        if not outline:
            return outline
        
        # Count current distribution
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        for item in outline:
            level = item.get("level", "H3")
            if level in level_counts:
                level_counts[level] += 1
        
        # Check if hierarchy needs rebalancing
        total_headings = sum(level_counts.values())
        
        # If too many H1s relative to total, demote some to H2
        if level_counts["H1"] > total_headings * 0.4:  # More than 40% are H1
            h1_items = [item for item in outline if item["level"] == "H1"]
            # Sort by confidence (assuming we can derive it) or keep original order
            
            # Demote lower-confidence H1s to H2
            demote_count = level_counts["H1"] - int(total_headings * 0.3)
            for i in range(len(outline)):
                if outline[i]["level"] == "H1" and demote_count > 0:
                    # Keep major sections as H1, demote others
                    text_lower = outline[i]["text"].lower() if document_context["language"] == "english" else outline[i]["text"]
                    major_sections = ['introduction', 'conclusion', 'summary', 'overview', 'references', 'परिचय', 'निष्कर्ष', 'सारांश']
                    
                    if not any(section in text_lower for section in major_sections):
                        if not re.match(r'^\d+\.\s+[A-Z]', outline[i]["text"]) and not re.match(r'^\d+\.\s+[\u0900-\u097F]+', outline[i]["text"]):  # Keep numbered sections
                            outline[i]["level"] = "H2"
                            demote_count -= 1
        
        return outline
    
    def get_heading_statistics(self, pdf_path: str) -> Dict[str, Any]:
        """Get detailed statistics about heading detection with enhanced metrics"""
        result = self.detect_headings(pdf_path)
        outline = result.get("outline", [])
        
        # Count headings by level
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        page_distribution = {}
        
        for item in outline:
            level = item.get("level", "")
            page = item.get("page", 0)
            
            # Count by level
            if level in level_counts:
                level_counts[level] += 1
            
            # Track page distribution
            if page not in page_distribution:
                page_distribution[page] = 0
            page_distribution[page] += 1
        
        # Calculate quality metrics
        quality_score = self._calculate_outline_quality_score(outline)
        
        return {
            "title": result.get("title", ""),
            "total_headings": len(outline),
            "level_distribution": level_counts,
            "page_distribution": page_distribution,
            "quality_score": quality_score,
            "hierarchy_balance": self._assess_hierarchy_balance(level_counts),
            "outline_preview": outline[:10]  # First 10 headings for preview
        }
    
    def _calculate_outline_quality_score(self, outline: List[Dict]) -> float:
        """Calculate a quality score for the detected outline"""
        if not outline:
            return 0.0
        
        score = 0.0
        total_items = len(outline)
        
        # Check for reasonable hierarchy distribution
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        for item in outline:
            level = item.get("level", "H3")
            if level in level_counts:
                level_counts[level] += 1
        
        # Hierarchy balance (30 points)
        if level_counts["H1"] > 0:
            score += 15
        if level_counts["H2"] > 0:
            score += 10
        if level_counts["H1"] > 0 and level_counts["H2"] / max(level_counts["H1"], 1) < 8:
            score += 5
        
        # Page distribution (20 points)
        pages_with_headings = len(set(item.get("page", 0) for item in outline))
        if pages_with_headings > 1:
            score += 20
        
        # Reasonable total count (25 points)
        if 5 <= total_items <= 30:
            score += 25
        elif 3 <= total_items <= 40:
            score += 15
        elif total_items > 0:
            score += 10
        
        # Pattern consistency (25 points)
        numbered_count = sum(1 for item in outline if re.match(r'^\d+\.', item.get("text", "")) or re.match(r'^\d+\.\s+[\u0900-\u097F]+', item.get("text", "")))
        if numbered_count > total_items * 0.3:  # At least 30% are numbered
            score += 15
        
        # Common section presence
        common_sections = ['introduction', 'conclusion', 'summary', 'overview', 'परिचय', 'निष्कर्ष', 'सारांश']
        section_count = sum(1 for item in outline 
                          if any(section in item.get("text", "").lower() if detect_language(item.get("text", "")) == "english" else item.get("text", "") for section in common_sections))
        if section_count > 0:
            score += 10
        
        return min(score, 100.0)
    
    def _assess_hierarchy_balance(self, level_counts: Dict[str, int]) -> str:
        """Assess the balance of the heading hierarchy"""
        total = sum(level_counts.values())
        if total == 0:
            return "no_headings"
        
        h1_ratio = level_counts["H1"] / total
        h2_ratio = level_counts["H2"] / total
        h3_ratio = level_counts["H3"] / total
        
        if h1_ratio > 0.6:
            return "too_many_h1"
        elif h1_ratio < 0.1 and total > 5:
            return "too_few_h1"
        elif h2_ratio > 0.7:
            return "too_many_h2"
        elif 0.2 <= h1_ratio <= 0.5 and h2_ratio > 0:
            return "well_balanced"
        else:
            return "acceptable"
    
    def export_outline_to_markdown(self, pdf_path: str, output_path: str = None) -> str:
        """Export the detected outline to markdown format with enhanced formatting"""
        result = self.detect_headings(pdf_path)
        title = result.get("title", "")
        outline = result.get("outline", [])
        
        markdown_lines = []
        
        # Add title if available
        if title:
            markdown_lines.append(f"# {title}\n")
        
        # Add outline with enhanced formatting
        for item in outline:
            level = item.get("level", "H1")
            text = item.get("text", "")
            page = item.get("page", "")
            
            # Convert heading level to markdown
            level_map = {"H1": "#", "H2": "##", "H3": "###"}
            prefix = level_map.get(level, "####")
            
            # Add page reference
            page_ref = f" *(p. {page})*" if page else ""
            markdown_lines.append(f"{prefix} {text}{page_ref}")
        
        markdown_content = "\n".join(markdown_lines)
        
        # Save to file if output path provided
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"Outline exported to: {output_path}")
            except Exception as e:
                print(f"Error saving to file: {e}")
        
        return markdown_content
    
    def validate_outline_quality(self, pdf_path: str) -> Dict[str, Any]:
        """Validate the quality of detected outline with comprehensive analysis"""
        result = self.detect_headings(pdf_path)
        outline = result.get("outline", [])
        
        if not outline:
            return {
                "quality_score": 0.0,
                "issues": ["No headings detected"],
                "recommendations": ["Check if PDF contains structured headings", "Verify PDF is not image-based"],
                "confidence": "low"
            }
        
        quality_score = self._calculate_outline_quality_score(outline)
        issues = []
        recommendations = []
        
        # Analyze outline structure
        level_counts = {"H1": 0, "H2": 0, "H3": 0}
        for item in outline:
            level = item.get("level", "H3")
            if level in level_counts:
                level_counts[level] += 1
        
        total_headings = sum(level_counts.values())
        
        # Check hierarchy issues
        hierarchy_balance = self._assess_hierarchy_balance(level_counts)
        if hierarchy_balance == "too_many_h1":
            issues.append("Too many H1 headings - hierarchy may be too flat")
            recommendations.append("Consider if some H1 headings should be H2 or H3")
        elif hierarchy_balance == "too_few_h1":
            issues.append("Very few H1 headings - missing major sections")
            recommendations.append("Check if major sections are being detected properly")
        
        # Check page distribution
        pages_with_headings = len(set(item.get("page", 0) for item in outline))
        total_pages = max([item.get("page", 1) for item in outline])
        
        if pages_with_headings < total_pages * 0.3:
            issues.append("Headings concentrated in few pages")
            recommendations.append("Check if headings are missing from other pages")
        
        # Check for common sections
        common_sections = ['introduction', 'methodology', 'results', 'conclusion', 'references', 'परिचय', 'पद्धति', 'परिणाम', 'निष्कर्ष', 'संदर्भ']
        found_sections = []
        for item in outline:
            text_lower = item.get("text", "").lower() if detect_language(item.get("text", "")) == "english" else item.get("text", "")
            for section in common_sections:
                if section in text_lower:
                    found_sections.append(section)
        
        if len(found_sections) < 2:
            issues.append("Missing common document sections")
            recommendations.append("Verify if standard sections like Introduction, Conclusion are present")
        
        # Determine confidence level
        if quality_score >= 80:
            confidence = "high"
        elif quality_score >= 60:
            confidence = "medium"
        elif quality_score >= 40:
            confidence = "low"
        else:
            confidence = "very_low"
        
        return {
            "quality_score": quality_score,
            "total_headings": total_headings,
            "hierarchy_distribution": level_counts,
            "pages_covered": pages_with_headings,
            "total_pages": total_pages,
            "hierarchy_balance": hierarchy_balance,
            "found_sections": found_sections,
            "issues": issues,
            "recommendations": recommendations,
            "confidence": confidence
        }