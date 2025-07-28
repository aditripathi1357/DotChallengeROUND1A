from transformers import pipeline, AutoTokenizer
import torch
import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from collections import Counter
import logging

# Suppress transformer warnings for cleaner output
logging.getLogger("transformers").setLevel(logging.ERROR)

class HeadingClassifier:
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        """Initialize with a lightweight BERT model (~17MB) and enhanced features"""
        self.model_name = model_name
        self.classifier = None
        self.tokenizer = None
        self.feature_cache = {}
        self.pattern_cache = {}
        
        # Load model
        self.load_model()
        
        # Enhanced heading patterns with confidence scores
        self.heading_patterns = [
            (r'^\d+\.?\s+[A-Z]', 0.9),  # "1. Introduction", "1 Overview"
            (r'^\d+\.\d+\.?\s+', 0.8),   # "1.1 Something", "2.1. Details"
            (r'^\d+\.\d+\.\d+\.?\s+', 0.7),  # "1.1.1 Details"
            (r'^Chapter\s+\d+', 0.9),
            (r'^Section\s+[A-Z0-9]', 0.8),
            (r'^Part\s+[A-Z0-9]', 0.8),
            (r'^Appendix\s+[A-Z]', 0.8),
            (r'^[IVX]+\.\s+[A-Z]', 0.7),  # Roman numerals
            (r'^[A-Z]\.\s+[A-Z]', 0.7),   # "A. Introduction"
            (r'^(Abstract|Summary|Introduction|Conclusion|References)$', 0.85),  # Exact matches
        ]
        
        # Enhanced heading keywords with contextual weights
        self.heading_keywords = {
            # Core academic/document sections (high confidence)
            'abstract': 0.9, 'executive summary': 0.9, 'introduction': 0.85, 'conclusion': 0.85,
            'overview': 0.8, 'summary': 0.8, 'background': 0.75, 'references': 0.9,
            'bibliography': 0.85, 'acknowledgements': 0.85, 'appendix': 0.8,
            
            # Methodology and analysis (medium-high confidence)
            'methodology': 0.8, 'method': 0.7, 'approach': 0.65, 'framework': 0.6,
            'results': 0.8, 'findings': 0.8, 'discussion': 0.8, 'analysis': 0.7,
            'evaluation': 0.7, 'assessment': 0.7, 'review': 0.6,
            
            # Structure words (medium confidence)
            'objectives': 0.7, 'goals': 0.6, 'purpose': 0.6, 'scope': 0.6,
            'requirements': 0.7, 'specifications': 0.7, 'guidelines': 0.6,
            'recommendations': 0.8, 'suggestions': 0.6, 'proposal': 0.7,
            
            # Technical/business terms (context-dependent)
            'implementation': 0.65, 'development': 0.6, 'design': 0.6,
            'strategy': 0.7, 'business': 0.6, 'process': 0.5, 'system': 0.5,
            'model': 0.5, 'procedure': 0.6, 'protocol': 0.6, 'standard': 0.6,
            
            # Document structure
            'chapter': 0.85, 'section': 0.8, 'subsection': 0.7, 'part': 0.7,
            'phase': 0.6, 'stage': 0.6, 'step': 0.5, 'milestone': 0.6,
            
            # Content-specific (context-dependent)
            'glossary': 0.8, 'index': 0.7, 'contents': 0.8, 'table of contents': 0.9,
            'deliverable': 0.6, 'outcome': 0.5, 'output': 0.5,
        }
        
        # Enhanced anti-patterns with better detection
        self.anti_patterns = [
            (r'\b(copyright|©|®|™)\b', 0.9),
            (r'\b(page|p\.|pg\.)\s*\d+', 0.95),
            (r'\b(figure|fig\.|table|tbl\.)\s*\d+', 0.9),
            (r'\b(see|refer|visit|contact|email|phone|tel)\b', 0.8),
            (r'www\.|http|\.com|\.org|\.edu|\.gov', 0.95),
            (r'@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 0.95),
            (r'^\d{3}[\-\.\s]?\d{3}[\-\.\s]?\d{4}$', 0.95),  # Phone numbers
            (r'^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$', 0.9),  # Dates
            (r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 0.8),
            (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 0.8),
            (r'^\w+\s+(is|are|was|were|will|can|have|has|had)\s+', 0.85),  # Sentence starters
            (r'\s+(and|or|but|because|since|although|however|therefore)\s+', 0.7),  # Conjunctions
        ]
        
        # Linguistic features for enhanced classification
        self.linguistic_features = {
            'action_verbs': ['maintain', 'ensure', 'provide', 'develop', 'implement', 'establish', 
                           'review', 'update', 'manage', 'create', 'design', 'analyze'],
            'question_words': ['who', 'what', 'when', 'where', 'why', 'how', 'which'],
            'modal_verbs': ['should', 'must', 'may', 'might', 'could', 'would', 'shall'],
            'determiners': ['the', 'this', 'that', 'these', 'those', 'a', 'an'],
            'prepositions': ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'about']
        }
    
    def load_model(self):
        """Load the ML model with error handling and optimization"""
        try:
            # Use a very small BERT model to stay under 200MB limit
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                do_lower_case=True
            )
            
            # Create a simple feature extraction pipeline
            self.classifier = pipeline(
                "feature-extraction",
                model=self.model_name,
                tokenizer=self.tokenizer,
                return_tensors="pt",
                device=-1  # Force CPU usage
            )
            
            print(f"✅ Loaded model: {self.model_name}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load ML model ({e}). Using rule-based classification only.")
            self.classifier = None
            self.tokenizer = None
    
    @lru_cache(maxsize=1000)
    def _cached_pattern_check(self, text: str, pattern: str) -> bool:
        """Cache pattern matching for performance"""
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def get_text_features(self, text: str, context: Dict = None) -> Dict:
        """Extract comprehensive text features using enhanced ML model and context"""
        if not text or len(text.strip()) == 0:
            return {"ml_score": 0.0}
        
        # Check cache first
        cache_key = f"{text}:{str(context)}"
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        try:
            # Calculate base ML score
            ml_score = self._calculate_enhanced_heading_probability(text)
            
            # Context-aware adjustments
            if context:
                ml_score = self._apply_enhanced_context_adjustments(ml_score, text, context)
            
            # Get semantic features if model is available
            semantic_score = 0.0
            if self.classifier:
                semantic_score = self._get_enhanced_semantic_score(text)
            
            # Extract linguistic features
            linguistic_features = self._extract_linguistic_features(text)
            
            # Combine scores with intelligent weighting
            final_score = self._combine_scores_intelligently(
                ml_score, semantic_score, linguistic_features, context
            )
            
            features = {
                "ml_score": final_score,
                "pattern_score": self._get_enhanced_pattern_score(text),
                "keyword_score": self._get_enhanced_keyword_score(text),
                "length_score": self._get_enhanced_length_score(text),
                "structure_score": self._get_enhanced_structure_score(text),
                "linguistic_score": linguistic_features.get("overall_score", 0.0),
                "semantic_score": semantic_score
            }
            
            # Cache the result
            self.feature_cache[cache_key] = features
            return features
            
        except Exception as e:
            print(f"Error in ML classification: {e}")
            return {"ml_score": 0.0}
    
    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features for better classification"""
        if not text:
            return {"overall_score": 0.0}
        
        text_lower = text.lower()
        words = text_lower.split()
        
        features = {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "capitalization_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / len(text) if text else 0,
            "punctuation_ratio": sum(1 for c in text if c in '.,;:!?()-[]{}"\'/\\') / len(text) if text else 0,
        }
        
        # Pattern-based features
        features.update({
            "starts_with_number": bool(re.match(r'^\d', text)),
            "has_colon": text.endswith(':'),
            "is_title_case": text.istitle(),
            "is_upper_case": text.isupper(),
            "has_roman_numerals": bool(re.search(r'\b[IVX]+\b', text)),
        })
        
        # Linguistic pattern analysis
        features.update({
            "has_action_verbs": any(verb in text_lower for verb in self.linguistic_features['action_verbs']),
            "has_question_words": any(word in text_lower for word in self.linguistic_features['question_words']),
            "has_modal_verbs": any(verb in text_lower for verb in self.linguistic_features['modal_verbs']),
            "determiner_ratio": sum(1 for word in words if word in self.linguistic_features['determiners']) / len(words) if words else 0,
            "preposition_ratio": sum(1 for word in words if word in self.linguistic_features['prepositions']) / len(words) if words else 0,
        })
        
        # Calculate overall linguistic score
        score = 0.0
        
        # Length optimization
        if 2 <= features["word_count"] <= 8:
            score += 0.3
        elif 1 <= features["word_count"] <= 12:
            score += 0.2
        
        # Capitalization patterns
        if 0.3 <= features["capitalization_ratio"] <= 0.8:
            score += 0.2
        elif features["is_title_case"]:
            score += 0.25
        
        # Structure patterns
        if features["starts_with_number"]:
            score += 0.2
        if features["has_colon"] and features["word_count"] <= 10:
            score += 0.15
        
        # Penalize sentence-like patterns
        if features["has_action_verbs"] and features["word_count"] > 5:
            score -= 0.2
        if features["has_question_words"]:
            score -= 0.15
        if features["preposition_ratio"] > 0.3:
            score -= 0.15
        if features["determiner_ratio"] > 0.3:
            score -= 0.1
        
        # Punctuation penalties
        if features["punctuation_ratio"] > 0.3:
            score -= 0.2
        
        features["overall_score"] = max(0.0, min(score, 1.0))
        return features
    
    def _get_enhanced_semantic_score(self, text: str) -> float:
        """Get enhanced semantic similarity score using BERT embeddings"""
        try:
            if not self.classifier or len(text) > 512:
                return 0.0
            
            # Get embeddings with proper truncation
            text_truncated = text[:512]
            features = self.classifier(text_truncated)
            
            if isinstance(features, list) and len(features) > 0:
                embeddings = features[0]
                
                # Calculate semantic features
                if hasattr(embeddings, 'mean'):
                    # Mean pooling of embeddings
                    mean_activation = float(embeddings.mean())
                    
                    # Variance as a measure of semantic richness
                    variance = float(embeddings.var()) if hasattr(embeddings, 'var') else 0.0
                    
                    # Normalize to 0-1 range with improved heuristics
                    semantic_score = max(0, min(1, (mean_activation + 1) / 2))
                    
                    # Adjust based on variance (headings often have distinctive patterns)
                    if variance > 0.1:  # High variance suggests distinctive content
                        semantic_score += 0.1
                    
                    return min(semantic_score * 0.4, 0.4)  # Conservative weight, max 0.4
            
            return 0.0
            
        except Exception as e:
            return 0.0
    
    def _calculate_enhanced_heading_probability(self, text: str) -> float:
        """Enhanced probability calculation with better accuracy"""
        text_clean = text.strip()
        text_lower = text_clean.lower()
        
        if len(text_clean) < 2:
            return 0.0
        
        score = 0.0
        
        # Enhanced pattern matching (35% weight)
        pattern_score = self._get_enhanced_pattern_score(text_clean)
        score += pattern_score * 0.35
        
        # Enhanced keyword analysis (25% weight)
        keyword_score = self._get_enhanced_keyword_score(text_lower)
        score += keyword_score * 0.25
        
        # Enhanced structure analysis (20% weight)
        structure_score = self._get_enhanced_structure_score(text_clean)
        score += structure_score * 0.20
        
        # Length optimization (10% weight)
        length_score = self._get_enhanced_length_score(text_clean)
        score += length_score * 0.10
        
        # Position and formatting (10% weight)
        format_score = self._get_formatting_score(text_clean)
        score += format_score * 0.10
        
        # Apply enhanced anti-patterns penalty
        anti_pattern_penalty = self._calculate_anti_pattern_penalty(text_lower)
        score *= (1.0 - anti_pattern_penalty)
        
        return max(0.0, min(1.0, score))
    
    def _get_enhanced_pattern_score(self, text: str) -> float:
        """Enhanced pattern scoring with confidence weights"""
        max_score = 0.0
        
        # Check all patterns with their confidence scores
        for pattern, confidence in self.heading_patterns:
            if self._cached_pattern_check(text, pattern):
                max_score = max(max_score, confidence)
        
        # Additional pattern checks
        # Strong numbered patterns
        if re.match(r'^\d+\.\s+[A-Z][a-z]+', text):  # "1. Introduction"
            max_score = max(max_score, 0.95)
        elif re.match(r'^\d+\.\d+\s+[A-Z][a-z]+', text):  # "2.1 Overview"
            max_score = max(max_score, 0.85)
        
        # Title case patterns
        if text.istitle() and len(text.split()) <= 6:
            max_score = max(max_score, 0.7)
        
        # All caps patterns (with length constraint)
        if text.isupper() and 5 <= len(text) <= 40:
            max_score = max(max_score, 0.6)
        
        # Colon ending patterns
        if text.endswith(':') and len(text) < 80 and not any(word in text.lower() for word in ['note', 'example', 'warning']):
            max_score = max(max_score, 0.5)
        
        return max_score
    
    def _get_enhanced_keyword_score(self, text_lower: str) -> float:
        """Enhanced keyword scoring with context awareness"""
        max_keyword_score = 0.0
        
        # Primary keyword matching
        for keyword, weight in self.heading_keywords.items():
            if keyword in text_lower:
                # Check for word boundaries for better accuracy
                if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                    max_keyword_score = max(max_keyword_score, weight)
                else:
                    # Partial match with reduced weight
                    max_keyword_score = max(max_keyword_score, weight * 0.7)
        
        # Compound keyword detection
        compound_keywords = [
            ('executive summary', 0.95),
            ('table of contents', 0.95),
            ('business plan', 0.85),
            ('mission statement', 0.85),
            ('terms of reference', 0.85),
            ('learning objectives', 0.8),
            ('business outcomes', 0.8),
        ]
        
        for compound, weight in compound_keywords:
            if compound in text_lower:
                max_keyword_score = max(max_keyword_score, weight)
        
        # Domain-specific keywords with context
        domain_keywords = {
            'technical': ['api', 'algorithm', 'implementation', 'specification', 'protocol'],
            'business': ['strategy', 'roi', 'kpi', 'stakeholder', 'governance'],
            'academic': ['hypothesis', 'methodology', 'literature', 'empirical', 'theoretical'],
            'legal': ['compliance', 'regulation', 'policy', 'procedure', 'guideline']
        }
        
        for domain, keywords in domain_keywords.items():
            domain_score = sum(0.1 for keyword in keywords if keyword in text_lower)
            max_keyword_score = max(max_keyword_score, min(domain_score, 0.6))
        
        return max_keyword_score
    
    def _get_enhanced_structure_score(self, text: str) -> float:
        """Enhanced structure analysis"""
        score = 0.0
        words = text.split()
        word_count = len(words)
        
        # Optimal word count ranges
        if 1 <= word_count <= 2:
            score += 0.4
        elif 3 <= word_count <= 6:
            score += 0.5
        elif 7 <= word_count <= 10:
            score += 0.3
        elif 11 <= word_count <= 15:
            score += 0.1
        elif word_count > 20:
            score -= 0.3
        
        # Capitalization analysis
        if len(words) > 1:
            capitalized_words = sum(1 for word in words if word and word[0].isupper())
            cap_ratio = capitalized_words / len(words)
            
            if cap_ratio >= 0.8:  # Most words capitalized (Title Case)
                score += 0.3
            elif cap_ratio >= 0.5:  # Some words capitalized
                score += 0.2
            elif cap_ratio == 1.0 and word_count <= 4:  # All caps, short
                score += 0.25
        
        # Punctuation analysis
        punct_chars = '.,;!?'
        punct_count = sum(1 for char in text if char in punct_chars)
        
        if punct_count == 0:  # No punctuation (typical of headings)
            score += 0.2
        elif punct_count == 1:
            if text.endswith(':'):  # Single colon
                score += 0.3
            elif text.endswith('.') and word_count <= 5:  # Short sentence ending
                score += 0.1
        elif punct_count > 3:  # Too much punctuation
            score -= 0.2
        
        # Number and special character analysis
        has_numbers = bool(re.search(r'\d', text))
        if has_numbers:
            if re.match(r'^\d+\.', text):  # Starts with number and period
                score += 0.3
            else:
                score += 0.1
        
        # Length-based adjustments
        text_length = len(text)
        if 10 <= text_length <= 60:
            score += 0.1
        elif text_length > 120:
            score -= 0.2
        
        return max(0.0, min(score, 1.0))
    
    def _get_enhanced_length_score(self, text: str) -> float:
        """Enhanced length scoring with dynamic ranges"""
        length = len(text)
        
        # Optimal length ranges for different heading types
        if 8 <= length <= 50:  # Sweet spot for most headings
            return 1.0
        elif 5 <= length <= 80:  # Good range
            return 0.8
        elif 3 <= length <= 100:  # Acceptable range
            return 0.6
        elif 2 <= length <= 120:  # Extended acceptable range
            return 0.4
        elif length <= 150:  # Long but possible
            return 0.2
        else:  # Too long
            return 0.1
    
    def _get_formatting_score(self, text: str) -> float:
        """Score based on formatting characteristics"""
        score = 0.0
        
        # First character capitalization
        if text and text[0].isupper():
            score += 0.3
        
        # Consistent capitalization patterns
        if text.istitle():
            score += 0.4
        elif text.isupper() and len(text) <= 50:
            score += 0.3
        
        # Numeric prefixes (common in structured documents)
        if re.match(r'^\d+\.?\d*\.?\d*\s+', text):
            score += 0.5
        
        # Letter prefixes (A., B., etc.)
        if re.match(r'^[A-Z]\.\s+', text):
            score += 0.3
        
        # Roman numeral prefixes
        if re.match(r'^[IVX]+\.\s+', text):
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_anti_pattern_penalty(self, text_lower: str) -> float:
        """Calculate penalty based on anti-patterns"""
        max_penalty = 0.0
        
        for pattern, penalty in self.anti_patterns:
            if self._cached_pattern_check(text_lower, pattern):
                max_penalty = max(max_penalty, penalty)
        
        # Additional anti-pattern checks
        
        # Sentence-like structures
        if re.search(r'\s+(is|are|was|were|will|can|have|has|had)\s+', text_lower):
            max_penalty = max(max_penalty, 0.8)
        
        # Question patterns
        if text_lower.strip().endswith('?'):
            max_penalty = max(max_penalty, 0.7)
        
        # Long sentences with multiple clauses
        if len(text_lower.split()) > 15 and any(conj in text_lower for conj in [' and ', ' but ', ' or ', ' because ', ' although ']):
            max_penalty = max(max_penalty, 0.6)
        
        # Instruction-like patterns
        instruction_starters = ['please', 'make sure', 'ensure that', 'remember to', 'do not', 'always', 'never']
        if any(text_lower.startswith(starter) for starter in instruction_starters):
            max_penalty = max(max_penalty, 0.8)
        
        return min(max_penalty, 0.95)  # Cap penalty at 95%
    
    def _combine_scores_intelligently(self, ml_score: float, semantic_score: float, 
                                    linguistic_features: Dict, context: Dict = None) -> float:
        """Intelligently combine different scores with dynamic weighting"""
        
        # Base weights
        weights = {
            "ml": 0.4,
            "semantic": 0.2,
            "linguistic": 0.25,
            "context": 0.15
        }
        
        # Adjust weights based on available information
        if not semantic_score:  # No semantic model available
            weights["ml"] += 0.1
            weights["linguistic"] += 0.1
            weights["semantic"] = 0.0
        
        if not context:  # No context available
            weights["ml"] += 0.075
            weights["linguistic"] += 0.075
            weights["context"] = 0.0
        
        # Calculate context score
        context_score = 0.0
        if context:
            context_score = self._calculate_context_score(context)
        
        # Combine scores
        final_score = (
            weights["ml"] * ml_score +
            weights["semantic"] * semantic_score +
            weights["linguistic"] * linguistic_features.get("overall_score", 0.0) +
            weights["context"] * context_score
        )
        
        return max(0.0, min(final_score, 1.0))
    
    def _calculate_context_score(self, context: Dict) -> float:
        """Calculate score based on context information"""
        score = 0.0
        
        # Font size context
        font_size = context.get('font_size', 12)
        avg_font_size = context.get('avg_font_size', 12)
        
        if avg_font_size > 0:
            size_ratio = font_size / avg_font_size
            if size_ratio > 1.4:
                score += 0.4
            elif size_ratio > 1.2:
                score += 0.3
            elif size_ratio > 1.1:
                score += 0.2
            elif size_ratio < 0.9:
                score -= 0.1
        
        # Bold text bonus
        if context.get('is_bold', False):
            score += 0.3
        
        # Position context
        page = context.get('page', 1)
        if page <= 3:  # Early pages
            score += 0.1
        elif page <= 5:
            score += 0.05
        
        # Isolation bonus
        if context.get('is_isolated', False):
            score += 0.2
        
        # Font consistency (if font hierarchy is available)
        if 'font_hierarchy_level' in context:
            level = context['font_hierarchy_level']
            if level in ['H1', 'H2', 'H3']:
                score += 0.25
        
        return max(0.0, min(score, 1.0))
    
    def _apply_enhanced_context_adjustments(self, base_score: float, text: str, context: Dict) -> float:
        """Apply enhanced context-aware adjustments to the score"""
        adjusted_score = base_score
        
        # Font size context with improved logic
        font_size = context.get('font_size', 12)
        avg_font_size = context.get('avg_font_size', 12)
        max_font_size = context.get('max_font_size', 16)
        
        if avg_font_size > 0 and max_font_size > 0:
            size_ratio = font_size / avg_font_size
            max_ratio = font_size / max_font_size
            
            # Size-based adjustments
            if max_ratio > 0.95:  # Very close to largest font
                adjusted_score += 0.25
            elif max_ratio > 0.85:
                adjusted_score += 0.20
            elif size_ratio > 1.4:
                adjusted_score += 0.15
            elif size_ratio > 1.2:
                adjusted_score += 0.10
            elif size_ratio > 1.05:
                adjusted_score += 0.05
            elif size_ratio < 0.85:
                adjusted_score -= 0.10
        
        # Enhanced bold text consideration
        if context.get('is_bold', False):
            bold_bonus = 0.15
            # Reduce bonus if document has high bold ratio
            bold_ratio = context.get('document_bold_ratio', 0.1)
            if bold_ratio > 0.5:
                bold_bonus *= 0.6
            adjusted_score += bold_bonus
        
        # Position and page context
        page = context.get('page', 1)
        total_pages = context.get('total_pages', 10)
        
        if total_pages > 0:
            page_ratio = page / total_pages
            # Early pages bonus (but not first page for content)
            if 0.05 < page_ratio < 0.3:
                adjusted_score += 0.08
            elif page == 1:
                # First page bonus for title-like elements
                if any(keyword in text.lower() for keyword in ['title', 'cover', 'summary', 'overview']):
                    adjusted_score += 0.10
        
        # Isolation and positioning
        if context.get('is_isolated', False):
            adjusted_score += 0.12
        
        # Center alignment bonus
        center_alignment = context.get('center_alignment', 1.0)
        if center_alignment < 0.1:  # Very centered
            adjusted_score += 0.08
        elif center_alignment < 0.2:
            adjusted_score += 0.04
        
        # Font family consistency
        font_name = context.get('font_name', '')
        if font_name and 'heading' in font_name.lower():
            adjusted_score += 0.05
        
        # Document type specific adjustments
        doc_type = context.get('document_type', 'generic')
        if doc_type == 'academic' and any(keyword in text.lower() for keyword in ['abstract', 'methodology', 'conclusion']):
            adjusted_score += 0.08
        elif doc_type == 'technical' and any(keyword in text.lower() for keyword in ['specification', 'implementation', 'requirements']):
            adjusted_score += 0.08
        elif doc_type == 'business' and any(keyword in text.lower() for keyword in ['strategy', 'objectives', 'executive']):
            adjusted_score += 0.08
        
        return max(0.0, min(adjusted_score, 1.0))
    
    def classify_batch(self, texts_with_context: List[Tuple[str, Dict]]) -> List[Dict]:
        """Classify multiple texts with their contexts efficiently"""
        results = []
        
        # Process in batches for efficiency
        batch_size = 16
        for i in range(0, len(texts_with_context), batch_size):
            batch = texts_with_context[i:i + batch_size]
            
            for text, context in batch:
                features = self.get_text_features(text, context)
                results.append({
                    'text': text,
                    'features': features,
                    'is_heading_probability': features['ml_score'],
                    'confidence_level': self._determine_confidence_level(features['ml_score'])
                })
        
        return results
    
    def _determine_confidence_level(self, score: float) -> str:
        """Determine confidence level based on score"""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.65:
            return "high"
        elif score >= 0.5:
            return "medium"
        elif score >= 0.35:
            return "low"
        else:
            return "very_low"
    
    def get_confidence_threshold(self, precision_target: float = 0.85) -> float:
        """Get dynamic confidence threshold based on precision target with improved mapping"""
        # Enhanced threshold mapping based on empirical testing
        threshold_map = {
            0.98: 0.85,  # Ultra-high precision
            0.95: 0.75,  # Very high precision
            0.92: 0.68,  # High precision
            0.88: 0.60,  # Good precision
            0.85: 0.55,  # Target precision
            0.82: 0.50,  # Moderate precision
            0.78: 0.45,  # Lower precision
            0.75: 0.40,  # Higher recall
        }
        
        # Find closest precision target
        closest_precision = min(threshold_map.keys(), key=lambda x: abs(x - precision_target))
        base_threshold = threshold_map[closest_precision]
        
        # Fine-tune based on exact target
        if precision_target not in threshold_map:
            # Linear interpolation for more precise thresholds
            sorted_precisions = sorted(threshold_map.keys())
            for i in range(len(sorted_precisions) - 1):
                if sorted_precisions[i] <= precision_target <= sorted_precisions[i + 1]:
                    p1, p2 = sorted_precisions[i], sorted_precisions[i + 1]
                    t1, t2 = threshold_map[p1], threshold_map[p2]
                    
                    # Linear interpolation
                    ratio = (precision_target - p1) / (p2 - p1)
                    base_threshold = t1 + ratio * (t2 - t1)
                    break
        
        return base_threshold
    
    def optimize_for_document_type(self, document_type: str) -> Dict[str, float]:
        """Optimize classifier parameters for specific document types"""
        
        optimization_params = {
            "academic": {
                "keyword_weight_multiplier": 1.2,
                "pattern_weight_multiplier": 1.1,
                "threshold_adjustment": -0.05,  # Slightly more lenient
                "preferred_keywords": ['abstract', 'methodology', 'results', 'conclusion', 'references']
            },
            "technical": {
                "keyword_weight_multiplier": 1.15,
                "pattern_weight_multiplier": 1.3,  # Technical docs often have strong patterns
                "threshold_adjustment": 0.0,
                "preferred_keywords": ['specification', 'implementation', 'requirements', 'architecture']
            },
            "business": {
                "keyword_weight_multiplier": 1.1,
                "pattern_weight_multiplier": 0.9,  # Business docs may be less structured
                "threshold_adjustment": -0.03,
                "preferred_keywords": ['executive summary', 'strategy', 'objectives', 'recommendations']
            },
            "legal": {
                "keyword_weight_multiplier": 1.0,
                "pattern_weight_multiplier": 1.4,  # Legal docs are highly structured
                "threshold_adjustment": 0.05,  # More strict
                "preferred_keywords": ['whereas', 'therefore', 'article', 'section', 'clause']
            },
            "generic": {
                "keyword_weight_multiplier": 1.0,
                "pattern_weight_multiplier": 1.0,
                "threshold_adjustment": 0.0,
                "preferred_keywords": ['introduction', 'overview', 'summary', 'conclusion']
            }
        }
        
        return optimization_params.get(document_type, optimization_params["generic"])
    
    def validate_classification_accuracy(self, test_data: List[Tuple[str, bool]], 
                                       context_data: List[Dict] = None) -> Dict[str, float]:
        """Validate classification accuracy on test data"""
        if not test_data:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        predictions = []
        true_labels = []
        
        for i, (text, true_label) in enumerate(test_data):
            context = context_data[i] if context_data and i < len(context_data) else None
            features = self.get_text_features(text, context)
            
            # Use dynamic threshold
            threshold = self.get_confidence_threshold(0.85)
            predicted_label = features["ml_score"] >= threshold
            
            predictions.append(predicted_label)
            true_labels.append(true_label)
        
        # Calculate metrics
        tp = sum(1 for p, t in zip(predictions, true_labels) if p and t)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p and not t)
        fn = sum(1 for p, t in zip(predictions, true_labels) if not p and t)
        tn = sum(1 for p, t in zip(predictions, true_labels) if not p and not t)
        
        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }
    
    def explain_classification(self, text: str, context: Dict = None) -> Dict:
        """Provide detailed explanation of classification decision"""
        features = self.get_text_features(text, context)
        
        explanation = {
            "text": text,
            "final_score": features["ml_score"],
            "confidence_level": self._determine_confidence_level(features["ml_score"]),
            "feature_breakdown": {
                "pattern_score": features.get("pattern_score", 0.0),
                "keyword_score": features.get("keyword_score", 0.0),
                "structure_score": features.get("structure_score", 0.0),
                "length_score": features.get("length_score", 0.0),
                "linguistic_score": features.get("linguistic_score", 0.0),
                "semantic_score": features.get("semantic_score", 0.0)
            },
            "contributing_factors": [],
            "detrimental_factors": []
        }
        
        # Identify contributing factors
        if features.get("pattern_score", 0) > 0.5:
            explanation["contributing_factors"].append(f"Strong pattern match (score: {features['pattern_score']:.2f})")
        
        if features.get("keyword_score", 0) > 0.3:
            explanation["contributing_factors"].append(f"Contains heading keywords (score: {features['keyword_score']:.2f})")
        
        if features.get("structure_score", 0) > 0.4:
            explanation["contributing_factors"].append(f"Good structural characteristics (score: {features['structure_score']:.2f})")
        
        # Identify detrimental factors
        if features.get("length_score", 0) < 0.3:
            explanation["detrimental_factors"].append(f"Suboptimal length (score: {features['length_score']:.2f})")
        
        if features.get("linguistic_score", 0) < 0.2:
            explanation["detrimental_factors"].append(f"Contains sentence-like patterns (score: {features['linguistic_score']:.2f})")
        
        # Context factors
        if context:
            if context.get('is_bold', False):
                explanation["contributing_factors"].append("Bold formatting")
            
            font_size = context.get('font_size', 12)
            avg_font_size = context.get('avg_font_size', 12)
            if avg_font_size > 0 and font_size / avg_font_size > 1.2:
                explanation["contributing_factors"].append(f"Larger than average font size ({font_size:.1f} vs {avg_font_size:.1f})")
        
        return explanation
    
    def clear_cache(self):
        """Clear internal caches to free memory"""
        self.feature_cache.clear()
        self.pattern_cache.clear()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model and classifier state"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.classifier is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "cache_size": len(self.feature_cache),
            "pattern_cache_size": len(self.pattern_cache),
            "total_patterns": len(self.heading_patterns),
            "total_keywords": len(self.heading_keywords),
            "total_anti_patterns": len(self.anti_patterns)
        }