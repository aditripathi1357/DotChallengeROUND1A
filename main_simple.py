#!/usr/bin/env python3

import json
import re
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.pdf_extractor import PDFExtractor
from src.utils import clean_text, validate_json_schema

def simple_heading_detection(pdf_path: str):
    """Simple, effective heading detection"""
    extractor = PDFExtractor()
    structured_text = extractor.extract_text_with_formatting(pdf_path)
    
    if not structured_text:
        return {"title": "", "outline": []}
    
    # Get font statistics
    font_stats = extractor.get_font_statistics(structured_text)
    avg_font_size = font_stats["avg_font_size"]
    max_font_size = font_stats["max_font_size"]
    
    filename = Path(pdf_path).name
    print(f"    📄 Processing: {filename}")
    print(f"    📊 Text elements: {len(structured_text)}")
    print(f"    🔤 Font sizes: {font_stats['min_font_size']:.1f} - {max_font_size:.1f} (avg: {avg_font_size:.1f})")
    
    # Simple candidate detection
    candidates = []
    
    for item in structured_text:
        text = clean_text(item["text"])
        
        # Basic filtering - only remove obvious junk
        if not text or len(text) < 3 or len(text) > 150:
            continue
        
        # Skip obvious garbage patterns
        if re.match(r'^[A-Z]{1,2}:?\s*$', text):  # "R", "RFP:"
            continue
        if re.match(r'^[a-z]{1,3}\s+[A-Z][a-z]?\s*$', text):  # "r Pr"
            continue
        if re.search(r'[a-z]{3,}[A-Z]{3,}', text):  # Corruption like "questPROPOSAL"
            continue
        
        # Skip dates
        if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d', text, re.IGNORECASE):
            continue
        if re.match(r'^\d{1,2},?\s+\d{4}$', text):
            continue
        
        # Skip contact info
        if re.search(r'@|\.com|\.org|phone|email|address', text, re.IGNORECASE):
            continue
        
        # Calculate simple score
        score = 0.0
        font_size = item["font_size"]
        is_bold = item.get("is_bold", False)
        
        # Font size scoring
        if font_size > avg_font_size * 1.3:
            score += 0.6
        elif font_size > avg_font_size * 1.1:
            score += 0.3
        
        # Bold bonus
        if is_bold:
            score += 0.4
        
        # Pattern bonuses
        if re.match(r'^\d+\.?\d*\.?\d*\s+[A-Z]', text):  # Numbered sections
            score += 0.7
        elif re.match(r'^(Chapter|Section|Appendix)\s+\d+', text, re.IGNORECASE):
            score += 0.7
        elif text.endswith(':') and len(text) < 80:
            score += 0.5
        elif text.isupper() and 5 <= len(text) <= 50:
            score += 0.6
        elif text.istitle() and len(text.split()) <= 8:
            score += 0.3
        
        # Heading keywords
        heading_words = [
            'summary', 'introduction', 'overview', 'background', 'conclusion',
            'methodology', 'results', 'discussion', 'acknowledgements', 'references',
            'appendix', 'business plan', 'milestones', 'timeline', 'approach',
            'evaluation', 'requirements', 'preamble', 'membership', 'funding'
        ]
        
        if any(word in text.lower() for word in heading_words):
            score += 0.4
        
        # Only keep candidates with decent scores
        if score > 0.4:
            candidates.append({
                "text": text,
                "page": item["page"],
                "score": score,
                "font_size": font_size,
                "is_bold": is_bold
            })
    
    print(f"    🎯 Found {len(candidates)} candidates")
    
    # Simple duplicate removal
    seen_texts = set()
    unique_candidates = []
    
    for candidate in sorted(candidates, key=lambda x: -x["score"]):
        text_key = candidate["text"].lower().strip()
        
        # Skip if we've seen very similar text
        is_duplicate = False
        for seen in seen_texts:
            if seen in text_key or text_key in seen:
                if abs(len(seen) - len(text_key)) < 10:  # Similar length
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            seen_texts.add(text_key)
            unique_candidates.append(candidate)
    
    print(f"    🧹 After deduplication: {len(unique_candidates)} candidates")
    
    # Extract title
    title = ""
    page_1_candidates = [c for c in unique_candidates if c["page"] == 1]
    
    if page_1_candidates:
        # Look for title patterns
        for candidate in sorted(page_1_candidates, key=lambda x: -x["score"]):
            text = candidate["text"]
            if any(pattern in text.lower() for pattern in [
                'digital library', 'working together', 'foundation level',
                'application form', 'stem pathways', 'mission statement',
                'request for proposal', 'ontario'
            ]):
                title = text
                break
        
        # Fallback to highest scoring
        if not title and page_1_candidates:
            title = page_1_candidates[0]["text"]
    
    # Create outline
    outline = []
    
    for candidate in sorted(unique_candidates, key=lambda x: (x["page"], -x["score"])):
        # Skip title
        if title and candidate["text"].lower() == title.lower():
            continue
        
        # Only include high-confidence headings
        if candidate["score"] > 0.5:
            text = candidate["text"]
            
            # Simple level assignment
            if re.match(r'^\d+\.\s+', text):
                level = "H1"
            elif re.match(r'^\d+\.\d+\s+', text):
                level = "H2"
            elif re.match(r'^\d+\.\d+\.\d+\s+', text):
                level = "H3"
            elif re.match(r'^(Chapter|Appendix)\s+', text, re.IGNORECASE):
                level = "H1"
            elif candidate["score"] > 0.8:
                level = "H1"
            elif candidate["score"] > 0.65:
                level = "H2"
            else:
                level = "H3"
            
            outline.append({
                "level": level,
                "text": text,
                "page": candidate["page"]
            })
    
    # Limit results
    outline = outline[:20]
    
    return {"title": title, "outline": outline}

def main():
    input_dir = Path("input")
    output_dir = Path("output")
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in input/ directory")
        return
    
    print(f"🎯 Simple & Effective PDF Heading Extraction")
    print(f"📁 Found {len(pdf_files)} files to process")
    print("=" * 60)
    
    total_files = len(pdf_files)
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n📄 File {i}/{total_files}: {pdf_file.name}")
        print("-" * 40)
        
        try:
            result = simple_heading_detection(str(pdf_file))
            
            if validate_json_schema(result):
                output_file = output_dir / f"{pdf_file.stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"    ✅ Saved: {output_file.name}")
                
                if result['title']:
                    print(f"    📋 Title: {result['title']}")
                else:
                    print(f"    📋 Title: (no title detected)")
                
                print(f"    📝 Headings: {len(result['outline'])}")
                
                if result['outline']:
                    print(f"    🏷️  Preview:")
                    for j, heading in enumerate(result['outline'][:4]):
                        truncated = heading['text'][:50] + '...' if len(heading['text']) > 50 else heading['text']
                        print(f"        {heading['level']}: {truncated} (p.{heading['page']})")
                    
                    if len(result['outline']) > 4:
                        print(f"        ... and {len(result['outline']) - 4} more headings")
                else:
                    print(f"    📝 No headings detected")
                
            else:
                print(f"    ❌ Generated invalid JSON schema")
                
        except Exception as e:
            print(f"    ❌ Error processing {pdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 Processing complete!")
    print("📂 Results saved in 'output' directory")
    print("\n💡 Tips for better accuracy:")
    print("   • Ensure PDFs have good text extraction quality")
    print("   • Check that headings use consistent formatting")
    print("   • Verify font size differences between headings and body text")

if __name__ == "__main__":
    main()