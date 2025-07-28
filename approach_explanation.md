# Approach Explanation - approach.md

## Overview
The **Document Outline Extractor** is designed to transform unstructured PDF documents into structured JSON outlines by identifying titles and headings (H1, H2, H3) along with their respective page numbers. This solution, developed for the "Connecting the Dots Through Docs" challenge in Round 1A of the DotChallenge hackathon, employs a hybrid approach that combines multiple techniques to ensure robust and accurate extraction. Below is a detailed explanation of the methodology, including the integration of pre-downloaded machine learning models for potential enhancements.

---

## Core Methodology

### 1. Text and Metadata Extraction
- **Tool**: Utilizes **PyMuPDF (1.23.26)** to extract raw text and associated metadata (e.g., font sizes, bold formatting, coordinates) from PDFs.
- **Process**: 
  - Parses each page of the PDF (up to 50 pages) to retrieve text blocks.
  - Collects font-related attributes to differentiate headings from body text.
- **Purpose**: Provides the foundational data for subsequent analysis, ensuring compatibility with diverse PDF structures.

### 2. Hybrid Heading Detection
The solution avoids relying solely on font size, as some PDFs may not follow consistent formatting. Instead, it integrates multiple signals, enhanced by pre-downloaded ML models where applicable:

- **Font Size Analysis**:
  - Calculates the average and maximum font sizes across the document.
  - Assigns higher confidence to text with significantly larger fonts (e.g., >1.25x average), weighted at 30% of the total score.
  - Adjusts scores based on document context to handle variations.

- **Bold Formatting**:
  - Detects bold text as a strong indicator of headings, contributing 15% to the confidence score.
  - Adjusts weight based on the document's overall bold usage to avoid over-reliance in documents where most text is bold.

- **Pattern Matching**:
  - Identifies structured patterns (e.g., `^\d+\.\s+[A-Z]` for "1. Introduction", `^\d+\.\s+[\u0900-\u097F]+` for Hindi sections), contributing 25% to the score.
  - Includes high-confidence (e.g., chapter numbers) and medium-confidence (e.g., all caps) patterns, with length constraints to filter noise.

- **Content Analysis**:
  - Matches text against a curated list of heading keywords (e.g., "Introduction", "Conclusion", "अध्याय", "परिचय") across categories like core, academic, business, and structural, weighted at 20%.
  - Leverages pre-downloaded ML models (e.g., `prajjwal1/bert-tiny`) to enhance keyword matching or contextual understanding if enabled, adapting to document type (e.g., academic, business) if context is provided.

- **Position Scoring**:
  - Rewards text appearing early (e.g., first 2-5 pages) or in isolated lines, contributing 10% to the score.
  - Considers document length for contextual adjustment.

- **Penalty Application**:
  - Applies penalties (up to 50%) for non-heading traits, such as excessive length (>150 characters), multiple sentences, or body text indicators (e.g., "lorem ipsum", "सामग्री").
  - Reduces confidence for lowercase-heavy text in English.

- **Confidence Calculation**:
  - Combines scores and applies penalties to produce a final confidence value (0.0 to 1.0).
  - Uses a threshold (e.g., 0.6) to classify text as a heading, with levels (H1, H2, H3) determined by relative font size and nesting.
  - ML models can refine confidence scores if integrated for natural language understanding tasks.

### 3. Candidate Refinement and Deduplication
- **Refinement**: Filters out corrupted text (e.g., OCR artifacts, fragments) and metadata (e.g., page numbers, dates) using dedicated functions (`is_corrupted_text`, `is_document_metadata`).
- **Deduplication**: Removes duplicate headings based on text similarity and proximity, ensuring a clean hierarchical outline.
- **Title Selection**: Identifies the most prominent text (e.g., largest font, early page) as the document title, with potential ML-assisted disambiguation.

### 4. JSON Output Generation
- **Format**: Structures the extracted data into a JSON file matching the required schema:
  ```json
  {
    "title": "Understanding AI",
    "outline": [
      { "level": "H1", "text": "Introduction", "page": 1 },
      { "level": "H2", "text": "What is AI?", "page": 2 },
      { "level": "H3", "text": "History of AI", "page": 3 }
    ]
  }
  ```
- **Process**: Saves output as `filename.json` for each processed `filename.pdf` in the `output` directory.

---

## Multilingual Support
- **Language Detection**: Uses Unicode range analysis to identify English (ASCII) and Hindi (Devanagari: U+0900 to U+097F), with a 0.5 threshold for mixed-language texts.
- **Adaptation**: Extends patterns and keywords (e.g., "अध्याय", "परिचय") to support Hindi, with ML models potentially improving language-specific context if trained on multilingual data.
- **Limitation**: Currently optimized for English and Hindi; support for other languages (e.g., Japanese) is a bonus feature with potential for future expansion using diverse model training.

---

## Optimization and Constraints
- **Performance**: Processes 50-page PDFs in ≤ 10 seconds on an 8 CPU, 16 GB RAM system by leveraging efficient libraries and minimal computation.
- **Size**: Keeps the model and dependencies (e.g., `tensorflow-cpu`, `torch`, and pre-downloaded ML models) under 200MB using a slim Docker base image. The `download_model.py` script ensures models like `prajjwal1/bert-tiny` (~17MB) stay within limits.
- **Offline**: Avoids network calls, relying on local processing with PyMuPDF and pre-cached models.

---

## Challenges and Solutions
- **Inconsistent Formatting**: Handles PDFs where headings lack uniform font sizes by integrating multiple detection signals and optional ML model support.
- **Noise**: Filters corrupted text and metadata using heuristic patterns and character analysis, with ML models offering additional noise reduction if activated.
- **Complexity**: Maintains modularity (e.g., `utils.py` functions) for reuse in Round 1B, ensuring scalability, with model integration managed via `download_model.py`.

---

## Model Integration
- **Script**: The `download_model.py` script pre-downloads and verifies ML models offline:
  - **Models**: Tests `prajjwal1/bert-tiny` (~17MB), `distilbert-base-uncased` (~67MB), and `microsoft/DialoGPT-small` (~117MB).
  - **Process**: Downloads to `./models`, checks size constraints (<200MB), and saves configuration in `models/model_info.json`.
  - **Usage**: Currently disabled for inference to meet offline and size constraints, but available for future enhancements (e.g., contextual analysis).
- **Purpose**: Ensures compliance with challenge rules while providing a framework for model-based improvements.

---

## Future Improvements
- Enhance Hindi accuracy with more language-specific patterns and multilingual model training.
- Add support for additional languages (e.g., Japanese) by expanding model datasets.
- Optimize runtime for larger documents by caching intermediate results and activating ML models selectively.

This approach balances accuracy, performance, and flexibility, leveraging a hybrid rule-based system with the potential for ML enhancement, making it a solid foundation for the hackathon challenge.

---

