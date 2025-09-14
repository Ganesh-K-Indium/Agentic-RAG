import os
import fitz  # PyMuPDF
import csv
import re
from typing import List

def extract_tables_to_csv(pdf_path: str):
    """Extract tables from PDF with comprehensive detection and consolidation."""
    if not os.path.exists(pdf_path):
        print(f"Error: File does not exist: {pdf_path}")
        return

    try:
        print(f"Processing: {os.path.basename(pdf_path)}")
        pdf_document = fitz.open(pdf_path)
        all_tables_data = []
        table_count = 0

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_tables = []
        
            try:
                words = page.get_text("words")
                if words:
                    structured_table = extract_structured_data(words)
                    if structured_table and is_valid_table(structured_table):
                        page_tables.append(structured_table)
            except Exception as e:
                print(f"Method 2 failed on page {page_num + 1}: {e}")
            
            # Process and consolidate tables
            processed_tables = process_page_tables(page_tables)
            
            # Add tables to output with better formatting
            if processed_tables:
                all_tables_data.append([f'=== PAGE {page_num + 1} ==='])
                for i, table_data in enumerate(processed_tables):
                    if i > 0:
                        all_tables_data.append(['--- Next Table ---'])
                    
                    # Add table metadata if available
                    table_info = analyze_table_content(table_data)
                    if table_info:
                        all_tables_data.append([f"Table Type: {table_info}"])
                    
                    all_tables_data.extend(table_data)
                    table_count += 1
                    
                    # Add spacing after each table for readability
                    all_tables_data.append([''])

        # Save to CSV with better formatting
        if all_tables_data:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            csv_path = f"{base_name}_tables.csv"
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in all_tables_data:
                    writer.writerow(row)
            
            print(f"Extracted {table_count} tables → {csv_path}")
        else:
            print("No tables found")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'pdf_document' in locals():
            pdf_document.close()

def clean_table_data(data):
    """Clean and standardize table data."""
    cleaned = []
    for row in data:
        clean_row = []
        for cell in row:
            if cell is None:
                clean_cell = ""
            else:
                clean_cell = str(cell).strip()
                # Remove common PDF artifacts
                clean_cell = re.sub(r'\*\*', '', clean_cell)  # Remove ** symbols
                clean_cell = re.sub(r'\s+', ' ', clean_cell)   # Normalize whitespace
            clean_row.append(clean_cell)
        cleaned.append(clean_row)
    return cleaned

def extract_structured_data(words):
    """Extract structured data from word positions."""
    # Group words into lines
    lines = {}
    for x0, y0, x1, y1, text, *_ in words:
        y_center = (y0 + y1) / 2
        found_line = False
        for existing_y in lines:
            if abs(y_center - existing_y) <= 3:
                lines[existing_y].append({'text': text.strip(), 'x0': x0, 'x1': x1})
                found_line = True
                break
        if not found_line:
            lines[y_center] = [{'text': text.strip(), 'x0': x0, 'x1': x1}]
    
    # Sort words in each line and detect table patterns
    sorted_lines = []
    for y in sorted(lines.keys()):
        line_words = sorted(lines[y], key=lambda x: x['x0'])
        if len(line_words) >= 2:
            # Check if line has tabular spacing
            spaces = [line_words[i]['x0'] - line_words[i-1]['x1'] for i in range(1, len(line_words))]
            avg_space = sum(spaces) / len(spaces) if spaces else 0
            if avg_space > 10:  # Significant gaps suggest columns
                sorted_lines.append(line_words)
    
    # Convert to table format
    if len(sorted_lines) > 1:
        # Find column boundaries
        all_x = []
        for line in sorted_lines:
            for word in line:
                all_x.extend([word['x0'], word['x1']])
        boundaries = sorted(set(all_x))
        
        table_data = []
        for line in sorted_lines:
            row = [''] * (len(boundaries) - 1)
            for word in line:
                word_center = (word['x0'] + word['x1']) / 2
                for i in range(len(boundaries) - 1):
                    if boundaries[i] <= word_center <= boundaries[i + 1]:
                        if row[i]:
                            row[i] += ' ' + word['text']
                        else:
                            row[i] = word['text']
                        break
            if any(cell.strip() for cell in row):
                table_data.append([cell.strip() for cell in row])
        
        return clean_table_data(table_data) if table_data else None
    
    return None

def process_page_tables(page_tables):
    """Process and consolidate tables from a page."""
    if not page_tables:
        return []
    
    # Remove duplicates
    unique_tables = []
    single_row_financial = []
    
    for table in page_tables:
        # Check for duplicates
        is_duplicate = any(tables_are_similar(table, existing) for existing in unique_tables)
        if not is_duplicate:
            # Check if single-row financial data
            if is_single_row_financial(table):
                single_row_financial.append(table)
            else:
                unique_tables.append(table)
    
    # Merge single-row financial tables if they seem related
    if len(single_row_financial) > 1:
        merged = merge_financial_tables(single_row_financial)
        if merged:
            unique_tables.append(merged)
    else:
        unique_tables.extend(single_row_financial)
    
    return unique_tables

def tables_are_similar(table1, table2, threshold=0.8):
    """Check if two tables are similar (to avoid duplicates)."""
    if len(table1) != len(table2):
        return False
    
    total_cells = sum(len(row) for row in table1)
    if total_cells == 0:
        return True
    
    matching_cells = 0
    for i, row1 in enumerate(table1):
        if i < len(table2):
            row2 = table2[i]
            for j, cell1 in enumerate(row1):
                if j < len(row2) and str(cell1).strip() == str(row2[j]).strip():
                    matching_cells += 1
    
    similarity = matching_cells / total_cells
    return similarity >= threshold

def is_single_row_financial(table):
    """Check if table is a single-row financial statement."""
    if len(table) != 1:
        return False
    
    row_text = ' '.join(str(cell) for cell in table[0]).lower()
    financial_terms = ['revenue', 'income', 'earnings', 'operating', 'diluted', 'adjusted', 'total']
    has_financial = any(term in row_text for term in financial_terms)
    has_numbers = any(re.search(r'[\d,.$%]', str(cell)) for cell in table[0])
    
    return has_financial and has_numbers and len(table[0]) >= 3

def merge_financial_tables(financial_tables):
    """Merge related single-row financial tables."""
    if not financial_tables:
        return None
    
    # Simple merge - combine all rows
    merged = []
    for table in financial_tables:
        merged.extend(table)
    
    return merged if len(merged) > 1 else None

def analyze_table_content(table_data):
    """Analyze table content to determine type."""
    if not table_data:
        return "Unknown"
    
    all_text = ' '.join(' '.join(str(cell) for cell in row) for row in table_data).lower()
    
    # Common table types in financial documents
    if any(term in all_text for term in ['revenue', 'income', 'earnings']):
        return "Financial Performance"
    elif any(term in all_text for term in ['balance sheet', 'assets', 'liabilities']):
        return "Balance Sheet"
    elif any(term in all_text for term in ['cash flow', 'operating activities']):
        return "Cash Flow"
    elif any(term in all_text for term in ['owned', 'leased', 'location']):
        return "Property/Location"
    elif any(term in all_text for term in ['shares', 'stockholder', 'equity']):
        return "Equity/Shares"
    elif 'date' in all_text or re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', all_text):
        return "Time-based Data"
    else:
        return "General Data"

def is_valid_table(table_data) -> bool:
    """Validate if data is actually a table with improved logic."""
    if not table_data or len(table_data[0]) < 2:
        return False
    
    # Allow single-row financial tables
    if len(table_data) == 1:
        return is_single_row_financial(table_data)
    
    # Multi-row table validation
    column_counts = [len(row) for row in table_data]
    max_cols, min_cols = max(column_counts), min(column_counts)
    
    if max_cols > min_cols * 2:  # Too inconsistent
        return False
    
    # Check for numeric data
    total_cells = sum(len(row) for row in table_data)
    numeric_cells = sum(1 for row in table_data for cell in row 
                      if re.search(r'^\$?[\d,]+\.?\d*$|^\d+\.?\d*%?$', str(cell).strip()))
    
    # Check for table indicators
    all_text = ' '.join(' '.join(str(cell) for cell in row) for row in table_data).lower()
    table_indicators = ['total', 'amount', 'date', 'shares', 'price', 'value', 'revenue', 
                       'income', 'earnings', 'operating', 'percent', '%', 'year', 'owned', 'leased']
    has_indicators = any(indicator in all_text for indicator in table_indicators)
    
    # Reject narrative text patterns
    narrative_patterns = ['as a result of', 'we are required', 'during fiscal year', 
                         'our board of directors', 'we will continue', 'this table shows']
    has_narrative = any(pattern in all_text for pattern in narrative_patterns)
    
    if has_narrative:
        return False
    
    # Empty ratio check
    empty_cells = sum(1 for row in table_data for cell in row if not str(cell).strip())
    empty_ratio = empty_cells / total_cells if total_cells > 0 else 1
    
    # Improved scoring system
    score = 0
    if numeric_cells / total_cells > 0.15:  # 15%+ numeric
        score += 3
    if max_cols >= 3:  # Multi-column
        score += 2
    if has_indicators:  # Has table terms
        score += 2
    if len(table_data) >= 3:  # Multiple rows
        score += 1
    if empty_ratio < 0.6:  # Not too empty
        score += 1
    if max_cols >= 5:  # Many columns suggests structured data
        score += 1
    
    return score >= 4  # Slightly higher threshold

# Usage
if __name__ == "__main__":
    pdf_file_path = "10k_PDFs/META.pdf"
    extract_tables_to_csv(pdf_file_path)