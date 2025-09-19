# file: extract_tables_to_real_csv.py
import asyncio
import os
import csv
from bs4 import BeautifulSoup
import unstructured_client
from unstructured_client.models import shared, errors, operations
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

client = unstructured_client.UnstructuredClient(
    api_key_auth=API_KEY
)

async def partition_file_via_api(filename):
    try:
        with open(filename, "rb") as f:
            files = shared.Files(
                content=f.read(),
                file_name=os.path.basename(filename),
            )

        req = operations.PartitionRequest(
            partition_parameters=shared.PartitionParameters(
                files=files,
                strategy=shared.Strategy.HI_RES,   # better for tables
                infer_table_structure=True,
                split_pdf_page=True,
                split_pdf_allow_failed=True,
                split_pdf_concurrency_level=15,
            )
        )

        res = await client.general.partition_async(request=req)
        return res.elements

    except errors.UnstructuredClientError as e:
        print(f"Error partitioning {filename}: {e.message}")
        return []

def html_table_to_rows(html: str):
    """Convert HTML <table> to list of row lists"""
    rows = []
    if not html:
        return rows
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    return rows

async def extract_tables_from_file(filename):
    elements = await partition_file_via_api(filename)

    all_rows = []
    table_index = 1

    for el in elements or []:
        if el.get("type", "").lower() == "table" or el.get("category", "").lower() == "table":
            metadata = el.get("metadata", {})
            html = metadata.get("text_as_html")
            rows = html_table_to_rows(html)
            for row in rows:
                all_rows.append({
                    "file": os.path.basename(filename),
                    "page_number": metadata.get("page_number"),
                    "table_index": table_index,
                    "row": row
                })
            table_index += 1
    return all_rows

def load_filenames_in_directory(input_dir):
    filenames = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.json'):
                filenames.append(os.path.join(root, file))
    return filenames

async def process_files_and_write_csv():
    input_dir = "/Users/I8798/Desktop/Agentic-RAG/data"
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)

    filenames = load_filenames_in_directory(input_dir)

    all_rows = []
    for filename in filenames:
        rows = await extract_tables_from_file(filename)
        all_rows.extend(rows)

    if not all_rows:
        print("No tables extracted.")
        return

    # find max row length for consistent CSV columns
    max_cols = max(len(r["row"]) for r in all_rows)

    csv_file = os.path.join(output_dir, "all_tables.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "page_number", "table_index"] + [f"col_{i+1}" for i in range(max_cols)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_rows:
            row_dict = {
                "file": r["file"],
                "page_number": r["page_number"],
                "table_index": r["table_index"],
            }
            for i, val in enumerate(r["row"]):
                row_dict[f"col_{i+1}"] = val
            writer.writerow(row_dict)

    print(f"Extracted {len(all_rows)} table rows from {len(filenames)} PDFs -> {csv_file}")

if __name__ == "__main__":
    asyncio.run(process_files_and_write_csv())
