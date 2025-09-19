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
                strategy=shared.Strategy.HI_RES,
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
    """Convert HTML <table> to list of row lists with proper alignment"""
    rows = []
    if not html:
        return rows

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return rows

    for tr in table.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            colspan = int(cell.get("colspan", 1))
            text = cell.get_text(strip=True)
            row.extend([text] * colspan)
        rows.append(row)

    # normalize to max length
    max_len = max(len(r) for r in rows)
    for r in rows:
        if len(r) < max_len:
            r.extend([""] * (max_len - len(r)))

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

    # --- NEW: per-PDF folders & per-table CSVs ---
    for filename in filenames:
        pdf_base = os.path.splitext(os.path.basename(filename))[0]
        pdf_output_dir = os.path.join(output_dir, pdf_base)
        os.makedirs(pdf_output_dir, exist_ok=True)

        rows = [r for r in all_rows if r["file"] == os.path.basename(filename)]
        if not rows:
            continue

        grouped = {}
        for r in rows:
            key = (r["page_number"], r["table_index"])
            grouped.setdefault(key, []).append(r["row"])

        for (page, tbl_idx), tbl_rows in grouped.items():
            max_cols = max(len(r) for r in tbl_rows)
            csv_path = os.path.join(pdf_output_dir, f"page{page}_table{tbl_idx}.csv")

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in tbl_rows:
                    padded = row + [""] * (max_cols - len(row))
                    writer.writerow(padded)

            print(f"✅ Saved {csv_path}")

if __name__ == "__main__":
    asyncio.run(process_files_and_write_csv())
