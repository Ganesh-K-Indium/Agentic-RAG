import requests
import json
from dotenv import load_dotenv
import os
import datetime

# API endpoint
url = "http://127.0.0.1:8000/ask"  


payload = {
    "query": "Show the financial performance of amazon in the last 5 years",   
}

# --- Formatter function ---
def format_report(data, indent=0):
    """Convert dict/list into a human-readable CLI report string."""
    report_lines = []
    space = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                report_lines.append(f"{space}{key}:")
                report_lines.append(format_report(value, indent + 1))
            else:
                report_lines.append(f"{space}{key}: {value}")
    elif isinstance(data, list):
        for i, item in enumerate(data, start=1):
            report_lines.append(f"{space}- Item {i}:")
            report_lines.append(format_report(item, indent + 1))
    else:
        report_lines.append(f"{space}{data}")

    return "\n".join(report_lines)

# --- Main code ---
response = requests.post(url, json=payload)

if response.status_code == 200:
    data = response.json()

    # Convert JSON into a clean report
    content = format_report(data)

    folder = "responses"
    os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist

    # Generate filename with current date and time
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".md"
    filepath = os.path.join(folder, filename)

    # Prepare Markdown content
    md_content = (
        "\n" + "="*50 + "\n"
        "Final Response\n"
        + "="*50 + "\n\n"
        + content
    )

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Response saved to {filepath}")

else:
    print("Error:", response.status_code, response.text)
