import pandas as pd
import glob
import os

def csvs_to_excel(input_folder, output_excel):
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))

    if not csv_files:
        print("No CSV files found in the given folder.")
        return

    # Create a Pandas Excel writer
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for csv_file in csv_files:
            try:
                # Read CSV into DataFrame
                df = pd.read_csv(csv_file)

                # Create a sheet name from file name (without extension)
                sheet_name = os.path.splitext(os.path.basename(csv_file))[0]

                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)  # Excel sheet names max 31 chars
                print(f"Added {sheet_name} to {output_excel}")

            except Exception as e:
                print(f"Error processing {csv_file}: {e}")

    print(f"✅ All CSVs have been combined into {output_excel}")

# Example usage
csvs_to_excel("/Users/I8798/Desktop/Agentic-RAG/output/META_Shrinked", "combined_output.xlsx")
