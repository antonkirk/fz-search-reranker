import pandas as pd


cui_mapping = {
    'Spondyloarthritis': 'c0012940',
    "Behcet's disease": 'c0004943',
    'SAPHO Syndrome': 'c0263859',
    'Systemic Lupus Erythematosus (SLE)' : 'c1866373',
    'Henoch-Schonlein Purpura (HSP)': 'c0034152',
    'Focal segmental glomerulosclerosis (FSGS)': 'c0017668',
    'Mixed Amyloidosis': 'c0002726',
    "Sjogren's syndrome": 'c1527336',
    'Chronic Hepatitis C': 'c0019196',
    'TNF receptor associated periodic syndrome (TRAPS)': 'c1275126',
    'Cryopyrin-associated periodic syndrome (CAPS)': 'c2316212',
    "Systemic Lupus Erythematosus (SLE)": 'c0024141',
    'Thrombangiitis obliterans': 'c0040021',
    'Panarteritis nodosa': 'c0006272',
    'Giant Cell Arteritis': 'c1956391',
    "Takayasu's arteritis": 'c0035615',
    "Chronic Polyarthritis": 'c0162323',
    'Antisynthetase Syndrome': 'c2609059', 
    'Familial Mediterranean Fever (FMF)': 'c0031069',
    'Stickler Syndrome': 'c0265253',
    'Eosinophilic Granulomatosis with Polyangiitis (EGPA)': 'c0008728',
    'Granulomatosis with Polyangiitis (GPA)': 'c3495801',
    'IgG4-related disease': 'c3203653',
    'CREST syndrome': 'c0206138',
    'Renal Cell Carcinoma': 'c0007134',
    'Whipple disease': 'c2930851',
    'Sarcoidosis': 'c0036202',
    'Gout arthritis': 'c0018099',
    'Kimura Disease': 'c0033838',
    'Hypophosphatasia': 'c0020630',
    'Mixed connective tissue disease (MCTD)': 'c0026272',
    'Fabry disease': 'c0002986',
    'Cryoglobulinemia': 'c0010403',
    'Systemic sclerosis (renal crisis)': 'c0036421',
    'Polymyositis': 'c0085655',
    'scleroderma overlap': 'c0011644',
    'Antiphospholipid Syndrome (APS)': 'c0085278',
    'Primary sclerosing cholangitis': 'c0566602',
    'Small Fiber Neuropathy ENaC': 'c0014805',
    'CFTR channelopathy': 'c1720983',
    'Tubulointerstitial nephritis and uveitis syndrome (TINU)': 'c1843273',
    'Relapsing Polychondritis': 'c0032453',
    "Ankylosing spondylitis (Spondyloarthritis)": 'c0038013',
    "Psoriatic arthritis (Spondyloarthritis)": 'c0003872',
    "Thrombotic thrombocytopenic purpura (TTP)": 'c0034155',
    "Retroperitoneal fibrosis": 'c0035357',
}


def clean_dataset(file_path):
    # Load the Excel file
    excel_data = pd.read_excel(file_path)

    # Initialize a list to store the final combined rows
    combined_rows = []
    current_combination = {}

    # Iterate through each row
    for _, row in excel_data.iterrows():
        if row['Visit'] == "Diagnosis":
            # If we encounter a new "Diagnosis" row, save the current combination and start a new one
            if current_combination:
                combined_rows.append(current_combination)
            current_combination = row.dropna().to_dict()  # Start a new combination
        else:
            # Combine non-null fields into the current combination
            for column in excel_data.columns:
                if pd.notnull(row[column]):
                    if column in current_combination:
                        current_combination[column] = f"{current_combination[column]}, {row[column]}"
                    else:
                        current_combination[column] = row[column]

    # Append the last combination after the loop ends
    if current_combination:
        combined_rows.append(current_combination)

    # Convert the list of dictionaries back to a DataFrame
    combined_data = pd.DataFrame(combined_rows)

    # Drop the "Visit" column
    combined_data = combined_data.drop(columns=["Visit"])

    # Rename columns - lowercase, replace spaces with underscores, and remove suffix starting with \/findings
    combined_data.columns = combined_data.columns.str.lower()
    combined_data.columns = combined_data.columns.str.replace(" ", "_")
    combined_data.columns = combined_data.columns.str.replace(r"\/findings.*", "", regex=True)

    # 

    # Add new column 'cui' based on the 'confirmed_diagnoses' column. 
    # confirmed_diagnoses may contain more than one diagnosis, 
    # so we have to split them and map them to their corresponding CUIs
    combined_data["cui"] = combined_data["confirmed_diagnoses"].apply(
        lambda x: ", ".join([cui_mapping.get(diagnosis, "Unknown CUI") for diagnosis in x.split(", ")])
    )

    combined_data = combined_data.iloc[:, [0,1,7,2,3,4,5,6]]

    return combined_data


if __name__ == '__main__':
    # Define the file path
    cleaned_dataset = clean_dataset("data/raw/rare_disease_eval.xlsx")
    # Print or return the final DataFrame
    print(cleaned_dataset)
    print(set(cleaned_dataset["confirmed_diagnoses"]))
    print(cleaned_dataset["cui"])
    # Save the cleaned dataset to as jsonl file
    cleaned_dataset.to_json("data/processed/rare_disease_eval.jsonl", orient="records", lines=True)
