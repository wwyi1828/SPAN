from typing import Dict

import pandas as pd
import torch

def preprocess_tsv(tsv_data, n_groups=5):
    duplicates = tsv_data.duplicated('Patient ID', keep=False)
    uniq_tsv_data = tsv_data[~duplicates].copy()
    status_mapping = {'1:DECEASED': 0, '0:LIVING': 1}
    uniq_tsv_data.loc[:, 'Overall Survival Status'] = uniq_tsv_data['Overall Survival Status'].map(status_mapping)
    uncensored_data = uniq_tsv_data[uniq_tsv_data['Overall Survival Status'] == 0]
    survival_months = uncensored_data['Overall Survival (Months)']
    discrete_labels, bins = pd.qcut(survival_months, q=n_groups, retbins=True, labels=False)
    bins[0] = uniq_tsv_data['Overall Survival (Months)'].min() - 1e-6
    bins[-1] = uniq_tsv_data['Overall Survival (Months)'].max() + 1e-6
    uniq_tsv_data.loc[:, 'Discrete Label'] = pd.cut(uniq_tsv_data['Overall Survival (Months)'], bins=bins, labels=False, include_lowest=True)
    data_dict = {}
    for idx, row in uniq_tsv_data.iterrows():
        if pd.notna(row['Overall Survival (Months)']) and row['Overall Survival (Months)'] != 0:
            key = row['Sample ID'][:15]
            censorship = torch.tensor([row['Overall Survival Status']], dtype=torch.float)
            survival_label = torch.tensor([row['Discrete Label']], dtype=torch.long)
            survival_time = row['Overall Survival (Months)']
            data_dict[key] = (censorship, survival_label, survival_time)
    return data_dict


def load_bracs_subtype_index_map(csv_path: str) -> Dict[str, int]:
    label_df = pd.read_csv(csv_path)
    type_to_index = {
        "Type_IC": 0,
        "Type_DCIS": 1,
        "Type_ADH": 2,
        "Type_FEA": 3,
        "Type_UDH": 4,
        "Type_N": 5,
        "Type_PB": 6,
    }
    mapping: Dict[str, int] = {}
    for _, row in label_df.iterrows():
        subtype = row["type"]
        if subtype not in type_to_index:
            continue
        mapping[row["filename"].replace(".svs", "")] = type_to_index[subtype]
    return mapping
