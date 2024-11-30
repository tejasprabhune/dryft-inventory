from typing import List, Tuple, Optional
from pathlib import Path
import math

import torch
import pandas as pd

from torch.utils.data import Dataset

class DTPartDataset(Dataset):
    def __init__(self, csv_file: str|Path):
        self.data: pd.DataFrame = pd.read_csv(csv_file)

        bool_cols: List[str] = ["is_sporadic", "supplier_provision"]
        category_cols: List[str] = ["material_id",
                                    "lead_time", 
                                    "demand", 
                                    "name",
                                    "correct_decision", 
                                    "suggested_decision", 
                                    "part_type", 
                                    "part_group", 
                                    "supplier",
                                    "serialnumber"]

        for col in bool_cols:
            self.data[col] = self.data[col].astype(float)

        for col in category_cols:
            self.data[col] = pd.Categorical(self.data[col]).codes
        

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx) -> Tuple[List[float], int, str]:
        values = self.data.iloc[idx].to_dict()

        features = []
        for key, value in values.items():
            if key not in ["correct_decision", "feedback"]:
                features.append(value)

        label = values["correct_decision"]

        return torch.tensor(features), torch.tensor(label)
    
    def get_feedback(self, idx) -> Optional[str]:
        if type(feedback := self.data.iloc[idx]["feedback"]) != str:
            return None, None
        feedback = self.data.iloc[idx]["feedback"]
        other_values = self.data.iloc[idx].to_dict()
        other_values.pop("feedback")
        other_values.pop("correct_decision")
        other_values = [f"{key}: {value}" for key, value in other_values.items()]
        return feedback, other_values if type(feedback) == str else None
    
if __name__ == '__main__':
    dataset = DTPartDataset('train_data_sample.csv')
    print(dataset.data.columns)
    print(dataset[12])
