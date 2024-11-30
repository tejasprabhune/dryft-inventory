from typing import Optional, Set

import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.tree import DecisionTreeClassifier

import base64
import dill
import chromadb

from ..data import DTPartDataset
from .rule import CompoundRule
from .rule_generator import RuleGenerator

class NaiveDecisionTree(nn.Module):
    """A naive decision tree model that uses sklearn's DecisionTreeClassifier."""

    def __init__(self, ckpt: Optional[str] = None) -> None:
        super().__init__()

        self.model: DecisionTreeClassifier = DecisionTreeClassifier()
        if ckpt is not None:
            self.load(ckpt)
    
    def load(self, ckpt: str) -> None:
        with open(ckpt, 'rb') as f:
            self.model = pickle.load(f)
        
    def save(self, ckpt: str) -> None:
        with open(ckpt, 'wb') as f:
            pickle.dump(self.model, f)
    
    def fit_xy(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.model.fit(x.numpy(), y.numpy())

    def fit(self, data: DTPartDataset) -> None:
        dataloader = DataLoader(data, batch_size=1000)
        data = next(iter(dataloader))

        self.fit_xy(data[0], data[1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.model.predict(x.numpy())).float()

class ElasticDecisionTree(NaiveDecisionTree):
    """
    An elastic decision tree model that uses sklearn's DecisionTreeClassifier and
    applies a set of rules to the predictions of the decision tree.
    """
    def __init__(self, 
                 rules_db_path: str = "rules_db", 
                 ckpt: Optional[str] = None, 
                 reset: bool = False,
                 verbose: bool = False) -> None:
        super().__init__(ckpt)

        self.rules: Set[CompoundRule] = set()
        self.rule_generator = RuleGenerator()
        self.client = chromadb.PersistentClient(path=rules_db_path)
        if reset:
            self.client.reset()

        self.client.get_or_create_collection(name="rules")
        self.collection = self.client.get_collection("rules")
        self.id_counter = self.collection.count()

        for rule in self.collection.get()["metadatas"]:
            rule = dill.loads(base64.b64decode(rule["rule"]))
            if verbose:
                print("Adding:", rule)
            self.add_rule(rule)
    
    def add_rule(self, rule: CompoundRule) -> bool:
        before_len = len(self.rules)
        self.rules.add(rule)
        return len(self.rules) != before_len
    
    def fit_to_feedback(self, data: DTPartDataset) -> None:
        for i in range(len(data)):
            feedback, row = data.get_feedback(i)
            if feedback is not None:
                self.add_rule_from_feedback(feedback, row)

    def add_rule_from_feedback(self, 
                               feedback: str, 
                               row: Optional[torch.Tensor], 
                               add_to_db: bool = True,
                               verbose: bool = False) -> None:
        if not self.check_rules_db(feedback):
            try:
                rule = self.rule_generator.generate_rule(feedback, row)
            except ValueError:
                return
            if self.add_rule(rule) and add_to_db:
                print("Feedback:", feedback)
                print(rule)
                print()
                rule_b64 = base64.b64encode(dill.dumps(rule)).decode()

                self.collection.add(
                    documents=[feedback],
                    metadatas=[{"rule": rule_b64}],
                    ids=[str(self.id_counter)]
                )
                self.id_counter += 1
    
    def check_rules_db(self, feedback: str) -> bool:
        result: chromadb.QueryResult = self.collection.query(
            query_texts=[feedback],
            n_results=1
        )

        if len(result["distances"][0]) == 0:
            return False

        if result["distances"][0][0] < 0.6:
            return True
        else:
            return False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tree_pred = torch.tensor(self.model.predict(x.numpy())).float()
        for rule in self.rules:
            rule.apply(x, tree_pred)
        
        return tree_pred
