from __future__ import annotations
from dataclasses import dataclass
from typing import List
from enum import Enum

import torch

class Operator(Enum):
    EQ = '=='
    NEQ = '!='
    GT = '>'
    LT = '<'
    GTE = '>='
    LTE = '<='

    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.__str__()

    def from_value(value: str) -> Operator:
        if value == '==':
            return Operator.EQ
        elif value == '!=':
            return Operator.NEQ
        elif value == '>':
            return Operator.GT
        elif value == '<':
            return Operator.LT
        elif value == '>=':
            return Operator.GTE
        elif value == '<=':
            return Operator.LTE

class Conjunction(Enum):
    AND = 'and'
    OR = 'or'
    BLANK = ''

    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return self.__str__()

class Feature(Enum):
    material_id = 0
    is_sporadic = 1
    lead_time = 2
    demand = 3
    part_type = 4
    part_group = 5
    monthly_demand = 6
    name = 7
    supplier = 8
    serialnumber = 9
    supplier_provision = 10
    suggested_decision = 11

    def __str__(self) -> str:
        return self.name

class Output(Enum):
    MTS = 1
    MTO = 0

    def __str__(self) -> str:
        return self.name
    
    def from_str(value: str) -> Output:
        if value == 'MTS':
            return Output.MTS
        elif value == 'MTO':
            return Output.MTO

class Level(Enum):
    high = 0
    low = 1
    medium = 2

@dataclass
class Rule:
    feature: Feature
    operator: Operator
    value: float

    def __str__(self) -> str:
        return f"{self.feature} {self.operator} {self.value}"

    def __repr__(self) -> str:
        return self.__str__()
    
    def create_rule(feature: str, operator: str, value: float | Level) -> Rule:
        feature = Feature[feature]
        try:
            operator = Operator[operator]
        except KeyError:
            operator = Operator.from_value(operator)

        if isinstance(value, Level):
            value = value.value

        return Rule(feature, operator, value)

@dataclass
class CompoundRule:
    rules: List[Rule]
    conjunctions: List[Conjunction]
    output: Output

    def __str__(self) -> str:
        conjunctions = self.conjunctions + [Conjunction.BLANK]
        output_str = ""
        for rule, conj in zip(self.rules, conjunctions):
            if conj == Conjunction.BLANK:
                output_str += f"{rule} "
            else:
                output_str += f"{rule} {conj} "

        return f"{output_str}=> {self.output}"
    
    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.__str__())

    def create_rule(rules: List[Rule], conjunctions: List[Conjunction], output: str) -> CompoundRule:
        output = Output[output]
        return CompoundRule(rules, conjunctions, output)
    
    def apply(self, x: torch.Tensor, pred: torch.Tensor, verbose: bool = False) -> None:
        for i in range(len(x)):
            if self.eval_rule(x[i]):
                if verbose:
                    print(f"`{self}` applied to `{x[i]}`\n")
                    print(f"Output: {self.output.value}\n")
                pred[i] = self.output.value
    
    def eval_rule(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.rules) == 0:
            return torch.tensor(True)
        
        conjunctions = self.conjunctions + [Conjunction.BLANK]

        command = ""
        for rule, conj in zip(self.rules, conjunctions):
            command += f"x[{rule.feature.value}] {rule.operator} {rule.value} {conj} "
        
        return eval(command)
