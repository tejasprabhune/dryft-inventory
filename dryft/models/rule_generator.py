from typing import List, Optional

from openai import OpenAI

from .rule import Rule, CompoundRule, Feature, Operator, Conjunction, Output

class RuleGenerator:
    """
    Generates rules based on feedback given by the user for the 
    MTO/MTS decision-making process.
    """
    def __init__(self) -> None:
        self.client = OpenAI()

        self.global_prompt = """
        You are given a dataset with the following columns:
        - material_id: integer
        - is_sporadic: 0 or 1
        - lead_time: Level
        - demand: Level
        - part_type: string
        - part_group: string
        - monthly_demand: integer
        - name: string
        - supplier: string
        - serialnumber: string
        - supplier_provision: 0 or 1
        - suggested_decision: 0 or 1

        where lead_time and demand Levels are defined as:
        - high: 0
        - low: 1
        - medium: 2

        part_type can take the following values:
        - assembly_group: 0
        - engineering_part: 1
        - purchase_part: 2

        part_group is an enum with the following values:
        - air_system: 0
        - bolt: 1
        - cable: 2
        - panel: 3
        - semiconductor: 4

        The operators you can use are:
        - EQ = '=='
        - NEQ = '!='
        - GT = '>'
        - LT = '<'
        - GTE = '>='
        - LTE = '<='

        For the prompt you receive, generate a series of rules that can be used to classify the dataset into two categories: MTO and MTS.
        Since Level is an enum, you should only use EQ and NEQ operators for `lead_time` and `demand`.

        An example of a compound rule is:
            is_sporadic EQ 1 AND lead_time LT high AND part_type EQ assembly
        where the rules are:
            is_sporadic EQ 1
            lead_time LT high
            part_type EQ assembly
        and the conjunctions you return are:
            AND AND
        
        Note that the number of conjunctions should be one less than the number of rules.
        You MUST use the following functions to generate the rules:
        - create_rule
        - create_conjunctions
        - create_suggested_decision

        Here is the row of features and the feedback you need to generate rules for:
        """

        self.conjunction_prompt = """
        You have just created a set of rules. Now, you need to specify the conjunctions between the rules.
        The conjunctions should be one less than the number of rules.
        The conjunctions you can use are:
        - and

        You MUST use the function create_conjunctions to specify the conjunctions between the rules.

        Here are the rules you have created:
        """

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_rule",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feature": {
                                "type": "string",
                                "enum": [str(feature) for feature in Feature]
                            },
                            "operator": {
                                "type": "string",
                                "enum": [str(operator.name) for operator in Operator]
                            },
                            "value": {
                                "type": "number"
                            }
                        },
                        "required": ["feature", "operator", "value"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_suggested_decision",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "decision": {
                                "type": "string",
                                "enum": [str(output) for output in Output]
                            }
                        },
                        "required": ["decision"],
                        "additionalProperties": False
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_conjunctions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "conjunctions": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["and"]
                                },
                                "description": "Conjunctions between rules. Should be one less than the number of rules."
                            }
                        },
                        "required": ["conjunctions"],
                        "additionalProperties": False
                    }
                }
            }
        ]

    def generate_rule(self, feedback: str, row: Optional[List[str]]) -> CompoundRule:
        """
        Generates a compound rule based on the feedback given by the user.

        Args:
            feedback (str): The feedback given by the user.
            row (List[str]): The row of features for which the feedback was given. Expected
                to be a list of {feature: value} string pairs.
        
        Returns:
            CompoundRule: The compound rule generated based on the feedback.
        """
        prompt = self.global_prompt + "\n" + str(row) + "\n" + feedback
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools
        )
        rules = []
        conjunctions = []
        output = Output.MTO
        for tool_call in response.choices[0].message.tool_calls:
            kwargs = eval(tool_call.function.arguments)
            if tool_call.function.name == "create_rule":
                rules.append(Rule.create_rule(**kwargs))
            elif tool_call.function.name == "create_suggested_decision":
                output = Output[kwargs["decision"]]
            elif tool_call.function.name == "create_conjunctions":
                conjunctions = kwargs["conjunctions"]

        if len(rules) != len(conjunctions) + 1:
            conjunctions = self._generate_conjunctions(rules)
            if len(rules) != len(conjunctions) + 1:
                raise ValueError("Number of conjunctions should be one less than the number of rules.")

        return CompoundRule(rules, conjunctions, output)
    
    def _generate_conjunctions(self, rules: List[Rule]) -> List[Conjunction]:
        """Backup method to generate conjunctions if the LLM initially fails to generate them correctly."""
        rule_str = " ".join([str(rule) for rule in rules])

        prompt = self.conjunction_prompt + rule_str

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools
        )

        conjunctions = []
        for tool_call in response.choices[0].message.tool_calls:
            kwargs = eval(tool_call.function.arguments)
            if tool_call.function.name == "create_conjunctions":
                conjunctions = kwargs["conjunctions"]

        return conjunctions