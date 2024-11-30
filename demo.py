import streamlit as st

from io import StringIO
import sys
import torch
import pandas as pd
import numpy as np

from dryft.models import ElasticDecisionTree, Rule, CompoundRule, Output, Conjunction
from dryft.data.part_dataset import DTPartDataset

st.set_page_config(layout="wide")


e_model = ElasticDecisionTree(reset=False, rules_db_path="rules_db", ckpt="dryft/ckpts/naive_dt.pkl")

part_dataset = DTPartDataset("test_data_sample.csv")

if st.session_state.get("rules", None) is None:
    st.session_state.rules = e_model.rules
else:
    e_model.rules = st.session_state.rules

st.title("Dryft Elastic Decision Tree Demo")

st.sidebar.subheader("Options:")
st.sidebar.write("""
                Lead Time and Demand:\n
                    - high: 0\n
                    - low: 1\n
                    - medium: 2\n\n

                Part Type:\n
                    - assembly_group: 0\n
                    - engineering_part: 1\n
                    - purchase_part: 2\n\n

                Part Group:\n
                    - air_system: 0\n
                    - bolt: 1\n
                    - cable: 2\n
                    - panel: 3\n
                    - semiconductor: 4\n\n

                 Output:\n
                    - MTO: 0\n
                    - MTS: 1""")

top_empty = st.empty()
results_empty = st.empty()

def show_results():
    results_empty.empty()
    with results_empty.container():
        st.subheader("Results")
        rows = []
        prior_decisions = []
        after_decisions = []
        correct_decisions = []

        for row, result in part_dataset:
            correct_decisions.append(result)
            edt_input = row.unsqueeze(0)
            edt_result = e_model(edt_input)
            row = list(row)
            row.append(result)
            row.append(edt_result.numpy().item())
            rows.append(row)

            prior_decisions.append(row[-3])

            after_decisions.append(edt_result.numpy().item())

        prior_decisions = torch.tensor(prior_decisions)
        prior_decisions = prior_decisions.float()
        print(prior_decisions)

        old_stdout = sys.stdout
        st_stdout = StringIO()
        sys.stdout = st_stdout

        for i in range(len(prior_decisions)):
            rule_stdout = StringIO()
            sys.stdout = rule_stdout

            previous_len = sys.stdout.tell()

            for rule in e_model.rules:
                rule.apply(part_dataset[i][0].unsqueeze(0).numpy().tolist(), prior_decisions, verbose=True)

            if sys.stdout.tell() != previous_len:
                st_stdout.write(f"##### Row: {i}\n")
                st_stdout.write(f"Suggested Decision: {prior_decisions[i]} \n\n EDT Decision: {after_decisions[i]}\n\n")
                st_stdout.write(rule_stdout.getvalue())



        st.write(st_stdout.getvalue())

        sys.stdout = old_stdout

        rows = np.array(rows).astype(int)

        correct_decisions = np.array(correct_decisions)
        after_decisions = np.array(after_decisions)
        print(correct_decisions)
        print(after_decisions)

        accuracy = np.sum(correct_decisions == after_decisions) / len(correct_decisions)

        after_decisions = after_decisions.tolist()

        st.write(f"Accuracy: {accuracy}")

        df = pd.read_csv("test_data_sample.csv")
        for i in range(len(after_decisions)):
            if after_decisions[i] == 1:
                after_decisions[i] = "MTS"
            else:
                after_decisions[i] = "MTO"

        df["edt_decision"] = after_decisions
        df = df.style.set_table_styles([
            dict(selector="td", props=[("font-size", "70%")]), 
            dict(selector="th", props=[("font-size", "70%")])
        ])


        st.table(df)



with top_empty.container():

    left_column, right_column = st.columns(2)

    with left_column:
        rules_empty = st.empty()

    def show_rules():
        with left_column:
            rules_empty.empty()
            with rules_empty.container():
                st.subheader("Rules")

                for rule in st.session_state.rules:
                    st.code(str(rule))

    show_rules()


    with right_column:
        num_rows = st.slider('Number of rules', min_value=1, max_value=5)

        grid = st.columns(4)

        def add_rule(rule_num):
            with grid[0]:
                st.selectbox("Feature", ["material_id", 
                                            "is_sporadic", 
                                            "lead_time", 
                                            "demand", 
                                            "part_type", 
                                            "part_group", 
                                            "monthly_demand", 
                                            "name", 
                                            "supplier", 
                                            "serialnumber", 
                                            "supplier_provision", 
                                            "suggested_decision"], key=f"feature{rule_num}")
            with grid[1]:
                st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<="], key=f"operator{rule_num}")
            with grid[2]:
                st.text_input("Value", key=f"value{rule_num}")
            with grid[3]:
                if rule_num != num_rows - 1:
                    st.selectbox("Conjunction", ["AND"], key=f"conjunction{rule_num}")

        for i in range(num_rows):
            add_rule(i)

        output = st.selectbox("Output", ["MTS", "MTO"], key="output")

        if st.button("Add Explicit Compound Rule"):

            rules = []
            conjunctions = []
            for i in range(num_rows):
                feature = st.session_state[f"feature{i}"]
                operator = st.session_state[f"operator{i}"]
                value = st.session_state[f"value{i}"]
                if i != num_rows - 1:
                    conjunction = st.session_state[f"conjunction{i}"]
                    conjunctions.append(Conjunction.AND)

                rule = Rule.create_rule(feature, operator, value)
                rules.append(rule)

            output = st.session_state.output
            output = Output.from_str(output)

            compound_rule = CompoundRule(rules, conjunctions, output)

            e_model.add_rule(compound_rule)
            st.session_state.rules = e_model.rules

            show_rules()
            show_results()

        nlp_feedback = st.text_input("Natural Language Feedback")

        if st.button("Add Natural Compound Rule"):
            e_model.add_rule_from_feedback(nlp_feedback, None, add_to_db=False)
            st.session_state.rules = e_model.rules

            show_rules()
            show_results()
        
        if st.button("Reset Rules"):
            e_model = ElasticDecisionTree(reset=False, rules_db_path="rules_db", ckpt="dryft/ckpts/naive_dt.pkl")
            st.session_state.rules = e_model.rules
            st.rerun()

            print("reset")

            show_rules()

show_results()
