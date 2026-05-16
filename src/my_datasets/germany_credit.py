import csv
import os
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from my_datasets.credit_entry import CreditEntry
from my_datasets.dataset import Dataset
from my_datasets.tabular_loaded_data import TabularLoadedData


class GermanCredit(Dataset):

    def __init__(self, name, dataset_path):
        super().__init__(name, dataset_path)

    def get_intervention_generator(self, arguments):
        from intervention_generation.tabular_intervention_generator import TabularInterventionGenerator
        return TabularInterventionGenerator(arguments)

    def format_prompt_basic(self, idx, context_idx=0, double_space=True, context_ans=False):
        data = self.load_data().rows[context_idx]
        prompt_path = os.path.join(self.dataset_path, "few_shot_cot_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            return f"{prompt}\n{data.to_prompt_string()}"

    def format_prompt_qa(self, basic_prompt, prompt_strategy, idx=None):
        return basic_prompt


    def parse_counterfactual_output(self, counterfactual_output, includes_quality_checks=False):
        raise ValueError(
            "This method is only to be used with natural language counterfactual generation. This database should not use this.")

    def get_cot_answer_trigger(self, prompt, add_instr=None):
        raise NotImplementedError

    def format_question_counterfactual(self, counterfactual_dict, double_space=False):
        return self.format_prompt_qa_counterfactual(counterfactual_dict, double_space)

    def format_prompt_qa_counterfactual(self, counterfactual_dict, prompt_strategy, idx=None):
        prompt_path = os.path.join(self.dataset_path, "few_shot_cot_prompt.txt")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            counterfactual_text = str(counterfactual_dict["edited_question"])
            return  f"{prompt}\n{counterfactual_text}"

    def get_answer_choices(self):
        return ["good credit risk", "bad credit risk"]

    def extract_answer(self, response, prompt_strategy,  idx=None):
        if "Final decision: good risk".upper() in response.upper() or "Final decision: good credit risk".upper() in response.upper():
            return 1
        elif "Final decision: bad risk".upper() in response.upper() or "Final decision: bad credit risk".upper() in response.upper():
            return 0
        else:
            raise ValueError(f"Could not extract answer from response for example {idx}. Response: {response}")

    def format_prompt_implied_concepts(self, implied_concepts_base_prompt_name, concepts, concept_values, question, response, answer):
        prompt_path = os.path.join(self.dataset_path, f"{implied_concepts_base_prompt_name}.txt")
        data = self.load_data()

        concept_blocks = []
        for i, concept in enumerate(concepts):
            # go through all rows for given concept (column) and extract the possible values
            values = set()
            for row in data.rows:
                row_value = row[concept]
                values.add(row_value)
            # If there are more than 6 possible values, only choose the first 5 ones
            if len(values) > 6:
                values = list(values)[:5]
                values.append("... and more")
            concept_blocks.append(f"{i + 1}: {concept} (possible values: {', '.join(map(str, values))})\n")

        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            prompt_with_values = prompt.replace("{CONCEPT_LIST}", "".join(concept_blocks)).replace("{QUESTION}", question).replace("{ANSWER}", answer).replace("{EXPLANATION}", response)
            # remove all examples (between "######### Examples #########" and "######### End of Examples #########") if they exist in the prompt
            return prompt_with_values.split("######### Examples #########")[0] + \
                prompt_with_values.split("######### End of Examples #########")[-1]

    from typing import List, Set
    from itertools import combinations

    import pandas as pd
    from scipy.stats import chi2_contingency

    def get_correlated_concepts_sets(
            self,
            concepts,
            max_in_set
    ) -> List[Tuple[Set[str], float]]:
        """
        Returns disjoint correlated concept groups.

        Each tuple:
            (
                set_of_concepts,
                average_correlation_strength
            )

        Groups are greedily built from strongest pairwise
        correlations first, while respecting max_in_set.
        """

        significance_threshold = 0.001
        min_effect_size = 0.2

        # -----------------------------------
        # Load dataframe
        # -----------------------------------
        data = self.load_data()

        rows = [row.to_dict() for row in data.rows]

        df = pd.DataFrame(rows)
        df = df[concepts].copy()

        # -----------------------------------
        # Discretize numeric columns
        # -----------------------------------
        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):

                try:
                    df[col] = pd.qcut(
                        df[col],
                        q=min(4, df[col].nunique()),
                        duplicates="drop"
                    )

                except ValueError:
                    df[col] = df[col].astype(str)

        # -----------------------------------
        # Compute pairwise correlations
        # -----------------------------------
        pair_scores = []

        for c1, c2 in combinations(concepts, 2):

            contingency = pd.crosstab(df[c1], df[c2])

            # Skip degenerate tables
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            chi2, p, dof, expected = chi2_contingency(contingency)

            n = contingency.values.sum()

            k = min(contingency.shape)

            cramers_v = np.sqrt(chi2 / (n * (k - 1)))

            if (
                    p < significance_threshold
                    and cramers_v >= min_effect_size
            ):
                pair_scores.append(
                    (c1, c2, cramers_v)
                )

        # -----------------------------------
        # Sort strongest first
        # -----------------------------------
        pair_scores.sort(
            key=lambda x: x[2],
            reverse=True
        )

        # -----------------------------------
        # Greedy constrained merging
        # -----------------------------------
        groups = [{c} for c in concepts]

        def find_group(concept):

            for g in groups:
                if concept in g:
                    return g

            return None

        for c1, c2, strength in pair_scores:

            g1 = find_group(c1)
            g2 = find_group(c2)

            # already merged
            if g1 is g2:
                continue

            merged_size = len(g1 | g2)

            if merged_size > max_in_set:
                continue

            # merge
            new_group = g1 | g2

            groups.remove(g1)
            groups.remove(g2)

            groups.append(new_group)

        # -----------------------------------
        # Compute group strengths
        # -----------------------------------
        result = []

        for g in groups:

            if len(g) == 1:
                result.append((g, 0.0))
                continue

            strengths = []

            for c1, c2, s in pair_scores:

                if c1 in g and c2 in g:
                    strengths.append(s)

            avg_strength = (
                float(np.mean(strengths))
                if strengths
                else 0.0
            )

            result.append((g, avg_strength))

        # strongest groups first
        result.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return result

    def load_data(self):
        parsed_csv = self._parse_credit_csv()
        if len(parsed_csv) == 0:
            raise ValueError(f"Parsed CSV is empty for dataset {self.name} at path {self.dataset_path}")
        return TabularLoadedData(columns=parsed_csv[0].get_concepts_and_categories(), rows=parsed_csv)

    def _parse_credit_csv(self) -> List[CreditEntry]:
        """Read the semicolon-separated CSV (first line header) and return a list of CreditEntry objects."""
        entries = []
        full_path = os.path.join(self.dataset_path, "data.csv")
        with open(full_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                # Directly create CreditEntry with string values - no maps needed!
                entry = CreditEntry(
                    status=row['status'],
                    duration=int(row['duration']),
                    credit_history=row['credit_history'],
                    purpose=row['purpose'],
                    amount=float(row['amount']),
                    savings=row['savings'],
                    employment_duration=row['employment_duration'],
                    installment_rate=row['installment_rate'],
                    personal_status_sex=row['personal_status_sex'],
                    other_debtors=row['other_debtors'],
                    present_residence=row['present_residence'],
                    property=row['property'],
                    age=int(row['age']),
                    other_installment_plans=row['other_installment_plans'],
                    housing=row['housing'],
                    number_credits=row['number_credits'],
                    job=row['job'],
                    people_liable=row['people_liable'],
                    telephone=row['telephone'],
                    foreign_worker=row['foreign_worker'],
                )
                entries.append(entry)
        return entries