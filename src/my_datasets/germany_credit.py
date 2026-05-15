import csv
import os
from enum import Enum
from typing import List, Optional

from my_datasets.credit_entry import CreditEntry, STATUS_MAP, CREDIT_HISTORY_MAP, PURPOSE_MAP, SAVINGS_MAP, EMPLOYMENT_DURATION_MAP, INSTALLMENT_RATE_MAP, PERSONAL_STATUS_SEX_MAP, OTHER_DEBTORS_MAP, PRESENT_RESIDENCE_MAP, PROPERTY_MAP, OTHER_INSTALLMENT_PLANS_MAP, HOUSING_MAP, NUMBER_CREDITS_MAP, JOB_MAP, PEOPLE_LIABLE_MAP, TELEPHONE_MAP, FOREIGN_WORKER_MAP
from my_datasets.dataset import Dataset
from my_datasets.tabular_loaded_data import TabularLoadedData


class GermanCredit(Dataset):

    def __init__(self, name, dataset_path):
        super().__init__(name, dataset_path)

    def get_intervention_generator(self, arguments):
        from intervention_generation.tabular_intervention_generator import TabularInterventionGenerator
        return TabularInterventionGenerator(arguments)

    def format_prompt_basic(self, idx, context_idx=0, double_space=True, context_ans=False):
        """
        Formats a single BBQ question for the LLM in most basic format (without CoT, few shot examples, etc.).
        Args:
            idx: index of the question
            context_idx: index of the context to use (weak evidence 0 or 1)
            context_ans: whether the answer changes depending on the context
        Returns:
            prompt: a formatted prompt for the LLM
        """
        raise NotImplementedError

    def parse_counterfactual_output(self, counterfactual_output, includes_quality_checks=False):
        raise ValueError(
            "This method is only to be used with natural language counterfactual generation. This database should not use this.")

    def get_cot_answer_trigger(self, prompt, add_instr=None):
        """
        Returns the CoT answer trigger for a given question.
        Args:
            prompt: the prompt to add CoT trigger to
            add_instr: additional instructions to add to prompt
        Returns:
            cot_answer_trigger: the CoT answer trigger for the question
        """
        raise NotImplementedError

    def format_question_counterfactual(self, counterfactual_dict, double_space=True):
        raise ValueError(
            "This method is only to be used with natural language counterfactual generation. This database should not use this.")

    def get_answer_choices(self):
        """
        Returns the answer choices for a given question.
        Returns:
            answer_choices: the answer choices for the question
        """
        raise NotImplementedError

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
                # Convert fields using the maps
                entry = CreditEntry(
                    status=STATUS_MAP[int(row['status'])],
                    duration=int(row['duration']),
                    credit_history=CREDIT_HISTORY_MAP[int(row['credit_history'])],
                    purpose=PURPOSE_MAP[int(row['purpose'])],
                    amount=float(row['amount']),   # already in Euro
                    savings=SAVINGS_MAP[int(row['savings'])],
                    employment_duration=EMPLOYMENT_DURATION_MAP[int(row['employment_duration'])],
                    installment_rate=INSTALLMENT_RATE_MAP[int(row['installment_rate'])],
                    personal_status_sex=PERSONAL_STATUS_SEX_MAP[int(row['personal_status_sex'])],
                    other_debtors=OTHER_DEBTORS_MAP[int(row['other_debtors'])],
                    present_residence=PRESENT_RESIDENCE_MAP[int(row['present_residence'])],
                    property=PROPERTY_MAP[int(row['property'])],
                    age=int(row['age']),
                    other_installment_plans=OTHER_INSTALLMENT_PLANS_MAP[int(row['other_installment_plans'])],
                    housing=HOUSING_MAP[int(row['housing'])],
                    number_credits=NUMBER_CREDITS_MAP[int(row['number_credits'])],
                    job=JOB_MAP[int(row['job'])],
                    people_liable=PEOPLE_LIABLE_MAP[int(row['people_liable'])],
                    telephone=TELEPHONE_MAP[int(row['telephone'])],
                    foreign_worker=FOREIGN_WORKER_MAP[int(row['foreign_worker'])],
                )
                entries.append(entry)
        return entries