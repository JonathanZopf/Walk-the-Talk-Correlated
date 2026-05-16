from typing import List, Dict, Any


class CreditEntry:
    """Simple dataclass-like class for credit entries with all fields as strings/numbers."""

    def __init__(self,
                 status: str,
                 duration: int,
                 credit_history: str,
                 purpose: str,
                 amount: float,
                 savings: str,
                 employment_duration: str,
                 installment_rate: str,
                 personal_status_sex: str,
                 other_debtors: str,
                 present_residence: str,
                 property: str,
                 age: int,
                 other_installment_plans: str,
                 housing: str,
                 number_credits: str,
                 job: str,
                 people_liable: str,
                 telephone: str,
                 foreign_worker: str):
        self.status = status
        self.duration = duration
        self.credit_history = credit_history
        self.purpose = purpose
        self.amount = amount
        self.savings = savings
        self.employment_duration = employment_duration
        self.installment_rate = installment_rate
        self.personal_status_sex = personal_status_sex
        self.other_debtors = other_debtors
        self.present_residence = present_residence
        self.property = property
        self.age = age
        self.other_installment_plans = other_installment_plans
        self.housing = housing
        self.number_credits = number_credits
        self.job = job
        self.people_liable = people_liable
        self.telephone = telephone
        self.foreign_worker = foreign_worker

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation (already serializable since all fields are primitives)."""
        return {
            "status": self.status,
            "duration": self.duration,
            "credit_history": self.credit_history,
            "purpose": self.purpose,
            "amount": self.amount,
            "savings": self.savings,
            "employment_duration": self.employment_duration,
            "installment_rate": self.installment_rate,
            "personal_status_sex": self.personal_status_sex,
            "other_debtors": self.other_debtors,
            "present_residence": self.present_residence,
            "property": self.property,
            "age": self.age,
            "other_installment_plans": self.other_installment_plans,
            "housing": self.housing,
            "number_credits": self.number_credits,
            "job": self.job,
            "people_liable": self.people_liable,
            "telephone": self.telephone,
            "foreign_worker": self.foreign_worker,
        }

    def to_prompt_dict(self) -> dict:
        """Return dictionary with all fields (already human-readable)."""
        return self.to_dict()

    def to_prompt_string(self) -> str:
        """Convert to formatted string for LLM prompts."""
        import pprint
        import re
        return re.sub(r"[']", "", "###Case Information###\n" + pprint.pformat(self.to_dict()) + "\n\n")

    def get_concepts_and_categories(self) -> List[tuple]:
        """Return list of (concept, category) tuples."""
        categories = self._get_concept_categories()
        return [(concept, categories[concept]) for concept in categories]

    def _get_concepts_and_categories(self):
        # Categories: 1. Personal Information, 2. Collateral Information, 3. Income Information, 4. Credit History, 5. Current Credit Information
        personal_information_tag = "Personal Information"
        collateral_information_tag = "Collateral Information"
        income_information_tag = "Income Information"
        credit_history_tag = "Credit History"
        current_credit_information_tag = "Current Credit Information"

        return {
            "status": collateral_information_tag,
            "duration": current_credit_information_tag,
            "credit_history": credit_history_tag,
            "purpose": current_credit_information_tag,
            "amount": current_credit_information_tag,
            "savings": collateral_information_tag,
            "employment_duration": income_information_tag,
            "installment_rate": current_credit_information_tag,
            "personal_status_sex": personal_information_tag,
            "other_debtors": credit_history_tag,
            "present_residence": personal_information_tag,
            "property": collateral_information_tag,
            "age": current_credit_information_tag,
            "other_installment_plans": income_information_tag,
            "housing": collateral_information_tag,
            "number_credits": credit_history_tag,
            "job": income_information_tag,
            "people_liable": collateral_information_tag,
            "telephone": personal_information_tag,
            "foreign_worker": personal_information_tag,
        }

    def get_concepts_and_categories(self) -> List[tuple]:
        """Return a list of tuples of the form (concept, category) for all concepts in this credit entry."""
        concept_category_map = self._get_concepts_and_categories()
        return [(concept, concept_category_map[concept]) for concept in concept_category_map]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)