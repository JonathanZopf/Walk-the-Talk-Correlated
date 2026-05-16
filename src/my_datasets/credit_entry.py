from typing import List, Dict, Any


class CreditEntry:
    """Simple dataclass-like class for credit entries with all fields as strings/numbers."""

    def __init__(self,
                 status: str,
                 duration: int,
                 credit_history: str,
                 purpose: str,
                 amount: float):
        self.status = status
        self.duration = duration
        self.credit_history = credit_history
        self.purpose = purpose
        self.amount = amount

    def to_dict(self) -> Dict[str, Any]:
        """Return dictionary representation (already serializable since all fields are primitives)."""
        return {
            "status": self.status,
            "duration": self.duration,
            "credit_history": self.credit_history,
            "purpose": self.purpose,
            "amount": self.amount,
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
            "amount": current_credit_information_tag
        }

    def get_concepts_and_categories(self) -> List[tuple]:
        """Return a list of tuples of the form (concept, category) for all concepts in this credit entry."""
        concept_category_map = self._get_concepts_and_categories()
        return [(concept, concept_category_map[concept]) for concept in concept_category_map]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)