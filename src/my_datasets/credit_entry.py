import csv
from enum import Enum
from typing import List, Optional

class Status(Enum):
    NO_CHECKING_ACCOUNT = (1, "no checking account")
    LESS_THAN_0 = (2, "balance less than 0 Euro")
    BETWEEN_0_AND_200 = (3, "balance between 0 and 200 Euro")
    AT_LEAST_200_OR_SALARY = (4, "balance at least 200 Euro or salary for at least 1 year")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class CreditHistory(Enum):
    DELAY_IN_PAYING = (0, "delay in paying off in the past")
    CRITICAL_ACCOUNT = (1, "critical account or other credits elsewhere")
    NO_CREDITS_ALL_PAID = (2, "no credits taken or all credits paid back duly")
    EXISTING_PAID_DULY = (3, "existing credits paid back duly till now")
    ALL_PAID_DULY = (4, "all credits at this bank paid back duly")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Purpose(Enum):
    OTHERS = (0, "others")
    CAR_NEW = (1, "car (new)")
    CAR_USED = (2, "car (used)")
    FURNITURE_EQUIPMENT = (3, "furniture/equipment")
    RADIO_TELEVISION = (4, "radio/television")
    DOMESTIC_APPLIANCES = (5, "domestic appliances")
    REPAIRS = (6, "repairs")
    EDUCATION = (7, "education")
    VACATION = (8, "vacation")
    RETRAINING = (9, "retraining")
    BUSINESS = (10, "business")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Savings(Enum):
    UNKNOWN_NO_SAVINGS = (1, "unknown or no savings account")
    LESS_THAN_100 = (2, "savings less than 100 Euro")
    BETWEEN_100_AND_500 = (3, "savings between 100 and 500 Euro")
    BETWEEN_500_AND_1000 = (4, "savings between 500 and 1000 Euro")
    AT_LEAST_1000 = (5, "savings at least 1000 Euro")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class EmploymentDuration(Enum):
    UNEMPLOYED = (1, "unemployed")
    LESS_THAN_1_YEAR = (2, "employed less than 1 year")
    BETWEEN_1_AND_4_YEARS = (3, "employed 1 to 4 years")
    BETWEEN_4_AND_7_YEARS = (4, "employed 4 to 7 years")
    AT_LEAST_7_YEARS = (5, "employed at least 7 years")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class InstallmentRate(Enum):
    GE_35 = (1, "installment rate >= 35%")
    BETWEEN_25_AND_35 = (2, "installment rate between 25% and 35%")
    BETWEEN_20_AND_25 = (3, "installment rate between 20% and 25%")
    LT_20 = (4, "installment rate < 20%")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class PersonalStatusSex(Enum):
    MALE_DIVORCED_SEPARATED = (1, "male: divorced or separated")
    FEMALE_NONSINGLE_OR_MALE_SINGLE = (2, "female: non‑single or male: single")
    MALE_MARRIED_WIDOWED = (3, "male: married or widowed")
    FEMALE_SINGLE = (4, "female: single")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class OtherDebtors(Enum):
    NONE = (1, "none")
    CO_APPLICANT = (2, "co‑applicant")
    GUARANTOR = (3, "guarantor")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class PresentResidence(Enum):
    LESS_THAN_1_YEAR = (1, "present residence less than 1 year")
    BETWEEN_1_AND_4_YEARS = (2, "present residence 1 to 4 years")
    BETWEEN_4_AND_7_YEARS = (3, "present residence 4 to 7 years")
    AT_LEAST_7_YEARS = (4, "present residence at least 7 years")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Property(Enum):
    UNKNOWN_NO_PROPERTY = (1, "no property or unknown")
    CAR_OR_OTHER = (2, "car or other property")
    BUILDING_SOC_SAVINGS_LIFE_INS = (3, "building society savings agreement or life insurance")
    REAL_ESTATE = (4, "real estate")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class OtherInstallmentPlans(Enum):
    BANK = (1, "bank")
    STORES = (2, "stores")
    NONE = (3, "none")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Housing(Enum):
    FOR_FREE = (1, "for free")
    RENT = (2, "rent")
    OWN = (3, "own")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class NumberCredits(Enum):
    ONE = (1, "1 credit")
    TWO_OR_THREE = (2, "2 or 3 credits")
    FOUR_OR_FIVE = (3, "4 or 5 credits")
    SIX_OR_MORE = (4, "6 or more credits")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Job(Enum):
    UNEMPLOYED_UNSKILLED_NONRES = (1, "unemployed or unskilled non‑resident")
    UNSKILLED_RESIDENT = (2, "unskilled resident")
    SKILLED_EMPLOYEE_OFFICIAL = (3, "skilled employee or official")
    MANAGER_SELF_EMPL_HIGHLY_QUAL = (4, "manager, self‑employed or highly qualified employee")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class PeopleLiable(Enum):
    THREE_OR_MORE = (1, "3 or more people liable")
    ZERO_TO_TWO = (2, "0 to 2 people liable")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class Telephone(Enum):
    NO = (1, "no telephone")
    YES = (2, "yes, under customer name")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

class ForeignWorker(Enum):
    YES = (1, "yes, foreign worker")
    NO = (2, "no, not a foreign worker")

    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description


    def __init__(self, code: int, description: str):
        self.code = code
        self.description = description

# ---------- CreditEntry class ----------
class CreditEntry:
    def __init__(self,
                 status: Status,
                 duration: int,
                 credit_history: CreditHistory,
                 purpose: Purpose,
                 amount: float,          # already in Euro
                 savings: Savings,
                 employment_duration: EmploymentDuration,
                 installment_rate: InstallmentRate,
                 personal_status_sex: PersonalStatusSex,
                 other_debtors: OtherDebtors,
                 present_residence: PresentResidence,
                 property: Property,
                 age: int,
                 other_installment_plans: OtherInstallmentPlans,
                 housing: Housing,
                 number_credits: NumberCredits,
                 job: Job,
                 people_liable: PeopleLiable,
                 telephone: Telephone,
                 foreign_worker: ForeignWorker):

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

    def _to_prompt_dict(self) -> dict:
        """Return a dictionary with plain English descriptions for all fields.
        Suitable for conversion to an LLM prompt."""
        return {
            "status": self.status.description,
            "duration_months": self.duration,
            "credit_history": self.credit_history.description,
            "purpose": self.purpose.description,
            "amount_euro": self.amount,
            "savings": self.savings.description,
            "employment_duration": self.employment_duration.description,
            "installment_rate": self.installment_rate.description,
            "personal_status_sex": self.personal_status_sex.description,
            "other_debtors": self.other_debtors.description,
            "present_residence_years": self.present_residence.description,
            "property": self.property.description,
            "age_years": self.age,
            "other_installment_plans": self.other_installment_plans.description,
            "housing": self.housing.description,
            "number_credits": self.number_credits.description,
            "job": self.job.description,
            "people_liable": self.people_liable.description,
            "telephone": self.telephone.description,
            "foreign_worker": self.foreign_worker.description,
        }

    def to_prompt_string(self) -> str:
        import pprint, re
        return re.sub(r"[']", "", "###Case Information###\n" + pprint.pformat(self._to_prompt_dict()) + "\n\n")

    def _get_concepts_and_categories(self):
        # Categories: 1. Personal Information, 2. Collateral Information, 3. Income Information, 4. Credit History, 5. Current Credit Information
        personal_information_tag = "Personal Information"
        collateral_information_tag = "Collateral Information"
        income_information_tag = "Income Information"
        credit_history_tag = "Credit History"
        current_credit_information_tag = "Current Credit Information"

        return {
            "status": collateral_information_tag,
            "duration_months": current_credit_information_tag,
            "credit_history": credit_history_tag,
            "purpose": current_credit_information_tag,
            "amount_euro": current_credit_information_tag,
            "savings": collateral_information_tag,
            "employment_duration": income_information_tag,
            "installment_rate": current_credit_information_tag,
            "personal_status_sex": personal_information_tag,
            "other_debtors": credit_history_tag,
            "present_residence_years": personal_information_tag,
            "property": collateral_information_tag,
            "age_years": current_credit_information_tag,
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

    def __getitem__(self, index):
        return self._to_prompt_dict()[index]

    def __setitem__(self, index, value):
        setattr(self, index, value)


STATUS_MAP = {e.code: e for e in Status}
CREDIT_HISTORY_MAP = {e.code: e for e in CreditHistory}
PURPOSE_MAP = {e.code: e for e in Purpose}
SAVINGS_MAP = {e.code: e for e in Savings}
EMPLOYMENT_DURATION_MAP = {e.code: e for e in EmploymentDuration}
INSTALLMENT_RATE_MAP = {e.code: e for e in InstallmentRate}
PERSONAL_STATUS_SEX_MAP = {e.code: e for e in PersonalStatusSex}
OTHER_DEBTORS_MAP = {e.code: e for e in OtherDebtors}
PRESENT_RESIDENCE_MAP = {e.code: e for e in PresentResidence}
PROPERTY_MAP = {e.code: e for e in Property}
OTHER_INSTALLMENT_PLANS_MAP = {e.code: e for e in OtherInstallmentPlans}
HOUSING_MAP = {e.code: e for e in Housing}
NUMBER_CREDITS_MAP = {e.code: e for e in NumberCredits}
JOB_MAP = {e.code: e for e in Job}
PEOPLE_LIABLE_MAP = {e.code: e for e in PeopleLiable}
TELEPHONE_MAP = {e.code: e for e in Telephone}
FOREIGN_WORKER_MAP = {e.code: e for e in ForeignWorker}