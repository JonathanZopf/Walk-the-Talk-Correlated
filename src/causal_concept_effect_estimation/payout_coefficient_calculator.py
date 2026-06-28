import ast
import itertools

import pandas as pd


class PayoutCoefficientCalculator:
    def __init__(self, ce_df):
        self.ce_df = ce_df.copy(deep=False)

    def parse_concepts(self, value):
        """Convert string representation of list to actual list, or return if already list."""
        if isinstance(value, str):
            return ast.literal_eval(value)
        return value

    def calculate_payout_coefficient(self):
        result_dict = {
            "treatment": [],
            "n_concepts_in_treatment": [],
            "total_axiomatic_kl_div": [],
            "payout_coefficient": []
        }
        for example in self.ce_df["example_idx"].unique():
            example_df = self.ce_df[(self.ce_df["example_idx"] == example)]
            for treatment in example_df["treatment"].unique():
                treatment_df = self.ce_df[(self.ce_df["treatment"] == treatment)]
                if treatment_df.shape[0] > 0:
                    intrv_concepts = self.parse_concepts(treatment_df["intrv_concepts"].iloc[0])
                    axiomatic_treatment_mask = example_df['intrv_concepts'].apply(
                        lambda x: len(self.parse_concepts(x)) == 1 and self.parse_concepts(x)[0] in intrv_concepts
                    )
                    axiomatic_treatment_df = example_df[axiomatic_treatment_mask]
                    assert len(axiomatic_treatment_df) == len(intrv_concepts)
                    total_axiomatic_kl_div = sum(axiomatic_treatment_df["kl_div"].values)
                    treatment_kl_div = treatment_df["kl_div"].values[0]
                    payout_coefficient = treatment_kl_div / total_axiomatic_kl_div

                    result_dict["treatment"].append(treatment)
                    result_dict["n_concepts_in_treatment"].append(len(intrv_concepts))
                    result_dict["total_axiomatic_kl_div"].append(total_axiomatic_kl_div)
                    result_dict["payout_coefficient"].append(payout_coefficient)

        return pd.DataFrame(result_dict)

    def calculate_adjusted_kl_div(self, payout_coefficient_df):
        result_dict = {
            "treatment": [],
            "adjusted_kl_div": [],
        }
        df = pd.merge(payout_coefficient_df, self.ce_df, on="treatment")
        max_length_of_concepts_in_df = self.parse_concepts(df["intrv_concepts"]).apply(len).max()
        for example_idx, example_df in df.groupby("example_idx"):
            for group_idx, group_df in example_df.groupby("intrv_str"):
                mean_payout_coefficients_masks_within_group = [self.parse_concepts(group_df["intrv_concepts"]).apply(lambda x: len(x) == l)
                                       for l in range(1, max_length_of_concepts_in_df + 1)]


                # Check that the masks total 'true' items is as long as items in the group_df row count
                assert sum(mask.sum() for mask in mean_payout_coefficients_masks_within_group) == len(group_df), \
                    "The masks do not partition all rows correctly"

                # If there is only one large coalition within an example, it will by definition use only its own interaction for the payout_coefficient.
                # This is not good. We need to borrow from 1. other groups within example and 2. from other examples, always using coalitions of the same size.
                mean_payout_coefficients_masks_within_example = [
                    (
                            self.parse_concepts(example_df["intrv_concepts"]).apply(lambda x: len(x) == l)
                            & (example_df["intrv_str"] != group_idx)
                    )
                    for l in range(1, max_length_of_concepts_in_df + 1)
                ]
                mean_payout_coefficients_mask_outside_example = [(self.parse_concepts(df["intrv_concepts"]).apply(lambda x: len(x) == l) & (df["example_idx"] != example_idx))
                                       for l in range(1, max_length_of_concepts_in_df + 1)]

                field_in_masks_within_group = sum(mask.sum() for mask in mean_payout_coefficients_masks_within_group)
                field_in_masks_within_example = sum(mask.sum() for mask in mean_payout_coefficients_masks_within_example)
                field_in_masks_outside_example = sum(mask.sum() for mask in mean_payout_coefficients_mask_outside_example)
                sum_of_masks = field_in_masks_within_group + field_in_masks_within_example + field_in_masks_outside_example
                try:
                    assert sum_of_masks== len(df), \
                        f"The masks do not partition all rows correctly. Sum of masks: {sum_of_masks}, len(df): {len(df)}"
                except AssertionError as e:
                    print(f"AssertionError: {e}")
                mean_payout_coefficients = []
                for l in range(0, max_length_of_concepts_in_df):
                    # Only take the first 5 items. In many cases just using values from within the group should suffice, so we are only borrowing if needed.
                    payout_coefficients = pd.concat(
                        [
                            group_df.loc[mean_payout_coefficients_masks_within_group[l], "payout_coefficient"],
                            example_df.loc[mean_payout_coefficients_masks_within_example[l], "payout_coefficient"],
                            df.loc[mean_payout_coefficients_mask_outside_example[l], "payout_coefficient"],
                        ],
                        ignore_index=True,
                    )
                    mean_payout_coefficients.append(payout_coefficients.head(5).mean())

                for _, row in group_df.iterrows():
                    original_kl_div = float(row["kl_div"])
                    concept_length = len(self.parse_concepts(row["intrv_concepts"]))
                    payout_coefficient = mean_payout_coefficients[concept_length - 1]
                    adjusted_kl_div = original_kl_div / payout_coefficient
                    result_dict["treatment"].append(row["treatment"])
                    result_dict["adjusted_kl_div"].append(adjusted_kl_div)

        return pd.DataFrame(result_dict)