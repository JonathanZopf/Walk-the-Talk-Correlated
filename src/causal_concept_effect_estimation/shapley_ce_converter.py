import ast
import math
from itertools import combinations
from collections import defaultdict
from shapley_value import ShapleyCombinations

import numpy as np
import pandas as pd
class ShapleyCEConverter:
    """
    Converts a CE dataframe with multi-concept interventions into a
    concept-level dataframe containing Shapley-attributed KL divergence.

    Output:
        one row per (example_idx, concept)

    Adds:
        shapley_kl_div
    """

    def __init__(
        self,
        ce_df,
        concept_col="intrv_concepts",
        category_col="intrv_categories",
        value_col="kl_div",
        example_col="example_idx",
        group_col = "intrv_str",
        approximate_missing_coalitions = True
    ):
        self.ce_df = ce_df.copy()

        self.concept_col = concept_col
        self.category_col = category_col
        self.value_col = value_col
        self.example_col = example_col
        self.group_col = group_col
        self.approximate_missing_coalitions = approximate_missing_coalitions

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def convert(self):
        """
        Returns exploded CE dataframe with one row per concept and
        Shapley-attributed KL divergence.
        """

        df = self.ce_df.copy()

        # --------------------------------------
        # Parse list columns if stored as strings
        # --------------------------------------

        df[self.concept_col] = df[self.concept_col].apply(
            self._parse_if_needed
        )

        df[self.category_col] = df[self.category_col].apply(
            self._parse_if_needed
        )

        shapley_rows = []

        for example_idx, example_df in df.groupby(self.example_col):

            for group_idx, group_df in example_df.groupby(self.group_col):
                coalition_value_map = {}

                category_lookup = {}

                # --------------------------
                # Build coalition -> value
                # --------------------------

                for _, row in group_df.iterrows():

                    coalition = frozenset(row[self.concept_col])
                    value = row[self.value_col]
                    coalition_value_map[coalition] = row[self.value_col]

                    for concept, category in zip(
                        row[self.concept_col],
                        row[self.category_col]
                    ):
                        category_lookup[concept] = category

                all_concepts = set()

                for coalition in coalition_value_map:
                    all_concepts.update(coalition)

                try:
                    shapley_values = self._compute_shapley_values(
                        all_concepts,
                        coalition_value_map
                    )
                except:
                    raise Exception(f"Could not compute shapley values for example {example_idx}, coalition {coalition_value_map} and players {all_concepts}")

                for concept in all_concepts:
                    category = category_lookup.get(concept)
                    shapley = shapley_values.get(concept)
                    if category is None or shapley is None:
                        raise ValueError(f"Missing category or shapley value for concept \"{concept}\" in example {example_idx}")

                    shapley_rows.append(
                        {
                            self.example_col: example_idx,
                            self.concept_col: concept,
                            self.category_col: category,
                            "shapley_kl_div": shapley
                        }
                    )

        shapley_df = pd.DataFrame(shapley_rows)
        return shapley_df

    # --------------------------------------------------
    # Shapley computation
    # --------------------------------------------------
    def _convert_coalition_map(self, coalition_value_map):
        converted = {
            tuple(sorted(coalition)): value
            for coalition, value in coalition_value_map.items()
        }
        return converted

    def _compute_shapley_values(
        self,
        players,
        coalition_value_map,
    ):

        calculator = ShapleyCombinations(players)

        coalitions = self._convert_coalition_map(coalition_value_map)
        possible_coalitions = {
            tuple(sorted(c))
            for c in calculator.get_all_coalitions()
        }

        existing_coalitions = set(coalitions.keys())
        missing_coalitions = possible_coalitions - existing_coalitions

        if missing_coalitions:

            if not self.approximate_missing_coalitions:
                raise ValueError(
                    f"Missing coalitions: {sorted(missing_coalitions)}"
                )

            print(
                f"Warning: {len(missing_coalitions)} coalition(s) missing. "
                f"Approximating values."
            )

            existing_values = list(coalitions.values())

            for missing in missing_coalitions:
                # simplest approximation: mean of observed coalitions
                approx_value = float(np.mean(existing_values))

                coalitions[missing] = approx_value

                print(
                    f"Approximated coalition {missing} "
                    f"with value {approx_value}"
                )

        shapley_values = calculator.calculate_shapley_values(coalitions)
        return shapley_values

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _parse_if_needed(self, x):

        if isinstance(x, list):
            return x

        if pd.isna(x):
            return []

        if isinstance(x, str):

            try:
                return ast.literal_eval(x)

            except Exception:

                x = x.strip()

                if (
                    x.startswith("[")
                    and x.endswith("]")
                ):
                    x = x[1:-1]

                if len(x) == 0:
                    return []

                return [
                    item.strip()
                    for item in x.split(",")
                ]

        return [x]