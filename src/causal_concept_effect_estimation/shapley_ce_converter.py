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

    def convert(self):
        """Return exploded CE dataframe with one row per concept and Shapley‑attributed KL divergence."""
        df = self.ce_df.copy()

        # Parse string representations to sorted tuples of concept names
        df[self.concept_col] = df[self.concept_col].apply(self._parse_to_tuple)
        df[self.category_col] = df[self.category_col].apply(self._parse_to_tuple)

        shapley_rows = []

        for example_idx, example_df in df.groupby(self.example_col):
            for group_idx, group_df in example_df.groupby(self.group_col):
                coalition_value = {}
                category_lookup = {}

                for _, row in group_df.iterrows():
                    concepts = row[self.concept_col]
                    categories = row[self.category_col]
                    coalition_value[concepts] = row[self.value_col]

                    # Build category lookup for all concepts appearing in this group
                    for concept, cat in zip(concepts, categories):
                        category_lookup[concept] = cat

                all_concepts = set().union(*coalition_value.keys())

                # Compute Shapley values
                shapley_values = self._compute_shapley_values(
                    all_concepts, coalition_value
                )

                for concept in all_concepts:
                    category = category_lookup.get(concept)
                    shapley = shapley_values.get(concept)

                    if category is None or shapley is None:
                        raise ValueError(
                            f"Missing category or Shapley value for concept {concept!r} "
                            f"in example {example_idx}"
                        )

                    shapley_rows.append({
                        self.example_col: example_idx,
                        self.concept_col: concept,
                        self.category_col: category,
                        "shapley_kl_div": shapley
                    })

        return pd.DataFrame(shapley_rows)


    def _compute_shapley_values(
        self,
        players,
        coalition_values,
    ):

        calculator = ShapleyCombinations(players)

        possible_coalitions = {
            tuple(sorted(c))
            for c in calculator.get_all_coalitions()
        }

        existing_coalitions = set(coalition_values.keys())
        missing_coalitions = possible_coalitions - existing_coalitions

        # Two problems with current "missing coalition" handling:
        # 1. Using the mean for approximating the value is simple and non-bloated but very primitive
        # 2. If the quality of the generated counterfactuals and their response is of low quality, many coalitions need
        # to be replicated, massively reducing the accuracy of the approximation and the shapley values.
        # We should be looking to not return shapley kl values in this case.

        if missing_coalitions:

            if not self.approximate_missing_coalitions:
                raise ValueError(
                    f"Missing coalitions: {sorted(missing_coalitions)}"
                )

            print(
                f"Warning: {len(missing_coalitions)} coalition(s) missing. "
                f"Approximating values."
            )

            existing_values = list(coalition_values.values())

            for missing in missing_coalitions:
                # simplest approximation: mean of observed coalitions
                approx_value = float(np.mean(existing_values))

                coalition_values[missing] = approx_value

                print(
                    f"Approximated coalition {missing} "
                    f"with value {approx_value}"
                )

        shapley_values = calculator.calculate_shapley_values(coalition_values)
        return shapley_values

    def _parse_to_tuple(self, x):
        """Parse various input formats into a sorted tuple of strings."""
        if isinstance(x, list):
            return tuple(sorted(x))
        if pd.isna(x):
            return tuple()
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                if isinstance(parsed, list):
                    return tuple(sorted(parsed))
            except Exception:
                pass  # fall through to manual parsing

            x = x.strip()
            if x.startswith("[") and x.endswith("]"):
                x = x[1:-1]
            if not x.strip():
                return tuple()
            # Split by comma and clean each item
            items = [item.strip() for item in x.split(",") if item.strip()]
            return tuple(sorted(items))
        # If it's something else, treat as single concept
        return tuple(sorted([str(x)]))