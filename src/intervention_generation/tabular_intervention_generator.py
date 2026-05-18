from itertools import combinations
from typing import List, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from intervention_generation.base_intervention_generator import InterventionGenerator
from my_datasets.tabular_loaded_data import TabularLoadedData

import copy
import json
import os


class TabularInterventionGenerator(InterventionGenerator):
    """Tabular implementation that borrows values from other rows, ensuring different values."""

    def identify_concepts_within_correlation_groups(self, max_in_group) -> List[Set[Tuple[str, str]]]:
        loaded_data = self._load_data()
        concepts_and_categories = loaded_data.columns
        concepts = [i[0] for i in concepts_and_categories]
        categories = [i[1] for i in concepts_and_categories]

        with open(os.path.join(self.output_dir, 'concepts.json'), 'w') as f:
            json.dump(concepts, f)
        with open(os.path.join(self.output_dir, 'categories.json'), 'w') as f:
            json.dump(categories, f)

        return self._get_correlated_concepts_sets(loaded_data, concepts_and_categories, max_in_group)

    def _get_correlated_concepts_sets(
            self,
            data: TabularLoadedData,
            concepts_and_categories,
            max_in_set,
            significance_threshold=0.001,
            min_effect_size=0.3
    ) -> List[Set[Tuple[str, str]]]:
        """
        Returns disjoint correlated concept groups.

        Each tuple:
            (
                concept name,
                category name
            )

        Groups are greedily built from strongest pairwise
        correlations first, while respecting max_in_set.
        """

        # -----------------------------------
        # Load dataframe
        # -----------------------------------
        rows = [row.to_dict() for row in data.rows]

        # Extract concepts and categories
        concepts = [item[0] for item in concepts_and_categories]
        categories = [item[1] for item in concepts_and_categories]

        # Create mapping from concept to category
        concept_to_category = dict(concepts_and_categories)

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
        # Convert concepts to (concept, category) tuples
        # -----------------------------------
        result = []

        for g in groups:
            # Convert each concept in the group to a tuple with its category
            concept_set = set()
            for concept in g:
                category = concept_to_category.get(concept, "unknown")
                concept_set.add((concept, category))


            result.append(concept_set)

        # Sort by set size (largest first) for better intervention planning
        result.sort(key=lambda x: len(x), reverse=True)

        return result


    def _find_different_value(self, concept: str, current_value, current_idx: int, loaded_data) -> any:
        """
        Find a different value for the given concept by scanning through rows.
        Returns the first different value found, or raises ValueError if none exists.
        """
        total_rows = len(loaded_data.rows)

        # Scan forward from current_idx + 1, wrapping around if needed
        for offset in range(1, total_rows):
            check_idx = (current_idx + offset) % total_rows
            candidate_row = loaded_data.rows[check_idx]
            candidate_value = candidate_row[concept]

            # Convert to comparable format if they are CreditEntry objects
            if hasattr(candidate_value, 'to_dict'):
                candidate_value = candidate_value.to_dict().get(concept, candidate_value)
            if hasattr(current_value, 'to_dict'):
                current_value = current_value.to_dict().get(concept, current_value)

            if candidate_value != current_value:
                return candidate_value

        # If we've scanned all rows and found no different value
        raise ValueError(
            f"No alternative value found for concept '{concept}' with current value '{current_value}'. "
            f"All {total_rows} rows have the same value for this concept."
        )

    def define_intervention_sets(self, concept_groups):
        """
        For tabular data, each concept can be set to a different value
        found by scanning through rows until a different value is found.

        However for correlated concepts, we have to intervene on all of them together, to form 2^n counterfactuals.
        """
        loaded_data = self._load_data()
        current_idx = self.example_idx
        current_row = loaded_data.rows[current_idx]

        concept_settings = []
        for current_group in concept_groups:
            try:
                concepts_in_group = [i[0] for i in current_group]
                current_values = [current_row[concept] for concept in concepts_in_group]
                alternative_value = [
                    self._find_different_value(concepts_in_group[i], current_values[i], current_idx, loaded_data)
                    for i in range(len(concepts_in_group))
                ]

                # Construct counterfactual grid
                # e.g. (A,B), (A'B), (AB'), (A'B')
                new_settings = []
                for i in range(1, 2 ** len(concepts_in_group)):
                    intervention_str = bin(i)[2:].zfill(len(concepts_in_group))
                    new_settings_local = []
                    for idx, bit in enumerate(intervention_str):
                        if bit == '1':
                            new_settings_local.append(alternative_value[idx])
                        else:
                            new_settings_local.append(current_values[idx])
                    new_settings.append(new_settings_local)


                concept_settings.append({
                    "concepts": concepts_in_group,
                    "current_setting": current_values,
                    "new_settings": new_settings
                })
            except ValueError as e:
                # Log warning but continue - this concept cannot be intervened on
                print(f"Warning: {e}")
                concept_settings.append({

                })

        with open(os.path.join(self.output_dir, 'concept_settings.json'), 'w') as f:
            json.dump(concept_settings, f)
        return concept_settings



    def _get_different_value_for_concept(self, concept: str, current_idx: int, loaded_data, current_row) -> str:
        """
        Get a different value for the given concept by scanning rows.
        Returns the different value or None if no different value exists.
        """
        total_rows = len(loaded_data.rows)

        for offset in range(1, total_rows):
            check_idx = (current_idx + offset) % total_rows
            candidate_row = loaded_data.rows[check_idx]
            candidate_value = candidate_row[concept]

            # Handle if rows are CreditEntry objects
            if hasattr(candidate_value, 'to_dict'):
                candidate_value = candidate_value.to_dict().get(concept, candidate_value)
            current_val = current_row[concept]
            if hasattr(current_val, 'to_dict'):
                current_val = current_val.to_dict().get(concept, current_val)

            if candidate_value != current_val:
                return candidate_row, candidate_value

        raise ValueError("No different value found for concept '{}' after scanning all rows.".format(concept))

    def apply_single_intervention(self, intervention_str, all_concepts, concept_settings):
        """
        Apply one group-level intervention.
        intervention_str format: "G{group_idx}_C{combo_idx}"
        """
        # Parse the intervention string
        parts = intervention_str[1:].split('_C')
        if len(parts) != 2:
            raise ValueError(f"Unexpected intervention string format: {intervention_str}")
        group_idx = int(parts[0])
        combo_idx = int(parts[1])

        group_setting = concept_settings[group_idx]
        group_concepts = group_setting["concepts"]
        alt_values_tuple = group_setting["new_settings"][combo_idx]  # list of values

        loaded_data = self._load_data()
        current_idx = self.example_idx
        current_row = loaded_data.rows[current_idx]

        # Deep copy the current row
        if hasattr(current_row, 'to_dict'):
            counterfactual_row = copy.deepcopy(current_row.to_dict())
        else:
            counterfactual_row = copy.deepcopy(current_row)

        # Overwrite values for the concepts in this group
        for concept, new_val in zip(group_concepts, alt_values_tuple):
            counterfactual_row[concept] = new_val

        # Build old_values and new_values lists in the order of all_concepts
        old_values = []
        new_values = []
        for concept in all_concepts:
            # Original value
            orig_val = current_row[concept]
            if hasattr(orig_val, 'to_prompt_string'):
                orig_val = orig_val.to_prompt_string().get(concept, orig_val)
            elif hasattr(orig_val, 'to_dict'):
                orig_val = orig_val.to_dict().get(concept, orig_val)

            # New value from counterfactual_row (only modified if concept was in the group)
            new_val = counterfactual_row.get(concept, orig_val)
            # Ensure consistent serialization
            if hasattr(new_val, 'to_prompt_string'):
                new_val = new_val.to_prompt_string().get(concept, new_val)
            elif hasattr(new_val, 'to_dict'):
                new_val = new_val.to_dict().get(concept, new_val)

            old_values.append(orig_val)
            new_values.append(new_val)

        intervention_dict = {
            "intervention_str": intervention_str,
            "old_values": old_values,
            "new_values": new_values,
            "parsed_counterfactual": {
                "edited_question": counterfactual_row,
            }
        }

        out_path = os.path.join(self.output_dir, f'counterfactual_{intervention_str}.json')
        with open(out_path, 'w') as f:
            json.dump(intervention_dict, f, indent=2, default=str)

        return intervention_dict

    def _get_row_index(self, target_row, rows_list):
        """Helper to find the index of a row in the rows list."""
        for idx, row in enumerate(rows_list):
            if row is target_row:
                return idx
            # Try comparing dictionaries if rows are CreditEntry objects
            if hasattr(row, 'to_dict') and hasattr(target_row, 'to_dict'):
                if row.to_dict() == target_row.to_dict():
                    return idx
            elif hasattr(row, 'to_dict'):
                if row.to_dict() == target_row:
                    return idx
            elif hasattr(target_row, 'to_dict'):
                if row == target_row.to_dict():
                    return idx
            elif row == target_row:
                return idx
        return -1

    def _load_data(self):
        loaded_data = self.dataset.load_data()
        if not isinstance(loaded_data, TabularLoadedData):
            raise ValueError("Expected dataset.load_data() to return a TabularLoadedData instance.")
        return loaded_data