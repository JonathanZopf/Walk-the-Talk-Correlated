from intervention_generation.base_intervention_generator import InterventionGenerator
from my_datasets.tabular_loaded_data import TabularLoadedData

import copy
import json
import os


class TabularInterventionGenerator(InterventionGenerator):
    """Tabular implementation that borrows values from other rows, ensuring different values."""

    def identify_concepts(self):
        loaded_data = self._load_data()
        concepts = []
        categories = []
        for i in loaded_data.columns:
            concepts.append(i[0])
            categories.append(i[1])

        with open(os.path.join(self.output_dir, 'concepts.json'), 'w') as f:
            json.dump(concepts, f)
        with open(os.path.join(self.output_dir, 'categories.json'), 'w') as f:
            json.dump(categories, f)

        return concepts, categories

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

    def define_intervention_sets(self, concepts):
        """
        For tabular data, each concept can be set to a different value
        found by scanning through rows until a different value is found.
        """
        loaded_data = self._load_data()
        current_idx = self.example_idx
        current_row = loaded_data.rows[current_idx]

        concept_settings = []
        for concept in concepts:
            current_value = current_row[concept]

            try:
                # Find a different value for this concept
                different_value = self._find_different_value(concept, current_value, current_idx, loaded_data)

                concept_settings.append({
                    "concept": concept,
                    "current_setting": current_value,
                    "new_settings": [different_value],
                })
            except ValueError as e:
                # Log warning but continue - this concept cannot be intervened on
                print(f"Warning: {e}")
                concept_settings.append({
                    "concept": concept,
                    "current_setting": current_value,
                    "new_settings": [],  # No alternative available
                })

        with open(os.path.join(self.output_dir, 'concept_settings.json'), 'w') as f:
            json.dump(concept_settings, f)
        return concept_settings

    def _get_different_value_for_concept(self, concept: str, current_idx: int, loaded_data, current_row) -> any:
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

        return None, None

    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """
        Apply interventions by borrowing values from rows with different values.
        """
        loaded_data = self._load_data()
        current_idx = self.example_idx
        current_row = loaded_data.rows[current_idx]

        # Start with the current row values
        if hasattr(current_row, 'to_dict'):
            counterfactual_row = copy.deepcopy(current_row.to_dict())
        else:
            counterfactual_row = copy.deepcopy(current_row)

        # Track which rows we're borrowing from for each concept (for logging)
        borrowed_sources = {}

        # intervention_str comes from enumerate_interventions and is a string like "10-"
        for idx, val in enumerate(intervention_str):
            concept = concepts[idx]

            if val == '-':
                # removal means unknown / missing
                counterfactual_row[concept] = "UNKNOWN"
                borrowed_sources[concept] = "REMOVED"

            elif val != '0':
                # Find a different value for this concept
                source_row, different_value = self._get_different_value_for_concept(
                    concept, current_idx, loaded_data, current_row
                )

                if different_value is not None:
                    counterfactual_row[concept] = different_value
                    borrowed_sources[concept] = f"row_{self._get_row_index(source_row, loaded_data.rows)}"
                else:
                    raise ValueError(f"Error in intervention {intervention_str}, no difference value found")

        # Prepare old and new values lists
        old_values = []
        new_values = []
        for concept in concepts:
            current_val = current_row[concept]
            if hasattr(current_val, 'to_prompt_string'):
                current_val = current_val.to_prompt_string().get(concept, current_val)

            new_val = counterfactual_row.get(concept, current_val)


            old_values.append(current_val)
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
            json.dump(intervention_dict, f, indent=2, default=str)  # default=str handles non-serializable objects

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