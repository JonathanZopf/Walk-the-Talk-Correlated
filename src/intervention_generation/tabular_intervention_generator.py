from intervention_generation.base_intervention_generator import InterventionGenerator
from my_datasets.tabular_loaded_data import TabularLoadedData

import copy
import json
import os


class TabularInterventionGenerator(InterventionGenerator):
    """Tabular implementation that borrows values from the next row."""

    def identify_concepts(self):
        loaded_data = self._load_data()
        concepts = []
        categories = []
        for i in loaded_data.columns:
            concepts.append(i[0])
            categories.append(i[1])
        return concepts, categories

    def define_intervention_sets(self, concepts):
        """
        For tabular data, each concept can simply be set to the value
        borrowed from the next row. This method returns a structure
        compatible with the base intervention workflow.
        """
        loaded_data = self._load_data()

        current_row = loaded_data.rows[self.example_idx]
        next_row = loaded_data.rows[(self.example_idx + 1) % len(loaded_data.rows)]

        concept_settings = []
        for concept in concepts:
            current_value = current_row[concept]
            borrowed_value = next_row[concept]

            # In this design, the only alternative setting is the value from the next row.
            # The base class will append "UNKNOWN" unless only_concept_removals=True.
            concept_settings.append({
                "concept": concept,
                "current_setting": current_value,
                "new_settings": [borrowed_value],
            })

        return concept_settings

    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """
        Apply interventions by borrowing values from the next row.
        """
        loaded_data = self._load_data()
        current_row = loaded_data.rows[self.example_idx]
        next_row = loaded_data.rows[(self.example_idx + 1) % len(loaded_data.rows)]

        # Start with the current row values
        counterfactual_row = copy.deepcopy(current_row)

        # intervention_str comes from enumerate_interventions and is a string like "10-"
        for idx, val in enumerate(intervention_str):
            if val == '-':
                # removal means unknown / missing
                counterfactual_row[concepts[idx]] = "UNKNOWN"
            elif val != '0':
                # any positive setting means "borrow from next row"
                counterfactual_row[concepts[idx]] = next_row[concepts[idx]]

        intervention_dict = {
            "intervention_str": intervention_str,
            "old_values": [current_row[c] for c in concepts],
            "new_values": [counterfactual_row[c] for c in concepts],
            "counterfactual": counterfactual_row,
        }

        out_path = os.path.join(self.output_dir, f'counterfactual_{intervention_str}.json')
        with open(out_path, 'w') as f:
            json.dump(intervention_dict, f)

        return intervention_dict

    def _load_data(self):
        loaded_data = self.dataset.load_data()
        if not isinstance(loaded_data, TabularLoadedData):
            raise ValueError("Expected dataset.load_data() to return a TabularLoadedData instance.")
        return loaded_data