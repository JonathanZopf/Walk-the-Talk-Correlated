import copy
import json
import os
import traceback
from IPython import embed

from intervention_generation.base_intervention_generator import InterventionGenerator
from utils import parse_llm_response_concepts_and_categories, parse_llm_response_factor_settings


class NLInterventionGenerator(InterventionGenerator):
    """Original natural‑language logic – all behavior unchanged."""

    def identify_concepts(self):
        """
             Identifies concepts to test for a given question.
             Args:
                 dataset: dataset object
             Returns:
                 concepts: a parsed list of concepts identified by the LLM
                 categories: a parsed list of categories associated with each concept
             """
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concepts.json')):
            print("Found existing concepts.json. Skipping concept identification...")
            with open(os.path.join(self.output_dir, 'concepts.json'), 'r') as f:
                concepts = json.load(f)
            with open(os.path.join(self.output_dir, 'categories.json'), 'r') as f:
                categories = json.load(f)
            return concepts, categories

        prompt = self.dataset.format_prompt_concept_id(self.example_idx,
                                                       self.concept_id_base_prompt_name)
        response = self.intervention_model.generate_response(prompt)[0]
        try:
            concepts, categories = parse_llm_response_concepts_and_categories(response)
            with open(os.path.join(self.output_dir, 'concepts.json'), 'w') as f:
                json.dump(concepts, f)
            with open(os.path.join(self.output_dir, 'categories.json'), 'w') as f:
                json.dump(categories, f)
            return concepts, categories
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Concept identification failed: {e}")

    def define_intervention_sets(self, concepts):
        """
        Defines intervention sets for a given question (i.e., the values to set when intervening on a concept).
        Args:
            concepts: list of the concepts identified by the LLM
        Returns:
            concept_settings: a list of settings for each concept
        """
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concept_settings.json')):
            print("Found existing concept_settings.json. Skipping...")
            with open(os.path.join(self.output_dir, 'concept_settings.json'), 'r') as f:
                return json.load(f)

        prompt = self.dataset.format_prompt_concept_values(self.example_idx,
                                                           self.concept_values_base_prompt_name,
                                                           concepts)
        response = self.intervention_model.generate_response(prompt)[0]
        try:
            concept_settings = parse_llm_response_factor_settings(response)
            with open(os.path.join(self.output_dir, 'concept_settings.json'), 'w') as f:
                json.dump(concept_settings, f)
            return concept_settings
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Concept settings identification failed: {e}")

    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """
           Apply interventions for a given example.
           Args:
               concepts: the concepts to intervene on
               concept_settings: the settings for each factor
           """
        old_values = [x["current_setting"] for x in concept_settings]
        new_values = copy.deepcopy(old_values)
        for idx, val in enumerate(intervention_str):
            if val == '-':
                new_values[idx] = "UNKNOWN"
            else:
                val_int = int(val)
                if val_int:
                    new_values[idx] = concept_settings[idx]["new_settings"][val_int - 1]

        intervention_bool = [c != "0" for c in intervention_str]
        counterfactual_gen_prompt = self.dataset.format_prompt_counterfactual_gen(
            self.example_idx,
            self.counterfactual_gen_base_prompt_name,
            concepts, intervention_bool,
            new_values,
            old_values
        )
        counterfactual = self.intervention_model.generate_response(counterfactual_gen_prompt)[0].strip()

        try:
            parsed_counterfactual = self.dataset.parse_counterfactual_output(counterfactual)
        except Exception as e:
            embed()
            print(traceback.format_exc())
            raise Exception(f"Parsing failed for counterfactual {counterfactual}. Error: {e}")

        intervention_dict = {
            "intervention_str": intervention_str,
            "old_values": old_values,
            "new_values": new_values,
            "counterfactual": counterfactual,
            "counterfactual_gen_prompt": counterfactual_gen_prompt,
            "parsed_counterfactual": parsed_counterfactual
        }
        out_path = os.path.join(self.output_dir, f'counterfactual_{intervention_str}.json')
        with open(out_path, 'w') as f:
            json.dump(intervention_dict, f)
        return intervention_dict