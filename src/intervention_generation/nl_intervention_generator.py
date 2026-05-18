import copy
import json
import os
import traceback
import random
from typing import List, Set, Tuple

from IPython import embed

from intervention_generation.base_intervention_generator import InterventionGenerator
from utils import parse_llm_response_concepts_and_categories, parse_llm_response_factor_settings


class NLInterventionGenerator(InterventionGenerator):

    def identify_concepts_within_correlation_groups(self, max_in_group) -> List[Set[Tuple[str, str]]]:
        """
        Identify concepts and (for now) put each into its own singleton group.
        Returns a list of sets of (concept, category) tuples.
        """
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concepts.json')):
            print("Found existing concepts.json. Skipping concept identification...")
            with open(os.path.join(self.output_dir, 'concepts.json'), 'r') as f:
                concepts = json.load(f)
            with open(os.path.join(self.output_dir, 'categories.json'), 'r') as f:
                categories = json.load(f)
        else:
            prompt = self.dataset.format_prompt_concept_id(self.example_idx,
                                                           self.concept_id_base_prompt_name)
            response = self.intervention_model.generate_response(prompt)[0]
            try:
                concepts, categories = parse_llm_response_concepts_and_categories(response)
            except Exception as e:
                print(traceback.format_exc())
                raise Exception(f"Concept identification failed: {e}")

            # Save the raw lists for later reuse
            with open(os.path.join(self.output_dir, 'concepts.json'), 'w') as f:
                json.dump(concepts, f)
            with open(os.path.join(self.output_dir, 'categories.json'), 'w') as f:
                json.dump(categories, f)

        # Create groups randomly. Set group size between 1 (singletons) and
        # max_in_group (inclusive). Partition the list of (concept,category)
        # pairs into consecutive groups of random sizes so order is preserved
        # (order matters for LLM prompts later).
        max_in_group = max(1, int(max_in_group))
        random_group_size = 3
        print("Random group size: ", random_group_size)


        # Random mechanism for verification, remove later
        pairs = list(zip(concepts, categories))
        groups = []
        i = 0
        while i < len(pairs):
            remaining = len(pairs) - i
            size = random.randint(1, random_group_size)
            if size > remaining:
                size = remaining
            group = set()
            for concept, category in pairs[i:i+size]:
                group.add((concept, category))
            groups.append(group)
            i += size

        return groups

    def define_intervention_sets(self, concept_groups: List[Set[Tuple[str, str]]]):
        """
        Given correlated groups of (concept, category) tuples,
        obtain alternative values for all concepts (using the LLM)
        and build group‑level intervention settings.

        Returns a list of dicts:
          [ { "concepts": [...], "current_setting": [...], "new_settings": [[...], ...] }, ... ]
        """
        # Try to reuse existing settings if they are in the new group format
        if self.restart_from_previous and os.path.exists(os.path.join(self.output_dir, 'concept_settings.json')):
            with open(os.path.join(self.output_dir, 'concept_settings.json'), 'r') as f:
                loaded = json.load(f)
            # Simple check: first item should have a "concepts" key
            if isinstance(loaded, list) and len(loaded) > 0 and "concepts" in loaded[0]:
                print("Found existing group-format concept_settings.json. Skipping...")
                return loaded

        # -----------------------------------------------------------------
        # 1. Flatten all concepts (order matters for the LLM prompt)
        # -----------------------------------------------------------------
        all_concepts = []
        for group in concept_groups:
            for concept, category in group:
                if concept not in all_concepts:
                    all_concepts.append(concept)

        # -----------------------------------------------------------------
        # 2. Get per‑concept current and alternative values from the LLM
        # -----------------------------------------------------------------
        prompt = self.dataset.format_prompt_concept_values(self.example_idx,
                                                           self.concept_values_base_prompt_name,
                                                           all_concepts)
        response = self.intervention_model.generate_response(prompt)[0]
        try:
            per_concept_settings = parse_llm_response_factor_settings(response)
            # per_concept_settings is a list of dicts:
            #   [{"current_setting": ..., "new_settings": [...]}, ...]
            # in the same order as all_concepts
        except Exception as e:
            print(traceback.format_exc())
            raise Exception(f"Concept settings identification failed: {e}")

        # Build a dictionary: concept -> its settings dict
        concept_setting_map = dict(zip(all_concepts, per_concept_settings))

        # -----------------------------------------------------------------
        # 3. Build group‑level intervention sets
        # -----------------------------------------------------------------
        group_settings_list = []

        for group in concept_groups:
            # Extract concept names in the group (order arbitrarily, but consistent)
            group_concepts = [concept for concept, _ in group]  # category unused here

            # Current values for these concepts
            group_current = []
            group_alternatives = []  # list of lists: alternatives for each concept in order

            for concept in group_concepts:
                setting = concept_setting_map[concept]
                group_current.append(setting["current_setting"])
                group_alternatives.append(setting["new_settings"])

            # Generate all combinations of alternative values (skip the all‑original one)
            # For concept i, the possible choices are:
            #   0 = keep current (but we'll later exclude the combination where all are 0)
            #   For each alternative, index 1..len(alternatives)
            # We'll create a list of combinations as tuples, each being a list of values
            new_combos = []
            num_concepts = len(group_concepts)
            # Iterate over all possible assignments (0=current, 1..N = alternative)
            from itertools import product
            # ranges: for each concept, 0..len(alternatives[i])
            ranges = [range(len(alts) + 1) for alts in group_alternatives]
            # exclude the all‑0 combination
            for assignment in product(*ranges):
                if all(a == 0 for a in assignment):
                    continue  # this is the original (no intervention)
                combo = []
                for i, choice_idx in enumerate(assignment):
                    if choice_idx == 0:
                        combo.append(group_current[i])
                    else:
                        combo.append(group_alternatives[i][choice_idx - 1])
                new_combos.append(combo)

            group_settings_list.append({
                "concepts": group_concepts,
                "current_setting": group_current,
                "new_settings": new_combos
            })

        # Save for future restarts
        with open(os.path.join(self.output_dir, 'concept_settings.json'), 'w') as f:
            json.dump(group_settings_list, f, indent=2)

        return group_settings_list

    def apply_single_intervention(self, intervention_str, all_concepts, concept_settings):
        """
        Apply one group‑level intervention.
        intervention_str format: "G{group_idx}_C{combo_idx}"
        """
        parts = intervention_str[1:].split('_C')
        if len(parts) != 2:
            raise ValueError(f"Unexpected intervention string format: {intervention_str}")
        group_idx = int(parts[0])
        combo_idx = int(parts[1])

        group_setting = concept_settings[group_idx]
        group_concepts = group_setting["concepts"]
        alt_values_tuple = group_setting["new_settings"][combo_idx]  # list of values

        # -----------------------------------------------------------------
        # Build a map concept -> current value from all groups
        # -----------------------------------------------------------------
        concept_current = {}
        for gs in concept_settings:
            for concept, cur_val in zip(gs["concepts"], gs["current_setting"]):
                concept_current[concept] = cur_val

        # Old values for all concepts (in the order of all_concepts)
        old_values = [concept_current.get(c, None) for c in all_concepts]

        # New values: copy old, then overwrite for the intervened group
        new_values = list(old_values)
        for concept, new_val in zip(group_concepts, alt_values_tuple):
            idx = all_concepts.index(concept)
            new_values[idx] = new_val

        # Intervention boolean vector: True where new value differs from old
        intervention_bool = [old != new for old, new in zip(old_values, new_values)]

        # -----------------------------------------------------------------
        # Generate counterfactual using the LLM
        # -----------------------------------------------------------------
        counterfactual_gen_prompt = self.dataset.format_prompt_counterfactual_gen(
            self.example_idx,
            self.counterfactual_gen_base_prompt_name,
            all_concepts,
            intervention_bool,
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

        # Save intervention result
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
            json.dump(intervention_dict, f, indent=2)
        return intervention_dict