import os
import random
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Set, Tuple


class InterventionGenerator(ABC):
    """Abstract base class – defines the interface and common workflow."""

    def __init__(self, intervention_generator_arguments):
        self.dataset = intervention_generator_arguments.dataset
        self.example_idx = intervention_generator_arguments.example_idx
        self.intervention_model = intervention_generator_arguments.intervention_model
        self.output_dir = intervention_generator_arguments.output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.concept_id_base_prompt_name = intervention_generator_arguments.concept_id_base_prompt_name
        self.concept_values_base_prompt_name = intervention_generator_arguments.concept_values_base_prompt_name
        self.counterfactual_gen_base_prompt_name = intervention_generator_arguments.counterfactual_gen_base_prompt_name
        self.n_workers = intervention_generator_arguments.n_workers
        self.verbose = intervention_generator_arguments.verbose
        self.debug = intervention_generator_arguments.debug
        self.seed = intervention_generator_arguments.seed
        self.include_unknown_concept_values = intervention_generator_arguments.include_unknown_concept_values
        self.only_concept_removals = intervention_generator_arguments.only_concept_removals
        self.restart_from_previous = intervention_generator_arguments.restart_from_previous
        random.seed(self.seed)

    @abstractmethod
    def identify_concepts_within_correlation_groups(self) -> List[Set[Tuple[str, str]]]:
        """Return a list of correlated concept groups.
        Each concept group is a set of (concept, category) tuples that are correlated with each other.
        If a concept is not correlated with any other, it's added as a single-item set (singleton).
        """
        pass

    @abstractmethod
    def define_intervention_sets(self, concepts):
        """Return concept_settings (list of dicts with 'new_settings', etc.)."""
        pass

    @abstractmethod
    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """Generate and save one intervention. Return a dict with results."""
        pass

    def apply_interventions(self, concept_settings):
        """
        Enumerate all interventions from concept_settings (group format),
        skip existing ones, and run them in parallel.
        Calls apply_single_intervention() for each.
        """
        existing_interventions = []
        if self.restart_from_previous:
            existing_interventions = [x.split('.')[0].split('_')[1]
                                      for x in os.listdir(self.output_dir)
                                      if x.startswith('counterfactual_')]
            if existing_interventions:
                print(f"Found {len(existing_interventions)} existing interventions. Skipping...")

        # -----------------------------------------------------------------
        # 1. Determine if concept_settings use the group format
        #    (i.e. each entry has a "concepts" key)
        # -----------------------------------------------------------------
        is_group_format = all("concepts" in cs for cs in concept_settings)

        if is_group_format:
            # Flatten all concepts across groups to give a complete list to apply_single_intervention
            all_concepts = []
            for cs in concept_settings:
                all_concepts.extend(cs["concepts"])
            # Make unique, preserving order
            seen = set()
            all_concepts = [c for c in all_concepts if not (c in seen or seen.add(c))]
        else:
            # Legacy flat format: we need a concepts list. Here we assume
            # concept_settings items have a "concept" key, but the original code
            # didn't store concepts; adapt to what you actually have.
            # For safety, raise an error if we can't obtain them.
            raise NotImplementedError(
                "apply_interventions with flat concept_settings is not supported in this version. "
                "Please ensure define_intervention_sets returns group format."
            )

        # -----------------------------------------------------------------
        # 2. Build the intervention list
        # -----------------------------------------------------------------
        intervention_list = []

        if is_group_format:
            # Group‑based enumeration: one intervention per combination per group
            for group_idx, group_setting in enumerate(concept_settings):
                n_combos = len(group_setting["new_settings"])
                for combo_idx in range(n_combos):
                    intrv_str = f"G{group_idx}_C{combo_idx}"
                    intervention_list.append(intrv_str)
        else:
            # (Old path – not used now, kept for reference)
            # intervention_list = enumerate_interventions(concepts, concept_settings, ...)
            pass

        # Skip those already generated
        intervention_list = [x for x in intervention_list if x not in existing_interventions]

        if self.debug and len(intervention_list) >= 10:
            intervention_list = intervention_list[:10]
            if self.verbose:
                print("DEBUG: executing 10 interventions only...")

        if not intervention_list:
            print("No interventions to apply. Exiting...")
            return

        print(f"Executing {len(intervention_list)} interventions with {self.n_workers} workers...")
        interventions = []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(self.apply_single_intervention,
                                       i_str, all_concepts, concept_settings)
                       for i_str in intervention_list]
            for cnt, future in enumerate(as_completed(futures)):
                interventions.append(future.result(timeout=300))
                if self.verbose and cnt % 100 == 0:
                    print(f"Finished {cnt+1}/{len(intervention_list)} interventions")
                    print(f"Threading active count: {threading.active_count()}")