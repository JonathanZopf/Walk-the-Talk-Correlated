import copy
import json
import os
import random
import threading
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import enumerate_interventions   # generic helper


class InterventionGenerator(ABC):
    """Abstract base class – defines the interface and common workflow."""

    def __init__(self, intervention_generator_arguments):
        """
        Stores shared attributes. Subclasses receive all parameters
        but may ignore those not needed for their own logic.
        """
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
    def identify_concepts(self):
        """Return (concepts_list, categories_list)."""
        pass

    @abstractmethod
    def define_intervention_sets(self, concepts):
        """Return concept_settings (list of dicts with 'new_settings', etc.)."""
        pass

    @abstractmethod
    def apply_single_intervention(self, intervention_str, concepts, concept_settings):
        """Generate and save one intervention. Return a dict with results."""
        pass

    def apply_interventions(self, concepts, concept_settings):
        """
        Enumerate all interventions, skip existing ones, and run them in parallel.
        Calls the abstract apply_single_intervention() for each.
        """
        existing_interventions = []
        if self.restart_from_previous:
            existing_interventions = [x.split('.')[0].split('_')[1]
                                      for x in os.listdir(self.output_dir)
                                      if x.startswith('counterfactual_')]
            if existing_interventions:
                print(f"Found {len(existing_interventions)} existing interventions. Skipping...")

        if self.only_concept_removals:
            for fs in concept_settings:
                fs["new_settings"] = ["UNKNOWN"]

        if self.include_unknown_concept_values and not self.only_concept_removals:
            for fs in concept_settings:
                fs["new_settings"].append("UNKNOWN")

        intervention_list = enumerate_interventions(
            concepts, concept_settings, k_hop=1,
            include_no_intervention=False, mark_removals=True
        )
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
                                       i_str, concepts, concept_settings)
                       for i_str in intervention_list]
            for cnt, future in enumerate(as_completed(futures)):
                interventions.append(future.result(timeout=300))
                if self.verbose and cnt % 100 == 0:
                    print(f"Finished {cnt+1}/{len(intervention_list)} interventions")
                    print(f"Threading active count: {threading.active_count()}")