class InterventionGeneratorArguments(object):
    def __init__(self, dataset, example_idx, intervention_model, output_dir,
                 concept_id_base_prompt_name, concept_values_base_prompt_name,
                 counterfactual_gen_base_prompt_name, n_workers=4, seed=42,
                 verbose=False, debug=False, include_unknown_concept_values=True,
                 only_concept_removals=False, restart_from_previous=True):
        """
        Stores shared attributes. Subclasses receive all parameters
        but may ignore those not needed for their own logic.
        """
        self.dataset = dataset
        self.example_idx = example_idx
        self.intervention_model = intervention_model
        self.output_dir = output_dir
        self.concept_id_base_prompt_name = concept_id_base_prompt_name
        self.concept_values_base_prompt_name = concept_values_base_prompt_name
        self.counterfactual_gen_base_prompt_name = counterfactual_gen_base_prompt_name
        self.n_workers = n_workers
        self.verbose = verbose
        self.debug = debug
        self.seed = seed
        self.include_unknown_concept_values = include_unknown_concept_values
        self.only_concept_removals = only_concept_removals
        self.restart_from_previous = restart_from_previous