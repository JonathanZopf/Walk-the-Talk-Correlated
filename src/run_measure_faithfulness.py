#!/usr/bin/env python3
"""
Faithfulness Measurement Script for LLM Explanations

This script measures the faithfulness of an LLM's explanations by:
1. Estimating Explanation-Implied Effects (EE) - which concepts the LLM claims are influential
2. Estimating Causal Concept Effects (CE) - actual causal impact of concepts on model outputs
3. Measuring Causal Concept Faithfulness - correlation between EE and CE

Based on the methodology from the BBQ example notebook.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Add the src directory to path if needed
sys.path.append('../src')

from causal_concept_effect_estimation.estimate_concept_effects import ConceptEffectEstimator
from explanation_implied_effect_estimation.estimate_explanation_implied_effects import ExplanationImpliedEffectEstimator
from faithfulness_estimation.estimate_faithfulness import FaithfulnessEstimator
from my_datasets.bbq import BBQDataset


def parse_args():
    """Parse command line arguments for faithfulness measurement."""
    parser = argparse.ArgumentParser(
        description='Measure faithfulness of LLM explanations using causal concept effects',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data paths
    parser.add_argument('--dataset', type=str, default='bbq',
                        help='Dataset name (default: bbq)')
    parser.add_argument('--dataset_path', type=str, default='data/bbq',
                        help='Path to dataset')
    parser.add_argument('--intervention_dir', type=str,
                        default='output/bbq/intervention_generation',
                        help='Path to counterfactual generation data')
    parser.add_argument('--model_response_dir', type=str,
                        default='output/bbq/model_responses',
                        help='Path to model responses directory')
    parser.add_argument('--implied_concepts_dir', type=str,
                        default='output/bbq/implied_concepts',
                        help='Path to implied concepts directory')
    parser.add_argument('--output_dir', type=str,
                        default='output/faithfulness_results',
                        help='Directory to save results')

    # Model specification
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the LLM to evaluate')

    # Example indices
    parser.add_argument('--example_idxs', type=int, nargs='+',
                        help='Specific example indices to analyze')
    parser.add_argument('--n_examples', type=int, default=None, help='Number of examples to analyze')

    # Analysis options
    parser.add_argument('--skip_ee', action='store_true',
                        help='Skip EE estimation (load from existing files)')
    parser.add_argument('--skip_ce', action='store_true',
                        help='Skip CE estimation (load from existing files)')
    parser.add_argument('--n_posterior_samples', type=int, default=2000,
                        help='Number of posterior samples for Bayesian models')
    parser.add_argument('--n_chains', type=int, default=4,
                        help='Number of MCMC chains')
    parser.add_argument('--n_tune', type=int, default=1000,
                        help='Number of tuning steps for MCMC')

    # Output options
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress information')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save faithfulness plots')
    parser.add_argument('--save_results', action='store_true',
                        help='Save results')
    parser.add_argument('--print_results', action='store_true',
                        help='Print results to console')

    return parser.parse_args()


def get_example_indices(args, dataset):
    """
    Get the example indices to analyze.

    Args:
        args: Command line arguments
        dataset: Dataset object

    Returns:
        List of example indices
    """
    if args.example_idxs:
        return args.example_idxs
    elif args.n_examples:
        return range(args.n_examples)
    else:
        # Use all examples in dataset
        return list(range(len(dataset)))


def estimate_explanation_implied_effects(args, dataset, example_idxs):
    """
    Estimate Explanation-Implied Effects (EE).

    Args:
        args: Command line arguments
        dataset: Dataset object
        example_idxs: List of example indices

    Returns:
        DataFrame with EE estimates
    """
    if args.verbose:
        print("\n" + "=" * 60)
        print("STEP 1: Estimating Explanation-Implied Effects (EE)")
        print("=" * 60)
        print(f"Loading implied concepts from: {args.implied_concepts_dir}")

    # Construct the implied concepts directory path (
    implied_concepts_path = args.implied_concepts_dir

    # Initialize estimator
    ee_estimator = ExplanationImpliedEffectEstimator(
        dataset,
        example_idxs,
        args.intervention_dir,
        implied_concepts_path,
        verbose=args.verbose
    )

    # Load data
    if args.verbose:
        print("Loading implied concepts data...")
    ic_df = ee_estimator.load_data(load_counterfactual_responses=False)

    if args.verbose:
        print(f"Loaded data for {len(ic_df)} concept-question pairs")
        print("\nEstimating implied effects...")

    # Estimate implied effects (average concept_decisions across responses)
    ee_df = ee_estimator.estimate_implied_effects(ic_df)

    if args.verbose:
        print(f"Estimated EE for {len(ee_df)} concepts")
        print("\nEE Summary:")
        print(f"  Mean probability: {ee_df['p(concept_in_explanation)'].mean():.3f}")
        print(f"  Std probability: {ee_df['p(concept_in_explanation)'].std():.3f}")

    return ee_df


def estimate_causal_concept_effects(args, dataset, example_idxs):
    """
    Estimate Causal Concept Effects (CE).

    Args:
        args: Command line arguments
        dataset: Dataset object
        example_idxs: List of example indices

    Returns:
        DataFrame with CE estimates
    """
    if args.verbose:
        print("\n" + "=" * 60)
        print("STEP 2: Estimating Causal Concept Effects (CE)")
        print("=" * 60)
        print(f"Loading model responses from: {args.model_response_dir}")
        print(f"Model: {args.model_name}")

    # Construct the model responses directory path
    model_responses_path = args.model_response_dir

    # Initialize estimator
    ce_estimator = ConceptEffectEstimator(
        dataset,
        example_idxs,
        args.intervention_dir,
        model_responses_path,
        verbose=args.verbose
    )

    # Load response data
    if args.verbose:
        print("Loading model responses...")
    response_df = ce_estimator.load_data(standardize_order=False)

    if args.verbose:
        print(f"Loaded {len(response_df)} responses")
        print("\nFitting hierarchical Bayesian model...")

    # Fit hierarchical Bayesian model
    samples, cats, treatments, treatment_ref_classes = ce_estimator.fit_logistic_regression_hierarchical_bayesian(
        response_df
    )

    if args.verbose:
        print("Model fitting complete")
        print("Extracting parameter estimates...")

    # Get parameter results from posterior samples
    cat_df, treatment_df = ce_estimator.get_parameter_results_from_posterior_samples(
        samples, cats, treatments, treatment_ref_classes, response_df
    )



    if args.verbose:
        print(f"\nCE Summary:")
        print(f"  Number of concepts: {len(treatment_df)}")
        print(f"  Mean CE (kl): {treatment_df['kl_div'].mean():.3f}")
        print(f"  Std CE (kl): {treatment_df['kl_div'].std():.3f}")
        if 'category' in treatment_df.columns:
            print(f"\nCategory-wise CE:")
            for cat in treatment_df['category'].unique():
                cat_mean = treatment_df[treatment_df['category'] == cat]['kl_div'].mean()
                print(f"  {cat}: {cat_mean:.3f}")

    return treatment_df


def measure_faithfulness(args, ee_df, ce_df):
    """
    Measure faithfulness by correlating EE and CE.

    Args:
        args: Command line arguments
        ee_df: DataFrame with explanation-implied effects
        ce_df: DataFrame with causal concept effects

    Returns:
        Tuple of (faithfulness_samples, beta_mean, beta_credible_interval)
    """
    if args.verbose:
        print("\n" + "=" * 60)
        print("STEP 3: Measuring Causal Concept Faithfulness")
        print("=" * 60)
        print("Correlating EE and CE estimates...")

    # Initialize faithfulness estimator
    faithfulness_estimator = FaithfulnessEstimator(ee_df, ce_df)

    # Estimate faithfulness (hierarchical Bayesian regression)
    if args.verbose:
        print("Fitting hierarchical Bayesian regression model...")

    faith_samples, beta_mean, beta_credible_interval = faithfulness_estimator.estimate_faithfulness()

    if args.verbose:
        print("\nFaithfulness Results:")
        print(f"  Beta (faithfulness) mean: {beta_mean:.3f}")

        ci_lower = float(beta_credible_interval[0])
        ci_upper = float(beta_credible_interval[1])
        print(f"  90% Credible Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")

        # Interpret the result
        if beta_credible_interval[0] > 0:
            print("\n  ✓ Positive faithfulness detected")
            print("    The LLM's explanations align with its causal behavior")
        elif beta_credible_interval[1] < 0:
            print("\n  ✗ Negative faithfulness detected")
            print("    The LLM's explanations contradict its causal behavior")
        else:
            print("\n  ? No significant faithfulness detected")
            print("    The LLM's explanations do not reliably indicate causal influences")

    # Save plots if requested
    if args.save_plots:
        faithfulness_estimator.plot_faithfulness(faith_samples, save_path=os.path.join(args.output_dir, f"{args.model_name}.png"))

    return faith_samples, beta_mean, beta_credible_interval


def save_results(args, ee_df, ce_df, beta_mean, beta_credible_interval, example_idxs):
    """
    Save all results to files.

    Args:
        args: Command line arguments
        ee_df: EE estimates DataFrame
        ce_df: CE estimates DataFrame
        beta_mean: Faithfulness beta mean
        beta_credible_interval: Faithfulness credible interval
        example_idxs: List of example indices used
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save EE results
    ee_path = os.path.join(args.output_dir, f"{args.model_name}_ee_estimates.csv")
    ee_df.to_csv(ee_path, index=False)
    if args.verbose:
        print(f"\nEE estimates saved to: {ee_path}")

    # Save CE results
    ce_path = os.path.join(args.output_dir, f"{args.model_name}_ce_estimates.csv")
    ce_df.to_csv(ce_path, index=False)
    if args.verbose:
        print(f"CE estimates saved to: {ce_path}")

    # Save faithfulness summary
    results = {
        'model_name': args.model_name,
        'example_indices': list(example_idxs),
        'n_examples': len(example_idxs),
        'n_concepts_ee': len(ee_df),
        'n_concepts_ce': len(ce_df),
        'faithfulness_beta_mean': float(beta_mean),
        'faithfulness_beta_ci_lower': float(beta_credible_interval[0]),
        'faithfulness_beta_ci_upper': float(beta_credible_interval[1]),
        'faithfulness_significant': bool(beta_credible_interval[0] > 0 or beta_credible_interval[1] < 0),
        'faithfulness_positive': bool(beta_credible_interval[0] > 0),
        'parameters': {
            'n_posterior_samples': args.n_posterior_samples,
            'n_chains': args.n_chains,
            'n_tune': args.n_tune
        }
    }

    results_path = os.path.join(args.output_dir, f"{args.model_name}_faithfulness_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if args.verbose:
        print(f"Faithfulness results saved to: {results_path}")


def print_results_to_console(args, ee_df, ce_df, beta_mean, beta_credible_interval):
    """
    Print formatted results to console.

    Args:
        args: Command line arguments
        ee_df: EE estimates DataFrame
        ce_df: CE estimates DataFrame
        beta_mean: Faithfulness beta mean
        beta_credible_interval: Faithfulness credible interval
    """
    print("\n" + "=" * 60)
    print("FAITHFULNESS MEASUREMENT RESULTS")
    print("=" * 60)
    print(f"\nModel: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Examples analyzed: {len(ee_df['example_idx'].unique()) if 'example_idx' in ee_df.columns else 'N/A'}")
    print(f"Concepts analyzed: {len(ee_df)}")

    print("\n" + "-" * 40)
    print("EXPLANATION-IMPLIED EFFECTS (EE)")
    print("-" * 40)
    print(f"  Mean probability concept mentioned: {ee_df['p(concept_in_explanation)'].mean():.3f}")
    print(f"  Std probability: {ee_df['p(concept_in_explanation)'].std():.3f}")
    print(f"  Min probability: {ee_df['p(concept_in_explanation)'].min():.3f}")
    print(f"  Max probability: {ee_df['p(concept_in_explanation)'].max():.3f}")

    # Top 5 most mentioned concepts (EE)
    if 'concept' in ee_df.columns:
        top_concepts = ee_df.nlargest(5, 'p(concept_in_explanation)')[['concept', 'p(concept_in_explanation)']]
        print("\n  Top 5 most mentioned concepts:")
        for _, row in top_concepts.iterrows():
            print(f"    - {row['concept']}: {row['p(concept_in_explanation)']:.3f}")

    print("\n" + "-" * 40)
    print("CAUSAL CONCEPT EFFECTS (CE)")
    print("-" * 40)
    # Use 'kl_div' instead of 'beta_mean'
    print(f"  Mean KL divergence (causal effect): {ce_df['kl_div'].mean():.3f}")
    print(f"  Std KL divergence: {ce_df['kl_div'].std():.3f}")
    print(f"  Min KL divergence: {ce_df['kl_div'].min():.3f}")
    print(f"  Max KL divergence: {ce_df['kl_div'].max():.3f}")

    # Concepts with strongest causal effects (highest KL divergence)
    if 'intrv_concept' in ce_df.columns:
        strong_concepts = ce_df.nlargest(5, 'kl_div')[['intrv_concept', 'kl_div']]
        print("\n  Top 5 concepts with strongest causal effects:")
        for _, row in strong_concepts.iterrows():
            print(f"    - {row['intrv_concept']}: {row['kl_div']:.3f}")

    print("\n" + "-" * 40)
    print("FAITHFULNESS (EE vs CE Correlation)")
    print("-" * 40)
    print(f"  Beta (faithfulness) mean: {beta_mean:.3f}")


    if isinstance(beta_credible_interval, (np.ndarray, tuple, list)):
        ci_lower = beta_credible_interval[0]
        ci_upper = beta_credible_interval[1]
        print(f"  90% Credible Interval: [{ci_lower}, {ci_upper}]")
    else:
        print(f"  90% Credible Interval: {beta_credible_interval}")

    # Interpretation
    if beta_credible_interval[0] > 0:
        print("\n  ✓ INTERPRETATION: Positive faithfulness detected")
        print("    The LLM's explanations accurately reflect its causal behavior.")
        print("    Concepts the model claims are influential actually affect its outputs.")
    elif beta_credible_interval[1] < 0:
        print("\n  ✗ INTERPRETATION: Negative faithfulness detected")
        print("    The LLM's explanations contradict its causal behavior.")
        print("    Concepts the model claims are influential do NOT affect its outputs.")
    else:
        print("\n  ? INTERPRETATION: No significant faithfulness detected")
        print("    The LLM's explanations do not reliably indicate which concepts")
        print("    actually influence its outputs.")

    print("\n" + "=" * 60)


def main():
    """Main execution function."""
    args = parse_args()

    if not args.print_results and not args.save_results:
        raise ValueError("Choose at least to print or save results")

    # Print configuration if verbose
    if args.verbose:
        print("\n" + "=" * 60)
        print("FAITHFULNESS MEASUREMENT CONFIGURATION")
        print("=" * 60)
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")

    # Initialize dataset
    if args.verbose:
        print(f"\nLoading dataset: {args.dataset} from {args.dataset_path}")
    dataset = BBQDataset(args.dataset, args.dataset_path)

    example_idxs = get_example_indices(args, dataset)
    if args.verbose:
        print(f"Analyzing {len(example_idxs)} examples")

    # Step 1: Estimate Explanation-Implied Effects
    if not args.skip_ee:
        ee_df = estimate_explanation_implied_effects(args, dataset, example_idxs)
    else:
        if args.verbose:
            print("\nSkipping EE estimation (using --skip_ee flag)")
        # Load existing EE results
        ee_path = os.path.join(args.output_dir, f"{args.model_name}_ee_estimates.csv")
        if os.path.exists(ee_path):
            ee_df = pd.read_csv(ee_path)
            if args.verbose:
                print(f"Loaded EE estimates from: {ee_path}")
        else:
            raise FileNotFoundError(f"EE estimates not found at {ee_path}. Remove --skip_ee to generate them.")

    # Step 2: Estimate Causal Concept Effects
    if not args.skip_ce:
        ce_df = estimate_causal_concept_effects(args, dataset, example_idxs)
    else:
        if args.verbose:
            print("\nSkipping CE estimation (using --skip_ce flag)")
        # Load existing CE results
        ce_path = os.path.join(args.output_dir, f"{args.model_name}_ce_estimates.csv")
        if os.path.exists(ce_path):
            ce_df = pd.read_csv(ce_path)
            if args.verbose:
                print(f"Loaded CE estimates from: {ce_path}")
        else:
            raise FileNotFoundError(f"CE estimates not found at {ce_path}. Remove --skip_ce to generate them.")

    # Step 3: Measure Faithfulness
    faith_samples, beta_mean, beta_credible_interval = measure_faithfulness(args, ee_df, ce_df)

    # Save
    if args.save_results:
        save_results(args, ee_df, ce_df, beta_mean, beta_credible_interval, example_idxs)

    # Print results to console if requested
    if args.print_results:
        print_results_to_console(args, ee_df, ce_df, beta_mean, beta_credible_interval)

    if args.verbose:
        print("\n" + "=" * 60)
        print("FAITHFULNESS MEASUREMENT COMPLETE")
        print("=" * 60)

    return beta_mean, beta_credible_interval


if __name__ == '__main__':
    main()
