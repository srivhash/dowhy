# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import numpy as np

import rca_task

# function to perform anomaly detection
def run(dataset_path, method, method_name, out_dir):
    # what does evaluate do?
    df = rca_task.evaluate(method, dataset_path)
    # stores the processed dataframe in a csv file
    filename = f"results_recall_{method_name}.csv"
    print(f"Storing results under {filename}")
    df.to_csv(os.path.join(out_dir, filename), index=False)

    # looping through all the possible scenario combinations
    for scenario in ["low_traffic", "high_traffic", "temporal_traffic"]:
        for issue_metric in ["latency", "availability"]:
            # selects the rows in the df with the relevant combination of scenario and issue_metric
            df_sel = df[df["scenario"].str.startswith(scenario)]
            df_sel = df_sel[df_sel["metric"] == issue_metric]

            # for k in [1, 3], calculates the mean of the intopk column and the proportion of empty results
            for k in [1, 3]:
                df_sel_k = df_sel[df_sel["topk"] == k]
                size_all = len(df_sel_k.intopk)
                res = np.mean(df_sel_k.intopk)
                empty = len(df_sel_k[df_sel_k["empty"] == True]) / size_all
                print(
                    f"for {scenario} with {issue_metric} with {size_all} ({empty:.2f} with no results) many issues at top{k} got {res:.3f}"
                )
    # Specificity : in the context of causal learning, specificity refers to the ability of a model to correctly identify
    # only the true causes and not any spurious correlations. high specifictity implies that the model is 
    # good at avoiding false positives
    # correlation does not imply causation and hence specificity is important
    # evaluate_specificity is a function to evaluate the specificity of the method on the original dataset
    df_specificity = rca_task.evaluate_specificity(method, dataset_path)
    filename = f"results_specificity_{method_name}.csv"
    print(f"Storing results under {filename}")
    df_specificity.to_csv(os.path.join(out_dir, filename), index=False)

    # the same loop again, but now for specificity
    for scenario in ["low_traffic", "high_traffic", "temporal_traffic"]:
        for issue_metric in ["latency", "availability"]:
            df_sel = df_specificity[df_specificity["scenario"].str.startswith(scenario)]
            df_sel = df_sel[df_sel["metric"] == issue_metric]
            print(
                f"for {scenario} with {issue_metric} got specificity of {np.mean(df_sel.specificity):.3f}"
            )

# argparser to take arguments from command line
def parse_args():
    parser = argparse.ArgumentParser(
        description="A script to perform a anomaly-traversal to identify root-causes based on performance metrics from the PetShop application."
    )
    parser.add_argument(
        "--dataset_path", type=str, help="Path to petshop metric dataset."
    )
    parser.add_argument("--out_path", type=str, help="Path to store results.")
    parser.add_argument(
        "--method",
        type=str,
        help="baseline, rcd, epsilon_diagnosis, circa and ranked_correlation currently implemented.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True) # creates the output directory if it does not exist already
    print(args.method)
    # check which method is used for anomaly detection
    if args.method == "baseline":
        import baseline_anomaly_traversal
        method = baseline_anomaly_traversal.make_baseline_analyze_root_causes()

    elif args.method == "rcd":
        import hierarchical_rcd
        method = hierarchical_rcd.make_hierarchical_rcd()

    elif args.method == "epsilon_diagnosis":
        import epsilon_diagnosis
        method = epsilon_diagnosis.make_epsilon_diagnosis()

    elif args.method == "circa":
        import circa
        method = circa.make_circa()

    elif args.method == "ranked_correlation":
        import ranked_correlation
        method = ranked_correlation.make_ranked_correlation()

    # following may not have been implemented based on argparser
    elif args.method == "random_walk":
        import random_walk
        method = random_walk.make_random_walk()

    elif args.method == "counterfactual_attribution":
        import counterfactual_attribution
        method = counterfactual_attribution.make_counterfactual_attribution_method()

    else:
        raise ValueError(f"Unsupported method {args.method}")
    # calls the final running function for anomaly detection
    run(args.dataset_path, method, args.method, args.out_path)


if __name__ == "__main__":
    main()
