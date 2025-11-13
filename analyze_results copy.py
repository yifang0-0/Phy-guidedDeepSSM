import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_multitrain_results():
    """
    Analyze multitrain CSV files and calculate mean RMSE, NRMSE, and correlation coefficients
    """
    
    # Base directory containing the results
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_multitrain_results():
    """
    Analyze multitrain CSV files and calculate mean RMSE, NRMSE, and correlation coefficients
    for multiple data lengths (100, 50, 20, 10).
    """
    # List of (base_dir, length) tuples
    base_dirs = [
        ("/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/toy_lgssm_5_pre", 100),
        ("/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_50/toy_lgssm_5_pre", 50),
        ("/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_20/toy_lgssm_5_pre", 20),
        ("/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_10/toy_lgssm_5_pre", 10),
    ]

    # Dictionary to store results for each (length, model)
    results = {}

    for base_dir, length in base_dirs:
        # Find all model directories
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        print(f"Found model directories for length {length}:", model_dirs)
        print("-" * 80)
        
        for model_dir in model_dirs:
            model_path = os.path.join(base_dir, model_dir)
            
            # Find multitrain CSV files in this model directory
            csv_files = glob.glob(os.path.join(model_path, "*multitrain.csv"))

            if not csv_files:
                print(f"No multitrain CSV files found in {model_dir} (length {length})")
                continue

            print(f"\nAnalyzing {model_dir} (length {length}):")
            print(f"Found {len(csv_files)} multitrain CSV files")

            # Prepare a table to collect results for this model_dir
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                print(filename, csv_file)


                # Extract config: only the substring matching "_A*_B*_C*"
                import re
                config_match = re.search(r'(_A[^_]*_B[^_]*_C[^_]*)', filename)
                config_str = config_match.group(1) if config_match else "N/A"

                mpw_match = re.search(r'(mpw-?\d+)', filename).group(0)

                try:
                    print("read: ", csv_file)
                    df = pd.read_csv(csv_file)
                    metrics = ['rmse', 'nrmse', 'Correlation Coefficient']
                    row = {
                        "Config": config_str,
                        "File": filename,
                    }
                    for metric in metrics:
                        if metric in df.columns:
                            row[f"{metric}_mean"] = df[metric].mean()
                            row[f"{metric}_std"] = df[metric].std()
                        else:
                            row[f"{metric}_mean"] = np.nan
                            row[f"{metric}_std"] = np.nan
                    row["num_runs"] = len(df)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    continue

                # Store results in a nested dict: results[length][model][config]
                if length not in results:
                    results[length] = {}
                if model_dir not in results[length]:
                    results[length][model_dir] = {}
                if mpw_match not in results[length][model_dir]:
                    results[length][model_dir][mpw_match] = {}
                results[length][model_dir][mpw_match][config_str]= row
                # print(mpw_match)
    # print(results)
    return results

def save_results_to_csv(results, output_prefix="multitrain_summary"):
    """
    Save the results dictionary to CSV files, one for each data length.
    """
    for length, model_data in results.items():
        rows = []
        for model_name, configs in model_data.items():
            for config, metrics in configs.items():
                row = {
                    'Length': length,
                    'Model': model_name,
                    'Configuration': config,
                    'RMSE_Mean': metrics.get('rmse_mean', np.nan),
                    'RMSE_Std': metrics.get('rmse_std', np.nan),
                    'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
                    'NRMSE_Std': metrics.get('nrmse_std', np.nan),
                    'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
                    'Correlation_Std': metrics.get('Correlation Coefficient_std', np.nan),
                    'Num_Runs': metrics.get('num_runs', np.nan)
                }
                rows.append(row)
                
        df_results = pd.DataFrame(rows)
        output_file = f"{output_prefix}_length{length}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nResults for length {length} saved to {output_file}")

def print_summary_table(results):
    """
    Print a summary table with columns: Length, Model, MPW, A*_B*_C*, RMSE, Correlation Coefficient
    """
    from tabulate import tabulate
    import re

    table_rows = []
    print(results)
    for length, model_data in results.items():
        for model_name, configs in model_data.items():
            for mpw, metrics_config in configs.items():
                # print(model_name, config)
                # Try to extract mpw* and A*_B*_C* from config string
                for config, metrics in metrics_config.items():
                    # mpw = "N/A"
                    abc = "N/A"
                    # mpw: match mpw followed by digits or -digits (e.g., mpw-10, mpw100)
                    # mpw_match = re.search(r'(mpw-?\d+)', mpw)
                    # if mpw_match:
                    #     mpw = mpw_match.group(1)
                    # A*_B*_C*
                    print("mpw", mpw)
                    abc_match = re.search(r'(A\d+_B\d+_C\d+)', config)
                    if abc_match:
                        abc = abc_match.group(1)
                    rmse = metrics.get('rmse_mean', np.nan)
                    cc = metrics.get('Correlation Coefficient_mean', np.nan)
                    table_rows.append([
                        length,
                        model_name,
                        mpw,
                        abc,
                        f"{rmse:.5f}" if not np.isnan(rmse) else "N/A",
                        f"{cc:.5f}" if not np.isnan(cc) else "N/A"
                    ])
    headers = ["Length", "Model", "mpw*", "A*_B*_C*", "RMSE", "CC"]
    print("\nSUMMARY TABLE")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))

if __name__ == "__main__":
    # Run the analysis for all lengths
    results = analyze_multitrain_results()
    # Save results to CSVs
    save_results_to_csv(results)
    print_summary_table(results)
    print("\nAnalysis complete!")
    
#     # Dictionary to store results for each model
#     results = {}
    
#     # Find all model directories
#     model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
#     print("Found model directories:", model_dirs)
#     print("-" * 80)
    
#     for model_dir in model_dirs:
#         model_path = os.path.join(base_dir, model_dir)
        
#         # Find multitrain CSV files in this model directory
#         csv_files = glob.glob(os.path.join(model_path, "*multitrain.csv"))

#         if not csv_files:
#             print(f"No multitrain CSV files found in {model_dir}")
#             continue

#         print(f"\nAnalyzing {model_dir}:")
#         print(f"Found {len(csv_files)} multitrain CSV files")

#         # Prepare a table to collect results for this model_dir
#         table_rows = []
#         for csv_file in csv_files:
#             filename = os.path.basename(csv_file)
#             # Only process files whose name starts with "mpw"
#             if not filename.startswith("mpw"):
#                 continue

#             # Extract config: only the substring matching "_A*_B*_C*"
#             import re
#             import pandas as pd
            
#             config_match = re.search(r'(_A[^_]*_B[^_]*_C[^_]*)', filename)
#             config_str = config_match.group(1) if config_match else "N/A"

#             try:
#                 # import pandas as pd
#                 print("read: ", csv_file)
#                 df = pd.read_csv(csv_file)
#                 metrics = ['rmse', 'nrmse', 'Correlation Coefficient']
#                 row = {
#                     "Config": config_str,
#                     "File": filename,
#                 }
#                 for metric in metrics:
#                     if metric in df.columns:
#                         row[f"{metric}_mean"] = df[metric].mean()
#                         row[f"{metric}_std"] = df[metric].std()
#                     else:
#                         row[f"{metric}_mean"] = None
#                         row[f"{metric}_std"] = None
#                 row["num_runs"] = len(df)
#                 table_rows.append(row)

#                 # Print results for this file
#                 print(f"  File: {filename}")
#                 print(f"    Config: {config_str}")
#                 for metric in metrics:
#                     if metric in df.columns:
#                         print(f"    - {metric}: {df[metric].mean():.4f} ± {df[metric].std():.4f}")
#                 print(f"    - Number of runs: {len(df)}")
#             except Exception as e:
#                 print(f"    Error reading {csv_file}: {str(e)}")
#                 continue
#        # Print table for this model_dir
#         if table_rows:
#             print("\nSummary Table for", model_dir)
#             import pandas as pd
#             table_df = pd.DataFrame(table_rows)
#             # Only show columns: Config, File, rmse_mean, nrmse_mean, Correlation Coefficient_mean, num_runs
#             show_cols = ["Config", "File", "rmse_mean", "nrmse_mean", "Correlation Coefficient_mean", "num_runs"]
#             print(table_df[show_cols].to_string(index=False))
        
#         model_results = {}
        
#         for csv_file in csv_files:
#             try:
#                 # Extract configuration from filename
#                 filename = os.path.basename(csv_file)
#                 print(f"  Reading: {filename}")
                
#                 # Read CSV file
#                 import pandas as pd
                
#                 df = pd.read_csv(csv_file)
                
#                 # Calculate statistics for each metric
#                 metrics = ['rmse', 'nrmse', 'Correlation Coefficient']
                
#                 for metric in metrics:
#                     if metric in df.columns:
#                         mean_val = df[metric].mean()
#                         std_val = df[metric].std()
                        
#                         # Store results with configuration info
#                         config_key = filename.replace('_multitrain.csv', '')
                        
#                         if config_key not in model_results:
#                             model_results[config_key] = {}
                        
#                         model_results[config_key][f'{metric}_mean'] = mean_val
#                         model_results[config_key][f'{metric}_std'] = std_val
#                         model_results[config_key]['num_runs'] = len(df)
                
#                 print(f"    - RMSE: {df['rmse'].mean():.4f} ± {df['rmse'].std():.4f}")
#                 print(f"    - NRMSE: {df['nrmse'].mean():.4f} ± {df['nrmse'].std():.4f}")
#                 print(f"    - Correlation: {df['Correlation Coefficient'].mean():.4f} ± {df['Correlation Coefficient'].std():.4f}")
#                 print(f"    - Number of runs: {len(df)}")
                
#             except Exception as e:
#                 print(f"    Error reading {csv_file}: {str(e)}")
#                 continue
        
#         results[model_dir] = model_results
    
#     # Print summary
#     print("\n" + "=" * 80)
#     print("SUMMARY OF ALL MODELS")
#     print("=" * 80)
    
#     for model_name, model_data in results.items():
#         print(f"\n{model_name}:")
#         print("-" * 40)
        
#         for config, metrics in model_data.items():
#             print(f"  Configuration: {config}")
#             print(f"    RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}")
#             print(f"    NRMSE: {metrics.get('nrmse_mean', 'N/A'):.4f} ± {metrics.get('nrmse_std', 'N/A'):.4f}")
#             print(f"    Correlation: {metrics.get('Correlation Coefficient_mean', 'N/A'):.4f} ± {metrics.get('Correlation Coefficient_std', 'N/A'):.4f}")
#             print(f"    Runs: {metrics.get('num_runs', 'N/A')}")
    
#     return results

# def save_results_to_csv(results, output_file="analysis_results.csv"):
#     """
#     Save the analysis results to a CSV file
#     """
#     rows = []
    
#     for model_name, model_data in results.items():
#         for config, metrics in model_data.items():
#             row = {
#                 'Model': model_name,
#                 'Configuration': config,
#                 'RMSE_Mean': metrics.get('rmse_mean', np.nan),
#                 'RMSE_Std': metrics.get('rmse_std', np.nan),
#                 'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
#                 'NRMSE_Std': metrics.get('nrmse_std', np.nan),
#                 'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
#                 'Correlation_Std': metrics.get('Correlation Coefficient_std', np.nan),
#                 'Num_Runs': metrics.get('num_runs', np.nan)
#             }
#             rows.append(row)
    
#     df_results = pd.DataFrame(rows)
#     df_results.to_csv(output_file, index=False)
#     print(f"\nResults saved to {output_file}")
    
#     return df_results
# def print_summary_table(results):
#     """
#     Print a summary table with columns: Model, MPW, A*_B*_C*, RMSE, Correlation Coefficient
#     """
#     from tabulate import tabulate

#     table_rows = []
#     import re

#     for model_name, model_data in results.items():
#         for config, metrics in model_data.items():
#             # Try to extract mpw* and A*_B*_C* from config string
#             mpw = "N/A"
#             abc = "N/A"
#             # mpw: match mpw followed by digits or -digits (e.g., mpw-10, mpw100)
#             mpw_match = re.search(r'(mpw-?\d+)', config)
#             if mpw_match:
#                 mpw = mpw_match.group(1)
#             # A*_B*_C*
#             abc_match = re.search(r'(A\d+_B\d+_C\d+)', config)
#             if abc_match:
#                 abc = abc_match.group(1)
#             rmse = metrics.get('rmse_mean', np.nan)
#             cc = metrics.get('Correlation Coefficient_mean', np.nan)
#             table_rows.append([
#                 model_name,
#                 mpw,
#                 abc,
#                 f"{rmse:.5f}" if not np.isnan(rmse) else "N/A",
#                 f"{cc:.5f}" if not np.isnan(cc) else "N/A"
#             ])
#     headers = ["Model", "mpw*", "A*_B*_C*", "RMSE", "CC"]
#     print("\nSUMMARY TABLE")
#     print(tabulate(table_rows, headers=headers, tablefmt="github"))


# if __name__ == "__main__":
#     # Run the analysis
#     results = analyze_multitrain_results()
    
#     # Save results to CSV
#     df_summary = save_results_to_csv(results)
#     print_summary_table(results)
#     print("\nAnalysis complete!")

