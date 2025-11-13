# Start of Selection
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_cascaded_tank_results_across_lengths():
    """
    Analyze cascaded_tank experiment results from different models and configurations,
    across different dataset lengths.
    """
    # List of dataset lengths and their corresponding base directories
    dataset_lengths = [
        ("10", "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_10/cascaded_tank"),
        ("20", "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_20/cascaded_tank"),
        ("50", "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_50/cascaded_tank"),
        ("100", "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/cascaded_tank"),
    ]

    # Types to analyze
    types = ['-10', '100']
    # Indices to analyze (0-2 for cascaded_tank)
    indices = list(range(3))

    # Store all results and file summaries, keyed by dataset length
    all_results = {}
    all_file_summaries = {}

    for length, base_dir in dataset_lengths:
        results = {}
        file_summary = {}

        if not os.path.exists(base_dir):
            print(f"Base directory {base_dir} does not exist, skipping length {length}")
            all_results[length] = results
            all_file_summaries[length] = file_summary
            continue

        # Find all model directories
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"\n{'*'*10} Dataset Length: {length} {'*'*10}")
        print("Found model directories:", model_dirs)
        print("-" * 80)

        for model_dir in model_dirs:
            model_path = os.path.join(base_dir, model_dir)
            data_path = os.path.join(model_path, 'data')

            if not os.path.exists(data_path):
                print(f"No data directory found in {model_dir}")
                continue

            print(f"\nAnalyzing {model_dir}:")

            model_results = {}
            model_file_summary = {}

            for type_val in types:
                print(f"  Processing type: mpw{type_val}")

                pattern = f"cascaded_tank_h16_z2_n1_mpw{type_val}_No[0-9]"
                data_files = glob.glob(os.path.join(data_path, pattern))

                if not data_files:
                    print(f"    No files found for mpw{type_val}")
                    model_file_summary[f'mpw{type_val}'] = {'files_found': 0, 'files_read': 0, 'file_numbers': []}
                    continue

                print(f"    Found {len(data_files)} files")

                # Dictionary to store results for each index
                index_results = {i: [] for i in indices}

                # Track file reading
                files_read = 0
                file_numbers = []

                for data_file in data_files:
                    try:
                        # Extract number from filename
                        filename = os.path.basename(data_file)
                        number = filename.split('No')[-1]
                        file_numbers.append(number)

                        # Read CSV data
                        df = pd.read_csv(data_file)
                        files_read += 1

                        # Process each index (row) separately
                        for idx in indices:
                            if idx < len(df):
                                row_data = df.iloc[idx].copy()
                                row_data['file_number'] = number
                                row_data['type'] = type_val
                                row_data['model'] = model_dir
                                row_data['index'] = idx
                                index_results[idx].append(row_data)

                    except Exception as e:
                        print(f"    Error reading {data_file}: {str(e)}")
                        continue

                # Store file summary for this type
                model_file_summary[f'mpw{type_val}'] = {
                    'files_found': len(data_files),
                    'files_read': files_read,
                    'file_numbers': sorted(file_numbers)
                }

                # Calculate statistics for each index
                type_results = {}

                for idx in indices:
                    if index_results[idx]:
                        # Combine all data for this index
                        combined_df = pd.DataFrame(index_results[idx])

                        # Calculate statistics
                        metrics = ['rmse', 'nrmse', 'Correlation Coefficient']

                        index_stats = {}
                        for metric in metrics:
                            if metric in combined_df.columns:
                                mean_val = combined_df[metric].replace([np.inf, -np.inf], np.nan).dropna().mean()
                                std_val = combined_df[metric].std()
                                min_val = combined_df[metric].min()
                                max_val = combined_df[metric].max()
                                index_stats[f'{metric}_mean'] = mean_val
                                index_stats[f'{metric}_std'] = std_val
                                index_stats[f'{metric}_min'] = min_val
                                index_stats[f'{metric}_max'] = max_val

                        index_stats['num_files'] = len(set(combined_df['file_number']))
                        index_stats['total_runs'] = len(combined_df)

                        type_results[f'index_{idx}'] = index_stats

                        # Print summary for this index
                        print(f"    Index {idx}:")
                        print(f"      - RMSE: {combined_df['rmse'].mean():.4f} ± {combined_df['rmse'].std():.4f}")
                        print(f"      - NRMSE: {combined_df['nrmse'].mean():.4f} ± {combined_df['nrmse'].std():.4f}")
                        print(f"      - Correlation: {combined_df['Correlation Coefficient'].mean():.4f} ± {combined_df['Correlation Coefficient'].std():.4f}")
                        print(f"      - Files: {len(set(combined_df['file_number']))}, Runs: {len(combined_df)}")

                model_results[f'mpw{type_val}'] = type_results

            results[model_dir] = model_results
            file_summary[model_dir] = model_file_summary

        all_results[length] = results
        all_file_summaries[length] = file_summary

    # Print comprehensive summary across dataset lengths, sorted by int value
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SUMMARY OF ALL MODELS BY INDEX AND DATASET LENGTH")
    print("=" * 120)

    sorted_length_keys = sorted(all_results.keys(), key=lambda x: int(x))
    for length_key in sorted_length_keys:
        print(f"\n{'*'*10} Dataset Length: {length_key} {'*'*10}")
        results = all_results[length_key]
        for model_name, model_data in results.items():
            print(f"\n{model_name}:")
            print("-" * 80)
            for config, index_data in model_data.items():
                print(f"  {config}:")
                for index_key, metrics in index_data.items():
                    index_num = index_key.split('_')[1]
                    print(f"    Index {index_num}:")
                    print(f"      RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}")
                    print(f"      NRMSE: {metrics.get('nrmse_mean', 'N/A'):.4f} ± {metrics.get('nrmse_std', 'N/A'):.4f}")
                    print(f"      Correlation: {metrics.get('Correlation Coefficient_mean', 'N/A'):.4f} ± {metrics.get('Correlation Coefficient_std', 'N/A'):.4f}")
                    print(f"      Files: {metrics.get('num_files', 'N/A')}, Runs: {metrics.get('total_runs', 'N/A')}")

    return all_results, all_file_summaries

def print_training_process_summary_across_lengths(all_file_summaries):
    """
    Print a comprehensive summary of the training process and file counts across dataset lengths
    """
    print("\n" + "=" * 120)
    print("TRAINING PROCESS SUMMARY - FILES READ (BY DATASET LENGTH)")
    print("=" * 120)

    sorted_length_keys = sorted(all_file_summaries.keys(), key=lambda x: int(x))
    for length in sorted_length_keys:
        file_summary = all_file_summaries[length]
        print(f"\n{'*'*10} Dataset Length: {length} {'*'*10}")
        total_files_found = 0
        total_files_read = 0

        for model_name, model_data in file_summary.items():
            print(f"\n{model_name}:")
            print("-" * 60)
            model_total_found = 0
            model_total_read = 0

            for config, file_info in model_data.items():
                files_found = file_info['files_found']
                files_read = file_info['files_read']
                file_numbers = file_info['file_numbers']

                model_total_found += files_found
                model_total_read += files_read
                total_files_found += files_found
                total_files_read += files_read

                print(f"  {config}:")
                print(f"    Files found: {files_found}")
                print(f"    Files successfully read: {files_read}")
                if file_numbers:
                    print(f"    File numbers: {', '.join(file_numbers)}")
                else:
                    print(f"    File numbers: None")

            print(f"  Model totals: {model_total_found} found, {model_total_read} read")

        print(f"\n  Dataset Length {length} totals: {total_files_found} found, {total_files_read} read")
        print(f"  Success rate: {(total_files_read/total_files_found*100):.1f}%" if total_files_found > 0 else "No files found")

    print("=" * 120)
    return all_file_summaries

def print_summary_tables_by_length_and_method(all_results):
    """
    Print summary tables (avg nrmse, avg rmse, avg correlation coefficient, min rmse)
    with dataset length as rows and methods as columns (each method has two columns for type -10 and 100).
    If data cannot be loaded, use nan.
    """
    # Define the methods and their directory names
    method_map = {
        "AE-RNN": "AE-RNN_None",
        "AE-RNN-U": "AE-RNN-U_None",
        "AE-RNN-XU": "AE-RNN-XU_None",
        "MLP-U": "MLP-U_None",
        "LIU-U": "LIU-U_None"
    }
    types = ['-10', '100']
    type_names = {'-10': 'mpw-10', '100': 'mpw100'}
    dataset_lengths = sorted(all_results.keys(), key=lambda x: int(x))

    # Prepare the table columns: for each method, two columns for each type
    columns = []
    for method in method_map:
        for t in types:
            columns.append(f"{method} ({type_names[t]})")

    # Helper to get stats for a given length, method, type
    def get_stats(length, method_dir, type_val, stat_key):
        try:
            model_data = all_results[length][method_dir]
            type_data = model_data.get(f"mpw{type_val}", {})
            # Average across all indices
            vals = []
            for idx in range(3):
                idx_key = f"index_{idx}"
                if idx_key in type_data and stat_key in type_data[idx_key]:
                    v = type_data[idx_key][stat_key]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        vals.append(v)
            if vals:
                return float(np.mean(vals))
            else:
                return np.nan
        except Exception:
            return np.nan

    # Print tables for each metric
    metrics = [
        ("Average NRMSE", "nrmse_mean"),
        ("Average RMSE", "rmse_mean"),
        ("Average Correlation Coefficient", "Correlation Coefficient_mean"),
        ("Min RMSE", "rmse_min"),
    ]
    for metric_name, stat_key in metrics:
        print("\n" + "="*80)
        print(f"{metric_name} (Rows: Length, Columns: Method-Type)")
        print("="*80)
        table = []
        for length in dataset_lengths:
            row = []
            for method, method_dir in method_map.items():
                for t in types:
                    val = get_stats(length, method_dir, t, stat_key)
                    row.append(val)
            table.append(row)
        df = pd.DataFrame(table, columns=columns, index=dataset_lengths)
        print(df.round(4))
    print("="*80)

if __name__ == "__main__":
    # Run the analysis
    all_results, all_file_summaries = analyze_cascaded_tank_results_across_lengths()

    # Print training process summary
    print_training_process_summary_across_lengths(all_file_summaries)

    # Print summary tables for performance across dataset lengths and methods
    print_summary_tables_by_length_and_method(all_results)

    print("\nAnalysis complete!")
# End of Selection
#         # Dictionary to store results for each model
#         results = {}
#         # Dictionary to store file counts for summary
#         file_summary = {}

#         # Find all model directories
#         if not os.path.exists(base_dir):
#             print(f"Base directory does not exist: {base_dir}")
#             continue
#         model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

#         print(f"\n{'='*10} Dataset Length: {length} ({base_dir}) {'='*10}")
#         print("Found model directories:", model_dirs)
#         print("-" * 80)

#         for model_dir in model_dirs:
#             model_path = os.path.join(base_dir, model_dir)
#             data_path = os.path.join(model_path, 'data')

#             if not os.path.exists(data_path):
#                 print(f"No data directory found in {model_dir}")
#                 continue

#             print(f"\nAnalyzing {model_dir}:")

#             model_results = {}
#             model_file_summary = {}

#             for type_val in types:
#                 print(f"  Processing type: mpw{type_val}")

#                 # Find all files for this type
#                 pattern = f"cascaded_tank_h16_z2_n1_mpw{type_val}_No[0-9]"
#                 data_files = glob.glob(os.path.join(data_path, pattern))

#                 if not data_files:
#                     print(f"    No files found for mpw{type_val}")
#                     model_file_summary[f'mpw{type_val}'] = {'files_found': 0, 'files_read': 0, 'file_numbers': []}
#                     continue

#                 print(f"    Found {len(data_files)} files")

#                 # Dictionary to store results for each index
#                 index_results = {i: [] for i in indices}

#                 # Track file reading
#                 files_read = 0
#                 file_numbers = []

#                 for data_file in data_files:
#                     try:
#                         # Extract number from filename
#                         filename = os.path.basename(data_file)
#                         number = filename.split('No')[-1]

#                         file_numbers.append(number)

#                         # Read CSV data
#                         df = pd.read_csv(data_file)
#                         files_read += 1

#                         # Process each index (row) separately
#                         for idx in indices:
#                             if idx < len(df):
#                                 row_data = df.iloc[idx].copy()
#                                 row_data['file_number'] = number
#                                 row_data['type'] = type_val
#                                 row_data['model'] = model_dir
#                                 row_data['index'] = idx
#                                 index_results[idx].append(row_data)

#                     except Exception as e:
#                         print(f"    Error reading {data_file}: {str(e)}")
#                         continue

#                 # Store file summary for this type
#                 model_file_summary[f'mpw{type_val}'] = {
#                     'files_found': len(data_files),
#                     'files_read': files_read,
#                     'file_numbers': sorted(file_numbers)
#                 }

#                 # Calculate statistics for each index
#                 type_results = {}

#                 for idx in indices:
#                     if index_results[idx]:
#                         # Combine all data for this index
#                         combined_df = pd.DataFrame(index_results[idx])

#                         # Calculate statistics
#                         metrics = ['rmse', 'nrmse', 'Correlation Coefficient']

#                         index_stats = {}
#                         for metric in metrics:
#                             if metric in combined_df.columns:
#                                 mean_val = combined_df[metric].replace([np.inf, -np.inf], np.nan).dropna().mean()
#                                 std_val = combined_df[metric].std()
#                                 min_val = combined_df[metric].min()
#                                 max_val = combined_df[metric].max()
#                                 index_stats[f'{metric}_mean'] = mean_val
#                                 index_stats[f'{metric}_std'] = std_val
#                                 index_stats[f'{metric}_min'] = min_val
#                                 index_stats[f'{metric}_max'] = max_val

#                         index_stats['num_files'] = len(set(combined_df['file_number']))
#                         index_stats['total_runs'] = len(combined_df)

#                         type_results[f'index_{idx}'] = index_stats

#                         # Print summary for this index
#                         print(f"    Index {idx}:")
#                         print(f"      - RMSE: {combined_df['rmse'].mean():.4f} ± {combined_df['rmse'].std():.4f}")
#                         print(f"      - NRMSE: {combined_df['nrmse'].mean():.4f} ± {combined_df['nrmse'].std():.4f}")
#                         print(f"      - Correlation: {combined_df['Correlation Coefficient'].mean():.4f} ± {combined_df['Correlation Coefficient'].std():.4f}")
#                         print(f"      - Files: {len(set(combined_df['file_number']))}, Runs: {len(combined_df)}")

#                 model_results[f'mpw{type_val}'] = type_results

#             results[model_dir] = model_results
#             file_summary[model_dir] = model_file_summary

#         all_results[length] = results
#         all_file_summaries[length] = file_summary

#     # Print comprehensive summary across dataset lengths, sorted by int value
#     print("\n" + "=" * 120)
#     print("COMPREHENSIVE SUMMARY OF ALL MODELS BY INDEX AND DATASET LENGTH")
#     print("=" * 120)

#     sorted_length_keys = sorted(all_results.keys(), key=lambda x: int(x))
#     for length_key in sorted_length_keys:
#         print(f"\n{'*'*10} Dataset Length: {length_key} {'*'*10}")
#         results = all_results[length_key]
#         for model_name, model_data in results.items():
#             print(f"\n{model_name}:")
#             print("-" * 80)
#             for config, index_data in model_data.items():
#                 print(f"  {config}:")
#                 for index_key, metrics in index_data.items():
#                     index_num = index_key.split('_')[1]
#                     print(f"    Index {index_num}:")
#                     print(f"      RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}")
#                     print(f"      NRMSE: {metrics.get('nrmse_mean', 'N/A'):.4f} ± {metrics.get('nrmse_std', 'N/A'):.4f}")
#                     print(f"      Correlation: {metrics.get('Correlation Coefficient_mean', 'N/A'):.4f} ± {metrics.get('Correlation Coefficient_std', 'N/A'):.4f}")
#                     print(f"      Files: {metrics.get('num_files', 'N/A')}, Runs: {metrics.get('total_runs', 'N/A')}")

#     return all_results, all_file_summaries

# def print_training_process_summary_across_lengths(all_file_summaries):
#     """
#     Print a comprehensive summary of the training process and file counts across dataset lengths
#     """
#     print("\n" + "=" * 120)
#     print("TRAINING PROCESS SUMMARY - FILES READ (BY DATASET LENGTH)")
#     print("=" * 120)

#     sorted_length_keys = sorted(all_file_summaries.keys(), key=lambda x: int(x))
#     for length in sorted_length_keys:
#         file_summary = all_file_summaries[length]
#         print(f"\n{'*'*10} Dataset Length: {length} {'*'*10}")
#         total_files_found = 0
#         total_files_read = 0

#         for model_name, model_data in file_summary.items():
#             print(f"\n{model_name}:")
#             print("-" * 60)
#             model_total_found = 0
#             model_total_read = 0

#             for config, file_info in model_data.items():
#                 files_found = file_info['files_found']
#                 files_read = file_info['files_read']
#                 file_numbers = file_info['file_numbers']

#                 model_total_found += files_found
#                 model_total_read += files_read
#                 total_files_found += files_found
#                 total_files_read += files_read

#                 print(f"  {config}:")
#                 print(f"    Files found: {files_found}")
#                 print(f"    Files successfully read: {files_read}")
#                 if file_numbers:
#                     print(f"    File numbers: {', '.join(file_numbers)}")
#                 else:
#                     print(f"    File numbers: None")

#             print(f"  Model totals: {model_total_found} found, {model_total_read} read")

#         print(f"\n  Dataset Length {length} totals: {total_files_found} found, {total_files_read} read")
#         print(f"  Success rate: {(total_files_read/total_files_found*100):.1f}%" if total_files_found > 0 else "No files found")

#     print("=" * 120)
#     return all_file_summaries

# if __name__ == "__main__":
#     # Run the analysis across all dataset lengths
#     all_results, all_file_summaries = analyze_cascaded_tank_results_across_lengths()

#     # Print training process summary across all lengths
#     # print_training_process_summary_across_lengths(all_file_summaries)
    

#     print("\nAnalysis complete!")
#         print(f"\n{'='*40}\nAnalyzing dataset length: {length}\nBase dir: {base_dir}\n{'='*40}")
#         results = {}
#         file_summary = {}

#         if not os.path.exists(base_dir):
#             print(f"  Base directory {base_dir} does not exist. Skipping.")
#             all_results[length] = results
#             all_file_summaries[length] = file_summary
#             continue

#         # Find all model directories
#         model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
#         print("  Found model directories:", model_dirs)
#         print("  " + "-" * 60)

#         for model_dir in model_dirs:
#             model_path = os.path.join(base_dir, model_dir)
#             data_path = os.path.join(model_path, 'data')

#             if not os.path.exists(data_path):
#                 print(f"  No data directory found in {model_dir}")
#                 continue

#             print(f"\n  Analyzing {model_dir}:")
#             model_results = {}
#             model_file_summary = {}

#             for type_val in types:
#                 print(f"    Processing type: mpw{type_val}")

#                 pattern = f"cascaded_tank_h16_z2_n1_mpw{type_val}_No[0-9]"
#                 data_files = glob.glob(os.path.join(data_path, pattern))

#                 if not data_files:
#                     print(f"      No files found for mpw{type_val}")
#                     model_file_summary[f'mpw{type_val}'] = {'files_found': 0, 'files_read': 0, 'file_numbers': []}
#                     continue

#                 print(f"      Found {len(data_files)} files")
#                 index_results = {i: [] for i in indices}
#                 files_read = 0
#                 file_numbers = []

#                 for data_file in data_files:
#                     try:
#                         filename = os.path.basename(data_file)
#                         number = filename.split('No')[-1]
#                         file_numbers.append(number)
#                         df = pd.read_csv(data_file)
#                         files_read += 1

#                         for idx in indices:
#                             if idx < len(df):
#                                 row_data = df.iloc[idx].copy()
#                                 row_data['file_number'] = number
#                                 row_data['type'] = type_val
#                                 row_data['model'] = model_dir
#                                 row_data['index'] = idx
#                                 index_results[idx].append(row_data)
#                     except Exception as e:
#                         print(f"      Error reading {data_file}: {str(e)}")
#                         continue

#                 model_file_summary[f'mpw{type_val}'] = {
#                     'files_found': len(data_files),
#                     'files_read': files_read,
#                     'file_numbers': sorted(file_numbers, key=lambda x: int(x) if x.isdigit() else x)
#                 }

#                 type_results = {}
#                 for idx in indices:
#                     if index_results[idx]:
#                         combined_df = pd.DataFrame(index_results[idx])
#                         metrics = ['rmse', 'nrmse', 'Correlation Coefficient']
#                         index_stats = {}
#                         for metric in metrics:
#                             if metric in combined_df.columns:
#                                 mean_val = combined_df[metric].replace([np.inf, -np.inf], np.nan).dropna().mean()
#                                 std_val = combined_df[metric].std()
#                                 min_val = combined_df[metric].min()
#                                 max_val = combined_df[metric].max()
#                                 index_stats[f'{metric}_mean'] = mean_val
#                                 index_stats[f'{metric}_std'] = std_val
#                                 index_stats[f'{metric}_min'] = min_val
#                                 index_stats[f'{metric}_max'] = max_val

#                         index_stats['num_files'] = len(set(combined_df['file_number']))
#                         index_stats['total_runs'] = len(combined_df)
#                         type_results[f'index_{idx}'] = index_stats

#                         # Print summary for this index
#                         print(f"      Index {idx}:")
#                         print(f"        - RMSE: {combined_df['rmse'].mean():.4f} ± {combined_df['rmse'].std():.4f}")
#                         print(f"        - NRMSE: {combined_df['nrmse'].mean():.4f} ± {combined_df['nrmse'].std():.4f}")
#                         print(f"        - Correlation: {combined_df['Correlation Coefficient'].mean():.4f} ± {combined_df['Correlation Coefficient'].std():.4f}")
#                         print(f"        - Files: {len(set(combined_df['file_number']))}, Runs: {len(combined_df)}")

#                 model_results[f'mpw{type_val}'] = type_results

#             results[model_dir] = model_results
#             file_summary[model_dir] = model_file_summary

#         all_results[length] = results
#         all_file_summaries[length] = file_summary

#     # Print comprehensive summary across dataset lengths, sorted by int value
#     print("\n" + "=" * 120)
#     print("COMPREHENSIVE SUMMARY OF ALL MODELS BY INDEX AND DATASET LENGTH")
#     print("=" * 120)

#     sorted_length_keys = sorted(all_results.keys(), key=lambda x: int(x))
#     for length_key in sorted_length_keys:
#         print(f"\n{'*'*10} Dataset Length: {length_key} {'*'*10}")
#         results = all_results[length_key]
#         for model_name, model_data in results.items():
#             print(f"\n{model_name}:")
#             print("-" * 80)
#             for config, index_data in model_data.items():
#                 print(f"  {config}:")
#                 for index_key, metrics in index_data.items():
#                     index_num = index_key.split('_')[1]
#                     print(f"    Index {index_num}:")
#                     print(f"      RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}")
#                     print(f"      NRMSE: {metrics.get('nrmse_mean', 'N/A'):.4f} ± {metrics.get('nrmse_std', 'N/A'):.4f}")
#                     print(f"      Correlation: {metrics.get('Correlation Coefficient_mean', 'N/A'):.4f} ± {metrics.get('Correlation Coefficient_std', 'N/A'):.4f}")
#                     print(f"      Files: {metrics.get('num_files', 'N/A')}, Runs: {metrics.get('total_runs', 'N/A')}")

#     return all_results, all_file_summaries

# def print_training_process_summary_across_lengths(all_file_summaries):
#     """
#     Print a comprehensive summary of the training process and file counts across dataset lengths
#     """
#     print("\n" + "=" * 120)
#     print("TRAINING PROCESS SUMMARY - FILES READ (BY DATASET LENGTH)")
#     print("=" * 120)

#     sorted_length_keys = sorted(all_file_summaries.keys(), key=lambda x: int(x))
#     for length in sorted_length_keys:
#         file_summary = all_file_summaries[length]
#         print(f"\n{'*'*10} Dataset Length: {length} {'*'*10}")
#         total_files_found = 0
#         total_files_read = 0

#         for model_name, model_data in file_summary.items():
#             print(f"\n{model_name}:")
#             print("-" * 60)
#             model_total_found = 0
#             model_total_read = 0

#             for config, file_info in model_data.items():
#                 files_found = file_info['files_found']
#                 files_read = file_info['files_read']
#                 file_numbers = file_info['file_numbers']

#                 model_total_found += files_found
#                 model_total_read += files_read
#                 total_files_found += files_found
#                 total_files_read += files_read

#                 print(f"  {config}:")
#                 print(f"    Files found: {files_found}")
#                 print(f"    Files successfully read: {files_read}")
#                 if file_numbers:
#                     print(f"    File numbers: {', '.join(file_numbers)}")
#                 else:
#                     print(f"    File numbers: None")

#             print(f"  Model totals: {model_total_found} found, {model_total_read} read")

#         print(f"\n  Dataset Length {length} totals: {total_files_found} found, {total_files_read} read")
#         print(f"  Success rate: {(total_files_read/total_files_found*100):.1f}%" if total_files_found > 0 else "No files found")

#     # print("=" * 120)
#     # return all_file_summaries

    
#     # Base directory containing the results
#     base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/cascaded_tank"
    
#     # Dictionary to store results for each model
#     results = {}
    
#     # Dictionary to store file counts for summary
#     file_summary = {}
    
#     # Find all model directories
#     model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
#     print("Found model directories:", model_dirs)
#     print("-" * 80)
    
#     # Types to analyze
#     types = ['-10', '100']
    
#     # Indices to analyze (0-2 for cascaded_tank)
#     indices = list(range(3))
    
#     for model_dir in model_dirs:
#         model_path = os.path.join(base_dir, model_dir)
#         data_path = os.path.join(model_path, 'data')
        
#         if not os.path.exists(data_path):
#             print(f"No data directory found in {model_dir}")
#             continue
            
#         print(f"\nAnalyzing {model_dir}:")
        
#         model_results = {}
#         model_file_summary = {}
        
#         for type_val in types:
#             print(f"  Processing type: mpw{type_val}")
            
#             # Find all files for this type
#             # pattern = f"cascaded_tank_h16_z2_n1_mpw{type_val}_No*"
#             pattern = f"cascaded_tank_h16_z2_n1_mpw{type_val}_No[0-9]"
            

#             data_files = glob.glob(os.path.join(data_path, pattern))
            
#             if not data_files:
#                 print(f"    No files found for mpw{type_val}")
#                 model_file_summary[f'mpw{type_val}'] = {'files_found': 0, 'files_read': 0, 'file_numbers': []}
#                 continue
                
#             print(f"    Found {len(data_files)} files")
            
#             # Dictionary to store results for each index
#             index_results = {i: [] for i in indices}
            
#             # Track file reading
#             files_read = 0
#             file_numbers = []
            
#             for data_file in data_files:
                
#                 try:
#                     # Extract number from filename
#                     filename = os.path.basename(data_file)
#                     number = filename.split('No')[-1]
    
#                     file_numbers.append(number)
                    
#                     # Read CSV data
#                     df = pd.read_csv(data_file)
#                     files_read += 1
                    
#                     # Process each index (row) separately
#                     for idx in indices:
#                         if idx < len(df):
#                             row_data = df.iloc[idx].copy()
#                             row_data['file_number'] = number
#                             row_data['type'] = type_val
#                             row_data['model'] = model_dir
#                             row_data['index'] = idx
#                             index_results[idx].append(row_data)
                    
#                 except Exception as e:
#                     print(f"    Error reading {data_file}: {str(e)}")
#                     continue
            
#             # Store file summary for this type
#             model_file_summary[f'mpw{type_val}'] = {
#                 'files_found': len(data_files),
#                 'files_read': files_read,
#                 'file_numbers': sorted(file_numbers)
#             }
            
#             # Calculate statistics for each index
#             type_results = {}
            
#             for idx in indices:
#                 if index_results[idx]:
#                     # Combine all data for this index
#                     combined_df = pd.DataFrame(index_results[idx])
                    
#                     # Calculate statistics
#                     metrics = ['rmse', 'nrmse', 'Correlation Coefficient']
                    
#                     index_stats = {}
#                     for metric in metrics:
#                         if metric in combined_df.columns:
#                             mean_val = combined_df[metric].replace([np.inf, -np.inf], np.nan).dropna().mean()
#                             std_val = combined_df[metric].std()
#                             min_val = combined_df[metric].min()
#                             max_val = combined_df[metric].max()
#                                             #    cleaned_series = cleaned_series[~cleaned_series.isin([float('inf'), float('-inf')])]
#                             index_stats[f'{metric}_mean'] = mean_val
#                             index_stats[f'{metric}_std'] = std_val
#                             index_stats[f'{metric}_min'] = min_val
#                             index_stats[f'{metric}_max'] = max_val
                    
#                     index_stats['num_files'] = len(set(combined_df['file_number']))
#                     index_stats['total_runs'] = len(combined_df)
                    
#                     type_results[f'index_{idx}'] = index_stats
                    
#                     # Print summary for this index
#                     print(f"    Index {idx}:")
#                     print(f"      - RMSE: {combined_df['rmse'].mean():.4f} ± {combined_df['rmse'].std():.4f}")
#                     print(f"      - NRMSE: {combined_df['nrmse'].mean():.4f} ± {combined_df['nrmse'].std():.4f}")
#                     print(f"      - Correlation: {combined_df['Correlation Coefficient'].mean():.4f} ± {combined_df['Correlation Coefficient'].std():.4f}")
#                     print(f"      - Files: {len(set(combined_df['file_number']))}, Runs: {len(combined_df)}")
            
#             model_results[f'mpw{type_val}'] = type_results
        
#         results[model_dir] = model_results
#         file_summary[model_dir] = model_file_summary
    
#     # Print comprehensive summary
#     print("\n" + "=" * 120)
#     print("COMPREHENSIVE SUMMARY OF ALL MODELS BY INDEX")
#     print("=" * 120)
    
#     for model_name, model_data in results.items():
#         print(f"\n{model_name}:")
#         print("-" * 80)
        
#         for config, index_data in model_data.items():
#             print(f"  {config}:")
#             for index_key, metrics in index_data.items():
#                 index_num = index_key.split('_')[1]
#                 print(f"    Index {index_num}:")
#                 print(f"      RMSE: {metrics.get('rmse_mean', 'N/A'):.4f} ± {metrics.get('rmse_std', 'N/A'):.4f}")
#                 print(f"      NRMSE: {metrics.get('nrmse_mean', 'N/A'):.4f} ± {metrics.get('nrmse_std', 'N/A'):.4f}")
#                 print(f"      Correlation: {metrics.get('Correlation Coefficient_mean', 'N/A'):.4f} ± {metrics.get('Correlation Coefficient_std', 'N/A'):.4f}")
#                 print(f"      Files: {metrics.get('num_files', 'N/A')}, Runs: {metrics.get('total_runs', 'N/A')}")
    
#     return results, file_summary

# def print_training_process_summary(file_summary):
#     """
#     Print a comprehensive summary of the training process and file counts
#     """
#     print("\n" + "=" * 120)
#     print("TRAINING PROCESS SUMMARY - FILES READ")
#     print("=" * 120)
    
#     total_files_found = 0
#     total_files_read = 0
    
#     for model_name, model_data in file_summary.items():
#         print(f"\n{model_name}:")
#         print("-" * 60)
        
#         model_total_found = 0
#         model_total_read = 0
        
#         for config, file_info in model_data.items():
#             files_found = file_info['files_found']
#             files_read = file_info['files_read']
#             file_numbers = file_info['file_numbers']
            
#             model_total_found += files_found
#             model_total_read += files_read
#             total_files_found += files_found
#             total_files_read += files_read
            
#             print(f"  {config}:")
#             print(f"    Files found: {files_found}")
#             print(f"    Files successfully read: {files_read}")
#             if file_numbers:
#                 print(f"    File numbers: {', '.join(file_numbers)}")
#             else:
#                 print(f"    File numbers: None")
        
#         print(f"  Model totals: {model_total_found} found, {model_total_read} read")
    
#     print(f"\n" + "=" * 60)
#     print(f"OVERALL SUMMARY:")
#     print(f"Total files found across all models: {total_files_found}")
#     print(f"Total files successfully read: {total_files_read}")
#     print(f"Success rate: {(total_files_read/total_files_found*100):.1f}%" if total_files_found > 0 else "No files found")
#     print("=" * 60)
    
#     return file_summary

# def save_cascaded_tank_results_to_csv(results, output_file="cascaded_tank_analysis_results.csv"):
#     """
#     Save the cascaded_tank analysis results to a CSV file
#     """
#     rows = []
    
#     for model_name, model_data in results.items():
#         for config, index_data in model_data.items():
#             for index_key, metrics in index_data.items():
#                 print(index_key)
#                 index_num = index_key
#                 row = {
#                     'Model': model_name,
#                     'Configuration': config,
#                     'Index': index_num,
#                     'RMSE_Mean': metrics.get('rmse_mean', np.nan),
#                     'RMSE_Std': metrics.get('rmse_std', np.nan),
#                     'RMSE_Min': metrics.get('rmse_min', np.nan),
#                     'RMSE_Max': metrics.get('rmse_max', np.nan),
#                     'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
#                     'NRMSE_Std': metrics.get('nrmse_std', np.nan),
#                     'NRMSE_Min': metrics.get('nrmse_min', np.nan),
#                     'NRMSE_Max': metrics.get('nrmse_max', np.nan),
#                     'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
#                     'Correlation_Std': metrics.get('Correlation Coefficient_std', np.nan),
#                     'Correlation_Min': metrics.get('Correlation Coefficient_min', np.nan),
#                     'Correlation_Max': metrics.get('Correlation Coefficient_max', np.nan),
#                     'Num_Files': metrics.get('num_files', np.nan),
#                     'Total_Runs': metrics.get('total_runs', np.nan)
#                 }
#                 rows.append(row)
    
#     df_results = pd.DataFrame(rows)
#     df_results.to_csv(output_file, index=False)
#     print(f"\nResults saved to {output_file}")
    
#     return df_results

# def save_file_summary_to_csv(file_summary, output_file="cascaded_tank_file_summary.csv"):
#     """
#     Save the file summary to a CSV file
#     """
#     rows = []
    
#     for model_name, model_data in file_summary.items():
#         for config, file_info in model_data.items():
#             row = {
#                 'Model': model_name,
#                 'Configuration': config,
#                 'Files_Found': file_info['files_found'],
#                 'Files_Read': file_info['files_read'],
#                 'File_Numbers': ', '.join(file_info['file_numbers']) if file_info['file_numbers'] else 'None'
#             }
#             rows.append(row)
    
#     df_summary = pd.DataFrame(rows)
#     df_summary.to_csv(output_file, index=False)
#     print(f"File summary saved to {output_file}")
    
#     return df_summary

# def compare_models_across_types_and_indices(results):
#     """
#     Compare models across different types and indices
#     """
#     print("\n" + "=" * 120)
#     print("MODEL COMPARISON ACROSS TYPES AND INDICES")
#     print("=" * 120)
    
#     # Create comparison table
#     comparison_data = []
    
#     for model_name, model_data in results.items():
#         for config, index_data in model_data.items():
#             for index_key, metrics in index_data.items():
#                 # print(index_key)
#                 index_num = index_key
#                 comparison_data.append({
#                     'Model': model_name,
#                     'Type': config,
#                     'Index': index_num,
#                     'RMSE_Mean': metrics.get('rmse_mean', np.nan),
#                     'RMSE_Min': metrics.get('rmse_min', np.nan),
#                     'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
#                     'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
#                     'Total_Runs': metrics.get('total_runs', np.nan)
#                 })
    
#     df_comparison = pd.DataFrame(comparison_data)
    
#     # Pivot tables for different views
#     print("\nRMSE Comparison by Model and Type (averaged across indices):")
#     pivot_rmse = df_comparison.groupby(['Model', 'Type'])['RMSE_Mean'].mean().unstack()
#     print(pivot_rmse.round(4))
    
#     print("\nNRMSE Comparison by Model and Type (averaged across indices):")
#     pivot_nrmse = df_comparison.groupby(['Model', 'Type'])['NRMSE_Mean'].mean().unstack()
#     print(pivot_nrmse.round(4))
    
#     print("\nCorrelation Comparison by Model and Type (averaged across indices):")
#     pivot_corr = df_comparison.groupby(['Model', 'Type'])['Correlation_Mean'].mean().unstack()
#     print(pivot_corr.round(4))\
        
#     print("\nMin rmse by Model and Type (averaged across indices):")
#     pivot_rmsemin = df_comparison.groupby(['Model', 'Type'])['RMSE_Min'].mean().unstack()
#     print(pivot_rmsemin.round(4))
    
#     # Index-specific comparisons - Only NRMSE
#     print("\n" + "-" * 80)
#     print("INDEX-SPECIFIC COMPARISONS - NRMSE ONLY")
#     print("-" * 80)
    
#     for idx in range(3):  # cascaded_tank has 3 indices (0, 1, 2)
#         print(f"\nIndex {idx} - NRMSE Comparison:")
#         idx_data = df_comparison[df_comparison['Index'] == str(idx)]
#         pivot_idx_nrmse = idx_data.pivot(index='Model', columns='Type', values='NRMSE_Mean')
#         print(pivot_idx_nrmse.round(4))
    
#     return df_comparison

# if __name__ == "__main__":
#     # Run the analysis
#     results, file_summary = analyze_cascaded_tank_results_across_lengths()
    
#     # Print training process summary
#     print_training_process_summary_across_lengths(file_summary)
    
#     # Save results to CSV
#     df_summary = save_cascaded_tank_results_to_csv(results)
#     # df_file_summary = save_file_summary_to_csv(file_summary)
    
#     # Compare models across types and indices
#     df_comparison = compare_models_across_types_and_indices(results)
    
#     print("\nAnalysis complete!")
