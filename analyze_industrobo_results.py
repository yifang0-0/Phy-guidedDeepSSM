import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def analyze_industrobo_results():
    """
    Analyze industrobo experiment results from different models and configurations
    """
    
    # Base directory containing the results
    base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/industrobo"
    base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100_a1/industrobo"
    # base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100_a0/industrobo"
    base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100_realshift/industrobo"
    base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/1010_robo_real_multi/industrobo"
    base_dir = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/1010_robo_sim_test/industrobo"


    # Support both a single directory and multiple directories for different lengths
    def get_base_dirs(base_input):
        """
        Given a string or list of strings, return a list of base directories.
        If a string contains '*', expand it as a glob pattern.
        """
        import glob
        if isinstance(base_input, str):
            if '*' in base_input:
                return sorted(glob.glob(base_input))
            else:
                return [base_input]
        elif isinstance(base_input, list):
            dirs = []
            for b in base_input:
                if '*' in b:
                    dirs.extend(sorted(glob.glob(b)))
                else:
                    dirs.append(b)
            return dirs
        else:
            raise ValueError("base_input must be a string or list of strings")

    # Example usage: set this to a pattern or a list of patterns/dirs
    # The correct pattern should point to the parent directory, not the 'industrobo' subdir
    # e.g. /home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_*_realshift
    base_input = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_*"
    base_input = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_*_realshift"
    base_input = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/1010_robo_real_multi"

    
    base_dirs = get_base_dirs(base_input)

    def print_avg_nrmse_table(base_dirs):
        """
        Print a table of averaged NRMSE for all trained models under each base_dir (for different lengths).
        This will look for files like industrobo_h64_z12_n3_mpw-10_No0, No1, ... (no extension) in the data/ subdir.
        """
        import re
        import pandas as pd
        import numpy as np
        import os
        import glob

        print("/////////////////////////////////////////////")
        print("print result for all lengths")
        table_rows = []
        for base_dir in base_dirs:
            # The model subdirs are under base_dir/industrobo
            industrobo_dir = os.path.join(base_dir, "industrobo")
            print(f"Scanning: {industrobo_dir}")
            if not os.path.exists(industrobo_dir):
                print(f"industrobo directory does not exist: {industrobo_dir}")
                continue

            # Extract the length from the path for labeling
            match = re.search(r'full_(\d+)_realshift', base_dir)
            length_label = match.group(1) if match else base_dir

            smodel_dirs = [d for d in os.listdir(industrobo_dir) if os.path.isdir(os.path.join(industrobo_dir, d))]
            
            for model_dir in smodel_dirs:
                # We want to group/average only among No* for the same pg_type (mpw-10/NN, mpw100/pg), not mix them.
                import re
                model_path = os.path.join(industrobo_dir, model_dir)
                data_path = os.path.join(model_path, 'data')
                if not os.path.exists(data_path):
                    continue

                # Find all files matching industrobo_h64_z12_n3_mpw*No* (no extension)
                data_files = glob.glob(os.path.join(data_path, "industrobo_h64_z12_n3_mpw*No*"))

                # Group files by pg_type (NN/pg/other) using the mpw value in the filename
                pgtype_to_files = {}
                for data_file in data_files:
                    # Extract mpw value from filename
                    fname = os.path.basename(data_file)
                    mpw_match = re.search(r'mpw(-?\d+)', fname)
                    if mpw_match:
                        mpw_val = mpw_match.group(1)
                        if mpw_val == '-10':
                            pg_type = 'NN'
                        elif mpw_val == '100':
                            pg_type = 'pg'
                        else:
                            pg_type = mpw_val  # fallback, use raw value
                    else:
                        pg_type = 'unknown'
                    pgtype_to_files.setdefault(pg_type, []).append(data_file)

                for pg_type, files in pgtype_to_files.items():
                    nrmse_list = []
                    cc_list = []
                    # For printing, replace mpw-10 with mpwNN, mpw100 with mpwpg in model_dir
                    if pg_type == 'NN':
                        model_dir_print = model_dir.replace('mpw-10', 'NN')
                    elif pg_type == 'pg':
                        model_dir_print = model_dir.replace('mpw100', 'pg')
                    else:
                        model_dir_print = model_dir
                    print(f"Model: {model_dir_print}, pg_type: {pg_type}, files: {files}")
                    for data_file in files:
                        try:
                            # Try reading as CSV first, if fails, try as txt
                            try:
                                df = pd.read_csv(data_file)
                            except Exception:
                                # Try with .csv extension
                                if os.path.exists(data_file + ".csv"):
                                    df = pd.read_csv(data_file + ".csv")
                                else:
                                    raise
                            if 'nrmse' in df.columns:
                                nrmse_list.extend(df['nrmse'].values)
                            if 'Correlation Coefficient' in df.columns:
                                cc_list.extend(df['Correlation Coefficient'].values)
                        except Exception as e:
                            print(f"Error reading {data_file}: {e}")
                            continue
                    if nrmse_list:
                        avg_nrmse = np.mean(nrmse_list)
                        std_nrmse = np.std(nrmse_list)
                        avg_cc = np.mean(cc_list)
                        table_rows.append({
                            "Length": length_label,
                            "Model": model_dir_print,
                            "pg_type": pg_type,
                            "Avg_NRMSE": avg_nrmse,
                            # "Std_NRMSE": std_nrmse,
                            "Avg_cc": avg_cc,
                            "Num_Files": len(files)
                        })
                    else:
                        table_rows.append({
                            "Length": length_label,
                            "Model": model_dir_print,
                            "pg_type": pg_type,
                            "Avg_NRMSE": np.nan,
                            "Std_NRMSE": np.nan,
                            "Num_Files": len(files)
                        })

        if table_rows:
            df_table = pd.DataFrame(table_rows)
            df_table = df_table.sort_values(["Length", "Model"])
            print("\nAveraged NRMSE Table (across all models and lengths):")
            print(df_table.to_string(index=False))
        else:
            print("No NRMSE data found.")

    # Call the function to print the table
    print_avg_nrmse_table(base_dirs)
    
    
    
    
    # Dictionary to store results for each model
    results = {}
    
    # Dictionary to store file counts for summary
    file_summary = {}
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    print("Found model directories:", model_dirs)
    print("-" * 80)
    
    # Types to analyze
    types = ['-10', '100']
    
    # Indices to analyze (0-5)
    indices = list(range(6))
    
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
            
            # Find all files for this type
            pattern = f"industrobo_h64_z12_n3_mpw{type_val}_No*"
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
                            mean_val = combined_df[metric].mean()
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
    
    # Print comprehensive summary
    print("\n" + "=" * 120)
    print("COMPREHENSIVE SUMMARY OF ALL MODELS BY INDEX")
    print("=" * 120)
    
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
    
    return results, file_summary

def print_training_process_summary(file_summary):
    """
    Print a comprehensive summary of the training process and file counts
    """
    print("\n" + "=" * 120)
    print("TRAINING PROCESS SUMMARY - FILES READ")
    print("=" * 120)
    
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
    
    print(f"\n" + "=" * 60)
    print(f"OVERALL SUMMARY:")
    print(f"Total files found across all models: {total_files_found}")
    print(f"Total files successfully read: {total_files_read}")
    print(f"Success rate: {(total_files_read/total_files_found*100):.1f}%" if total_files_found > 0 else "No files found")
    print("=" * 60)
    
    return file_summary

def save_industrobo_results_to_csv(results, output_file="industrobo_analysis_results.csv"):
    """
    Save the industrobo analysis results to a CSV file
    """
    rows = []
    
    for model_name, model_data in results.items():
        for config, index_data in model_data.items():
            for index_key, metrics in index_data.items():
                index_num = index_key.split('_')[1]
                row = {
                    'Model': model_name,
                    'Configuration': config,
                    'Index': index_num,
                    'RMSE_Mean': metrics.get('rmse_mean', np.nan),
                    'RMSE_Std': metrics.get('rmse_std', np.nan),
                    'RMSE_Min': metrics.get('rmse_min', np.nan),
                    'RMSE_Max': metrics.get('rmse_max', np.nan),
                    'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
                    'NRMSE_Std': metrics.get('nrmse_std', np.nan),
                    'NRMSE_Min': metrics.get('nrmse_min', np.nan),
                    'NRMSE_Max': metrics.get('nrmse_max', np.nan),
                    'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
                    'Correlation_Std': metrics.get('Correlation Coefficient_std', np.nan),
                    'Correlation_Min': metrics.get('Correlation Coefficient_min', np.nan),
                    'Correlation_Max': metrics.get('Correlation Coefficient_max', np.nan),
                    'Num_Files': metrics.get('num_files', np.nan),
                    'Total_Runs': metrics.get('total_runs', np.nan)
                }
                rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    return df_results

def save_file_summary_to_csv(file_summary, output_file="industrobo_file_summary.csv"):
    """
    Save the file summary to a CSV file
    """
    rows = []
    
    for model_name, model_data in file_summary.items():
        for config, file_info in model_data.items():
            row = {
                'Model': model_name,
                'Configuration': config,
                'Files_Found': file_info['files_found'],
                'Files_Read': file_info['files_read'],
                'File_Numbers': ', '.join(file_info['file_numbers']) if file_info['file_numbers'] else 'None'
            }
            rows.append(row)
    
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(output_file, index=False)
    print(f"File summary saved to {output_file}")
    
    return df_summary

def compare_models_across_types_and_indices(results):
    """
    Compare models across different types and indices
    """
    print("\n" + "=" * 120)
    print("MODEL COMPARISON ACROSS TYPES AND INDICES")
    print("=" * 120)
    
    # Create comparison table
    comparison_data = []
    
    for model_name, model_data in results.items():
        for config, index_data in model_data.items():
            for index_key, metrics in index_data.items():
                index_num = index_key.split('_')[1]
                comparison_data.append({
                    'Model': model_name,
                    'Type': config,
                    'Index': index_num,
                    'RMSE_Mean': metrics.get('rmse_mean', np.nan),
                    'NRMSE_Mean': metrics.get('nrmse_mean', np.nan),
                    'Correlation_Mean': metrics.get('Correlation Coefficient_mean', np.nan),
                    'Total_Runs': metrics.get('total_runs', np.nan)
                })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Pivot tables for different views
    print("\nRMSE Comparison by Model and Type (averaged across indices):")
    pivot_rmse = df_comparison.groupby(['Model', 'Type'])['RMSE_Mean'].mean().unstack()
    print(pivot_rmse.round(4))
    
    print("\nNRMSE Comparison by Model and Type (averaged across indices):")
    pivot_nrmse = df_comparison.groupby(['Model', 'Type'])['NRMSE_Mean'].mean().unstack()
    print(pivot_nrmse.round(4))
    
    print("\nCorrelation Comparison by Model and Type (averaged across indices):")
    pivot_corr = df_comparison.groupby(['Model', 'Type'])['Correlation_Mean'].mean().unstack()
    print(pivot_corr.round(4))
    
    # Index-specific comparisons - Only NRMSE
    print("\n" + "-" * 80)
    print("INDEX-SPECIFIC COMPARISONS - NRMSE ONLY")
    print("-" * 80)
    
    for idx in range(6):
        print(f"\nIndex {idx} - NRMSE Comparison:")
        idx_data = df_comparison[df_comparison['Index'] == str(idx)]
        pivot_idx_nrmse = idx_data.pivot(index='Model', columns='Type', values='NRMSE_Mean')
        print(pivot_idx_nrmse.round(4))
    
    return df_comparison

if __name__ == "__main__":
    # Run the analysis
    results, file_summary = analyze_industrobo_results()
    
    # Print training process summary
    print_training_process_summary(file_summary)
    
    # Save results to CSV
    df_summary = save_industrobo_results_to_csv(results)
    df_file_summary = save_file_summary_to_csv(file_summary)
    
    # Compare models across types and indices
    df_comparison = compare_models_across_types_and_indices(results)
    
    print("\nAnalysis complete!")
