import os
import re
from datetime import datetime, timedelta

def analyze_training_timing():
    """
    Analyze training timing information from log files and estimate finishing times
    """
    
    # Log file paths
    log_files = {
        'mpw-10': '/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/industrobo/AE-RNN_None/data/industrobo_h64_z12_n3_mpw-10.log',
        'mpw100': '/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/log/0813_full_100/industrobo/AE-RNN_None/data/industrobo_h64_z12_n3_mpw100.log'
    }
    
    print("ANALYZING TRAINING TIMING INFORMATION")
    print("=" * 80)
    
    results = {}
    
    for config, log_file in log_files.items():
        print(f"\nAnalyzing {config} log file...")
        
        if not os.path.exists(log_file):
            print(f"  Log file not found: {log_file}")
            continue
        
        # Read the log file
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        print(f"  Total lines in log: {len(lines)}")
        
        # Extract training information
        training_info = extract_training_info(lines, config)
        results[config] = training_info
        
        # Print summary
        print(f"  Training rounds found: {training_info['num_rounds']}")
        print(f"  Total epochs across all rounds: {training_info['total_epochs']}")
        print(f"  Average epochs per round: {training_info['avg_epochs_per_round']:.1f}")
        print(f"  Max epochs in a round: {training_info['max_epochs']}")
        print(f"  Min epochs in a round: {training_info['min_epochs']}")
        
        # Estimate timing
        timing_info = estimate_timing(training_info, config)
        results[config].update(timing_info)
    
    return results

def extract_training_info(lines, config):
    """
    Extract training information from log lines
    """
    training_info = {
        'num_rounds': 0,
        'rounds': [],
        'total_epochs': 0,
        'avg_epochs_per_round': 0,
        'max_epochs': 0,
        'min_epochs': float('inf'),
        'config': config
    }
    
    current_round = None
    current_epochs = 0
    
    for line in lines:
        # Check for round start
        round_match = re.search(r'(\d+)/10 round starts', line)
        if round_match:
            # Save previous round info
            if current_round is not None:
                training_info['rounds'].append({
                    'round': current_round,
                    'epochs': current_epochs
                })
                training_info['total_epochs'] += current_epochs
                training_info['max_epochs'] = max(training_info['max_epochs'], current_epochs)
                training_info['min_epochs'] = min(training_info['min_epochs'], current_epochs)
            
            current_round = int(round_match.group(1))
            current_epochs = 0
            training_info['num_rounds'] += 1
        
        # Check for epoch information
        epoch_match = re.search(r'Train Epoch: \[\s*(\d+)/\s*200\]', line)
        if epoch_match:
            epoch_num = int(epoch_match.group(1))
            current_epochs = max(current_epochs, epoch_num + 1)  # +1 because epochs are 0-indexed
    
    # Save the last round
    if current_round is not None:
        training_info['rounds'].append({
            'round': current_round,
            'epochs': current_epochs
        })
        training_info['total_epochs'] += current_epochs
        training_info['max_epochs'] = max(training_info['max_epochs'], current_epochs)
        training_info['min_epochs'] = min(training_info['min_epochs'], current_epochs)
    
    # Calculate averages
    if training_info['num_rounds'] > 0:
        training_info['avg_epochs_per_round'] = training_info['total_epochs'] / training_info['num_rounds']
    
    if training_info['min_epochs'] == float('inf'):
        training_info['min_epochs'] = 0
    
    return training_info

def estimate_timing(training_info, config):
    """
    Estimate timing based on training information
    """
    print(f"\n  TIMING ESTIMATES FOR {config}:")
    print(f"  {'-' * 50}")
    
    # Assumptions for timing estimates
    # These are rough estimates - you may need to adjust based on your actual hardware
    avg_time_per_epoch = 30  # seconds (adjust based on your system)
    time_per_round_setup = 60  # seconds for data loading, model initialization, etc.
    
    # Calculate estimated times
    total_training_time = training_info['total_epochs'] * avg_time_per_epoch
    total_setup_time = training_info['num_rounds'] * time_per_round_setup
    total_estimated_time = total_training_time + total_setup_time
    
    # Convert to hours and minutes
    hours = total_estimated_time // 3600
    minutes = (total_estimated_time % 3600) // 60
    seconds = total_estimated_time % 60
    
    print(f"  Estimated training time per epoch: {avg_time_per_epoch} seconds")
    print(f"  Estimated setup time per round: {time_per_round_setup} seconds")
    print(f"  Total training epochs: {training_info['total_epochs']}")
    print(f"  Total setup time: {total_setup_time} seconds ({total_setup_time/60:.1f} minutes)")
    print(f"  Total estimated time: {total_estimated_time} seconds")
    print(f"  Total estimated time: {hours}h {minutes}m {seconds}s")
    
    # Estimate completion time if training is ongoing
    total_remaining_time = 0
    completion_time = None
    
    if training_info['num_rounds'] < 10:  # Assuming 10 rounds total
        remaining_rounds = 10 - training_info['num_rounds']
        avg_epochs_remaining = training_info['avg_epochs_per_round'] * remaining_rounds
        remaining_training_time = avg_epochs_remaining * avg_time_per_epoch
        remaining_setup_time = remaining_rounds * time_per_round_setup
        total_remaining_time = remaining_training_time + remaining_setup_time
        
        remaining_hours = total_remaining_time // 3600
        remaining_minutes = (total_remaining_time % 3600) // 60
        
        print(f"\n  ESTIMATED REMAINING TIME:")
        print(f"  Remaining rounds: {remaining_rounds}")
        print(f"  Estimated remaining epochs: {avg_epochs_remaining:.0f}")
        print(f"  Estimated remaining time: {remaining_hours}h {remaining_minutes}m")
        
        # Estimate completion time
        current_time = datetime.now()
        completion_time = current_time + timedelta(seconds=total_remaining_time)
        print(f"  Estimated completion time: {completion_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'total_estimated_time': total_estimated_time,
        'total_remaining_time': total_remaining_time,
        'completion_time': completion_time
    }

def print_summary(results):
    """
    Print a summary of remaining hours needed
    """
    print("\n" + "=" * 80)
    print("SUMMARY OF REMAINING HOURS NEEDED")
    print("=" * 80)
    
    total_remaining_hours = 0
    
    for config, info in results.items():
        remaining_time = info.get('total_remaining_time', 0)
        remaining_hours = remaining_time // 3600
        remaining_minutes = (remaining_time % 3600) // 60
        
        total_remaining_hours += remaining_hours
        
        if remaining_time > 0:
            print(f"\n{config}:")
            print(f"  Remaining time: {remaining_hours}h {remaining_minutes}m")
            if info.get('completion_time'):
                print(f"  Estimated completion: {info['completion_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"\n{config}: COMPLETED")
    
    print(f"\n" + "=" * 50)
    print(f"TOTAL REMAINING HOURS: {total_remaining_hours} hours")
    print("=" * 50)

if __name__ == "__main__":
    # Run the timing analysis
    results = analyze_training_timing()
    
    # Print summary
    print_summary(results)
    
    print("\nTiming analysis complete!")

