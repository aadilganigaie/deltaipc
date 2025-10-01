import pandas as pd
import numpy as np
# 1. Improved Data Loading Functions
def load_application_data(app_name, base_path):
    """Load all way configurations for a single application"""
    data = {}

    # Handle both naming patterns: "app1w" and "app_1w"
    for ways in range(1, 12):
        # Try both possible file naming patterns
        file_pattern1 = f"{base_path}/{app_name}{ways}w*"  # astar1w.csv
        file_pattern2 = f"{base_path}/{app_name}_{ways}w*"  # astar_1w.csv

        matching_files = glob.glob(file_pattern1) + glob.glob(file_pattern2)

        if matching_files:
            file_path = matching_files[0]
            try:
                # Try reading as TSV first (interval data), then as CSV (average data)
                df = pd.read_csv(file_path, sep='\t' if '\t' in open(file_path).readline() else ',')
                df['ways'] = ways  # Add ways as a column
                data[ways] = df
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

    if not data:
        print(f"No data files found for {app_name}")
        return None

    return pd.concat(data.values())

def load_average_data(app_name, base_path):
    """Load average files for an application"""
    avg_files = glob.glob(f"{base_path}/*{app_name}*average*")
    data_frames = []

    for file_path in avg_files:
        try:
            df = pd.read_csv(file_path)
            if 'ways' in df.columns:  # Only include if it has the ways column
                data_frames.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")

    return data_frames

import glob
applications = [
        'astar', 'blender_r', 'bwaves', 'bwaves_r', 'bzip2', 'cactuBSSN_r',
        'cactusADM', 'calculix', 'cam4_r', 'dealII', 'deepsjeng_r',
        'exchange2_r', 'fotonik3d_r', 'gamess', 'gcc_r', 'GemsFDTD',
        'gobmk', 'gromacs', 'h264ref', 'hmmer', 'imagick_r', 'lbm',
        'lbm_r', 'leela_r', 'leslie3d', 'libquantum', 'mcf', 'mcf_r',
        'milc', 'nab_r', 'namd', 'namd_r', 'omnetpp', 'omnetpp_r',
        'parest_r', 'perlbench', 'perlbench_r', 'povray', 'povray_r',
        'roms_r', 'sjeng', 'soplex', 'sphinx3', 'tonto', 'wrf', 'wrf_r',
        'x264_r', 'xalancbmk', 'xalancbmk_r', 'zeusmp'
    ]

'''applications = [
    'GemsFDTD', 'astar', 'bwaves', 'bzip2', 'cactusADM', 'calculix', 'dealII',
    'gamess', 'gobmk', 'gromacs', 'h264ref', 'hmmer', 'lbm', 'leslie3d',
    'libquantum', 'mcf', 'milc', 'namd', 'omnetpp', 'perlbench', 'povray',
    'sjeng', 'soplex', 'sphinx3', 'tonto', 'wrf', 'xalancbmk', 'zeusmp', 'xalancbmk_r',
    'omnetpp_r'
]'''


# 1. Data Preparation Function
def prepare_training_data(base_path, applications):
    """Combine all application data into a single DataFrame for training"""
    all_data = []

    for app in applications:
        try:
            data = load_application_data(app, base_path)
            if data is not None:
                # Add application name as a feature
                data['application'] = app
                all_data.append(data)
        except Exception as e:
            print(f"Error processing {app}: {str(e)}")

    if not all_data:
        raise ValueError("No valid data found for training")

    combined = pd.concat(all_data)

    # Feature engineering - you can add more features here
    #combined['l1_miss_ratio'] = combined['MPKI_L1'] / (combined['MPKI_L1'].max() + 1e-6)
    #combined['l2_miss_ratio'] = combined['MPKI_L2'] / (combined['MPKI_L2'].max() + 1e-6)
    #combined['l3_miss_ratio'] = combined['MPKI_L3'] / (combined['MPKI_L3'].max() + 1e-6)

    return combined

df = prepare_training_data('low_memory_interference_cache_ways_INSTRUCTIONS_v2/tables_instructions', applications)
df.to_csv('low_mem_newfeatures.csv')
