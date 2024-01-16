import argparse
import pandas as pd
from SurrealGAN.Surreal_GAN_representation_learning import apply_saved_model

# Assuming the SurrealGAN package has a version
VERSION = "0.0.1"

def main():
    prog="SurrealGAN"
    description = "Apply a saved SurrealGAN model for inference"
    usage = """
    SurrealGAN v{VERSION}.
    Apply a saved SurrealGAN model for inference
    required arguments:
        [MODEL_DIR]      The directory containing the saved model.
        [-d, --model_dir]

        [DATA]           The dataset to be used for inference. Can be a
        [-i, --input]    filepath string of a .csv file.

        [EPOCH]          The epoch number of the saved model to be used.
        [-e, --epoch]    Default: 50000

    optional arguments:
        [OUTPUT]         The path where the output .csv file should be 
        [-o, --output]   saved. If none given, nothing will be saved.

        [COVARIATE]      The covariate dataset. Can be a filepath string
        [-c, --covariate]of a .csv file. If not given, no covariate data
                         will be used.

        [HELP]           Show this help message and exit.
        [-h, --help]

        [VERSION]        Display the version of the package.
        [-V, --version]
    """.format(VERSION=VERSION)

    parser = argparse.ArgumentParser(prog=prog,
                                     usage=usage,
                                     description=description,
                                     add_help=False)

    # MODEL_DIR argument
    help = "The directory containing the saved model."
    parser.add_argument("-d", 
                        "--model_dir", 
                        type=str,
                        help=help, 
                        required=True)
    
    # DATA argument
    help = "The dataset to be used for inference. Can be a filepath string of a .csv file."
    parser.add_argument("-i", 
                        "--input",
                        type=str,
                        help=help, 
                        required=True)
    
    # EPOCH argument
    help = "The epoch number of the saved model to be used."
    parser.add_argument("-e", 
                        "--epoch",
                        type=int,
                        help=help,
                        default=50000,
                        required=False)
    
    # OUTPUT argument
    help = "The path where the output .csv file should be saved. If none given, nothing will be saved."
    parser.add_argument("-o", 
                        "--output",
                        type=str,
                        help=help,
                        default=None,
                        required=False)   

    # COVARIATE argument
    help = "The covariate dataset. Can be a filepath string of a .csv file. "\
           "If not given, no covariate data will be used."
    parser.add_argument("-c", 
                        "--covariate",
                        type=str,
                        help=help, 
                        default=None,
                        required=False)

    # VERSION argument
    help = "Show the version and exit"
    parser.add_argument("-V", 
                        "--version", 
                        action='version',
                        version=prog + ": v{VERSION}.".format(VERSION=VERSION),
                        help=help)

    # HELP argument
    help = 'Show this message and exit'
    parser.add_argument('-h', 
                        '--help',
                        action='store_true', 
                        help=help)

    arguments = parser.parse_args()

    # Apply logic for checking the data and demographic conventions
    data = pd.read_csv(arguments.input)
    transformed_input = transform_data(data)
    if arguments.covariate:
        demographic_data = pd.read_csv(arguments.covariate)
        covariates = transform_covariate(demographic_data, data)
    else:
        covariates = None

    # Call the apply_saved_model function with the CLI arguments
    results = apply_saved_model(arguments.model_dir, 
                                transformed_input, 
                                arguments.epoch, 
                                covariates)

    # Handling output
    if arguments.output:
        surreal_df = pd.DataFrame(results, columns=[f'SurrealGAN_r_index_{i}' for i in range(results.shape[1])])
        surreal_df['ID'] = data['ID']
        merged_data = pd.merge(data, surreal_df, on='ID')
        merged_data.to_csv(arguments.output, index=False)
        print(f"Results saved to {arguments.output}")
    else:
        print(results)
        print("No output path specified. Results are not saved.")

    return

def transform_data(data):
    # Rename 'ID' column to 'participant_id' and add 'diagnosis'
    data = data.rename(columns={'ID': 'participant_id'})
    data['diagnosis'] = 1  # Assuming all participants have a diagnosis

    # Handle the unmerged ROIs
    unmerged_rois = ['35', '71', '72', '73', '95']
    for roi in unmerged_rois:
        data[f'MUSE_Volume_{roi}'] = data[roi]

    # Define the ROIs that need to be merged
    single_rois = [roi for roi in data.columns if roi.isdigit() and int(roi) < 300 and int(roi) not in [4, 11, 49, 50, 51, 52] and roi not in unmerged_rois]
    merged_rois = []
    for i in range(len(single_rois)//2):
        roi1, roi2 = single_rois[i*2], single_rois[i*2+1]
        new_col_name = f'MUSE_Volume_{roi1}_{roi2}'
        data[new_col_name] = data[roi1] + data[roi2]
        merged_rois.append(new_col_name)

    # Select necessary columns
    selected_vars = ['participant_id', 'diagnosis'] + [f'MUSE_Volume_{roi}' for roi in unmerged_rois] + merged_rois
    transformed_data = data[selected_vars]

    return transformed_data

def transform_covariate(demographic_data, raw_input_data):
    # Rename 'ID' column to 'participant_id' and add 'diagnosis'
    demographic_data = demographic_data.rename(columns={'ID': 'participant_id'})
    demographic_data['diagnosis'] = 1  # Assuming all participants have a diagnosis
    
    # Convert 'Sex' from 'F'/'M' to 0/1
    demographic_data['sex'] = demographic_data['Sex'].replace({'F': 0, 'M': 1})
    
    # Source 'DLICV_baseline' values from ROI 702 in the raw input data
    raw_input_data = raw_input_data.rename(columns={'ID': 'participant_id'})
    roi_702_data = raw_input_data[['participant_id', '702']]
    roi_702_data = roi_702_data.rename(columns={'702': 'DLICV_baseline'})
    
    # Merge demographic data with DLICV_baseline values
    transformed_covariate = pd.merge(demographic_data, roi_702_data, on='participant_id', how='left')
    transformed_covariate = transformed_covariate[['participant_id', 'diagnosis', 'sex', 'DLICV_baseline']]
    
    return transformed_covariate

if __name__ == "__main__":
    main()
