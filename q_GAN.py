# Import the ml_pipeline_function from the external module
from ml_pipeline_function import ml_pipeline_function
# Import pandas for data manipulation
import pandas as pd
# Import the LabelEncoder class from sklearn.preprocessing for converting categorical data into numerical form.
from sklearn.preprocessing import LabelEncoder

# Define the file path for the dataset as a string variable named 'mtype'.
mtype = '../qGAN/metypes_data/e-type.csv'

# Load the dataset using pandas 'read_csv' function, specifying the delimiter as ';', and store it in a DataFrame named 'df'.
df = pd.read_csv(mtype, delimiter=';')

# Rename the "e-type" column to "Target" in the 'df' DataFrame.
df = df.rename(columns={"e-type": "Target"})

# Select only one class
# Set the type_number variable to the class label you want to filter for
#type_number = 0

# Filter the df_train DataFrame to include only instances where the 'Target' column
# is equal to the specified type_number
df = df[df['Target'] == "Exc_1"]


# Creating instance of labelencoder

# Instantiate a LabelEncoder object and store it in the variable 'labelencoder'.
labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

# Apply the 'fit_transform' method of the 'labelencoder' object to the 'Target' column of the DataFrame 'df',
# converting the categorical values to numerical values, and store the results back in the 'Target' column.
df['Target'] = labelencoder.fit_transform(df['Target'])

# Import the KNNImputer class from the sklearn.impute module for handling missing values using k-Nearest Neighbors.
from sklearn.impute import KNNImputer

# Instantiate a KNNImputer object with 5 nearest neighbors and store it in the variable 'KNN_imputer'.
KNN_imputer = KNNImputer(n_neighbors=5)

# Apply the 'fit_transform' method of the 'KNN_imputer' object to the entire DataFrame 'df',
# imputing missing values based on the 5 nearest neighbors, and store the results back in 'df'.
df.iloc[:, :] = KNN_imputer.fit_transform(df)

print(df)

# Run ML Pipeline
ml_pipeline_function(
    df,
    output_folder='./Outputs/',  # Store the output in the './Outputs/' folder
    test_size=0.5,  # Set the test set size to 20% of the dataset
    generative_algorithms=['q_gan'],  # Apply Support Vector Machine with a linear kernel
    reps=2,  # Set the number of repetitions for the quantum circuits
    ibm_account='f788498a9bb1808e0d9c491721fa5ce8cdf66d26c3bb39ae71500ecc1a17cb0804c14e0d6d1c003fc50418cda3b7a11db31381bb75528bf27076a7cb17cf3a13',  # Replace with your IBM Quantum API key
    quantum_backend='simulator_statevector',  # Use the QASM simulator as the quantum backend
    data_columns = [
         'adaptation',
         'avg_isi',
         'electrode_0_pa',
         'f_i_curve_slope',
         'fast_trough_t_long_square',
         'fast_trough_t_ramp',
         'fast_trough_t_short_square',
         'fast_trough_v_long_square',
         'fast_trough_v_ramp',
         'fast_trough_v_short_square',
         'input_resistance_mohm',
         'latency',
         'peak_t_long_square',
         'peak_t_ramp',
         'peak_t_short_square',
         'peak_v_long_square',
         'peak_v_ramp',
         'peak_v_short_square',
         'ri',
         'sag',
         'seal_gohm',
         'slow_trough_t_long_square',
         'slow_trough_t_ramp',
         'slow_trough_t_short_square',
         'slow_trough_v_long_square',
         'slow_trough_v_ramp',
         'slow_trough_v_short_square',
         'tau',
         'threshold_i_long_square',
         'threshold_i_ramp',
         'threshold_i_short_square',
         'threshold_t_long_square',
         'threshold_t_ramp',
         'threshold_t_short_square',
         'threshold_v_long_square',
         'threshold_v_ramp',
         'threshold_v_short_square',
         'trough_t_long_square',
         'trough_t_ramp',
         'trough_t_short_square',
         'trough_v_long_square',
         'trough_v_ramp',
         'trough_v_short_square',
         'upstroke_downstroke_ratio_long_square',
         'upstroke_downstroke_ratio_ramp',
         'upstroke_downstroke_ratio_short_square',
         'vm_for_sag',
         'vrest'
         ]                 
)
