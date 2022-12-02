from ml_pipeline_function import ml_pipeline_function

# Import dataset
from data.datasets import neurons
df = neurons()

ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'row_removal')
