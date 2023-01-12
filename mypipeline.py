from ml_pipeline_function import ml_pipeline_function

from data.datasets import etype
df = etype()
print(df)

#standard_scaler
#minmax_scaler
#maxabs_scaler
#robust_scaler
#normalizer
#log_transformation
#square_root_transformation
#reciprocal_transformation
#box_cox
#yeo_johnson
#quantile_gaussian
#quantile_uniform


# Run ML Pipeline
ml_pipeline_function(df, output_folder = './Outputs/', missing_method = 'knn', test_size = 0.2, categorical = ['label_encoding'],features_label = ['Target'], rescaling = 'yeo_johnson', classification_algorithms=['mlp_neural_network_auto'], cv = 5)

