## Note!:
Unlike the parent repository, this is an package, does not work as a command line script, *AND* it requires >=Python 3.5.

AutoImpute: Autoencoder based imputation of single cell RNA-seq data
================
Divyanshu Talwar, Aanchal Mongia, Debarka Sengupta, and Angshul Majumdar

## Introduction
`AutoImpute` is an auto-encoder based gene-expression (sparse) matrix imputation.

For detailed information refer to our paper titled "[AutoImpute : Autoencoder based imputation of single cell RNA-seq data](https://www.nature.com/articles/s41598-018-34688-x)".

-	For technical problems, please report to [Issues](https://github.com/divyanshu-talwar/AutoImpute/issues).

## Description
The input to `AutoImpute.py` is a pre-processed count single cell matrix, with columns representing genes and rows representing cells. It's output is an imputed count matrix with the same dimension. The complete pipeline is described with the following image : 

![AutoImpute-Pipeline](./images/pipeline.jpg).

## Dependencies
* For Python (3.5):
    > numpy, scikit-learn, tensorflow, matplotlib.
* For R (pre-processing):
	> R.matlab, Matrix, ggplot2, Rtsne, svd, plyr, dplyr, data.table, mclust, flexclust, reshape2, irlba, dynamicTreeCut, RColorBrewer, GenSA, gplots

## Contents
* `AutoImpute\ Model/AutoImpute.py` - is the AutoImpute imputation model.
* `AutoImpute\ Model/data/raw` - contains the raw data in `.csv` format.
* `AutoImpute\ Model/data/` - contains the pre-processed data in `<dataset_name>.{mat, csv}` format.
* `AutoImpute\ Model/Pre-processing/` - contains the R scripts for pre-processing.

## Execution
* To run, import `AutoImpute` and use the `autoimpute()` function
```
Options :
	debug: type = bool, default=True, Want debug statements
	debug_display_step: type=int, default=1, Display loss after
	# Hyper-parameters
	hidden_units: type=int, default=2000, Size of hidden layer or latent space dimensions
	lambda_val: type=int, default=1, Regularization coefficient, to control the contribution of regularization term in the cost function
	initial_learning_rate: type=float, default=0.0001, Initial value of learning rate
	iterations: type=int, default=7000, Number of iterations to train the model for
	threshold: type=int, default=0.0001, To stop gradient descent after the change in loss function value in consecutive iterations is less than the threshold, implying convergence
	# Data
	data: type = str representing the filename of a delimited file or a numpy array.
	# Run the masked matrix recovery test
	masked_matrix_test: type = bool, default=False, nargs = '+', help = "Run the masked matrix recovery test?
	masking_percentage: type = float, default=10, nargs = '+', help = "Percentage of masking required. Like 10, 20, 12.5 etc
	# Model save and restore options
	save_model_location: type=str, default='checkpoints/model1.ckpt', Location to save the learnt model
	load_model_location: type=str, default='checkpoints/model0.ckpt', Load the saved model from.
	log_file: type=str, default='log.txt', text file to save training logs
	load_saved: type=bool, default=False, flag to indicate if a saved model will be loaded
	# masked and imputed matrix save location / name
	imputed_save: type=str, default='imputed_matrix', save the imputed matrix as
	masked_save: type=str, default='masked_matrix', save the masked matrix as
```
* To pre-process any dataset, place the raw data (in `.csv` format) into `AutoImpute\ Model/data/raw/` and change your directory to `AutoImpute\ Model/Pre-processing/`.
* Run the R-script `pre-process.R` using the following command:
```bash
Rscript pre-process.R <input-file-name> <dataset-name>
```
* For example, the sample dataset can be pre-processed using the following command:
```bash
Rscript pre-process.R Blakeley_raw_data.csv blakeley
```
