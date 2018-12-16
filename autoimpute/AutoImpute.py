import os
import numpy as np
import scipy.io
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def autoimpute(data: str, debug: bool = True, debug_display_step: int = 1, 
	hidden_units: int = 2000, lambda_val: int = 1, initial_learning_rate: float = 0.0001,	iterations: int = 7000,
	threshold: int = 0.0001,
	masked_matrix_test: bool = False,
	masking_percentage: float = 10,
	log_file: str = 'log.txt',
	load_saved: bool = False):
	"""
		# Print debug statements
		debug: type = bool, default=True, Want debug statements
		debug_display_step: type=int, default=1, Display loss after

		# Hyper-parameters
		hidden_units: type=int, default=2000, Size of hidden layer or latent space dimensions
		lambda_val: type=int, default=1, Regularization coefficient, to control the contribution of regularization term in the cost function
		initial_learning_rate: type=float, default=0.0001, Initial value of learning rate
		iterations: type=int, default=7000, Number of iterations to train the model for
		threshold: type=int, default=0.0001, To stop gradient descent after the change in loss function value in consecutive iterations is less than the threshold, implying convergence

		# Data
		data: type = str, default='blakeley.csv', help = "Dataset to run the script on. In the paper we choose from : ['blakeley.csv', 'jurkat-293T.mat', 'kolodziejczyk.csv', 'PBMC.csv', 'preimplantation.mat', 'quake.csv', 'usoskin.csv', 'zeisel.csv']

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
	"""
	# reading dataset
	if(type(data) == np.ndarray):
		processed_count_matrix = data
	elif(type(data) != np.ndarray):
		if(type(data) == str & data[-3:-1] == "csv"):
			processed_count_matrix = np.loadtxt(data, delimiter=',')
		elif(type(data) == str & data[-3:-1] == "mtx"):
			processed_count_matrix = scipy.io.mmread(data)
			processed_count_matrix = processed_count_matrix.toarray()
			processed_count_matrix = np.array(processed_count_matrix)

	if(masked_matrix_test):
		masking_percentage = masking_percentage/100.0

		idxi, idxj = np.nonzero(processed_count_matrix)

		ix = np.random.choice(len(idxi), int(np.floor(masking_percentage * len(idxi))), replace = False)
		store_for_future = processed_count_matrix[idxi[ix], idxj[ix]]
		indices = idxi[ix], idxj[ix]

		processed_count_matrix[idxi[ix], idxj[ix]] = 0  # making masks 0
		matrix_mask = processed_count_matrix.copy()
		matrix_mask[matrix_mask.nonzero()] = 1 

		mae = []
		rmse = []
		nmse = []

	# finding number of genes and cells.
	genes = processed_count_matrix.shape[1]
	cells = processed_count_matrix.shape[0]
	print(f"[info] Genes : {genes}, Cells : {cells}")

	# placeholder definitions
	X = tf.placeholder("float32", [None, genes])
	mask = tf.placeholder("float32", [None, genes])

	matrix_mask = processed_count_matrix.copy()
	matrix_mask[matrix_mask.nonzero()] = 1

	print(f"[info] Hyper-parameters"
		f"\n\t Hidden Units : {hidden_units}"
		f"\n\t Lambda : {lambda_val}"
		f"\n\t Threshold : {threshold}"
		f"\n\t Iterations : {iterations}"
		f"\n\t Initial learning rate : {initial_learning_rate}")

	# model definition
	weights = {
		'encoder_h': tf.Variable(tf.random_normal([genes, hidden_units])),
		'decoder_h': tf.Variable(tf.random_normal([hidden_units, genes])),
		}
	biases = {
		'encoder_b': tf.Variable(tf.random_normal([hidden_units])),
		'decoder_b': tf.Variable(tf.random_normal([genes])),
	}

	def encoder(x):
		layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h']), biases['encoder_b']))
		return layer_1

	def decoder(x):
		layer_1 = tf.add(tf.matmul(x, weights['decoder_h']), biases['decoder_b'])
		return layer_1

	encoder_op = encoder(X)
	decoder_op = decoder(encoder_op)

	# loss definition
	y_pred = decoder_op
	y_true = X 
	rmse_loss = tf.pow(tf.norm(y_true - y_pred * mask), 2)
	regularization = tf.multiply(tf.constant(lambda_val/2.0, dtype="float32"), tf.add(tf.pow(tf.norm(weights['decoder_h']), 2), tf.pow(tf.norm(weights['encoder_h']), 2)))
	loss = tf.add(tf.reduce_mean(rmse_loss), regularization)
	optimizer = tf.train.RMSPropOptimizer(initial_learning_rate).minimize(loss)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()

	with tf.Session() as sess:
		if(load_saved):
			saver.restore(sess, load_model_location)
			print("[info] model restored.")
		else:
			sess.run(init)
		prev_loss = 0
		for k in range(1, iterations+1):
			_, loss = sess.run([optimizer, rmse_loss], feed_dict={X: processed_count_matrix, mask: matrix_mask})
			lpentry = loss/cells
			change = abs(prev_loss - lpentry)
			if ( change <= threshold ):
				print("Reached the threshold value.")
				break
			prev_loss = lpentry
			if(debug):
				if (k - 1) % debug_display_step == 0:
					print(f'Step {k} : Total loss: {loss}, Loss per Cell : {lpentry}, Change : {change}')
					with open(log_file, 'a') as log:
						log.write('{0}\t{1}\t{2}\t{3}\n'.format(
						    k,
						    loss,
						    lpentry,
						    change
						))
			# if((k-1) % 5 == 0):
			# 	save_path = saver.save(sess, save_model_location)
		imputed_count_matrix = sess.run([y_pred], feed_dict={X: processed_count_matrix, mask: matrix_mask})
		
		if(masked_matrix_test):	
			predictions = []

			for idx, value in enumerate(store_for_future):
				prediction = imputed_count_matrix[0][indices[0][idx], indices[1][idx]]
				predictions.append(prediction)
		else:
			predictions = None
	
	return imputed_count_matrix, predictions