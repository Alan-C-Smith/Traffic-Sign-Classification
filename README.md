# Final Project

## Dependencies
All development was done using packages used internal to the course - Tensorflow-2.7.0 was used for ANN development. The final model was developed using HiPerGator on an NVIDIA A100 GPU with 1 core and 16 GB RAM.

## Installation
1. Clone the repo:
   ```sh
   git clone (https://github.com/EEL5840-EEE4773-Summer2022/final-project-captcha_bots.git)
   ```
2. Create a new Python project or Jupyter Notebook file and import the test.py and train.py functions using the following code:
   ```sh
   from train import *
   from test import *
   ```
3. Import training/test data and associated labels to the main workspace as a NumPy array, see Function Parameters for specific formatting.
4. Download the 'Xception DNN Traffic Sign Model Fine Tuned.h5' model from Canvas. NOTE: If the model file is in the same directory as the train.py script when it is run, the model may be overwritten. Please run the test.py script first with the trained model from Canvas.
## Function Parameters
### train.py
|Parameter | Description|
|----------|------------|
|data         | NumPy array of 270000 x N integers ranging from 0 to 255, inclusive|
| t           | NumPy array of N x 1 integers ranging from 0 to 9, inclusive|
|n_iter       | Scalar positive integer > 0, number of cross validation (CV) rounds for use in RandomizedSearchCV, default is 10|
|cv           | Scalar positive integer > 1, value of K used in K-fold CV, default is 3|
|epochs_top   | Scalar positive integer > 0, number of epochs used for training the ANN above the base Xception model, default is 100|
|epochs_fine  | Scalar positive integer > 0, number of epochs used for training the entire ANN (including the base Xception model), default is 20|
|patience_top | Scalar positive integer > 0, patience early stopping criteria for training the ANN above the base Xception model, default is 10|
|patience_fine| Scalar positive integer > 0, patience early stopping criteria for training the ANN (including the base Xception model), default is 5|
|learning_rate| Range or distribution real number > 0, used for RandomizedSearchCV, default is a reciprocal probability distribution from 3e-4 to 3e-2|
|dropout_rate| Range or distribution real number > 0 and < 1, used for RandomizedSearchCV, default is a range from 0.15 to 0.3 in increments of 0.05|
|learning_rate_fine| Scalar positive real number > 0 and < 1, used for tuning the entire ANN, default is 1e-5|
|class_names| List of 10 strings containing class label names, default is 'Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only','Do Not Enter','Crosswalk','Handicap Parking','No Parking'|

|Returns  | Description|
|---------|------------|
|t_pred_all     |Labels corresponding to each datapoint, NumPy array of integers ranging from 0 to 9, inclusive|

The train.py script will save the preliminary model as 'Xception DNN Traffic Sign Model.h5' and the final, refined model as 'Xception DNN Traffic Sign Model Fine Tuned.h5' to the local directory.

### test.py
|Parameter | Description|
|----------|------------|
|data         | NumPy array of 270000 x N integers ranging from 0 to 255, inclusive|
| t           | NumPy array of N x 1 integers ranging from 0 to 9, inclusive|
|model        | Instance of tf.keras.Model() object, default = keras.models.load_model('Xception DNN Traffic Sign Model Fine Tuned.h5')|
|class_names| List of 10 strings containing class label names, default is 'Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only','Do Not Enter','Crosswalk','Handicap Parking','No Parking'|

|Returns  | Description|
|---------|------------|
|score    |Accuracy score of testing set, scalar real number >0 and <1|
|y        |Labels corresponding to each datapoint, NumPy array of integers ranging from 0 to 9, inclusive|

### test_hard.py
|Parameter | Description|
|----------|------------|
|data         | NumPy array of 270000 x N integers ranging from 0 to 255, inclusive|
| t           | NumPy array of N x 1 integers ranging from 0 to 9, inclusive|
|model        | Instance of tf.keras.Model() object, default = keras.models.load_model('Xception DNN Traffic Sign Model Fine Tuned.h5')|
|class_names| List of 11 strings containing class label names, default is 'Unknown','Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only','Do Not Enter','Crosswalk','Handicap Parking','No Parking'|

|Returns  | Description|
|---------|------------|
|score    |Accuracy score of testing set, scalar real number >0 and <1|
|y        |Labels corresponding to each datapoint, NumPy array of integers ranging from -1 to 9, inclusive|
  

