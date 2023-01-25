import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split as split
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

def train(data, t, n_iter=10, cv=3, epochs_top = 100, epochs_fine = 20,
          patience_top = 10, patience_fine = 5,
          n_hidden = np.array([0, 1, 2]), n_neurons = np.arange(1,100), learning_rate = reciprocal(3e-4, 3e-2), 
          dropout_rate = np.arange(0.15, 0.3, 0.05), learning_rate_fine = 1e-5,
          class_names = ['Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only',
                'Do Not Enter','Crosswalk','Handicap Parking','No Parking']
         ):
    numImg = data.shape[1]

    # Reshape the image data into 300x300x3 arrays
    X_ = data.reshape(300, 300, 3, numImg)
    X = np.moveaxis(X_, 3, 0)

    # Split data into training and test sets

    X_train_set, X_test, t_train_set, t_test = split(X, t, test_size=0.1, random_state=42)
    # Further split training set into train and validation sets
    X_train, X_valid, t_train, t_valid = split(X_train_set, t_train_set, test_size=0.2, random_state=42)
    X_train.shape, X_valid.shape, X_test.shape, t_train.shape, t_valid.shape, t_test.shape

    # Set up a wrapper for CV
    keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)


    # Establish a parameter grid of values for CV to test
    param_distribs = {"n_hidden": n_hidden,
                      "n_neurons": n_neurons,
                      "learning_rate": learning_rate,
                      "dropout_rate": dropout_rate,
                      }

    # CV scikit-learn object
    rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=n_iter, cv=cv, n_jobs=-1)

    rnd_search_cv.fit(X_train, t_train, epochs=epochs_top,
                      validation_data=(X_valid, t_valid),
                      callbacks=[keras.callbacks.EarlyStopping(patience=patience_top),
                                 keras.callbacks.ModelCheckpoint('Xception DNN Traffic Sign Model.h5',
                                                                 save_best_only=True)])

    # Save the best model (highest accuracy) from CV
    model = rnd_search_cv.best_estimator_.model
    model.save('Xception DNN Traffic Sign Model.h5')

    # Make test set classification predictions using the trained model
    t_predict_prob = model.predict(X_test)
    t_predict_prob.shape

    # Show predictions
    t_pred = np.argmax(t_predict_prob, 1)

    print("\033[1mCNN Test Accuracy\033[0m:", metrics.accuracy_score(t_test, t_pred))
    # Confusion Matrix
    cm = confusion_matrix(t_test, t_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    model.summary()
    print('\n=============Fine tune parameters within base model==============\n')
    # Fine tune parameters for the output layer
    model.get_layer('xception').trainable = True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate_fine),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    model.summary()

    model.fit(X_train, t_train, epochs=epochs_fine,
              validation_data=(X_valid, t_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=patience_fine),
                         keras.callbacks.ModelCheckpoint('Xception DNN Traffic Sign Model Fine Tuned.h5',
                                                         save_best_only=True)])

    # Check to see if saved model is saved and loaded properly
#     model_verify0 = keras.models.load_model('Xception DNN Traffic Sign Model.h5')
#     model_verify1 = keras.models.load_model('Xception DNN Traffic Sign Model Fine Tuned.h5')

    # Compare the model accuracy before and after fine tuning of the output layer
    model_top = keras.models.load_model('Xception DNN Traffic Sign Model.h5')
    t_top = np.argmax(model_top.predict(X_test), 1)
    t_fine = np.argmax(model.predict(X_test), 1)

    print("\033[1mTraining Top Layer Only Accuracy\033[0m:", metrics.accuracy_score(t_test, t_top))
    # Confusion Matrix
    cm = confusion_matrix(t_test, t_top)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print("\033[1mTraining Full Model Accuracy\033[0m:", metrics.accuracy_score(t_test, t_fine))
    # Confusion Matrix
    cm = confusion_matrix(t_test, t_fine)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    probabilities = model.predict(X)
    t_pred_all = np.argmax(probabilities, 1)

    print("\033[1mAll Sample Accuracy (Full Model)\033[0m:", metrics.accuracy_score(t, t_pred_all))
    # Confusion Matrix
    cm = confusion_matrix(t, t_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Check model confidence of correctly classified images
    # Get logical index of correct predictions
    correct_idx = t == t_pred_all
    correct_prob = np.max(probabilities[correct_idx], 1)
    avg_correct_prob = np.average(correct_prob)
    min_correct_prob = np.min(correct_prob)
    max_correct_prob = np.max(correct_prob)
    std_correct_prob = np.std(correct_prob)

    plt.hist(correct_prob)
    plt.xlabel('Confidence') 
    plt.ylabel('Frequency') 
    plt.title("Histogram of Softmax Confidence for Correctly Classified Images")
    plt.show()
    
    print('Average confidence of correct guess: ', avg_correct_prob)
    print('Minimum confidence of correct guess: ', min_correct_prob)
    print('Maximum confidence of correct guess: ', max_correct_prob)
    print('Standard deviation of confidence of correct guess: ', std_correct_prob)
    
    # Confidence for Incorrectly Classified Images
    incorrect_idx = t != t_pred_all
    incorrect_prob = np.max(probabilities[incorrect_idx], 1)
    avg_incorrect_prob = np.average(incorrect_prob)
    min_incorrect_prob = np.min(incorrect_prob)
    max_incorrect_prob = np.max(incorrect_prob)
    std_incorrect_prob = np.std(incorrect_prob)    
    
    plt.hist(incorrect_prob)
    plt.xlabel('Confidence') 
    plt.ylabel('Frequency') 
    plt.title("Histogram of Softmax Confidence for Incorrectly Classified Images")
    plt.show()
    
    print('Average confidence of incorrect guess: ', avg_incorrect_prob)
    print('Minimum confidence of incorrect guess: ', min_incorrect_prob)
    print('Maximum confidence of incorrect guess: ', max_incorrect_prob)
    print('Standard deviation of confidence of incorrect guess: ', std_incorrect_prob)


    y_train = np.argmax(model.predict(X_train), axis=1)

    y_test = np.argmax(model.predict(X_test), axis=1)
    print('\nClassification Report on Training Set:')
    print(classification_report(t_train, y_train, target_names=class_names))
    print('\nClassification Report on Testing Set:')
    print(classification_report(t_test, y_test, target_names=class_names))
    print('\nTesting Set Accuracy Score:')
    print(accuracy_score(t_test, y_test))
    
    return t_pred_all




# Define a function to build a CNN model that can be used as input for CV below
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, dropout_rate=0.5, input_shape=[300, 300, 3]):
    # Import the Xception model as the base
    base_model = keras.applications.Xception(
        weights="imagenet",  # Load weights pre-trained on ImageNet.
        input_shape=(300, 300, 3),
        include_top=False,  # Do not include the ImageNet classifier at the top
    )

    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = keras.Input(shape=(300, 300, 3))
    x = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), ])(
        inputs)  # Augment data to add noise

    # Xception weights requires that input be scaled from 0, 255 to a range of -1., +1
    x = tf.keras.applications.mobilenet.preprocess_input(x)

    # From Transfer Learning Tutorial
    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)  # Regularize with dropout

    # Make as many layers as there are n_hidden with n_layers
    for layer in range(n_hidden):
        x = keras.layers.Dense(n_neurons, activation="relu")(x)
        x = keras.layers.Dropout(dropout_rate)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    print("Number of hidden layers: ", n_hidden)
    print("Number of neurons per top hidden layer: ", n_neurons)
    print("Learning rate: ", learning_rate)
    print("Dropout Rate: ", dropout_rate)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Utilize the Adam optimizer for gradient descent
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
