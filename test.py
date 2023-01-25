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

def test(data, t,
         model = keras.models.load_model('Xception DNN Traffic Sign Model Fine Tuned.h5'),
         class_names = ['Stop','Yield','Red Light','Green Light','Roundabout','Right Turn Only',
                'Do Not Enter','Crosswalk','Handicap Parking','No Parking']
        ):
    
    numImg = data.shape[1]

    # Reshape the image data into 300x300x3 arrays
    X_ = data.reshape(300, 300, 3, numImg)
    X = np.moveaxis(X_, 3, 0)
    
    # Predict class labels
    y_prob = model.predict(X)
    y = np.argmax(y_prob, 1)
    
    
    print("\033[1mModel Test Accuracy\033[0m:", metrics.accuracy_score(t, y))
    # Confusion Matrix
    cm = confusion_matrix(t, y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    # Confidence for Correctly Classified Images
    correct_idx = t == y
    correct_prob = np.max(y_prob[correct_idx], 1)
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
    incorrect_idx = t != y
    incorrect_prob = np.max(y_prob[incorrect_idx], 1)
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
    
    # Classification report
    print('\nClassification Report on Testing Set:')
    print(classification_report(t, y, target_names=class_names))
    print('\nTesting Set Accuracy Score:')
    score = accuracy_score(t,y)
    print(score)
    
    return score, y