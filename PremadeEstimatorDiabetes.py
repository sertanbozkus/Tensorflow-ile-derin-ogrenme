from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import argparse
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


#Serving as restAPI
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/todo/api/v1.0/tasks', methods=['GET'])
@cross_origin()

def get_tasks():

    predict_patient ={
    'Pregnancies' : request.args.getlist('pregnancies', type=float),
    'Glucose' : request.args.getlist('glucose', type=float),
    'BloodPressure' : request.args.getlist('bloodPressure', type=float),
    'SkinThickness' : request.args.getlist('skinThickness', type=float),
    'Insulin' : request.args.getlist('insulin', type=float),
    'BMI' : request.args.getlist('bmi', type=float),
    'DiabetesPedigreeFunction' : request.args.getlist('diabetesPedigreeFunction', type=float),
    'Age' : request.args.getlist('age', type=float)}

    

    result = main(predict_patient)

    return jsonify(result)




def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset



CSV_COLUMN_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome','Group']

OUTCOME = [ 'Healthy' , 'Diabetes']

train = pd.read_csv("pima-indians-diabetes.csv", names = CSV_COLUMN_NAMES, header=0)
X,Y = train , train.pop('Outcome')

def main(predict_x):



    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)

    my_feature_columns = [
        tf.feature_column.numeric_column(key='Pregnancies'),
        tf.feature_column.numeric_column(key='Glucose'),
        tf.feature_column.numeric_column(key='BloodPressure'),
        tf.feature_column.numeric_column(key='SkinThickness'),
        tf.feature_column.numeric_column(key='Insulin'),
        tf.feature_column.numeric_column(key='BMI'),
        tf.feature_column.numeric_column(key='DiabetesPedigreeFunction'),
        tf.feature_column.numeric_column(key='Age')
    ]

    classifier =tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns, hidden_units = [10,10,10], n_classes = 2)

    classifier.train(
    input_fn = lambda: train_input_fn(train_x,train_y, 100), steps =2000)


    eval_result = classifier.evaluate(
        input_fn = lambda: eval_input_fn(test_x,test_y,100)
    )

    #This will show our program's prediction accuracy on our Test-set
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


    predictions = classifier.predict(
        input_fn=lambda:eval_input_fn(predict_x,labels=None,batch_size=100))

    template = ('Prediction is "{}"')


    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

    
    # Show our prediction and accurancy of the prediction
    template = ('\nPrediction is "{}" ({:.1f}%)')
    print(template.format(OUTCOME[class_id],
                              100 * probability))
    return OUTCOME[class_id]

if __name__ == '__main__':
    app.run(debug=True)
