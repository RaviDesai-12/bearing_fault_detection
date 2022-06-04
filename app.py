from flask import Flask, render_template, request
import pickle

#machine learning modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#modules for statistics
from scipy.stats import kurtosis
import math

app = Flask(__name__)

#defining scaler
train_data= pd.read_csv('training set.csv').drop(['sd','skewness', 'fault'], 1)
scaler = StandardScaler()
scaler.fit_transform(train_data)

model = pickle.load(open('bearing_fault_detection_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    if request.method=='GET':
            return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        dataset = request.form['dataset']
        data_list = dataset.split()
        try:
            data_list = [float(data_point) for data_point in data_list]
            data = np.array(data_list)

            #statistical parameters calculation
            mean= round(np.mean(data),4)
            max = round(np.max(data),4)
            min = round(np.min(data),4)
            rms = round(math.sqrt(np.mean(np.square(data))),4)
            crest=round(max/rms,4)
            form=round(rms/mean,4)
            ks = round(kurtosis(data),4)
    
            #scaling 
            scaled_features = scaler.transform(np.array([max, min, mean, rms, ks, crest, form]).reshape(1,-1))

            #predicting defect type 
            defect = model.predict(scaled_features)
            if defect == 0:
                output = 'No'
            elif defect == 1:
                output = 'Inner race'
            else: output = 'Outer race' 

            return render_template('index.html', max=max, min=min, mean=mean, rms=rms, kurtosis=ks, crest=crest, form=form, prediction_text = '{} defect is present in the bearing'.format(output), dataset= dataset )
        except:
            return render_template('index.html', max='-', min='-', mean='-', rms='-', kurtosis='-', crest='-', form='-', prediction_text='', dataset= 'Please input data correctly')

if __name__=="__main__":
    app.run(debug=True)  

