from flask import Flask,request, render_template

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = [GET, POST])
def predic_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            radius_mean =   float(request.form.get('radius_mean')),
            texture_mean  =  float(request.form.get('texture_mean')),
            perimeter_mean =   float(request.form.get('perimeter_mean')),
            area_mean =   float(request.form.get('area_mean')),
            smoothness_mean =   float(request.form.get('smoothness_mean')),
            compactness_mean =  float(request.form.get('compactness_mean')),
            concavity_mean =   float(request.form.get('concavity_mean')),
            concave_points_mean   = float(request.form.get('concave_points_mean')),
            symmetry_mean =   float(request.form.get('symmetry_mean')),
            fractal_dimension_mean   = float(request.form.get('fractal_dimension_mean')),
            radius_se =   float(request.form.get('radius_se')),
            texture_se =   float(request.form.get('texture_se')),
            perimeter_se =   float(request.form.get('perimeter_se')),
            area_se =   float(request.form.get('area_se')),
            smoothness_se =   float(request.form.get('smoothness_se')),
            compactness_se  = float(request.form.get('compactness_se')),
            concavity_se =   float(request.form.get('concavity_se')),
            concave_points_se   = float(request.form.get('concave_points_se')),
            symmetry_se =   float(request.form.get('symmetry_se')),
            fractal_dimension_se   = float(request.form.get('fractal_dimension_se')),
            radius_worst =   float(request.form.get('radius_worst')),
            texture_worst =  float(request.form.get('texture_worst')),
            perimeter_worst =   float(request.form.get('perimeter_worst')),
            area_worst =   float(request.form.get('area_worst')),
            smoothness_worst  = float(request.form.get('smoothness_worst')),
            compactness_worst  = float(request.form.get('compactness_worst')),
            concavity_worst =   float(request.form.get('concavity_worst')),
            concave_points_worst   = float(request.form.get('concave_points_worst')),
            symmetry_worst   = float(request.form.get('symmetry_worst')),
            fractal_dimension_worst   = float(request.form.get('fractal_dimension_worst'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print('Before Prediction')


        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results = results[0])

if __name__=="__main__":
    app.run(host = "0.0.0.0",debug = True)