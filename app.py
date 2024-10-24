from flask import Flask,request,render_template
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler

#step 1: create the app
app=Flask(__name__)

#step 2:define route for a homepage
app.route('/')
def index():
    return render_template('index.html')
#step 3: define route for prediction page

@app.route('/predictiondata',methods=['GET','POST'])
def predict_datapoint():
    #get the information from home page else get info from custom data
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get(' parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            math_score=float(request.form.get('math_score')),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        #step 4: Initialize data convertion to data frame
        pred_df=data.get_data_as_dataframe()
        print(pred_df)
        print("Before Prediction")
        #initialize prediction
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("After prediction")
        return render_template('home.html',results=results[0])
if __name__=="__main__":
    app.run(host="0.0.0.0")

