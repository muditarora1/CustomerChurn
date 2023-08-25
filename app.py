from flask import Flask,request,render_template
from flask_cors import cross_origin

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 


@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data=CustomData(
            Age = float(request.form.get('Age')),
            Gender = request.form.get('Gender'),
            Location = request.form.get('Location'),
            Subscription_Length_Months = float(request.form.get('Subscription_Length_Months')),
            Monthly_Bill = float(request.form.get('Monthly_Bill')),
            Total_Usage_GB = float(request.form.get('Total_Usage_GB')) 
        )

        pred_df=data.get_data_as_data_frame()

        print(pred_df)
        print("Before Prediction")
        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        output=round(results[0],2)
        print(output)

        return render_template('home.html',prediction_text="Result: {}".format(output))

#http://127.0.0.1:5000/predict
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)   