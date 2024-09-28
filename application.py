from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.exception import CustomException
import sys
application = Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/predict-data",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template("home.html")
    else:
        try:
            gender = request.form['gender']
            race_ethnicity = request.form['race_ethnicity']
            parent_education = request.form['parent_education']
            lunch = request.form['lunch']
            test_prep_score = request.form['test_prep_score']
            reading_score = int(request.form['reading_score'])
            writing_score = int(request.form['writing_score'])
            
            custom_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parent_education=parent_education,
                lunch=lunch,
                test_prep_score=test_prep_score,
                reading_score=reading_score,
                writing_score=writing_score
            )
            
            data_df = custom_data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(data_df)

            return render_template("home.html", prediction=prediction[0])

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__=="__main__":
    app.run(host="0.0.0.0")