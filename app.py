from src.logger import logging
import os , sys
from src.exception import CustomException
from src.pipelines.training_pipeline import TrainPipeline
from src.pipelines.predict_pipeline import PredictionPipeline
from flask import Flask , request , render_template , jsonify , send_file

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train")
def train_route():
    try:

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return jsonify("Training Successfull")

    except Exception as e:
        logging.info("Exception occured in train_route")
        raise CustomException(e , sys)


@app.route("/predict" , methods = ["GET" , "POST"])
def predict():
    try:

        if request.method=='POST':
            data = dict(request.form.items())
            print(data)
            return jsonify("Done")
    except Exception as e:
        logging.info("Exception occured in predict")
        raise CustomException(e , sys)


    

@app.route('/upload' , methods = ["GET" , "POST"])
def upload():
    try:

        if request.method=='POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            logging.info("Prediction completed Downloading prediction file")
            return send_file(prediction_file_detail.prediction_file_path , 
                             download_name=prediction_file_detail.prediction_file_name,
                             as_attachment=True)
        
        else:
            return render_template("upload_file.html")

    except Exception as e:
        logging.info("Exception occured in upload")
        raise CustomException(e , sys)
    


if __name__=='__main__':
    app.run(host="127.0.0.1" , port=5000 , debug=True)