from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
import os
import main

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Test(Resource):
    def get(self):
        return 'Welcome to, Test App API!'

    def post(self):
        try:
            value = request.get_json()
            if(value):
                return {'Post Values': value}, 201
            return {"error":"Invalid format."}
        except Exception as error:
            return {'error': str(error)}
    
class GetPredictionOutput(Resource):
    def get(self):
        try:
            pred = main.plot()
            return {'prediction_file': pred}
        
        except Exception as error:
            return {"error": str(error)}, 500

    
    def post(self):
        return {"error": "Invalid Method."}
        

api.add_resource(Test, '/')
api.add_resource(GetPredictionOutput, '/getPredictionOutput')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5500, debug=True)