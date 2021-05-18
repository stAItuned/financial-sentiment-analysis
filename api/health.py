from flask import make_response, jsonify
from flask import current_app as app


@app.route('/health', methods=['GET'])
def predict_sentiment():

    return make_response(jsonify({'health': 'ok'}))


