from flask import make_response, jsonify, request
from flask import current_app as app

from scripts.models.utils import init_model

import logging

logger = logging.getLogger()


@app.route('/predict/sentiment', methods=['POST'])
def predict_sentiment():
    data = request.json

    sentence = data['sentence']
    model_name = data['model']

    model_params = data.get('params')

    model = init_model(model_name)(model_name, model_params)
    logger.debug(f'Input sentence: {sentence} --> {type(sentence)}')

    raw_sentiment = model.predict(sentence)
    sentiment = model.postprocessing(raw_sentiment)

    return make_response(jsonify({'sentence': sentence,
                                  'sentiment': sentiment}))


