from flask import Flask

app = Flask('Financial Platform API')

with app.app_context():
    import api.read
    import api.predict
