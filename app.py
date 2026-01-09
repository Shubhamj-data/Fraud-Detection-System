from flask import Flask
import joblib

from src.api.routes import api_blueprint
from src.api.db import init_db


def create_app():
    app = Flask(__name__)

    # Load ML model once
    app.model = joblib.load("artifacts/xgboost_model.pkl")

    # Initialize database
    init_db()

    # Register routes
    app.register_blueprint(api_blueprint)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
