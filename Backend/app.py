from flask import Flask

from flask_cors import CORS
from db.database import init_db
from api.routes import api

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Initialize database on app start
    init_db()

    # Register API routes
    app.register_blueprint(api, url_prefix="/api")

    @app.route("/")
    def home():
        return "Server is running"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
