# webapp/app.py

from flask import Flask, request, render_template
from webapp.routes import main
from ml_model import analyze_stock


def create_app():
    app = Flask(__name__)
    app.register_blueprint(main)
    


    @app.route('/prediction', methods=['GET', 'POST'])
    def prediction():
        prediction_result = None
        stock_name = ''
        error = None

        if request.method == 'POST':
            stock_name = request.form['stock_name']
            try:
                prediction = analyze_stock(stock_name)
                prediction_result = "🔼 Going Up" if prediction == 1 else "🔽 Going Down"
            except Exception as e:
                error = str(e)

        return render_template('prediction.html', stock_name=stock_name, prediction=prediction_result, error=error)
    
    return app