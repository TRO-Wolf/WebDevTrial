from flask import Flask, render_template, url_for, request
import joblib

import DataHacker as dh


app = Flask(__name__, template_folder='templates')




@app.route('/', endpoint='ind')
def index():
    return render_template('index.html')

@app.route('/Home/', endpoint='home')
def home():
    return render_template('Home.html')


@app.route('/Nuts/', endpoint='nuts')
def nuts():
    return render_template('nuts.html')


@app.route('/Ticker/', endpoint='tickers')
def tic():
    return render_template('Ticker.html')

@app.route('/process_ticker', methods=['POST'])
def process_ticker():
    ticker = request.form['tick']
    
    msg1 = print('You have selected {}'.format(ticker.upper()))

    og = dh.Organizer(ticker=ticker)
    out_put = og.get_pred()

    return render_template('ticker_results.html', predictions=out_put)



@app.route('/process_nuts', methods=['POST'])
def processs_nuts():
    nut_type = request.form['nut_type']
    num_nuts = int(request.form['num_nuts'])

    predictions = f"I predict you will eat {num_nuts * 2} {nut_type} nuts."

    ticker = request.form['tick']

    og = dh.Organizer(ticker=ticker)
    out_put = og.get_pred()


    
    # render the output in an HTML template
    return render_template('nuts_results.html', predictions=out_put)




@app.route('/Predictions/', endpoint='preds')
def pred():
    return render_template('Predictions.html')

@app.route('/Results/',endpoint='res')
def res():
    return render_template('Results.html')

if __name__ == '__main__':
    app.debug = True
    ip = '127.0.0.1'
    app.run()