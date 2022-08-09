from flask import Flask, render_template, request
import numpy as np
import pickle
import os

model = pickle.load(
    open(
        os.path.join(os.path.dirname(__file__), 'model/income_base.pkl'),
        'rb'
))
app = Flask(__name__)

# flask app root directory initialize
@app.route('/')
def main():
    return render_template('start.html')

@app.route('/predict', methods=['POST'])
def start():
    val1 = request.form['a']
    val2 = request.form['b']
    val3 = request.form['c']
    val4 = request.form['d']
    val5 = request.form['e']
    val6 = request.form['f']
    val7 = request.form['g']
    val8 = request.form['h']
    arr = np.array([[val1, val2, val3, val4, val5, val6, val7, val8]])
    pred = model.predict(arr)
    print("start pred ", pred)
    return render_template('after.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)
