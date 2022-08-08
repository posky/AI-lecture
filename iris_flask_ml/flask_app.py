from flask import Flask, render_template, request
import numpy as np
import pickle
import os

model = pickle.load(
    open(
        os.path.join(os.path.dirname(__file__), 'model/iris_base.pkl'),
        'rb'
    )
)
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('start.html')

@app.route('/predict', methods=['POST'])
def start():
    val1 = float(request.form['a'])
    val2 = float(request.form['b'])
    val3 = float(request.form['c'])
    val4 = float(request.form['d'])
    arr = np.array([[val1, val2, val3, val4]])
    pred = model.predict(arr)
    print("start pred ", pred)
    return render_template('after.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)
