from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('IBM prediction model.pkl', 'rb'))


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        company = (request.form['company'])

        name = (request.form['car_models'])
        year = int(request.form['year'])
        fuel = (request.form['fuel_type'])
        km_driven = int(request.form['kilo_driven'])
        seller_type = (request.form['seller_type'])
        transmission = (request.form['transmission_type'])
        owner = (request.form['owner_type'])
        mileage = (request.form['mileage'])
        engine = (request.form['engine'])
        max_power = (request.form['max_power'])
        torque = (request.form['torque_type'])
        seats = (request.form['seat_type'])

        prediction = model.predict(pd.DataFrame(
            columns=["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine",
                     "max_power", "torque", "seats", "company"],
            data=np.array(
                [name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque,
                 seats, company]).reshape(1, 13)))

        print(prediction)

        output = round(prediction[0], 2)

        if output < 0:
            return render_template('index.html', prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html', prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)