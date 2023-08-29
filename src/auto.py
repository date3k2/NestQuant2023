#
from submit import Submission
from sklearn.impute import SimpleImputer
import numpy as np
from ulti import make_train_set
from sklearn.preprocessing import MinMaxScaler
from crawl import Crawler
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import tensorflow as tf

load_dotenv(find_dotenv())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
crawler = Crawler(api_key=os.getenv("API_KEY"))
crawler.download_historical_data(category="crypto", symbol="BTC", location="./data")

#
new_data = pd.read_parquet("./data/BTCUSDT/")

#
price = ((new_data["HIGH"] + new_data["LOW"] + new_data["CLOSE"]) / 3).loc[
    1546300800000:
]
price.index = pd.to_datetime(price.index, unit="ms") + pd.Timedelta(hours=7)

#

# Scale price to train
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(price.values.reshape(-1, 1))[-289:]

#

training_data = make_train_set(
    12,
    scaled_train.reshape(-1, 1),
    1,
)

# Load the pre-trained model
tf.random.set_seed(1012)
model = tf.keras.models.load_model("./final_model.keras")
model.fit(training_data, epochs=7)


#
model.save("./final_model.keras")


#
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, drop_remainder=True, shift=1)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    forecast = model.predict(ds.batch(32))
    return forecast


#
resample = price.resample("H").mean().shift(1).dropna()

#
one_hour_next = model_forecast(model, scaled_train[-12:], 12)

#
new = scaler.inverse_transform(one_hour_next.reshape(-1, 1)).flatten() - 900

# add new data to the end of the training data

btc_price = pd.DataFrame(
    np.concatenate((resample.values, new))[-1500:], columns=["PRICE"]
)


def r_v(i, btc_price: pd.DataFrame):
    # R
    btc_price["R"] = 100 * (btc_price["PRICE"].shift(-i) / btc_price["PRICE"] - 1)

    # V
    std_backward = SimpleImputer(strategy="mean").fit_transform(
        btc_price["PRICE"].rolling(round(1.5 * i)).std().values.reshape(-1, 1)
    )
    std_forward = SimpleImputer(strategy="mean").fit_transform(
        btc_price["PRICE"].rolling(i).std().shift(-i).values.reshape(-1, 1)
    )
    btc_price["V"] = std_forward / std_backward
    btc_price = btc_price.dropna()
    return btc_price["R"].iloc[-1], btc_price["V"].iloc[-1]


#
r, v = r_v(5, btc_price)

#
pred_labels = r / (2 * np.sqrt(v))
pred_labels

#
scaled_labels = (pred_labels - (-3.73357831)) / 9.47783281 + 0.04

# Calculate scaled prediction
submit_hour = (pd.Timestamp.now().ceil(freq="H") - pd.Timedelta("7H")).value // 10**6
print(submit_hour)

#
submission = pd.DataFrame(
    index=[0], data={"OPEN_TIME": submit_hour, "PREDICTION": scaled_labels}
)
submission = submission.to_dict(orient="records", index=True)

#

api_submit = Submission(api_key=os.getenv("API_KEY"))
submit_time = api_submit.submit(False, submission, "BTCUSDT")

with open("./newlog.txt", "a") as f:
    f.writelines(f"{(pd.Timestamp.now().ceil(freq='H'))} : {submission}  at {submit_time}\n")
