from flask import Flask, request, render_template
import pandas as pd
import pickle
import requests
import urllib3
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

model_path = "model_phishing.h5"
vocab_path = "vocab.pkl"
data_path = r"C:\Disk D\Download\archive\raw_data.csv"

max_len = 150

# ================================
# TRAIN MODEL NẾU CHƯA CÓ
# ================================
if not os.path.exists(model_path) or not os.path.exists(vocab_path):

    print("🔥 Chưa có model → TRAIN...")

    df = pd.read_csv(data_path)

    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    all_text = ''.join(df['url'].values)
    vocab = sorted(list(set(all_text)))
    char2idx = {char: idx + 1 for idx, char in enumerate(vocab)}

    with open(vocab_path, "wb") as f:
        pickle.dump(char2idx, f)

    def encode_url(url):
        return [char2idx.get(c, 0) for c in url]

    df['encoded'] = df['url'].apply(encode_url)

    X = pad_sequences(df['encoded'], maxlen=max_len, padding='post')
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential()
    model.add(Embedding(input_dim=len(char2idx)+1, output_dim=32, input_length=max_len))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("🚀 Training...")
    model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))

    model.save(model_path)
    print("✅ Train xong")

else:
    print("✅ Load model")

# ================================
# LOAD MODEL + VOCAB
# ================================
model = load_model(model_path)

with open(vocab_path, "rb") as f:
    char2idx = pickle.load(f)

# ================================
# CHECK REDIRECT
# ================================
def check_redirect(url):
    try:
        r = requests.get(url, timeout=5, allow_redirects=True, verify=False)
        return r.url
    except:
        return url

# ================================
# PREDICT
# ================================
def predict_url(url):
    original_url = url
    final_url = check_redirect(url)

    encoded = [char2idx.get(c, 0) for c in final_url]
    padded = pad_sequences([encoded], maxlen=max_len, padding='post')

    pred = model.predict(padded, verbose=0)[0][0]

    percent = round(pred * 100, 2)   # 🔥 đổi sang %

    if pred > 0.7:
        result = "PHISHING ⚠️"
    elif pred > 0.5:
        result = "NGHI NGỜ ⚠️"
    else:
        result = "AN TOÀN ✅"

    is_shortened = original_url != final_url  # 🔥 check rút gọn

    return result, percent, original_url, final_url, is_shortened

# ================================
# ROUTE
# ================================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    percent = None
    original_url = None
    final_url = None
    is_shortened = False

    if request.method == "POST":
        url = request.form["url"]
        result, percent, original_url, final_url, is_shortened = predict_url(url)

    return render_template(
        "index.html",
        result=result,
        percent=percent,
        original_url=original_url,
        final_url=final_url,
        is_shortened=is_shortened
    )

# ================================
# RUN
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))