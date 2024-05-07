from collections import defaultdict
import os
import pathlib
import requests
from flask import Flask, abort, jsonify, request, render_template, redirect, session, url_for, send_file  # Import send_file

from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from pip._vendor import cachecontrol
import google.auth.transport.requests
from pymongo import MongoClient
import csv
from flask import make_response
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder

app = Flask("Google Login App")
app.secret_key = "CodeSpecialist.com"

GOOGLE_CLIENT_ID = "68744667544-ghpi9886mtv27ku5bl8qpuk1tiu9apin.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

flow = Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="https://127.0.0.1:5000/callback"
)


client = MongoClient("mongodb+srv://devarshmistry25:devarsh123@cluster0.udmdq3e.mongodb.net/")
db = client["ibm_project"]

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/uploads")
def uploads():
    return render_template("upload_file.html")

@app.route("/")
def login():
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return render_template("login.html", authorization_url=authorization_url)

@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url)

    if not session["state"] == request.args["state"]:
        abort(500)

    credentials = flow.credentials
    if not credentials.token:
        abort(500, "Access token not available")

    request_session = requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token(
        id_token=credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )

    session["google_id"] = id_info.get("sub")
    session["name"] = id_info.get("name")
    return redirect("/index")

@app.route("/logout")
def logout():
    credentials = flow.credentials
    if credentials and credentials.token:
        revoke_url = f"https://accounts.google.com/o/oauth2/revoke?token={credentials.token}"
        requests.get(revoke_url)

    session.clear()
    return redirect("/")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        return redirect(request.url)

    if file:
        filename = os.path.splitext(file.filename)[0]
        collection_name = "temp_" + filename
        collection = db[collection_name]
        csv_data = csv.reader(file.stream.read().decode("UTF-8").splitlines())
        header = next(csv_data)
        for row in csv_data:
            data = {header[i]: row[i] for i in range(len(header))}
            collection.insert_one(data)
        return jsonify({"status": "success"})

# def clean_data(df, drop_null=False, fill_mean=False, fill_std=False):
#     if drop_null:
#         df = df.dropna()
#     if fill_mean:
#         df = df.fillna(df.mean())
#     if fill_std:
#         df = df.fillna(df.std())
#     return df

# @app.route("/collections", endpoint="list_collections")
# def list_collections_route():
#     collections = db.list_collection_names()
#     return render_template("collections.html", collections=collections)

# @app.route("/collection/<collection_name>", methods=['GET', 'POST'])
# def view_collection(collection_name):
#     collection = db[collection_name]
#     documents = collection.find().limit(10)
#     df = pd.DataFrame(list(documents))
#     df.fillna('N/A', inplace=True)
#     df_summary = df.describe()
#     return render_template("view_collection.html", df=df, df_summary=df_summary, collection_name=collection_name)

@app.route("/clean_data/<collection_name>", methods=['GET', 'POST'])
def clean_data_route(collection_name):
    if request.method == 'POST':
        drop_null = 'drop_null' in request.form
        fill_mean = 'fill_mean' in request.form
        fill_std = 'fill_std' in request.form
        collection = db[collection_name]
        documents = collection.find().limit(10)
        df = pd.DataFrame(list(documents))
        df_cleaned = clean_data(df, drop_null=drop_null, fill_mean=fill_mean, fill_std=fill_std)

        return render_template("cleaned_data.html", df_cleaned=df_cleaned, collection_name=collection_name)

    return redirect(url_for('view_collection', collection_name=collection_name))

def clean_data(df, drop_null=False, fill_mean=False, fill_std=False):
    if '_id' in df.columns:
        df.drop('_id', axis=1, inplace=True)   
    if drop_null:
        df = df.dropna()
    if fill_mean:
        df = df.fillna(df.mean())
    if fill_std:
        df = df.fillna(df.std())
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].astype(str)
    return df

def select_model_and_visualize(df, model_type, target_column):
    try:
        for col in df.columns:
            if df[col].dtype == 'object':
                if col != target_column:
                    label_encoder = LabelEncoder()
                    df[col] = label_encoder.fit_transform(df[col])

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'classification':
            model = RandomForestClassifier()
        elif model_type == 'regression':
            model = RandomForestRegressor()

        model.fit(X_train, y_train)

        if model_type == 'classification':
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            model_performance = f"Accuracy: {accuracy:.2f}"
        elif model_type == 'regression':
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            model_performance = f"Root Mean Squared Error: {rmse:.2f}"

        sns.pairplot(df)
        plt.savefig('static/pairplot.png')  # Save plot
        plt.close()

        return model_performance

    except ValueError as e:
        return f"Error occurred: {str(e)}"

@app.route("/collections", endpoint="list_collections")
def list_collections_route():
    collections = db.list_collection_names()
    return render_template("collections.html", collections=collections)

@app.route("/collection/<collection_name>", methods=['GET', 'POST'])
def view_collection(collection_name):
    collection = db[collection_name]
    documents = collection.find().limit(10)
    df = pd.DataFrame(list(documents))
    df.fillna('N/A', inplace=True)
    df_summary = df.describe()

    if request.method == 'POST':
        drop_null = 'drop_null' in request.form
        fill_mean = 'fill_mean' in request.form
        fill_std = 'fill_std' in request.form
        df_cleaned = clean_data(df, drop_null=drop_null, fill_mean=fill_mean, fill_std=fill_std)
        return render_template("view_collection.html", df_cleaned=df_cleaned, df_summary=df_summary, collection_name=collection_name)
    return render_template("view_collection.html", df=df, df_summary=df_summary, collection_name=collection_name)

@app.route("/model_selection_and_visualization/<collection_name>", methods=['POST'])
def model_selection_and_visualization(collection_name):
    model_type = request.form['model_type']
    target_column = request.form['target_column']

    collection = db[collection_name]
    documents = collection.find().limit(10)
    df = pd.DataFrame(list(documents))
    df.fillna('N/A', inplace=True)
    df_cleaned = clean_data(df)

    model_performance = select_model_and_visualize(df_cleaned, model_type, target_column)

    return render_template("model_selection_and_visualization.html", model_performance=model_performance)


def detect_delimiter(line):
    possible_delimiters = ['\t', ',', ';', '|', ' ']
    delimiter_counts = defaultdict(int)

    for char in line:
        if char in possible_delimiters:
            delimiter_counts[char] += 1

    return max(delimiter_counts, key=delimiter_counts.get)

def convert_log_to_csv(log_file_path):
    file_name, file_extension = os.path.splitext(log_file_path)
    csv_file_path = file_name + ".csv"

    with open(log_file_path, 'r') as log_file:
        first_line = log_file.readline().strip()
        delimiter = detect_delimiter(first_line)
        
        headers = first_line.split(delimiter)

        with open(csv_file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(headers)

            for line in log_file:
                row = line.strip().split(delimiter)
                csv_writer.writerow(row)
    
    return csv_file_path

@app.route('/log_to_csv', methods=['GET', 'POST'])
def log_to_csv():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('', file.filename)
            file.save(file_path)
            csv_path = convert_log_to_csv(file_path)
            return redirect(url_for('download_csv', filename=os.path.basename(csv_path)))
    
    return render_template('log_to_csv.html')

@app.route('/download/<filename>')
def download_csv(filename):
    csv_path = os.path.join('', filename)
    return send_file(csv_path, as_attachment=True)

if __name__ == "__main__":
    ssl_context = ('cert.pem', 'key.pem'    )
    app.run(debug=True, ssl_context=ssl_context, host="127.0.0.1", port=5000)
