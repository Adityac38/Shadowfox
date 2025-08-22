from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from datetime import datetime
import os
import csv
from car_price_prediction import predict_car_price, train_car_price_model
from bike_price_prediction import predict_bike_price, train_bike_model
app = Flask(__name__, static_folder="static", template_folder="templates")
HISTORY_FILE = "history.csv"
MAX_HISTORY = 30
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Type", "Name", "Year", "Present Price (₹ Lakh)", "Kms Driven",
            "Owner", "Fuel Type", "Seller Type", "Transmission",
            "Predicted Price (₹ Lakh)", "Date"
        ])
@app.route("/")
def home():
    return render_template("car.html")
@app.route("/car")
def car_page():
    return render_template("car.html")
@app.route("/bike")
def bike_page():
    return render_template("bike.html")
@app.route("/history")
def history_page():
    records = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    records = list(reversed(records))
    return render_template("history.html", records=records)
def append_history(row: dict):
    rows = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            rows = reader[-(MAX_HISTORY-1):] if len(reader) >= (MAX_HISTORY-1) else reader
    rows.append(row)
    with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Type", "Name", "Year", "Present Price (₹ Lakh)", "Kms Driven",
            "Owner", "Fuel Type", "Seller Type", "Transmission",
            "Predicted Price (₹ Lakh)", "Date"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
@app.route("/predict_car", methods=["POST"])
def predict_car():
    try:
        data = request.get_json(force=True)
        present_price = float(data.get("present_price", 0))
        kms_driven = float(data.get("kms_driven", 0))
        owner = data.get("owner", "First Owner")
        year = int(data.get("year", datetime.now().year))
        fuel = data.get("fuel", "Petrol")
        seller = data.get("seller", "Dealer")
        transmission = data.get("transmission", "Manual")
        car_name = data.get("car_name", "").strip() or "Unknown"
        features = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Year": year,
            "Fuel_Type": fuel,
            "Seller_Type": seller,
            "Transmission": transmission
        }
        try:
            prediction_val = predict_car_price(features)
        except Exception:
            train_car_price_model()
            prediction_val = predict_car_price(features)
        predicted_text = f"₹{prediction_val} Lakh"
        row = {
            "Type": "Car",
            "Name": car_name,
            "Year": year,
            "Present Price (₹ Lakh)": present_price,
            "Kms Driven": kms_driven,
            "Owner": owner,
            "Fuel Type": fuel,
            "Seller Type": seller,
            "Transmission": transmission,
            "Predicted Price (₹ Lakh)": prediction_val,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        append_history(row)
        return jsonify({"prediction": predicted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route("/predict_bike", methods=["POST"])
def predict_bike():
    try:
        data = request.get_json(force=True)
        present_price = float(data.get("present_price", 0))
        kms_driven = float(data.get("kms_driven", 0))
        owner = data.get("owner", "First Owner")
        year = int(data.get("year", datetime.now().year))
        fuel_type = data.get("fuel_type", "Petrol")
        seller_type = data.get("seller_type", "Dealer")
        bike_name = data.get("vehicle_name", "").strip() or "Unknown"
        features = {
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Owner": owner,
            "Year": year,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type
        }
        try:
            pred_val = predict_bike_price(features)
        except Exception:
            train_bike_model()
            pred_val = predict_bike_price(features)
        predicted_text = f"₹{pred_val} Lakh"
        row = {
            "Type": "Bike",
            "Name": bike_name,
            "Year": year,
            "Present Price (₹ Lakh)": present_price,
            "Kms Driven": kms_driven,
            "Owner": owner,
            "Fuel Type": fuel_type,
            "Seller Type": seller_type,
            "Transmission": "",
            "Predicted Price (₹ Lakh)": pred_val,
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        append_history(row)
        return jsonify({"prediction": predicted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
@app.route("/download_history")
def download_history():
    if os.path.exists(HISTORY_FILE):
        return send_file(HISTORY_FILE, as_attachment=True)
    return redirect(url_for("history_page"))
@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        with open(HISTORY_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Type", "Name", "Year", "Present Price (₹ Lakh)", "Kms Driven",
                "Owner", "Fuel Type", "Seller Type", "Transmission",
                "Predicted Price (₹ Lakh)", "Date"
            ])
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)
