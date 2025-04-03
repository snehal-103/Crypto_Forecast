from django.shortcuts import render, redirect
import yfinance as yf
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from datetime import datetime, timedelta
import joblib
import os
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.http import JsonResponse


def home(request):
    """Home page: Fetches crypto prices, BTC history, and predicts future prices."""

    # Fetch latest crypto prices
    crypto_symbols = ["BTC-USD", "ETH-USD", "DOGE-USD"]
    crypto_data = []

    for symbol in crypto_symbols:
        crypto = yf.Ticker(symbol)
        history = crypto.history(period="1d")
        
        price = round(history["Close"].iloc[-1], 2) if not history.empty else "N/A"
        
        crypto_data.append({
            "symbol": symbol,
            "price": price
        })

    # Fetch historical BTC data
    btc = yf.Ticker("BTC-USD")
    history = btc.history(period="6mo")  # Fetch last 6 months' data

    btc_dates = [date.strftime("%b %d") for date in history.index]
    btc_prices = history["Close"].tolist()

    # Prepare data for SVM model
    df = history.reset_index()
    df['Date'] = df['Date'].map(datetime.toordinal)

    X = df[['Date']].values  
    y = df['Close'].values  

    # Train SVM model
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    model.fit(X, y)

    # Save model
    joblib.dump(model, "svm_model.pkl")

    # Predict future prices
    future_dates = [datetime.today() + timedelta(days=i) for i in [1, 7, 30, 365]]
    future_dates_ordinal = np.array([[date.toordinal()] for date in future_dates])
    predicted_prices = model.predict(future_dates_ordinal)

    predictions = {
        "Next_Day": round(predicted_prices[0], 2),
        "Next_Week": round(predicted_prices[1], 2),
        "Next_Month": round(predicted_prices[2], 2),
        "Next_Year": round(predicted_prices[3], 2),
    }

    return render(request, 'home.html', {
        'crypto_data': crypto_data,
        'btc_dates': json.dumps(btc_dates),
        'btc_prices': json.dumps(btc_prices),
        'predictions': predictions,
    })


def predict(request):
    """Prediction page: Allows user to select a cryptocurrency and predict future prices."""

    predicted_price = None
    selected_crypto = None
    prediction_period = None

    if request.method == "POST":
        # Get user inputs
        selected_crypto = request.POST.get("crypto_symbol")
        prediction_period = float(request.POST.get("prediction_days"))  # Convert to float for hours

        if selected_crypto:
            # Fetch historical data with a 30-minute interval
            crypto = yf.download(selected_crypto, period="60d", interval="30m")

            if crypto.empty:
                return render(request, "predict.html", {
                    "predicted_price": "N/A",
                    "selected_crypto": selected_crypto,
                    "prediction_period": prediction_period,
                })

            # Prepare the dataset
            df = crypto.reset_index()
            df['Timestamp'] = df['Datetime'].map(datetime.toordinal)  # Convert datetime to numeric values

            X = df[['Timestamp']].values
            y = df['Close'].values

            # Train SVM Model
            model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
            model.fit(X, y)

            # Convert selected period (minutes/hours) into the correct future timestamp
            future_time = datetime.now() + timedelta(hours=prediction_period)  # Handle minutes/hours
            future_time_ordinal = np.array([[future_time.toordinal()]])

            # Predict for the selected period
            predicted_price = round(model.predict(future_time_ordinal)[0], 2)

    return render(request, "predict.html", {
        "predicted_price": predicted_price,
        "selected_crypto": selected_crypto,
        "prediction_period": prediction_period,
    })

def news(request):
    """News page."""
    return render(request, "news.html")


# ✅ User Registration
def user_register(request):
    """Handles user registration."""
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "Registration successful! Please log in.")
            return redirect("login")
        else:
            messages.error(request, "Registration failed. Please check your details.")
    else:
        form = UserCreationForm()
    
    return render(request, "register.html", {"form": form})


# ✅ User Login
def user_login(request):
    """Handles user login."""
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            messages.success(request, "Logged in successfully!")
            return redirect("home")
        else:
            messages.error(request, "Invalid username or password.")
    
    return render(request, "login.html")


# ✅ User Logout
def user_logout(request):
    """Handles user logout and redirects to the home page."""
    logout(request)
    messages.success(request, "Logged out successfully!")
    return redirect("home")  # Redirects to home instead of login

from django.contrib.auth.models import User
from django.shortcuts import render

def user_list(request):
    users = User.objects.all()
    return render(request, 'user_list.html', {'users': users})