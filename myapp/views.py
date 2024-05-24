from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.core import serializers
import json
import csv
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse, JsonResponse
from flask import Flask, render_template, request, redirect, flash, session,jsonify
from sklearn.metrics import accuracy_score

from flask_pymongo import PyMongo
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from simpleeval import simple_eval
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
views = Flask(__name__)
df = None
answer = None
def home(request):
    return render(request, "registration.html")

def registration(request):
    if request.method == 'POST':
        username = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        email = request.POST['email']
        pass1 = request.POST['pass1']
        pass2 = request.POST['pass2']
    
        if User.objects.filter(username=username):
            messages.error(request, "Username already exist! Please try some other username")
            return redirect('registration')
        if User.objects.filter(email=email):
            messages.error(request, "Email already registered")
            return redirect('registration')
        if len(username)>10:
            messages.error(request, "Username must be under 10 characters")
            return redirect('registration')
        if pass1 != pass2:
            messages.error(request, "Passwords didn't match")
            return redirect('registration')
        if not username.isalnum():
            messages.error(request, "Username must be Alpha-Numeric")
            return redirect('registration')
        
        myuser = User.objects.create_user(username, email, pass1)
        myuser.first_name=fname
        myuser.last_name=lname
        myuser.is_active = True
        myuser.save()
    
        messages.success(request, "Your Account has been successfully created.")
        return redirect('signin')
        
    return render(request, "registration.html")

def signin(request):
    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username=username, password=pass1)
        
        if user is not None:
            login(request, user)
            fname = user.first_name
            return render(request, "index.html", { 'fname': fname})
        else:
            messages.error(request, "Bad Credentials!")
            return redirect('registration')
    
    return render(request, "signin.html")

def signout(request):
    logout(request)
    messages.success(request, "Logged Out Successfully!")
    return render(request, 'registration.html')

def index(request):
    return render(request, 'index.html')

def dashboard(request):
    return render(request, 'dashboard.html')

@views.route('/dashboard_chartjs', methods=['POST', 'GET'])
def dashboard_chartjs(request):
    global df  # Use the global variable
    error = None
    if request.method == 'POST':
        uploaded_file = request.FILES.get('file')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Get user input for column numbers
            x_column = int(request.POST.get('x_column'))
            y_column = int(request.POST.get('y_column'))

            # Validate user input to ensure column numbers are within the range of the DataFrame
            if x_column < 0 or x_column >= len(df.columns) or y_column < 0 or y_column >= len(df.columns):
                error = "Invalid column numbers. Please enter valid numbers."
            else:
                # Get column names based on user input
                x_column_name = df.columns[x_column]
                y_column_name = df.columns[y_column]

                # Extract data for Chart.js
                labels = df[x_column_name].tolist()
                data = df[y_column_name].tolist()
                print(x_column_name, y_column_name)
                return render(request, 'dashboard_chartjs.html', {'labels': labels, 'data': data, 'x_column_name': x_column_name, 'y_column_name': y_column_name, 'df': df, 'error': error})

    # For GET requests, check if df is None
    if df is None:
        error = "Upload a CSV file first."

    # Render the template with df
    return render(request, 'dashboard_chartjs.html', {'df': df, 'error': error})

@views.route('/index_chartjs', methods=['POST', 'GET'])
def index_chartjs(request):
    return render(request, 'index_chartjs.html')

@views.route('/ask_question', methods=['POST', 'GET'])
def ask_question(request):
    global df
    return render(request, 'ask_question.html', {'df': df})

@views.route('/ml_output', methods=['POST', 'GET'])
def ml_output(request):
    global df
    global answer
    if request.method == 'POST':
        try:
            question_type = request.POST.get('question_type')
            question_column = request.POST.get('question_column')
            if not question_type or not question_column:
                return render(request, 'ml_output.html', {'answer': "Invalid form data. Missing required keys."})

            answer = "Placeholder answer."

            if question_type == 'sort':
                sorted_df = df.sort_values(by=question_column).reset_index(drop=True)
                sorted_html = sorted_df.to_html(index=False)
                df = sorted_df.copy()
                answer = f"Sorted DataFrame by {question_column}:\n{sorted_html}"

            elif question_type == 'search':
                search_value = request.POST.get('search_value')
                if not search_value:
                    return render(request, 'ml_output.html', {'answer': "Invalid form data. Missing search_value for search."})
                result_df = df[df[question_column] == search_value]
                answer = f"Search results for {question_column} = {search_value}: \n{result_df}"

            elif question_type == 'aggregate':
                aggregation_function = request.POST.get('aggregation_function')
                if not aggregation_function:
                    return render(request, 'ml_output.html', {'answer': "Invalid form data. Missing aggregation_function for aggregate."})
                if aggregation_function == 'sum':
                    result = df[question_column].sum()
                elif aggregation_function == 'mean':
                    result = df[question_column].mean()
                elif aggregation_function == 'max':
                    result = df[question_column].max()
                elif aggregation_function == 'min':
                    result = df[question_column].min()
                else:
                    return render(request, 'ml_output.html', {'answer': 'Invalid aggregation_function. Supported functions: sum, mean, max, min.'})
                answer = f"Aggregated {question_column} using {aggregation_function}: {result}"

            elif question_type == 'ml_regression':
                train_file = request.FILES.get('train_file')
                test_file = request.FILES.get('test_file')
                target_column = request.POST.get('target_column')
                
                if not train_file or not test_file or not target_column:
                    return render(request, 'ml_output.html', {'answer': 'Missing training file, testing file, or target column.'})
                
                try:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                except Exception as e:
                    return render(request, 'ml_output.html', {'answer': f'Error reading CSV files: {str(e)}'})
                
                if target_column not in train_df.columns or target_column not in test_df.columns:
                    return render(request, 'ml_output.html', {'answer': f'Target column {target_column} does not exist in the provided DataFrames.'})
                
                X_train = train_df.drop(columns=[target_column])
                y_train = train_df[target_column]
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
                
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Optionally, you can also print the coefficients and intercept
                print("Coefficients:", model.coef_)
                print("Intercept:", model.intercept_)

                # Make predictions
                predictions = model.predict(X_test)

                # Evaluate the model using Mean Squared Error
                mse = mean_squared_error(y_test, predictions)

                # Prepare the answer
                r2 = r2_score(y_test, predictions)

                # Prepare the answer
                answer = {
                    'questionType': 'ml_regression',
                    'r2_score': r2,
                    'predictions': predictions.tolist(),
                    'true_values': y_test.tolist()
                }
                request.session['predictions'] = predictions.tolist()
                request.session['true_values'] = y_test.tolist()

            elif question_type == 'ml_classification':
                train_file = request.FILES.get('train_file')
                test_file = request.FILES.get('test_file')
                target_column = request.POST.get('target_column')
                
                if not train_file or not test_file or not target_column:
                    return render(request, 'ml_output.html', {'answer': 'Missing training file, testing file, or target column.'})
                
                try:
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)
                except Exception as e:
                    return render(request, 'ml_output.html', {'answer': f'Error reading CSV files: {str(e)}'})
                
                if target_column not in train_df.columns or target_column not in test_df.columns:
                    return render(request, 'ml_output.html', {'answer': f'Target column {target_column} does not exist in the provided DataFrames.'})
                
                X_train = train_df.drop(columns=[target_column])
                y_train = train_df[target_column]
                X_test = test_df.drop(columns=[target_column])
                y_test = test_df[target_column]
                # Train the model
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                
                # Make predictions and evaluate the model
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                
                # Format the answer as a table
                headers = ['Feature', 'Importance']
                rows = list(zip(X_train.columns, model.feature_importances_))
                answer = {
                    #'headers': headers,
                    #'rows': rows,
                    'questionType': 'ml_classification',
                    'accuracy': accuracy,
                    'predictions': predictions.tolist(),
                    'true_values': y_test.tolist()
                }
                request.session['predictions'] = predictions.tolist()
                request.session['true_values'] = y_test.tolist()

            elif question_type == 'basic_analysis':
                descriptive_stats = df.describe()
                descriptive_stats_transposed = descriptive_stats.transpose()
                answer = {
                    'questionType': 'basic_analysis',
                    'headers': list(descriptive_stats_transposed.columns),
                    'rows': [list(descriptive_stats_transposed.iloc[i]) for i in range(len(descriptive_stats_transposed))]
                }
            return render(request, 'ml_output.html', {'answer': answer, 'df':df})

        except Exception as e:
            return render(request, 'ml_output.html', {'answer': f'Error: {str(e)}'})
    return render(request, 'ml_output.html', {'df': df})

def download_csv(request):
    # Here you will generate the CSV file on the fly
    predictions = request.session.get('predictions')
    true_values = request.session.get('true_values')

    if predictions is None or true_values is None:
        return HttpResponse("No data to download")

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="predictions.csv"'

    writer = csv.writer(response)
    writer.writerow(["Prediction", "True Value"])
    for pred, true_val in zip(predictions, true_values):
        writer.writerow([pred, true_val])

    return response

if __name__ == '__main__':
    views.run(debug=True)
