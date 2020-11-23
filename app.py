from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

from flask import Flask
import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = '12345'
#mysql://bb084372ec497f:ae140677@us-cdbr-east-02.cleardb.com/heroku_95bd855e45bc403?reconnect=true
# Enter your database connection details below
app.config['MYSQL_HOST'] = 'us-cdbr-east-02.cleardb.com'
app.config['MYSQL_USER'] = 'bb084372ec497f'
app.config['MYSQL_PASSWORD'] = 'ae140677'
app.config['MYSQL_DB'] = 'heroku_95bd855e45bc403'

# Intialize MySQL
mysql = MySQL(app)

# http://localhost:5000/pythonlogin/ - this will be the login page, we need to use both GET and POST requests
@app.route('/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
        # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
                # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('index.html', msg='')
# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
                # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)

# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home')
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/timeseries')
def timeseries():
    # Check if user is loggedin
    if 'loggedin' in session:
        # futute forecasting of sales
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the forecast
        return render_template('timeseries.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

#Time-seriesForecasting Model
url='https://raw.githubusercontent.com/Bansari8189/Insights/main/data/table_sales.csv'
    
Insights = pd.read_csv(url,sep=",",
                       index_col ='Month',
                       parse_dates = True , engine='python')

result = seasonal_decompose(Insights['Sales'],
                            model ='multiplicative')

# ETS plot

stepwise_fit = auto_arima(Insights['Sales'], start_p = 1, start_q = 1,
                          max_p = 3, max_q = 3, m = 12,
                          start_P = 0, seasonal = True,
                          d = None, D = 1, trace = True,
                          error_action ='ignore',   # we don't want to know if an order does not work
                          suppress_warnings = True,  # we don't want convergence warnings
                          stepwise = True)           # set to stepwise

# To print the summary
stepwise_fit.summary()

train = Insights.iloc[:len(Insights)-12]
test = Insights.iloc[len(Insights)-12:] # set one year(12 months) for testing

# Fit a SARIMAX(0, 0, 0)x(0, 1, 0, 12) on the training set
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train['Sales'],
                order = (0, 0, 0),
                seasonal_order =(0, 1, 0, 12))

result = model.fit()
#result.summary()

start = len(train)
end = len(train) + len(test) - 1

# Predictions for one-year against the test set
predictions = result.predict(start, end,
                             typ = 'levels').rename("Predictions")
print(predictions)

rmse(test["Sales"], predictions)

# Calculate mean squared error
mean_squared_error(test["Sales"], predictions)

model = SARIMAX(Insights['Sales'],
                        order = (0, 0, 0),
                        seasonal_order =(0, 1, 0, 12))
result = model.fit()

# Forecast for the next 1 year
forecast = result.predict(start = len(Insights),
                          end = (len(Insights)-1) + 1*12,
                          typ = 'levels').rename('Forecast')

# Plot the forecast values
Insights['Sales'].plot(figsize = (10, 6), legend = True)
forecast.plot(legend = True)
#plt.savefig('forecasted_sales.png')
