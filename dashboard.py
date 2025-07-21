import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from streamlit_option_menu import option_menu
from matplotlib.dates import DateFormatter, MonthLocator
from urllib.request import urlopen, Request
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
import ssl
import plotly.graph_objects as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import MinMaxScaler



# Page configuration
st.set_page_config(page_title="📉Stock Price!!!", layout="wide")

# Title and CSS
st.title(" 📈 Stock Price Prediction Dashboard")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)

# File uploader
fl = st.file_uploader(":file_folder: Upload a file", type=(["csv", "txt", "xlsx", "xls"]))
if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv(filename, encoding="ISO-8859-1")
else:
    os.chdir(r"C:\Users\varun\Desktop\Sahithya")
    df = pd.read_csv("input_data_NWG.csv", encoding="ISO-8859-1")
    

hsba=df.copy()
hsba1=df.copy()


df["Date"] = pd.to_datetime(df["Date"])

# Getting the min and max date
startDate = pd.to_datetime(df["Date"]).min()
endDate = pd.to_datetime(df["Date"]).max()

col1, col2 = st.columns((2))

with col1:
    date1 = pd.to_datetime(st.date_input("Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("Date", endDate))

#df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()

# Filter data based on the selected date range
filtered_df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()


###########################################################

# Function to scrape news from Finviz
def scrape_news(stock_ticker):
    finviz_url = f'https://finviz.com/quote.ashx?t={stock_ticker}&p=d'
    ssl._create_default_https_context = ssl._create_stdlib_context

    req = Request(url=finviz_url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')

    # Parse the HTML content
    soup = BeautifulSoup(str(news_table), 'html.parser')

    # Extract the data
    articles = []
    for row in soup.find_all('tr'):
        time_td = row.find('td', align='right')
        link_div = row.find('div', class_='news-link-left')
        source_span = row.find('div', class_='news-link-right').find('span')

        if time_td and link_div and source_span:
            time = time_td.get_text(strip=True)
            link = link_div.find('a')['href']
            title = link_div.find('a').get_text(strip=True)
            source = source_span.get_text(strip=True)

            articles.append({
                'time': time,
                'title': title,
                'link': link,
                'source': source
            })
    return articles

# Display news articles
stock_ticker = 'HSBC'


articles = scrape_news(stock_ticker)


nltk.download('vader_lexicon')



mean_df = pd.DataFrame({
    'date': pd.date_range(start='2024-01-01', periods=74, freq='D'),
    'compound': np.random.uniform(-0.5, 0.5, size=74)
})

# Convert comparison dates to Timestamp
start_date = pd.Timestamp('2024-01-01')
end_date = pd.Timestamp('2024-07-05')

# Ensure date columns in DataFrames are in datetime format
hsba['Date'] = pd.to_datetime(hsba['Date'])
mean_df['date'] = pd.to_datetime(mean_df['date'])

# Filter data by date range
filtered_df = hsba[(hsba['Date'] >= start_date) & (hsba['Date'] <= end_date)]
filtered_mean_df = mean_df[(mean_df['date'] >= start_date) & (mean_df['date'] <= end_date)]


# Create the Streamlit sidebar with navigation options
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=[ "Model 1", "Model 2", "Sentiment Score"],
        icons=["bar-chart-line", "bar-chart-line", "sentiment-dissatisfied"],
        default_index=0
    )

            
############################################################


if selected == "Sentiment Score":
    # Example DataFrame
    start_date = '2024-01-01'
    end_date = '2024-07-05'

    df = pd.DataFrame({
        'Date': pd.date_range(start=start_date, end=end_date, freq='D'),
        'Close': np.random.uniform(100, 200, size=(pd.date_range(start=start_date, end=end_date, freq='D').size,))
    })
    df['Date'] = pd.to_datetime(df['Date'])

    # Example mean_df for sentiment
    mean_df = pd.DataFrame({
        'date': pd.date_range(start=start_date, end=end_date, freq='W'),
        'compound': np.random.uniform(-1, 1, size=(pd.date_range(start=start_date, end=end_date, freq='W').size,))
    })
    mean_df['date'] = pd.to_datetime(mean_df['date'])

    # Ensure proper date filtering
    start_date = pd.Timestamp('2024-01-01')
    end_date = pd.Timestamp('2024-07-05')

    filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    filtered_mean_df = mean_df[(mean_df['date'] >= start_date) & (mean_df['date'] <= end_date)]

    # Plot sentiment scores
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=mean_df, x='date', y='compound', color='b', linewidth=2, ax=ax)
    ax.set_title('Average Sentiment Score by Date', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Compound Sentiment Score', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)

    # Plot Closing Price and Sentiment Score with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot the closing price on the primary y-axis without markers
    color = 'tab:blue'
    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Closing Price', color=color, fontsize=14, fontweight='bold')
    ax1.plot(filtered_df['Date'], filtered_df['Close'], color=color, linestyle='-', label='Closing Price', linewidth=2, marker=None)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.xaxis.set_major_locator(MonthLocator())
    ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a second y-axis for the average sentiment score
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Sentiment Score', color=color, fontsize=14, fontweight='bold')
    ax2.plot(filtered_mean_df['date'], filtered_mean_df['compound'], color=color, linestyle='--', marker='x', label='Average Sentiment Score', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add titles and legends
    plt.title('Closing Price and Average Sentiment Score from January 1, 2024, to July 5, 2024', fontsize=16, fontweight='bold')
    fig.tight_layout()  # Adjust layout to fit labels

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
############################################################
    
    
# Model 1
if selected == "Model 1":
    st.subheader("Model 1 - Stock Price Prediction with Sequence LSTM")
    
    
    # Drop rows with NaN values created by rolling and exponential moving averages
    hsba.dropna(inplace=True)
    
    # Separate features and target
    hsba_features = hsba.drop(['Close'], axis=1, errors='ignore')
    hsba_target = hsba['Close']
    
    # Normalize features and target separately
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # Ensure only numeric data is passed to the scaler
    hsba_features_numeric = hsba_features.select_dtypes(include=[np.number])
    hsba_target_numeric = hsba_target.values.reshape(-1, 1)

    hsba_scaled_features = pd.DataFrame(scaler_features.fit_transform(hsba_features_numeric), columns=hsba_features_numeric.columns, index=hsba.index)
    hsba_scaled_target = pd.DataFrame(scaler_target.fit_transform(hsba_target_numeric), columns=['Close'], index=hsba.index)
    
    # Combine scaled features and target for sequence creation
    hsba_scaled = hsba_scaled_features.join(hsba_scaled_target)
    
    # Prepare sequences for LSTM
    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:i+seq_length].values
            y = data.iloc[i+seq_length]['Close']
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 3
    X, y = create_sequences(hsba_scaled, seq_length)

    # Train-Test Split
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(seq_length, X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                        verbose=1, callbacks=[early_stop])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    # Prepare data with zeros for inverse transformation
    test_data_with_zeros = np.zeros((X_test.shape[0], X_test.shape[2]))
    test_data_with_zeros[:, -1] = y_test
    y_test_with_zeros = np.concatenate((test_data_with_zeros, y_test.reshape(-1, 1)), axis=1)
    
    y_pred_with_zeros = np.concatenate((test_data_with_zeros, y_pred), axis=1)

    # Apply inverse transformation
    y_test_unscaled = scaler_target.inverse_transform(y_test_with_zeros)[:, -1]
    y_pred_unscaled = scaler_target.inverse_transform(y_pred_with_zeros)[:, -1]
    
   
    
    import matplotlib.dates as mdates

    # Ensure 'Date' is the index and is in datetime format
    hsba['Date'] = pd.to_datetime(hsba['Date'])
    hsba.set_index('Date', inplace=True)


# Create a result DataFrame with the index from hsba and predictions
    result_df = pd.DataFrame({
    'Actual': y_test_unscaled,
    'Predicted': y_pred_unscaled
}, index=hsba.index[-len(y_test_unscaled):])

# Visualize predictions vs actual values
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(result_df.index, result_df['Actual'], label='Actual Close Price', color='b')
    ax.plot(result_df.index, result_df['Predicted'], label='Predicted Close Price', color='r')
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Close Price', fontsize=14, fontweight='bold')
    ax.set_title('Actual vs Predicted Close Price', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

# Format the x-axis to show only month and year in YYYY-MM format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show ticks for each month

# Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    st.pyplot(fig)
    
   # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
    mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
    r2_test = r2_score(y_test_unscaled, y_pred_unscaled)

    # Predict on training set to compare
    y_train_pred = model.predict(X_train)

    # Inverse transform predictions and actual values for training set
    # Use scaler_target for inverse transformation
    y_train_with_zeros = np.concatenate((np.zeros((X_train.shape[0], X_train.shape[2] - 1)), y_train.reshape(-1, 1)), axis=1)
    y_train_pred_with_zeros = np.concatenate((np.zeros((X_train.shape[0], X_train.shape[2] - 1)), y_train_pred), axis=1)

    y_train_unscaled = scaler_target.inverse_transform(y_train_with_zeros)[:, -1]
    y_train_pred_unscaled = scaler_target.inverse_transform(y_train_pred_with_zeros)[:, -1]

    # Calculate R-squared (R^2) score for training set
    r2_train = r2_score(y_train_unscaled, y_train_pred_unscaled)

    # Print evaluation metrics
    st.markdown("### Model Evaluation Metrics")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**R-squared (R^2) score for Test set:** {r2_test:.2f}")
    st.write(f"**R-squared (R^2) score for Train set:** {r2_train:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Interpretation of Model Performance")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**Interpretation:** RMSE measures the average magnitude of the errors between predicted and actual values. An RMSE of {rmse:.2f} indicates that, on average, the predictions are off by {rmse:.2f} units. In the context of stock prices, this is relatively low, suggesting that the model’s predictions are quite close to the actual values.")
        
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Interpretation:** MAE represents the average absolute difference between predicted and actual values. An MAE of {mae:.2f} means that, on average, the model's predictions are off by {mae:.2f} units. This metric provides a straightforward interpretation of prediction accuracy and confirms that the model performs well with relatively small prediction errors.")
        
    st.write(f"**R-squared (R²) score for Test set:** {r2_test:.2f}")
    st.write(f"**Interpretation:** R-squared indicates the proportion of the variance in the dependent variable that is predictable from the independent variables. An R² score of {r2_test:.2f} suggests that {r2_test * 100:.0f}% of the variability in the stock’s closing price is explained by the model. This is a high value, indicating that the model fits the test data well.")

    st.markdown("<br>", unsafe_allow_html=True)
    # Display result DataFrame
    st.markdown("### Result of Stock Price Prediction")
    st.write(result_df)




    
    
if selected == "Model 2":
    st.subheader("Model 2 - Stock Price Prediction with Traditional Models")
    
    # Ensure the 'Date' column is in datetime format and set as index
    hsba1['Date'] = pd.to_datetime(hsba1['Date'])
    hsba1.set_index('Date', inplace=True)
    hsba1 = hsba1.sort_index()
    

    # Apply exponential moving average smoothing to the closing price
    hsba1['Close_ema'] = hsba1['Close'].ewm(span=20, adjust=False).mean()

    # Apply moving average smoothing to the closing price
    hsba1['Close_ma'] = hsba1['Close'].rolling(window=20).mean()

    # Extract features and target
    features = ['Volume', 'Close_ma', 'Close_ema']
    target = 'Close'
    X = hsba1[features]
    y = hsba1[target]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, shuffle=False)

    # Normalize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # Train and evaluate models
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Calculate performance metrics
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        results[name] = {
            'RMSE Train': rmse_train,
            'MAE Train': mae_train,
            'R2 Train': r2_train,
            'RMSE Test': rmse_test,
            'MAE Test': mae_test,
            'R2 Test': r2_test
        }

    # Convert results to DataFrame for better display
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(2)  # Round to 2 decimal places

    # Display results in Streamlit as a table
    st.subheader("Model Evaluation Metrics")
    st.dataframe(results_df)

    # Plotting the actual vs predicted prices for all models
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 12))
    fig.tight_layout(pad=5.0)

    for i, (name, model) in enumerate(models.items()):
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        ax = axes[i]
        ax.plot(hsba1.index[:len(y_train)], y_train, label='Actual Train', color='blue')
        ax.plot(hsba1.index[:len(y_train)], y_train_pred, label='Predicted Train', color='orange')
        ax.plot(hsba1.index[len(y_train):len(y_train)+len(y_test)], y_test, label='Actual Test', color='green')
        ax.plot(hsba1.index[len(y_train):len(y_train)+len(y_test)], y_test_pred, label='Predicted Test', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title(f'{name} - Actual vs Predicted Prices')
        ax.legend()

        # Format the x-axis to show only the desired months and years
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        # Set the x-axis limits to match the range of the data
        ax.set_xlim(hsba1.index.min(), hsba1.index.max())

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Show the plots in Streamlit
    st.pyplot(fig)