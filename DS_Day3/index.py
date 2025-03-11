import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import datetime

app = Flask(__name__)

# Load and prepare the data
sales_df = pd.read_csv('./sales_data_sample.csv')
orders_df = pd.read_csv('./orders.csv.csv')

# Convert date columns to datetime
sales_df['ORDERDATE'] = pd.to_datetime(sales_df['ORDERDATE'])

class QueryProcessor:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Predefined query patterns
        self.query_patterns = {
            "top earning": self._get_top_earning,
            "best sales city": self._get_best_sales_city,
            "sales performance": self._get_sales_performance,
            "customer segmentation": self._get_customer_segmentation,
            "product demand": self._get_product_demand,
            "regional sales": self._get_regional_sales,
            "fulfillment efficiency": self._get_fulfillment_efficiency,
            "sales trend": self._get_sales_trend
        }
        
    def process_query(self, query):
        # Vectorize the query
        query_vec = self.vectorizer.fit_transform([query])
        
        # Find best matching pattern
        best_match = None
        highest_similarity = 0
        
        for pattern in self.query_patterns.keys():
            pattern_vec = self.vectorizer.transform([pattern])
            similarity = cosine_similarity(query_vec, pattern_vec)[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = pattern
                
        if best_match and highest_similarity > 0.3:
            return self.query_patterns[best_match]()
        else:
            return "I couldn't understand your question."
            
    def _get_top_earning(self):
        top_product = self.df.groupby('PRODUCTCODE')['SALES'].sum().sort_values(ascending=False).head(1)
        return f"The top earning product is {top_product.index[0]} with total sales of ${top_product.values[0]:,.2f}"


class QueryProcessor:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Predefined query patterns
        self.query_patterns = {
            "top earning": self._get_top_earning,
            "best sales city": self._get_best_sales_city,
            "sales performance": self._get_sales_performance,
            "customer segmentation": self._get_customer_segmentation,
            "product demand": self._get_product_demand,
            "regional sales": self._get_regional_sales,
            "fulfillment efficiency": self._get_fulfillment_efficiency,
            "sales trend": self._get_sales_trend
        }
    
    def process_query(self, query):
        # Vectorize the query
        query_vec = self.vectorizer.fit_transform([query])
        
        # Find best matching pattern
        best_match = None
        highest_similarity = 0
        
        for pattern in self.query_patterns.keys():
            pattern_vec = self.vectorizer.transform([pattern])
            similarity = cosine_similarity(query_vec, pattern_vec)[0][0]
            
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = pattern
                
        if best_match and highest_similarity > 0.3:
            return self.query_patterns[best_match]()
        else:
            return "I couldn't understand your question."

    def _get_top_earning(self):
        top_product = self.df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).head(1)
        return f"The top earning product line is {top_product.index[0]} with total sales of ${top_product.values[0]:,.2f}"

    def _get_best_sales_city(self):
        city_sales = self.df.groupby('CITY')['SALES'].sum().sort_values(ascending=False).head(1)
        return f"The city with highest sales is {city_sales.index[0]} with total sales of ${city_sales.values[0]:,.2f}"

    def _get_sales_performance(self):
        # Top 5 products in Q4 2003 with shipped status and min quantity 40
        q4_2003 = self.df[
            (self.df['ORDERDATE'].dt.year == 2003) & 
            (self.df['ORDERDATE'].dt.quarter == 4) &
            (self.df['STATUS'] == 'Shipped') &
            (self.df['QUANTITYORDERED'] >= 40)
        ]
        top_products = q4_2003.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).head(5)
        return f"Top 5 products in Q4 2003: {', '.join(top_products.index.tolist())}"

    def _get_customer_segmentation(self):
        # Customers with >3 orders above $5000 in 2003 from USA/France
        high_value_orders = self.df[
            (self.df['ORDERDATE'].dt.year == 2003) &
            (self.df['SALES'] > 5000) &
            (self.df['COUNTRY'].isin(['USA', 'France']))
        ]
        customer_counts = high_value_orders.groupby('CUSTOMERNAME').size()
        qualified_customers = customer_counts[customer_counts > 3]
        return f"Found {len(qualified_customers)} high-value customers"

    def _get_product_demand(self):
        # Month with highest avg order quantity in 2003
        monthly_stats = self.df[
            (self.df['ORDERDATE'].dt.year == 2003) &
            (self.df['PRICEEACH'] > 80)
        ].groupby(self.df['ORDERDATE'].dt.month).agg({
            'QUANTITYORDERED': 'mean',
            'SALES': 'sum'
        })
        qualified_months = monthly_stats[monthly_stats['SALES'] > 100000]
        top_month = qualified_months['QUANTITYORDERED'].idxmax()
        return f"Month {top_month} had the highest average order quantity"

    def _get_regional_sales(self):
        # Compare CA vs NY average order value in 2003
        state_comparison = self.df[
            (self.df['ORDERDATE'].dt.year == 2003) &
            (self.df['STATUS'] == 'Shipped') &
            (self.df['QUANTITYORDERED'] >= 20) &
            (self.df['STATE'].isin(['CA', 'NY']))
        ].groupby('STATE')['SALES'].mean()
        return f"CA avg: ${state_comparison.get('CA', 0):,.2f}, NY avg: ${state_comparison.get('NY', 0):,.2f}"

    def _get_fulfillment_efficiency(self):
        # Countries with highest proportion of quick shipments
        return "Fulfillment efficiency analysis requires order processing time data"

    def _get_sales_trend(self):
        # Monthly sales trend analysis
        monthly_sales = self.df.groupby([
            self.df['ORDERDATE'].dt.year,
            self.df['ORDERDATE'].dt.month
        ])['SALES'].sum()
        return "Sales trend analysis completed"

class RecommendationEngine:
    def __init__(self, sales_df, orders_df):
        self.sales_df = sales_df
        self.orders_df = orders_df
        
    def get_recommendations(self, month):
        # Prepare features for the specified month
        monthly_data = self.sales_df[self.sales_df['ORDERDATE'].dt.month == month]
        
        # Create feature matrix
        features = [
            'QUANTITYORDERED',
            'PRICEEACH',
            'SALES',
            'QTR_ID',
            'MONTH_ID'
        ]
        
        X = monthly_data[features].values
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimension reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Find anomalies using distance from mean
        distances = np.sqrt(np.sum(X_pca**2, axis=1))
        threshold = np.mean(distances) + 2*np.std(distances)
        
        anomalies = monthly_data[distances > threshold]
        
        if len(anomalies) > 0:
            return {
                'anomalies_found': True,
                'message': f"Found {len(anomalies)} unusual orders in month {month}",
                'details': anomalies[['PRODUCTCODE', 'SALES', 'QUANTITYORDERED']].to_dict('records')
            }
        else:
            return {
                'anomalies_found': False,
                'message': f"No significant anomalies found in month {month}"
            }


@app.route('/query', methods=['POST'])
def process_query():
    try:
        # Try to get JSON data
        if request.is_json:
            data = request.get_json()
            query = data.get('query')
        else:
            # Try to get form data
            query = request.form.get('query')
            if not query:
                # Try to get direct data
                query = request.data.decode('utf-8')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
            
        # Process the query
        processor = QueryProcessor(sales_df)
        result = processor.process_query(query)
    
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/recommend/<int:month>', methods=['GET'])
def get_recommendations(month):
    if month < 1 or month > 12:
        return jsonify({'error': 'Invalid month'}), 400
        
    engine = RecommendationEngine(sales_df, orders_df)
    recommendations = engine.get_recommendations(month)
    
    return jsonify(recommendations)
if __name__ == '__main__':
    app.run(debug=True)
