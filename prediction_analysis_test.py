"""
Prediction Analysis Module - Part 1: Imports and Core Classes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

class PredictionAnalysis:
    """Main prediction analysis class that handles all forecasting functionality"""
    
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        self.predictions_cache = {}
    
    def get_most_recent_file(self, pattern: str, folder: str = ".") -> Optional[str]:
        """Get the most recent file matching the pattern"""
        search_path = os.path.join(folder, pattern) if folder != "." else pattern
        files = glob.glob(search_path)
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    
    def load_country_data(self, country_key: str) -> Dict:
        """Load country data from Excel files with Singapore Property Price filter"""
        country_data = {}
        
        # Map country key to file pattern
        if country_key.lower() == 'singapore':
            pattern = "standardized_cleaned_macro_data_singapore_*.xlsx"
        else:
            pattern = f"cleaned_macro_data_{country_key.lower()}_*.xlsx"
        
        file_path = self.get_most_recent_file(pattern, "extracted_data")
        
        if not file_path:
            print(f"No file found for {country_key}")
            return {}
        
        try:
            indicators = ['GDP', 'CPI', 'Interest_Rate', 'Population', 'Property_Price']
            
            for indicator in indicators:
                try:
                    df = pd.read_excel(file_path, sheet_name=indicator)
                    
                    # Singapore Property Filter: Filter for "All Residential" only
                    if country_key.lower() == 'singapore' and indicator == 'Property_Price':
                        print(f"   Filtering Singapore Property Price data for 'All Residential' only")
                        
                        if 'property_type' in df.columns:
                            original_count = len(df)
                            df_filtered = df[df['property_type'].str.contains('All Residential', case=False, na=False)]
                            
                            if df_filtered.empty:
                                df_filtered = df[df['property_type'].str.contains('All', case=False, na=False)]
                            
                            if df_filtered.empty:
                                df_filtered = df.head(50)  # Fallback
                            
                            df = df_filtered
                            print(f"   Property Price filtered: {original_count} → {len(df)} rows")
                    
                    # Find date and value columns
                    date_col = next((col for col in df.columns if 'date' in col.lower()), None)
                    value_col = 'value' if 'value' in df.columns else df.columns[1]
                    
                    if date_col and value_col:
                        clean_df = df[[date_col, value_col]].copy()
                        clean_df.columns = ['date', 'value']
                        clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce')
                        clean_df = clean_df.dropna().sort_values('date').reset_index(drop=True)
                        
                        if len(clean_df) > 0:
                            country_data[indicator] = clean_df
                            print(f"   {indicator}: {len(clean_df)} data points loaded")
                
                except Exception as e:
                    print(f"Error loading {indicator} for {country_key}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading country data for {country_key}: {e}")
        
        return country_data
    
    def load_all_countries_data(self) -> Dict:
        """Load data for all available countries"""
        countries = [
            'china', 'euro area', 'india', 'indonesia', 'japan', 
            'malaysia', 'thailand', 'uk', 'us', 'vietnam', 'singapore'
        ]
        
        all_data = {}
        for country in countries:
            country_data = self.load_country_data(country)
            if country_data:
                all_data[country] = country_data
        
        return all_data
    
"""
Prediction Analysis Module - Part 2: MacroPredictor Class
Individual indicator prediction using multiple time series models
"""

class MacroPredictor:
    """Individual indicator prediction using multiple time series models"""
    
    def __init__(self, indicator_type):
        self.indicator_type = indicator_type
        self.model = None
        self.last_training_date = None
        self.model_performance = {}
        
    def prepare_data(self, data, target_col='value', date_col='date'):
        """Prepare data for time series modeling"""
        df = data.copy()
        
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
        
        # Remove duplicates and handle missing values
        df = df.drop_duplicates(subset=[date_col]).dropna(subset=[target_col])
        
        return df
    
    def detect_frequency(self, data, date_col='date'):
        """Auto-detect data frequency"""
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        if len(df) < 2:
            return 'unknown'
        
        date_diffs = df[date_col].diff().dropna()
        
        if len(date_diffs) == 0:
            return 'unknown'
        
        median_diff = date_diffs.median()
        
        if median_diff <= pd.Timedelta(days=7):
            return 'weekly'
        elif median_diff <= pd.Timedelta(days=32):
            return 'monthly'
        elif median_diff <= pd.Timedelta(days=100):
            return 'quarterly'
        else:
            return 'yearly'
    
    def fit_simple_model(self, data, periods_ahead=12):
        """Fit simple trend + seasonal model - FIXED trend calculation"""
        df = self.prepare_data(data)
        
        if len(df) < 12:
            return None, "Insufficient data for prediction"
        
        # Use recent data for consistent trend calculation
        trend_data = df.tail(min(24, len(df)))
        
        # Calculate trend using recent data
        trend_indices = np.arange(len(trend_data))
        trend_coef = np.polyfit(trend_indices, trend_data['value'], 1)
        recent_trend_slope = trend_coef[0]
        
        explanation_parts = ["Simple trend model fitted successfully"]
        
        # Determine trend direction
        if recent_trend_slope > 0.1:
            explanation_parts.append(f"Recent trend: INCREASING (+{recent_trend_slope:.2f} per period)")
            expected_direction = "UPWARD"
        elif recent_trend_slope < -0.1:
            explanation_parts.append(f"Recent trend: DECREASING ({recent_trend_slope:.2f} per period)")
            expected_direction = "DOWNWARD"
        else:
            explanation_parts.append(f"Recent trend: STABLE (≈{recent_trend_slope:.2f} per period)")
            expected_direction = "STABLE"
        
        # Simple seasonal pattern for monthly data
        frequency = self.detect_frequency(df)
        seasonal_avg = pd.Series()
        
        if frequency == 'monthly' and len(df) >= 24:
            df['month'] = pd.to_datetime(df['date']).dt.month
            seasonal_avg = df.groupby('month')['value'].mean()
            explanation_parts.append("Seasonal patterns detected and applied")
        
        # Generate future dates
        last_date = pd.to_datetime(df['date']).max()
        
        if frequency == 'quarterly':
            future_dates = [last_date + pd.DateOffset(months=(i+1)*3) for i in range(periods_ahead)]
            explanation_parts.append(f"Quarterly forecast: {periods_ahead} quarters ahead")
        else:
            future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]
            explanation_parts.append(f"Monthly forecast: {periods_ahead} months ahead")
        
        # FIXED: Generate predictions using CONSISTENT indexing
        predictions = []
        last_trend_value = trend_data['value'].iloc[-1]  # Use last actual value as baseline
        
        for i, future_date in enumerate(future_dates):
            # FIXED: Use incremental trend from last actual value
            trend_increment = recent_trend_slope * (i + 1)
            trend_value = last_trend_value + trend_increment
            
            if frequency == 'monthly' and not seasonal_avg.empty:
                seasonal_value = seasonal_avg.get(future_date.month, 0) - trend_data.groupby(pd.to_datetime(trend_data['date']).dt.month)['value'].mean().mean()
                pred_value = trend_value + seasonal_value
            else:
                pred_value = trend_value
                
            predictions.append({
                'date': future_date,
                'predicted_value': pred_value,
                'model': 'Simple Trend'
            })
        
        explanation_parts.append(f"Forecast direction: {expected_direction} (continues recent linear trend)")
        
        # Add explanation for Property Price trends
        if expected_direction == "DOWNWARD" and self.indicator_type == 'Property_Price':
            explanation_parts.append("Recent data shows downward trend, model extrapolates this pattern.")
        elif expected_direction == "UPWARD" and self.indicator_type == 'Property_Price':
            explanation_parts.append("Recent data shows upward trend, model extrapolates this pattern.")
        
        status_message = " | ".join(explanation_parts)
        
        return pd.DataFrame(predictions), status_message
    
    def fit_prophet_model(self, data, periods_ahead=12):
        """Fit Prophet model for forecasting - FIXED to focus on recent trend"""
        if not HAS_PROPHET:
            return self.fit_simple_model(data, periods_ahead)
        
        df = self.prepare_data(data)
        
        if len(df) < 24:
            return self.fit_simple_model(data, periods_ahead)
        
        # FOR PROPERTY PRICES: Use only recent data to capture current trend
        if self.indicator_type == 'Property_Price' and len(df) > 40:
            print(f"Property Price detected: Using recent {min(40, len(df))} data points for trend focus")
            df = df.tail(40)  # Use only last 40 quarters (~10 years) for property prices
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df['value']
        })
        
        try:
            # Configure Prophet for property prices - more aggressive trend following
            model_params = {
                'weekly_seasonality': False, 
                'daily_seasonality': False,
                'yearly_seasonality': True,
                'changepoint_prior_scale': 0.05,  # More sensitive to trend changes
                'seasonality_prior_scale': 10.0,   # Stronger seasonality
            }
            
            if self.indicator_type == 'Property_Price':
                model_params['changepoint_prior_scale'] = 0.1  # Even more sensitive for property
                model_params['yearly_seasonality'] = False     # Less seasonality for property
            
            model = Prophet(**model_params)
            model.fit(prophet_data)
            
            # Create future dates
            frequency = self.detect_frequency(df)
            last_training_date = prophet_data['ds'].max()
            
            if frequency == 'quarterly':
                future_dates = [last_training_date + pd.DateOffset(months=(i+1)*3) for i in range(periods_ahead)]
            else:
                future_dates = [last_training_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]
            
            future = pd.DataFrame({'ds': prophet_data['ds'].tolist() + future_dates})
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Return only future predictions
            future_predictions = forecast[forecast['ds'] > prophet_data['ds'].max()].copy()
            future_predictions = future_predictions.rename(columns={
                'ds': 'date', 
                'yhat': 'predicted_value',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'upper_bound'
            })
            future_predictions['model'] = 'Prophet'
            
            self.model = model
            
            # Check if Prophet predictions are below current value (problematic)
            current_value = df['value'].iloc[-1]
            first_prediction = future_predictions['predicted_value'].iloc[0]
            
            if first_prediction < current_value * 0.95:  # If prediction drops >5%
                print(f"WARNING: Prophet predicting decline from {current_value:.2f} to {first_prediction:.2f}")
                print("This may indicate Prophet is being overly conservative for recent trends")
            
            return future_predictions[['date', 'predicted_value', 'lower_bound', 'upper_bound', 'model']], "Prophet model fitted successfully"
            
        except Exception as e:
            print(f"Prophet model failed for {self.indicator_type}: {e}")
            return self.fit_simple_model(data, periods_ahead)
    
    def fit_arima_model(self, data, periods_ahead=12):
        """
        Industry-grade ARIMA with seasonal components and extended parameter search
        """
        if not HAS_STATSMODELS:
            return self.fit_simple_model(data, periods_ahead)
        
        df = self.prepare_data(data)
        
        if len(df) < 50:
            return self.fit_simple_model(data, periods_ahead)
        
        try:
            print(f"Advanced ARIMA fitting with {len(df)} data points...")
            
            # IMPROVEMENT 1: Extended parameter search for property data
            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_seasonal = (0, 0, 0, 0)
            best_model = None
            
            frequency = self.detect_frequency(df)
            seasonal_period = 4 if frequency == 'quarterly' else 12  # 4 quarters or 12 months
            
            # IMPROVEMENT 2: Test both regular ARIMA and SARIMA
            model_configs = []
            
            # Regular ARIMA with extended parameter range
            for p in range(0, 6):  # Extended from 3 to 6
                for d in range(0, 3):  # Extended from 2 to 3
                    for q in range(0, 4):  # Extended from 3 to 4
                        model_configs.append({
                            'order': (p, d, q),
                            'seasonal_order': (0, 0, 0, 0),
                            'type': 'ARIMA'
                        })
            
            # SARIMA models for property seasonality
            if len(df) >= 2 * seasonal_period:  # Need at least 2 full cycles
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            for P in range(0, 2):  # Seasonal AR
                                for D in range(0, 2):  # Seasonal differencing
                                    for Q in range(0, 2):  # Seasonal MA
                                        model_configs.append({
                                            'order': (p, d, q),
                                            'seasonal_order': (P, D, Q, seasonal_period),
                                            'type': 'SARIMA'
                                        })
            
            print(f"Testing {len(model_configs)} model configurations...")
            
            # IMPROVEMENT 3: Robust model selection with multiple criteria
            for i, config in enumerate(model_configs):
                try:
                    if config['seasonal_order'] == (0, 0, 0, 0):
                        model = ARIMA(df['value'].values, order=config['order'])
                    else:
                        # SARIMA model
                        from statsmodels.tsa.statespace.sarimax import SARIMAX
                        model = SARIMAX(df['value'].values, 
                                    order=config['order'],
                                    seasonal_order=config['seasonal_order'])
                    
                    fitted_model = model.fit(method_kwargs={"warn_convergence": False}, disp=False)
                    
                    # IMPROVEMENT 4: Multi-criteria model selection
                    aic_score = fitted_model.aic
                    
                    # Penalize overly complex models
                    complexity_penalty = (sum(config['order']) + sum(config['seasonal_order'][:3])) * 2
                    adjusted_score = aic_score + complexity_penalty
                    
                    if adjusted_score < best_aic:
                        best_aic = adjusted_score
                        best_order = config['order']
                        best_seasonal = config['seasonal_order']
                        best_model = fitted_model
                        
                    if i % 20 == 0:  # Progress indicator
                        print(f"  Tested {i+1}/{len(model_configs)} configurations...")
                        
                except Exception as e:
                    continue
            
            if best_model is None:
                print("All advanced ARIMA configurations failed, falling back to simple model")
                return self.fit_simple_model(data, periods_ahead)
            
            model_type = "SARIMA" if best_seasonal != (0, 0, 0, 0) else "ARIMA"
            print(f"Best {model_type} order: {best_order}, seasonal: {best_seasonal}, AIC: {best_aic:.2f}")
            
            # IMPROVEMENT 5: Enhanced diagnostics
            # Check residuals for remaining patterns
            residuals = best_model.resid
            ljung_box = None
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
                p_value = ljung_box['lb_pvalue'].iloc[-1]
                print(f"Ljung-Box test p-value: {p_value:.4f} ({'PASS' if p_value > 0.05 else 'FAIL - residuals have patterns'})")
            except:
                pass
            
            # Generate forecast
            forecast_result = best_model.forecast(steps=periods_ahead)
            
            # IMPROVEMENT 6: Better confidence intervals
            try:
                forecast_obj = best_model.get_forecast(steps=periods_ahead)
                conf_int = forecast_obj.conf_int()
            except:
                conf_int = None
            
            # Create future dates
            last_date = pd.to_datetime(df['date']).max()
            
            if frequency == 'quarterly':
                future_dates = [last_date + pd.DateOffset(months=(i+1)*3) for i in range(periods_ahead)]
            else:
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]
            
            predictions = []
            for i, future_date in enumerate(future_dates):
                pred_value = forecast_result[i]
                
                # Handle confidence intervals
                if conf_int is not None:
                    try:
                        if hasattr(conf_int, 'iloc'):
                            lower_bound = conf_int.iloc[i, 0]
                            upper_bound = conf_int.iloc[i, 1]
                        else:
                            lower_bound = conf_int[i, 0]
                            upper_bound = conf_int[i, 1]
                    except:
                        # IMPROVEMENT 7: Realistic confidence intervals for property
                        volatility = df['value'].pct_change().std() * df['value'].iloc[-1]
                        lower_bound = pred_value - (1.96 * volatility * np.sqrt(i+1))
                        upper_bound = pred_value + (1.96 * volatility * np.sqrt(i+1))
                else:
                    # Fallback confidence intervals
                    volatility = df['value'].pct_change().std() * df['value'].iloc[-1]
                    lower_bound = pred_value - (1.96 * volatility * np.sqrt(i+1))
                    upper_bound = pred_value + (1.96 * volatility * np.sqrt(i+1))
                
                predictions.append({
                    'date': future_date,
                    'predicted_value': pred_value,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'model': f'{model_type}{best_order}' + (f'x{best_seasonal}' if model_type == 'SARIMA' else '')
                })
            
            # IMPROVEMENT 8: Enhanced status message with diagnostics
            status_parts = [
                f"Advanced {model_type} model fitting",
                f"Best order: {best_order}" + (f", seasonal: {best_seasonal}" if model_type == 'SARIMA' else ""),
                f"AIC: {best_aic:.2f}",
                f"Tested {len(model_configs)} configurations"
            ]
            
            if ljung_box is not None:
                status_parts.append(f"Residual diagnostics: {'PASS' if p_value > 0.05 else 'PATTERNS DETECTED'}")
            
            status_message = " | ".join(status_parts)
            
            self.model = best_model
            return pd.DataFrame(predictions), status_message
            
        except Exception as e:
            print(f"Advanced ARIMA model failed: {e}")
            import traceback
            traceback.print_exc()
            return self.fit_simple_model(data, periods_ahead)
    
    def predict(self, data, periods_ahead=12, model_type='auto'):
        """Generate predictions with model selection"""
        if len(data) < 12:
            return None, "Insufficient historical data (minimum 12 periods required)"
        
        # Auto-selection with explanation
        selection_explanation = []
        
        if model_type == 'auto':
            selection_explanation.append("AUTO MODE SELECTION:")
            
            if len(data) >= 50 and HAS_STATSMODELS:
                model_type = 'arima'
                selection_explanation.append(f"Selected ARIMA (50+ data points, advanced model available)")
            elif len(data) >= 24 and HAS_PROPHET:
                model_type = 'prophet'
                selection_explanation.append(f"Selected Prophet (24+ data points, seasonality detection)")
            else:
                model_type = 'simple'
                selection_explanation.append(f"Selected Simple Trend ({len(data)} data points, basic linear model)")
            
            if not HAS_PROPHET:
                selection_explanation.append("Prophet not available - install with: pip install prophet")
            if not HAS_STATSMODELS:
                selection_explanation.append("Statsmodels not available - install with: pip install statsmodels")
        
        # Fit model based on type
        if model_type == 'arima':
            predictions, status = self.fit_arima_model(data, periods_ahead)
        elif model_type == 'prophet':
            predictions, status = self.fit_prophet_model(data, periods_ahead)
        else:  # simple or fallback
            predictions, status = self.fit_simple_model(data, periods_ahead)
        
        # Add selection explanation to status
        if selection_explanation:
            status = " | ".join(selection_explanation) + " | " + status
        
        return predictions, status

"""
Prediction Analysis Module - Part 3: PredictionEvaluator Class
Evaluate prediction model performance with proper train/test splits
"""

class PredictionEvaluator:
    """Evaluate prediction model performance with proper train/test splits"""
    
    @staticmethod
    def split_data_by_date(data, test_start_date='2024-01-01'):
        """Split data using date cutoff (recommended: 2024-01-01)"""
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        cutoff_date = pd.to_datetime(test_start_date)
        
        train_data = df[df['date'] < cutoff_date].copy()
        test_data = df[df['date'] >= cutoff_date].copy()
        
        return train_data, test_data
    
    @staticmethod
    def calculate_metrics(actual, predicted):
        """Calculate standard prediction metrics"""
        # Remove NaN values for fair comparison
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return None
        
        mae = mean_absolute_error(actual_clean, predicted_clean)
        mse = mean_squared_error(actual_clean, predicted_clean)
        rmse = np.sqrt(mse)
        
        # Avoid division by zero in MAPE
        mape_mask = actual_clean != 0
        if np.sum(mape_mask) > 0:
            mape = np.mean(np.abs((actual_clean[mape_mask] - predicted_clean[mape_mask]) / actual_clean[mape_mask])) * 100
        else:
            mape = np.inf
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'test_points': len(actual_clean)
        }
    
    @staticmethod
    def evaluate_model_with_date_split(data, model_class, target_indicator, test_start_date='2024-01-01'):
        """Evaluate model using date-based train/test split"""
        train_data, test_data = PredictionEvaluator.split_data_by_date(data, test_start_date)
        
        if len(train_data) < 24:
            return None, "Insufficient training data (minimum 24 points required)"
        
        if len(test_data) == 0:
            return None, f"No test data available after {test_start_date}"
        
        # Determine prediction horizon based on test data length
        test_periods = len(test_data)
        
        # Fit model on training data only
        predictions, status = model_class.predict(train_data, periods_ahead=test_periods)
        
        if predictions is None:
            return None, f"Model fitting failed: {status}"
        
        # Align predictions with test data
        test_dates = test_data['date'].values
        pred_dates = pd.to_datetime(predictions['date']).values
        
        # Match predictions to actual test data by date
        test_actual = []
        test_predicted = []
        
        for i, test_date in enumerate(test_dates):
            # Find closest prediction date
            date_diffs = np.abs(pred_dates - test_date)
            closest_idx = np.argmin(date_diffs)
            
            # Only include if dates are reasonably close (within 45 days)
            if date_diffs[closest_idx] <= pd.Timedelta(days=45):
                test_actual.append(test_data['value'].iloc[i])
                test_predicted.append(predictions['predicted_value'].iloc[closest_idx])
        
        if len(test_actual) == 0:
            return None, "No matching dates between predictions and test data"
        
        # Calculate metrics
        metrics = PredictionEvaluator.calculate_metrics(
            np.array(test_actual), np.array(test_predicted)
        )
        
        if metrics is None:
            return None, "Could not calculate metrics"
        
        # Add additional context
        metrics.update({
            'model_type': status,
            'train_periods': len(train_data),
            'test_periods_available': len(test_data),
            'test_periods_used': len(test_actual),
            'test_start_date': test_start_date,
            'train_end_date': train_data['date'].max().strftime('%Y-%m-%d'),
            'test_end_date': test_data['date'].max().strftime('%Y-%m-%d')
        })
        
        return metrics, f"Model evaluation completed using {test_start_date} cutoff"


# Utility functions for Streamlit integration
def validate_prediction_requirements():
    """Check if prediction libraries are available"""
    status = {
        'statsmodels': HAS_STATSMODELS,
        'prophet': HAS_PROPHET,
        'basic_prediction': True  # Always available with simple models
    }
    
    return status

def create_prediction_chart(historical_data, predictions, indicator, country):
    """Create visualization combining historical data with predictions - FIXED timestamp arithmetic"""
    import plotly.graph_objects as go
    
    # Prepare historical data
    hist_df = historical_data.copy()
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=hist_df['date'],
        y=hist_df['value'],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    pred_df = predictions.copy()
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['predicted_value'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence bands if available
    if 'lower_bound' in pred_df.columns and 'upper_bound' in pred_df.columns:
        # Upper bound
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['upper_bound'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name='Upper Bound'
        ))
        
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=pred_df['date'],
            y=pred_df['lower_bound'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.1)',
            name='Confidence Interval',
            showlegend=True
        ))
    
    # FIXED: Add vertical line at forecast start - handle timestamp arithmetic properly
    try:
        # Get the first prediction date and ensure it's a proper datetime
        first_pred_date = pred_df['date'].iloc[0]
        
        # Handle different possible date formats
        if isinstance(first_pred_date, str):
            forecast_start_dt = pd.to_datetime(first_pred_date)
        else:
            forecast_start_dt = pd.Timestamp(first_pred_date)
        
        fig.add_vline(
            x=forecast_start_dt,
            line_dash="dot",
            line_color="gray",
            annotation_text="Forecast Start",
            annotation_position="top"
        )
    except Exception as e:
        # Silently skip the vertical line if there are any datetime issues
        pass

    # Update layout
    country_display = country.replace('_', ' ').title()
    fig.update_layout(
        title=f"{country_display} - {indicator} Forecast",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

"""
Prediction Analysis Module - Part 4: Streamlit Interface
Complete Streamlit interface for economic forecasting
"""

def create_prediction_analysis_page():
    """Create prediction analysis page for Streamlit dashboard"""
    import streamlit as st
    import plotly.graph_objects as go
    from datetime import datetime
    
    st.markdown('<h1 class="main-header">Economic Forecasting</h1>', unsafe_allow_html=True)
    st.markdown("Generate economic forecasts using multiple time series models with comprehensive evaluation.")
    
    # Instructions
    with st.expander("How to Use This Page", expanded=False):
        st.markdown("""
        ## Step-by-Step Guide
        1. **Select Country/Region**: Choose the economy you want to analyze
        2. **Select Indicator**: Pick the economic indicator to forecast (GDP, CPI, etc.)
        3. **Configure Model**: Choose forecasting approach and settings
        4. **Generate Forecast**: Click to create predictions
        5. **Review Results**: Examine forecasts and confidence intervals
        6. **Evaluate Performance**: Test model accuracy on recent data
        
        ## Model Types
        - **Auto**: Automatically selects best available model based on data
        - **Simple Trend**: Linear trend with seasonal patterns (always available)
        - **Prophet**: Advanced forecasting with seasonality (requires Prophet library)
        - **ARIMA**: Statistical time series model (requires statsmodels library)
        """)
    
    # Initialize prediction analyzer
    if 'prediction_analyzer' not in st.session_state:
        st.session_state.prediction_analyzer = PredictionAnalysis()
    
    analyzer = st.session_state.prediction_analyzer
    
    # Load available countries (cached)
    if 'all_countries_data' not in st.session_state:
        with st.spinner("Loading available countries..."):
            st.session_state.all_countries_data = analyzer.load_all_countries_data()

    all_countries_data = st.session_state.all_countries_data
    
    if not all_countries_data:
        st.error("No country data available. Please ensure data files are in the extracted_data folder.")
        return
    
    available_countries = list(all_countries_data.keys())
    
    # Main interface
    st.subheader("Forecasting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Country selection
        selected_country = st.selectbox(
            "Select Country/Region",
            available_countries,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        # Get available indicators for selected country
        country_data = all_countries_data[selected_country]
        available_indicators = list(country_data.keys())
        
        # Check for quick prediction from overview page
        if 'quick_predict_indicator' in st.session_state and st.session_state.quick_predict_indicator:
            if st.session_state.quick_predict_indicator in available_indicators:
                default_indicator_idx = available_indicators.index(st.session_state.quick_predict_indicator)
            else:
                default_indicator_idx = 0
            # Clear the session state after using it
            st.session_state.quick_predict_indicator = None
        else:
            default_indicator_idx = 0
        
        selected_indicator = st.selectbox(
            "Select Indicator to Predict",
            available_indicators,
            index=default_indicator_idx
        )
    
    with col2:
        # Prediction settings
        forecast_periods = st.slider("Forecast Periods", 3, 12, 6)
        
        model_type = st.selectbox(
            "Model Type",
            ['auto', 'simple', 'prophet', 'arima'],
            help="Auto selects best available model based on data"
        )
    
    # Show current data summary
    current_data = country_data[selected_indicator]
    latest_value = current_data['value'].iloc[-1]
    latest_date = current_data['date'].iloc[-1]
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            f"Latest {selected_indicator}",
            f"{latest_value:,.2f}",
            delta=f"as of {latest_date.strftime('%Y-%m-%d')}"
        )
    with col4:
        st.metric("Historical Data Points", len(current_data))
    
    # Generate forecast button
    if st.button("Generate Forecast", type="primary"):
        with st.spinner(f"Generating {forecast_periods}-period forecast..."):
            # Get appropriate predictor
            predictor = MacroPredictor(selected_indicator)
            predictions, status = predictor.predict(current_data, periods_ahead=forecast_periods, model_type=model_type)

        # Display results
        if predictions is not None and not predictions.empty:
            st.success(f"Status: {status}")
            
            # Create prediction visualization
            fig = create_prediction_chart(
                current_data, 
                predictions, 
                selected_indicator, 
                selected_country
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show prediction table
            st.subheader("Forecast Values")
            display_predictions = predictions.copy()
            display_predictions['date'] = display_predictions['date'].dt.strftime('%Y-%m-%d')
            display_predictions['predicted_value'] = display_predictions['predicted_value'].round(2)
            
            if 'lower_bound' in display_predictions.columns:
                display_predictions['lower_bound'] = display_predictions['lower_bound'].round(2)
                display_predictions['upper_bound'] = display_predictions['upper_bound'].round(2)
            
            st.dataframe(display_predictions, use_container_width=True, hide_index=True)
            
            # # Model performance evaluation
            # st.subheader("Model Performance Evaluation")
            
            # eval_col1, eval_col2 = st.columns([1, 2])
            
            # with eval_col1:
            #     if st.button("Evaluate on 2024+ Data", key="eval_button"):
            #         try:
            #             with st.spinner("Running evaluation on 2024+ data..."):
            #                 # Check if we have sufficient 2024+ data first
            #                 test_data_check = current_data[pd.to_datetime(current_data['date']) >= '2024-01-01']
                            
            #                 if len(test_data_check) == 0:
            #                     st.error("No data available from 2024 onwards for evaluation")
            #                     st.info("This evaluation requires actual data from 2024+ to test prediction accuracy against recent outcomes")
            #                 elif len(test_data_check) < 3:
            #                     st.warning(f"Limited test data: Only {len(test_data_check)} points from 2024+")
            #                     st.info("More recent data points would provide more reliable evaluation results")
            #                 else:
            #                     # Proceed with evaluation
            #                     if use_multi_indicator:
            #                         # For multi-indicator, evaluate on single indicator for consistency
            #                         eval_predictor = MacroPredictor(selected_indicator)
            #                         metrics, eval_status = PredictionEvaluator.evaluate_model_with_date_split(
            #                             current_data, eval_predictor, selected_indicator, test_start_date='2024-01-01'
            #                         )
            #                     else:
            #                         metrics, eval_status = PredictionEvaluator.evaluate_model_with_date_split(
            #                             current_data, predictor, selected_indicator, test_start_date='2024-01-01'
            #                         )
                                
            #                     # Enhanced results display
            #                     if metrics and isinstance(metrics, dict) and 'MAE' in metrics:
            #                         st.success(f"Status: {eval_status}")
                                    
            #                         # Display metrics with enhanced formatting
            #                         metric_cols = st.columns(4)
            #                         with metric_cols[0]:
            #                             st.metric("MAE", f"{metrics['MAE']:.2f}")
            #                         with metric_cols[1]:
            #                             st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            #                         with metric_cols[2]:
            #                             mape_display = f"{metrics['MAPE']:.1f}%" if metrics['MAPE'] < 1000 else "High"
            #                             st.metric("MAPE", mape_display)
            #                         with metric_cols[3]:
            #                             st.metric("Test Points", metrics['test_points'])
                                    
            #                         # Evaluation context
            #                         st.caption(f"Evaluation Context:")
            #                         st.caption(f"• Trained on: {metrics.get('train_periods', 'N/A')} points before {metrics.get('test_start_date', '2024-01-01')}")
            #                         st.caption(f"• Tested on: {metrics['test_periods_used']} points from {metrics.get('test_start_date', '2024-01-01')} to {metrics.get('test_end_date', 'latest')}")
            #                         st.caption(f"• Performance: {'Good' if metrics['MAPE'] < 10 else 'Moderate' if metrics['MAPE'] < 25 else 'Poor'} (MAPE: {metrics['MAPE']:.1f}%)")
                                    
            #                     else:
            #                         error_msg = str(metrics) if metrics else eval_status
            #                         st.error(f"Evaluation failed: {error_msg}")
                                    
            #                         # Provide helpful guidance
            #                         if "2024" in error_msg or "test data" in error_msg.lower():
            #                             st.info("This indicator may not have sufficient 2024+ data for evaluation. Try with a different indicator or check data coverage.")
            #                         else:
            #                             st.info("Evaluation requires historical data patterns. Some indicators may be too volatile or have insufficient data for reliable evaluation.")
                        
            #         except Exception as e:
            #             st.error(f"Evaluation error: {str(e)}")
            #             st.info("Try refreshing the page or selecting a different indicator/country combination.")
            
            # with eval_col2:
            #     # Enhanced Model information
            #     with st.expander("Model Information & Guidance", expanded=True):
            #         st.write(f"**Model Used:** {predictions['model'].iloc[0]}")
            #         st.write(f"**Forecast Horizon:** {forecast_periods} periods")
            #         st.write(f"**Training Data Points:** {len(current_data)}")
                    
            #         # Model-specific explanations
            #         model_name = predictions['model'].iloc[0]
            #         if 'Prophet' in model_name:
            #             st.info("**Prophet Model**: Advanced time series model that automatically detects seasonal patterns, handles missing data, and provides confidence intervals. Best for data with clear seasonality.")
                        
            #         elif 'Simple' in model_name:
            #             st.info("**Simple Trend Model**: Linear trend continuation with basic seasonal patterns. Uses recent historical trend to project future values. Reliable for stable, trending data.")
                        
            #         elif 'ARIMA' in model_name:
            #             st.info("**ARIMA Model**: Statistical time series model that uses autoregression, differencing, and moving averages. Good for stationary time series with complex patterns.")
                        
            #         elif 'Multi' in model_name:
            #             st.info("**Multi-Indicator Model**: Uses related economic variables as additional features to improve forecast accuracy. More sophisticated but requires aligned data across indicators.")
                    
            #         # Add multi-indicator feature info
            #         if use_multi_indicator and hasattr(predictor, 'feature_indicators'):
            #             related_indicators = predictor.feature_indicators
            #             if related_indicators:
            #                 available_features = [ind for ind in related_indicators if ind in country_data and not country_data[ind].empty]
            #                 st.write(f"**Features Used:** {', '.join(available_features) if available_features else 'None (fell back to single indicator)'}")
                    
            #         # Data info with enhanced details
            #         data_start = current_data['date'].min().strftime('%Y-%m-%d')
            #         data_end = current_data['date'].max().strftime('%Y-%m-%d')
            #         data_span_years = (current_data['date'].max() - current_data['date'].min()).days / 365.25
                    
            #         st.write(f"**Data Coverage:** {data_start} to {data_end} ({data_span_years:.1f} years)")
                    
            #         # Usage recommendations
            #         st.markdown("**Usage Recommendations:**")
            #         if forecast_periods <= 6:
            #             st.text("Short-term forecasts: Generally reliable")
            #         elif forecast_periods <= 12:  
            #             st.text("Medium-term forecasts: Use with caution")
            #         else:
            #             st.text("Long-term forecasts: High uncertainty")

        else:
            st.error(f"Prediction failed: {status}")
    
    # Add information section
    st.markdown("---")
    st.subheader("Understanding Economic Forecasts")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Model Reliability by Indicator:**
        - **GDP**: Generally predictable, quarterly patterns
        - **CPI/Inflation**: Moderately predictable, policy dependent
        - **Interest Rates**: Hard to predict, policy driven
        - **Population**: Highly predictable, slow changes
        - **Property Prices**: Volatile, market dependent
        """)
    
    with info_col2:
        st.markdown("""
        **Forecast Limitations:**
        - Models assume historical patterns continue
        - Cannot predict policy changes or shocks
        - Confidence intervals show uncertainty ranges
        - Shorter forecasts generally more reliable
        - Use for planning, not guaranteed outcomes
        """)


if __name__ == "__main__":
    # Test prediction capabilities
    print("Economic Prediction Analysis")
    print("=" * 40)
    
    status = validate_prediction_requirements()
    print("Library Status:")
    for lib, available in status.items():
        print(f"  {lib}: {'Available' if available else 'Not Available'}")
    
    # Test data loading
    analyzer = PredictionAnalysis()
    countries_data = analyzer.load_all_countries_data()
    print(f"\nCountries loaded: {list(countries_data.keys())}")
    
    if countries_data:
        sample_country = list(countries_data.keys())[0]
        sample_data = countries_data[sample_country]
        print(f"Sample indicators for {sample_country}: {list(sample_data.keys())}")
        
        # Test prediction
        if 'GDP' in sample_data:
            predictor = MacroPredictor('GDP')
            predictions, status = predictor.predict(sample_data['GDP'])
            if predictions is not None:
                print(f"Prediction test: {status}")
                print(f"Generated {len(predictions)} predictions")
            else:
                print(f"Prediction failed: {status}")
        
        # # Test multi-indicator prediction
        # if 'Property_Price' in sample_data:
        #     multi_predictor = MultiIndicatorPredictor('Property_Price')
        #     multi_predictions, multi_status = multi_predictor.predict_with_features(sample_data)
        #     if multi_predictions is not None:
        #         print(f"Multi-indicator prediction test: {multi_status}")
        #         print(f"Generated {len(multi_predictions)} multi-indicator predictions")
        #     else:
        #         print(f"Multi-indicator prediction failed: {multi_status}")
    
    print("\nSupported Model Types:")
    print(f"  - Simple Trend: Always available")
    print(f"  - Prophet: {'Available' if HAS_PROPHET else 'Not Available'}")
    print(f"  - ARIMA: {'Available' if HAS_STATSMODELS else 'Not Available'}")
    # print(f"  - Multi-Indicator: Depends on data alignment")