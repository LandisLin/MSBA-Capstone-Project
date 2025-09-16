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
        """Enhanced Prophet model with adaptive parameters and trend momentum"""
        if not HAS_PROPHET:
            return self.fit_simple_model(data, periods_ahead)
        
        df = self.prepare_data(data)
        
        if len(df) < 24:
            return self.fit_simple_model(data, periods_ahead)
        
        # IMPROVED: Better data window selection for Property_Price
        if self.indicator_type == 'Property_Price' and len(df) > 40:
            # Check for structural breaks around 2024 first
            recent_trend = np.polyfit(range(6), df['value'].tail(6), 1)[0]
            
            # Look for 2024+ data to detect structural breaks
            break_point_2024 = df[pd.to_datetime(df['date']) >= '2024-01-01']
            
            if len(break_point_2024) > 6:
                # Calculate pre-2024 and post-2024 trends
                pre_2024 = df[pd.to_datetime(df['date']) < '2024-01-01']
                post_2024 = break_point_2024
                
                if len(pre_2024) >= 12 and len(post_2024) >= 6:
                    pre_trend = np.polyfit(range(len(pre_2024.tail(12))), pre_2024['value'].tail(12), 1)[0]
                    post_trend = np.polyfit(range(len(post_2024)), post_2024['value'], 1)[0]
                    
                    trend_change_ratio = abs(post_trend - pre_trend) / (abs(pre_trend) + 0.001)
                    
                    if trend_change_ratio > 0.8:
                        # Use data that includes the trend change
                        optimal_window = len(post_2024) + 18  # Include some pre-break context
                        df = df.tail(optimal_window)
                        print(f"Property Price: Structural break detected, using {optimal_window} points")
                    else:
                        # Normal adaptive window
                        older_trend = np.polyfit(range(20), df['value'].iloc[-40:-20], 1)[0]
                        trend_consistency = abs(recent_trend - older_trend) / abs(recent_trend + 0.001)
                        
                        if trend_consistency > 0.5:
                            df = df.tail(36)  # Slightly more data than original 30
                            print(f"Property Price: Trend change detected, using recent 36 points")
                        else:
                            df = df.tail(48)  # Balanced approach
                            print(f"Property Price: Stable trend, using 48 data points")
            else:
                # Original logic for cases without 2024+ data
                older_trend = np.polyfit(range(20), df['value'].iloc[-40:-20], 1)[0]
                trend_consistency = abs(recent_trend - older_trend) / abs(recent_trend + 0.001)
                
                if trend_consistency > 0.5:
                    df = df.tail(30)
                    print(f"Property Price: Trend change detected, using recent 30 points")
                else:
                    df = df.tail(50)
                    print(f"Property Price: Stable trend, using 50 data points")
        
        # Calculate data characteristics for adaptive parameters
        volatility = df['value'].pct_change().std()
        trend_strength = abs(np.polyfit(range(len(df)), df['value'], 1)[0])
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': pd.to_datetime(df['date']),
            'y': df['value']
        })
        
        try:
            # IMPROVED: More conservative base parameters
            base_changepoint_scale = 0.05
            base_seasonality_scale = 8.0  # Reduced from 10.0 to prioritize trends
            
            # Adjust based on volatility (keeping your logic but refined)
            if volatility > 0.05:  # High volatility data
                changepoint_scale = min(0.15, base_changepoint_scale + volatility * 1.5)  # Less aggressive
                seasonality_scale = max(4.0, base_seasonality_scale - volatility * 40)    # Less impact
            else:  # Low volatility data
                changepoint_scale = base_changepoint_scale
                seasonality_scale = base_seasonality_scale
            
            # IMPROVED: More conservative changepoint count
            n_changepoints = min(20, max(5, len(df) // 10))  # Reduced from // 8
            
            model_params = {
                'weekly_seasonality': False,
                'daily_seasonality': False,
                'yearly_seasonality': False,
                'changepoint_prior_scale': changepoint_scale,
                'seasonality_prior_scale': seasonality_scale,
                'n_changepoints': n_changepoints,
                'changepoint_range': 0.92,  # CRITICAL FIX: Increased from 0.85
                'interval_width': 0.8,
            }
            
            # IMPROVED: Property-specific adjustments with better parameters
            if self.indicator_type == 'Property_Price':
                # Calculate recent trend for better cap setting
                recent_trend_6m = np.polyfit(range(6), df['value'].tail(6), 1)[0]
                last_value = df['value'].iloc[-1]
                
                # CRITICAL FIX: Much more generous cap calculation
                recent_max = df['value'].tail(12).max()
                
                if recent_trend_6m > 0.3:
                    # For upward trends: allow significant growth above current levels
                    projected_growth = last_value + (recent_trend_6m * 12)  # 12-month projection
                    cap = max(projected_growth * 1.15, last_value * 1.30)   # At least 30% above current
                elif recent_trend_6m > 0:
                    # Mild upward trend
                    cap = last_value * 1.25  # 25% above for mild trends
                else:
                    # Flat or declining trends - still allow upward movement
                    cap = last_value * 1.20  # 20% above
                
                floor = df['value'].min() * 0.85  # Slightly higher floor
                prophet_data['cap'] = cap
                prophet_data['floor'] = floor
                model_params['growth'] = 'logistic'
                
                # IMPROVED: Better changepoint parameters for property
                model_params['changepoint_prior_scale'] = min(0.08, changepoint_scale * 1.2)  # More flexible
                model_params['changepoint_range'] = 0.95  # CRITICAL: Much higher for property
                
                print(f"Property bounds: cap={cap:.2f} (allows {((cap/last_value-1)*100):.1f}% growth), floor={floor:.2f}")
                print(f"Recent trend: {recent_trend_6m:.4f}, changepoint_range: {model_params['changepoint_range']}")
            
            model = Prophet(**model_params)
            
            # IMPROVED: Reduced seasonality influence to prioritize trends
            frequency = self.detect_frequency(df)
            
            if frequency == 'quarterly':
                # Annual seasonality for quarterly data with reduced influence
                model.add_seasonality(name='annual', period=365.25, fourier_order=2, prior_scale=2.0)  # Reduced
                
                # Multi-year cycle detection for property (reduced influence)
                if self.indicator_type == 'Property_Price' and len(df) > 20:
                    model.add_seasonality(name='medium_cycle', period=365.25 * 3, fourier_order=2, prior_scale=1.0)
            
            elif frequency == 'monthly':
                # Reduced seasonality influence for monthly data
                model.add_seasonality(name='quarterly', period=365.25/4, fourier_order=2, prior_scale=1.5)
                model.add_seasonality(name='annual', period=365.25, fourier_order=3, prior_scale=2.5)  # Reduced
            
            # Fit model
            model.fit(prophet_data)
            
            # Create future dates with proper frequency
            last_training_date = prophet_data['ds'].max()
            
            if frequency == 'quarterly':
                future_dates = [last_training_date + pd.DateOffset(months=(i+1)*3) for i in range(periods_ahead)]
            else:
                future_dates = [last_training_date + pd.DateOffset(months=i+1) for i in range(periods_ahead)]
            
            # Prepare future dataframe
            future = pd.DataFrame({'ds': prophet_data['ds'].tolist() + future_dates})
            
            # Add cap/floor for future periods if using logistic growth
            if model_params.get('growth') == 'logistic':
                future['cap'] = cap
                future['floor'] = floor
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Return only future predictions
            future_predictions = forecast[forecast['ds'] > prophet_data['ds'].max()].copy()
            
            if len(future_predictions) > 0:
                first_prediction = future_predictions['yhat'].iloc[0]
                last_actual = df['value'].iloc[-1]
                prediction_jump = first_prediction - last_actual
                
                # Calculate recent trend characteristics
                recent_trend_3m = np.polyfit(range(3), df['value'].tail(3), 1)[0] if len(df) >= 3 else 0
                recent_trend_6m = np.polyfit(range(6), df['value'].tail(6), 1)[0] if len(df) >= 6 else 0
                
                # Define reasonable jump threshold (adaptive based on current value and volatility)
                base_threshold = max(2.5, abs(last_actual) * 0.015)  # 1.5% or 2.5 units
                volatility_factor = df['value'].pct_change().tail(12).std() * 10  # Scale volatility
                max_reasonable_jump = base_threshold + volatility_factor
                
                print(f"Jump Analysis: {last_actual:.2f} → {first_prediction:.2f} (change: {prediction_jump:+.2f})")
                print(f"Recent trends - 3M: {recent_trend_3m:.3f}, 6M: {recent_trend_6m:.3f}")
                print(f"Max reasonable jump: ±{max_reasonable_jump:.2f}")
                
                # Apply correction if jump exceeds reasonable bounds
                if abs(prediction_jump) > max_reasonable_jump:
                    print(f"APPLYING JUMP CORRECTION: {prediction_jump:+.2f} exceeds ±{max_reasonable_jump:.2f}")
                    
                    # Calculate conservative target based on recent trends
                    if prediction_jump > 0:  # Excessive upward jump
                        # Use more conservative of recent trends, capped at max_reasonable_jump
                        conservative_change = min(
                            max(recent_trend_3m * 0.6, recent_trend_6m * 0.8),  # Conservative trend
                            max_reasonable_jump  # Hard cap
                        )
                    else:  # Excessive downward jump
                        # Similar logic for downward moves
                        conservative_change = max(
                            min(recent_trend_3m * 0.6, recent_trend_6m * 0.8),  # Conservative trend
                            -max_reasonable_jump  # Hard cap
                        )
                    
                    smoothed_first = last_actual + conservative_change
                    print(f"Smoothed first prediction: {first_prediction:.2f} → {smoothed_first:.2f}")
                    
                    # Apply graduated smoothing to first 3 predictions
                    num_predictions_to_smooth = min(3, len(future_predictions))
                    blend_weight = 0.0  # INITIALIZE HERE to prevent error
                    
                    for i in range(num_predictions_to_smooth):
                        try:
                            original = future_predictions['yhat'].iloc[i]
                            
                            if i == 0:
                                corrected = smoothed_first
                            else:
                                blend_weight = 0.6 * (0.7 ** i)  # This was causing the error
                                trend_continuation = smoothed_first + (conservative_change * 0.4 * i)
                                corrected = blend_weight * trend_continuation + (1 - blend_weight) * original
                            
                            # Update prediction using .loc to avoid warnings
                            future_predictions.loc[future_predictions.index[i], 'yhat'] = corrected
                            
                            # Adjust bounds if they exist
                            if 'yhat_lower' in future_predictions.columns and original != 0:
                                adjustment_ratio = corrected / original
                                future_predictions.loc[future_predictions.index[i], 'yhat_lower'] *= adjustment_ratio
                                future_predictions.loc[future_predictions.index[i], 'yhat_upper'] *= adjustment_ratio
                            
                            print(f"  Period {i+1}: {original:.2f} → {corrected:.2f}")
                            
                        except Exception as e:
                            print(f"Error correcting prediction {i}: {e}")
                            break
                
                else:
                    print(f"No correction needed - jump {prediction_jump:+.2f} within reasonable bounds ±{max_reasonable_jump:.2f}")
            
            future_predictions = future_predictions.rename(columns={
                'ds': 'date', 
                'yhat': 'predicted_value',
                'yhat_lower': 'lower_bound',
                'yhat_upper': 'upper_bound'
            })
            
            # Enhanced model name with key parameters
            model_name = f'Enhanced_Prophet(cp={changepoint_scale:.2f},range={model_params["changepoint_range"]:.2f})'
            if model_params.get('growth') == 'logistic':
                model_name += '_logistic'
            
            future_predictions['model'] = model_name
            
            self.model = model
            
            # Enhanced validation with better messaging
            current_value = df['value'].iloc[-1]
            first_prediction = future_predictions['predicted_value'].iloc[0]
            
            # Dynamic threshold based on volatility
            threshold = 0.95 - volatility
            
            if first_prediction < current_value * threshold:
                print(f"NOTE: Prediction shows decline from {current_value:.2f} to {first_prediction:.2f}")
                print(f"Model parameters: cp_scale={changepoint_scale:.3f}, cp_range={model_params['changepoint_range']:.2f}")
            
            # Forecast reasonableness check
            max_growth = self.forecast_validation(future_predictions, volatility)
            
            status_parts = [
                "Enhanced Prophet model fitted",
                f"Changepoint range: {model_params['changepoint_range']:.2f}",
                f"Parameters: cp_scale={changepoint_scale:.3f}",
                f"Growth: {model_params.get('growth', 'linear')}"
            ]
            
            return future_predictions[['date', 'predicted_value', 'lower_bound', 'upper_bound', 'model']], " | ".join(status_parts)
            
        except Exception as e:
            print(f"Enhanced Prophet model failed for {self.indicator_type}: {e}")
            return self.fit_simple_model(data, periods_ahead)
    
    def forecast_validation(self, predictions, volatility):
        """Validate forecast for unrealistic growth rates"""
        max_reasonable_growth = 0.1 + volatility  # Adaptive based on data volatility
        
        for i in range(1, len(predictions)):
            growth = (predictions['predicted_value'].iloc[i] / 
                    predictions['predicted_value'].iloc[i-1]) - 1
            if abs(growth) > max_reasonable_growth:
                print(f"WARNING: High growth rate detected: {growth:.1%} (period {i})")
        
        return max_reasonable_growth
    
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
    def evaluate_model_with_date_split(data, model_class, target_indicator, test_start_date='2024-01-01', model_type='auto'):
        """Evaluate model using date-based train/test split"""
        train_data, test_data = PredictionEvaluator.split_data_by_date(data, test_start_date)
        
        if len(train_data) < 24:
            return None, "Insufficient training data (minimum 24 points required)"
        
        if len(test_data) == 0:
            return None, f"No test data available after {test_start_date}"
        
        # Determine prediction horizon based on test data length
        test_periods = len(test_data)
        
        # Fit model on training data only
        predictions, status = model_class.predict(train_data, periods_ahead=test_periods, model_type=model_type)
        
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
    
    @staticmethod
    def evaluate_walk_forward(data, predictor_class, indicator_type, periods_ahead=1, model_type='auto'):
        """
        Evaluate using walk-forward validation - more realistic than single split
        This mimics how the model would actually be used in practice
        """
        test_data_check = data[pd.to_datetime(data['date']) >= '2024-01-01'].copy()
        
        if len(test_data_check) < 6:  # Need sufficient test data
            return None, "Insufficient 2024+ data for walk-forward evaluation"
        
        errors = []
        predictions_made = 0
        
        # Start from 2024-01-01 and walk forward
        for i in range(len(test_data_check) - periods_ahead):
            try:
                # Get training data up to current point (simulate real-world usage)
                current_date = test_data_check.iloc[i]['date']
                train_data = data[pd.to_datetime(data['date']) < current_date].copy()
                
                if len(train_data) < 24:  # Need minimum training data
                    continue
                
                # Create fresh predictor and make prediction
                fresh_predictor = predictor_class(indicator_type)
                pred_result, _ = fresh_predictor.predict(train_data, periods_ahead=periods_ahead, model_type=model_type)
                
                if pred_result is not None and len(pred_result) > 0:
                    # Get actual value at target date
                    target_idx = i + periods_ahead
                    if target_idx < len(test_data_check):
                        actual_value = test_data_check.iloc[target_idx]['value']
                        predicted_value = pred_result['predicted_value'].iloc[0]
                        
                        # Calculate error
                        if actual_value != 0:
                            mape_error = abs(actual_value - predicted_value) / abs(actual_value)
                            errors.append(mape_error)
                            predictions_made += 1
            
            except Exception as e:
                continue
        
        if len(errors) > 0:
            walk_forward_mape = np.mean(errors) * 100
            return {
                'walk_forward_mape': walk_forward_mape,
                'predictions_made': predictions_made,
                'test_period': len(test_data_check),
                'method': 'walk_forward'
            }, f"Walk-forward evaluation: {walk_forward_mape:.1f}% MAPE ({predictions_made} predictions)"
        else:
            return None, "Walk-forward evaluation failed - no valid predictions"


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
            fillcolor='rgba(255, 255, 0, 0.3)',
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

def create_enhanced_prediction_chart(historical_data, predictions, test_predictions, indicator, country, evaluation_results=None):
    """Create comprehensive visualization with historical data, test predictions, and future forecasts"""
    import plotly.graph_objects as go
    
    # Prepare historical data
    hist_df = historical_data.copy()
    hist_df['date'] = pd.to_datetime(hist_df['date'])
    hist_df = hist_df.sort_values('date')
    
    # Create figure
    fig = go.Figure()
    
    # Split historical data into train/test for visualization
    train_cutoff = pd.to_datetime('2024-01-01')
    train_hist = hist_df[hist_df['date'] < train_cutoff]
    test_hist = hist_df[hist_df['date'] >= train_cutoff]
    
    # Historical training data
    fig.add_trace(go.Scatter(
        x=train_hist['date'],
        y=train_hist['value'],
        mode='lines',
        name='Historical Data (Training)',
        line=dict(color='blue', width=2)
    ))
    
    # Historical test data (actual values from 2024+)
    if len(test_hist) > 0:
        fig.add_trace(go.Scatter(
            x=test_hist['date'],
            y=test_hist['value'],
            mode='lines+markers',
            name='Actual Data (2024+)',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
    
    # Test predictions (model predictions for 2024+ period)
    if test_predictions is not None and not test_predictions.empty:
        test_pred_df = test_predictions.copy()
        test_pred_df['date'] = pd.to_datetime(test_pred_df['date'])
        
        fig.add_trace(go.Scatter(
            x=test_pred_df['date'],
            y=test_pred_df['predicted_value'],
            mode='lines+markers',
            name='Model Test Predictions (2024+)',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=5, symbol='diamond')
        ))
    
    # Future predictions
    pred_df = predictions.copy()
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['predicted_value'],
        mode='lines+markers',
        name='Future Forecast',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add confidence bands for future predictions if available
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
            fillcolor='rgba(255, 255, 0, 0.2)',
            name='Confidence Interval',
            showlegend=True
        ))
    
    # Add vertical lines
    try:
        # Training/test split line
        fig.add_vline(
            x=train_cutoff,
            line_dash="solid",
            line_color="gray",
            annotation_text="Training/Test Split (2024-01-01)",
            annotation_position="top"
        )
        
        # Forecast start line
        if len(hist_df) > 0:
            forecast_start_dt = hist_df['date'].max()
            fig.add_vline(
                x=forecast_start_dt,
                line_dash="dot",
                line_color="purple",
                annotation_text="Forecast Start",
                annotation_position="bottom"
            )
    except Exception:
        pass

    # Enhanced title with evaluation info
    title_text = f"{country.replace('_', ' ').title()} - {indicator} Forecast"
    if evaluation_results:
        mape = evaluation_results.get('MAPE', 0)
        title_text += f" (Test MAPE: {mape:.1f}%)"
    
    # Update layout
    fig.update_layout(
        title=title_text,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
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
        with st.spinner(f"Generating {forecast_periods}-period forecast and evaluation..."):
            # Get appropriate predictor
            predictor = MacroPredictor(selected_indicator)
            predictions, status = predictor.predict(current_data, periods_ahead=forecast_periods, model_type=model_type)

        # Display results
        if predictions is not None and not predictions.empty:
            st.success(f"Status: {status}")
            
            # AUTOMATIC EVALUATION - Run evaluation during prediction
            evaluation_results = None
            test_predictions = None
            walk_forward_results = None

            # Check if we have 2024+ data for evaluation
            test_data_check = current_data[pd.to_datetime(current_data['date']) >= '2024-01-01']

            if len(test_data_check) >= 3:
                try:
                    with st.spinner("Running comprehensive model evaluation..."):
                        # 1. Standard evaluation (original method)
                        train_data, test_data = PredictionEvaluator.split_data_by_date(current_data, '2024-01-01')
                        eval_predictor = MacroPredictor(selected_indicator)
                        test_periods = len(test_data)
                        test_predictions, eval_status = eval_predictor.predict(train_data, periods_ahead=test_periods, model_type=model_type)
                        
                        if test_predictions is not None:
                            metrics, eval_message = PredictionEvaluator.evaluate_model_with_date_split(
                                current_data, eval_predictor, selected_indicator, test_start_date='2024-01-01', model_type=model_type
                            )
                            
                            if metrics and isinstance(metrics, dict) and 'MAE' in metrics:
                                evaluation_results = metrics
                        
                        # 2. Walk-forward evaluation (more realistic)
                        if model_type == 'arima':
                            walk_forward_results = None
                            st.info("Walk-forward evaluation skipped for ARIMA (computationally expensive)")
                        else:
                            walk_forward_results, wf_message = PredictionEvaluator.evaluate_walk_forward(
                                current_data, MacroPredictor, selected_indicator, periods_ahead=1, model_type=model_type
                            )
                        
                        # Display evaluation info
                        if evaluation_results and walk_forward_results:
                            standard_mape = evaluation_results['MAPE']
                            wf_mape = walk_forward_results['walk_forward_mape']
                            st.info(f"Standard Evaluation: {standard_mape:.1f}% MAPE | Walk-Forward: {wf_mape:.1f}% MAPE")
                            
                            # Flag significant differences
                            if abs(standard_mape - wf_mape) > 5:
                                st.warning(f"Large difference between evaluation methods suggests model instability")
                        
                        elif evaluation_results:
                            st.info(f"Standard evaluation: MAPE {evaluation_results['MAPE']:.1f}%")
                        elif walk_forward_results:
                            st.info(f"Walk-forward evaluation: MAPE {walk_forward_results['walk_forward_mape']:.1f}%")

                except Exception as e:
                    st.warning(f"Evaluation failed: {str(e)}")

            else:
                st.info("Insufficient 2024+ data for evaluation (need 3+ points)")
            
            # CREATE ENHANCED VISUALIZATION with evaluation
            fig = create_enhanced_prediction_chart(
                current_data, 
                predictions, 
                test_predictions,
                selected_indicator, 
                selected_country,
                evaluation_results
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
            
            # EVALUATION RESULTS DISPLAY
            if evaluation_results or walk_forward_results:
                st.subheader("Model Performance Evaluation")
                
                eval_col1, eval_col2 = st.columns(2)
                
                with eval_col1:
                    st.markdown("**Standard Evaluation (2024+ Test Set)**")
                    if evaluation_results:
                        metric_cols = st.columns(2)
                        with metric_cols[0]:
                            st.metric("MAE", f"{evaluation_results['MAE']:.2f}")
                            st.metric("RMSE", f"{evaluation_results['RMSE']:.2f}")
                        with metric_cols[1]:
                            mape_display = f"{evaluation_results['MAPE']:.1f}%" if evaluation_results['MAPE'] < 1000 else "High"
                            st.metric("MAPE", mape_display)
                            st.metric("Test Points", evaluation_results['test_points'])
                    else:
                        st.info("Standard evaluation not available")
                
                with eval_col2:
                    st.markdown("**Walk-Forward Evaluation (Realistic)**")
                    if walk_forward_results:
                        st.metric("Walk-Forward MAPE", f"{walk_forward_results['walk_forward_mape']:.1f}%")
                        st.metric("Predictions Made", walk_forward_results['predictions_made'])
                        
                        # Performance assessment using walk-forward results
                        wf_mape = walk_forward_results['walk_forward_mape']
                        if wf_mape < 10:
                            performance = "Good"
                        elif wf_mape < 25:
                            performance = "Moderate"
                        else:
                            performance = "Poor"
                        
                        st.caption(f"**Realistic Performance: {performance}**")
                        st.caption("Walk-forward simulates real-world usage")
                    else:
                        st.info("Walk-forward evaluation not available")
                
                # Show comparison if both available
                if evaluation_results and walk_forward_results:
                    st.markdown("---")
                    standard_mape = evaluation_results['MAPE']
                    wf_mape = walk_forward_results['walk_forward_mape']
                    
                    if wf_mape > standard_mape * 1.5:
                        st.warning(f"Walk-forward MAPE ({wf_mape:.1f}%) significantly higher than standard evaluation ({standard_mape:.1f}%) - model may be overfitting")
                    elif abs(wf_mape - standard_mape) < 2:
                        st.success("Both evaluation methods show consistent results - model appears stable")
                    else:
                        st.info("Moderate difference between evaluation methods - consider walk-forward as more realistic")
            
            # Model information
            with st.expander("Model Information & Guidance", expanded=False):
                st.write(f"**Model Used:** {predictions['model'].iloc[0]}")
                st.write(f"**Forecast Horizon:** {forecast_periods} periods")
                st.write(f"**Training Data Points:** {len(current_data)}")
                
                # Model-specific explanations
                model_name = predictions['model'].iloc[0]
                if 'Prophet' in model_name:
                    st.info("**Prophet Model**: Advanced time series model that automatically detects seasonal patterns, handles missing data, and provides confidence intervals. Best for data with clear seasonality.")
                    
                elif 'Simple' in model_name:
                    st.info("**Simple Trend Model**: Linear trend continuation with basic seasonal patterns. Uses recent historical trend to project future values. Reliable for stable, trending data.")
                    
                elif 'ARIMA' in model_name:
                    st.info("**ARIMA Model**: Statistical time series model that uses autoregression, differencing, and moving averages. Good for stationary time series with complex patterns.")
                
                # Data info
                data_start = current_data['date'].min().strftime('%Y-%m-%d')
                data_end = current_data['date'].max().strftime('%Y-%m-%d')
                data_span_years = (current_data['date'].max() - current_data['date'].min()).days / 365.25
                
                st.write(f"**Data Coverage:** {data_start} to {data_end} ({data_span_years:.1f} years)")
                
                # Usage recommendations
                st.markdown("**Usage Recommendations:**")
                if forecast_periods <= 6:
                    st.text("Short-term forecasts | Generally reliable")
                elif forecast_periods <= 12:  
                    st.text("Medium-term forecasts | Use with caution")
                else:
                    st.text("Long-term forecasts | High uncertainty")

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
