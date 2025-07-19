"""
SISTEMA AVANZADO DE ANÁLISIS INFERENCIAL DE DATOS
==================================================
Autor: Analista de Datos Senior
Versión: 3.0
Descripción: Framework completo para análisis estadístico inferencial, 
            machine learning avanzado y visualización de datos profesional.
"""

# ============================================================================
# IMPORTACIÓN DE LIBRERÍAS ESPECIALIZADAS
# ============================================================================

# Análisis de datos fundamentales
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    normaltest, shapiro, kstest, anderson, jarque_bera,
    pearsonr, spearmanr, kendalltau, chi2_contingency,
    mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    ttest_ind, ttest_rel, f_oneway, bartlett, levene
)

# Machine Learning y análisis predictivo
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, BayesianRidge
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Análisis de series temporales
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Análisis estadístico avanzado
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Análisis bayesiano
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("PyMC no disponible. Análisis bayesiano deshabilitado.")

# Visualización avanzada
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Utilities y warnings
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import itertools

# Configuración global
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CLASES Y ESTRUCTURAS DE DATOS AVANZADAS
# ============================================================================

@dataclass
class AnalysisResults:
    """
    Clase para almacenar resultados de análisis estadístico.
    Permite organizar y acceder fácilmente a múltiples métricas.
    """
    test_statistic: float
    p_value: float
    critical_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: Optional[str] = None
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Determina si el resultado es estadísticamente significativo"""
        return self.p_value < alpha

class AdvancedDataAnalyzer:
    """
    Clase principal para análisis de datos inferenciales avanzados.
    Integra múltiples metodologías estadísticas y de machine learning.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Inicializa el analizador con configuraciones por defecto.
        
        Args:
            confidence_level: Nivel de confianza para intervalos (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results_cache = {}
        self.models_trained = {}
        
        # Configuración de estilos para visualización
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        print(f"🔬 AdvancedDataAnalyzer inicializado")
        print(f"📊 Nivel de confianza: {confidence_level*100}%")
        print(f"⚠️  Alpha: {self.alpha}")

    def comprehensive_normality_test(self, data: np.ndarray) -> Dict[str, AnalysisResults]:
        """
        Realiza múltiples pruebas de normalidad para validación cruzada.
        
        Args:
            data: Array de datos numéricos
            
        Returns:
            Dict con resultados de todas las pruebas de normalidad
        """
        
        # Limpiar datos de valores NaN
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) < 3:
            raise ValueError("Insuficientes datos para pruebas de normalidad")
        
        results = {}
        
        # Test de Shapiro-Wilk (mejor para n < 5000)
        if len(clean_data) <= 5000:
            stat, p_val = shapiro(clean_data)
            results['shapiro_wilk'] = AnalysisResults(
                test_statistic=stat,
                p_value=p_val,
                interpretation="Normal" if p_val > self.alpha else "No Normal"
            )
        
        # Test de D'Agostino y Pearson
        stat, p_val = normaltest(clean_data)
        results['dagostino_pearson'] = AnalysisResults(
            test_statistic=stat,
            p_value=p_val,
            interpretation="Normal" if p_val > self.alpha else "No Normal"
        )
        
        # Test de Jarque-Bera
        stat, p_val = jarque_bera(clean_data)
        results['jarque_bera'] = AnalysisResults(
            test_statistic=stat,
            p_value=p_val,
            interpretation="Normal" if p_val > self.alpha else "No Normal"
        )
        
        # Test de Kolmogorov-Smirnov contra distribución normal
        # Estandarizar datos para comparar con N(0,1)
        standardized = (clean_data - np.mean(clean_data)) / np.std(clean_data)
        stat, p_val = kstest(standardized, 'norm')
        results['kolmogorov_smirnov'] = AnalysisResults(
            test_statistic=stat,
            p_value=p_val,
            interpretation="Normal" if p_val > self.alpha else "No Normal"
        )
        
        # Test de Anderson-Darling
        result = anderson(clean_data, dist='norm')
        # Usar el valor crítico para alpha=0.05 (índice 2)
        critical_val = result.critical_values[2] if len(result.critical_values) > 2 else None
        is_normal = result.statistic < critical_val if critical_val else False
        
        results['anderson_darling'] = AnalysisResults(
            test_statistic=result.statistic,
            p_value=None,  # Anderson-Darling no proporciona p-value directo
            critical_value=critical_val,
            interpretation="Normal" if is_normal else "No Normal"
        )
        
        return results

    def advanced_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Realiza análisis de correlación múltiple con diferentes métodos.
        
        Args:
            df: DataFrame con variables numéricas
            
        Returns:
            Dict con matrices de correlación de diferentes métodos
        """
        
        # Filtrar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_cols]
        
        correlations = {}
        
        # Correlación de Pearson (lineal)
        correlations['pearson'] = numeric_df.corr(method='pearson')
        
        # Correlación de Spearman (monotónica)
        correlations['spearman'] = numeric_df.corr(method='spearman')
        
        # Correlación de Kendall (ordinal)
        correlations['kendall'] = numeric_df.corr(method='kendall')
        
        # Matriz de correlación parcial (requiere statsmodels)
        try:
            from statsmodels.stats.correlation_tools import corr_nearest
            # Calcular correlación parcial manualmente
            inv_corr = np.linalg.pinv(correlations['pearson'].values)
            partial_corr = np.zeros_like(inv_corr)
            for i in range(len(inv_corr)):
                for j in range(len(inv_corr)):
                    if i != j:
                        partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
                    else:
                        partial_corr[i, j] = 1
            
            correlations['partial'] = pd.DataFrame(
                partial_corr, 
                index=correlations['pearson'].index,
                columns=correlations['pearson'].columns
            )
        except Exception as e:
            print(f"⚠️  No se pudo calcular correlación parcial: {e}")
        
        return correlations

    def ensemble_regression_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                   test_size: float = 0.3) -> Dict[str, Dict]:
        """
        Realiza análisis de regresión con múltiples algoritmos y ensamblado.
        
        Args:
            X: Variables predictoras
            y: Variable objetivo
            test_size: Proporción de datos para testing
            
        Returns:
            Dict con resultados de todos los modelos
        """
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Definir modelos base
        models = {
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0),
            'lasso_regression': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr_rbf': SVR(kernel='rbf', gamma='scale'),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'bayesian_ridge': BayesianRidge()
        }
        
        results = {}
        predictions = {}
        
        # Entrenar y evaluar cada modelo
        for name, model in models.items():
            print(f"🔄 Entrenando {name}...")
            
            try:
                # Usar datos escalados para modelos que lo requieren
                if name in ['svr_rbf', 'neural_network']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calcular métricas
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                # Validación cruzada
                cv_scores = cross_val_score(
                    model, X_train_scaled if name in ['svr_rbf', 'neural_network'] else X_train, 
                    y_train, cv=5, scoring='r2'
                )
                
                results[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': rmse,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred
                }
                
                predictions[name] = y_pred
                
                print(f"✅ {name}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
                
            except Exception as e:
                print(f"❌ Error en {name}: {e}")
                continue
        
        # Crear modelo de ensamble (Voting Regressor)
        if len(results) >= 3:
            try:
                # Seleccionar los 3 mejores modelos basados en R²
                best_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)[:3]
                
                ensemble_models = [(name, data['model']) for name, data in best_models]
                ensemble = VotingRegressor(ensemble_models)
                
                # Entrenar ensemble
                ensemble.fit(X_train, y_train)
                y_pred_ensemble = ensemble.predict(X_test)
                
                # Métricas del ensemble
                mse_ens = mean_squared_error(y_test, y_pred_ensemble)
                mae_ens = mean_absolute_error(y_test, y_pred_ensemble)
                r2_ens = r2_score(y_test, y_pred_ensemble)
                rmse_ens = np.sqrt(mse_ens)
                
                results['ensemble'] = {
                    'model': ensemble,
                    'mse': mse_ens,
                    'mae': mae_ens,
                    'r2': r2_ens,
                    'rmse': rmse_ens,
                    'predictions': y_pred_ensemble,
                    'base_models': [name for name, _ in best_models]
                }
                
                print(f"🏆 Ensemble: R² = {r2_ens:.4f}, RMSE = {rmse_ens:.4f}")
                
            except Exception as e:
                print(f"⚠️  No se pudo crear ensemble: {e}")
        
        # Guardar modelos entrenados
        self.models_trained.update(results)
        
        return results

    def time_series_analysis(self, ts: pd.Series, freq: str = 'D') -> Dict[str, any]:
        """
        Análisis completo de series temporales con múltiples técnicas.
        
        Args:
            ts: Serie temporal con índice de fechas
            freq: Frecuencia de la serie ('D', 'M', 'W', etc.)
            
        Returns:
            Dict con todos los resultados del análisis
        """
        
        # Asegurar que el índice sea datetime
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("La serie debe tener un índice de tipo DatetimeIndex")
        
        results = {}
        
        # 1. Estadísticas descriptivas
        results['descriptive_stats'] = {
            'count': len(ts),
            'mean': ts.mean(),
            'std': ts.std(),
            'min': ts.min(),
            'max': ts.max(),
            'skewness': ts.skew(),
            'kurtosis': ts.kurtosis()
        }
        
        # 2. Test de estacionariedad
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(ts.dropna())
        results['adf_test'] = AnalysisResults(
            test_statistic=adf_result[0],
            p_value=adf_result[1],
            critical_value=adf_result[4]['5%'],
            interpretation="Estacionaria" if adf_result[1] < 0.05 else "No Estacionaria"
        )
        
        # KPSS Test
        kpss_result = kpss(ts.dropna())
        results['kpss_test'] = AnalysisResults(
            test_statistic=kpss_result[0],
            p_value=kpss_result[1],
            critical_value=kpss_result[3]['5%'],
            interpretation="Estacionaria" if kpss_result[1] > 0.05 else "No Estacionaria"
        )
        
        # 3. Descomposición estacional
        try:
            # Determinar el período estacional automáticamente
            if freq == 'D':
                period = 365  # Anual para datos diarios
            elif freq == 'M':
                period = 12   # Anual para datos mensuales
            elif freq == 'W':
                period = 52   # Anual para datos semanales
            else:
                period = 12   # Default
            
            # Solo descomponer si hay suficientes datos
            if len(ts) >= 2 * period:
                decomposition = seasonal_decompose(ts, model='additive', period=period)
                results['decomposition'] = {
                    'trend': decomposition.trend,
                    'seasonal': decomposition.seasonal,
                    'residual': decomposition.resid,
                    'original': ts
                }
            else:
                print(f"⚠️  Insuficientes datos para descomposición estacional (necesarios: {2*period}, disponibles: {len(ts)})")
                
        except Exception as e:
            print(f"⚠️  Error en descomposición estacional: {e}")
        
        # 4. Modelado ARIMA automático
        try:
            # Encontrar los mejores parámetros ARIMA usando grid search
            best_aic = np.inf
            best_params = None
            best_model = None
            
            # Rangos de parámetros a probar
            p_values = range(0, 4)
            d_values = range(0, 3)
            q_values = range(0, 4)
            
            print("🔄 Buscando mejores parámetros ARIMA...")
            
            for p, d, q in itertools.product(p_values, d_values, q_values):
                try:
                    model = ARIMA(ts, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_params = (p, d, q)
                        best_model = fitted_model
                        
                except Exception:
                    continue
            
            if best_model is not None:
                results['arima_model'] = {
                    'model': best_model,
                    'params': best_params,
                    'aic': best_aic,
                    'bic': best_model.bic,
                    'forecast': best_model.forecast(steps=30)  # Pronóstico 30 períodos
                }
                
                print(f"✅ Mejor ARIMA{best_params}: AIC = {best_aic:.2f}")
            else:
                print("❌ No se pudo ajustar modelo ARIMA")
                
        except Exception as e:
            print(f"⚠️  Error en modelado ARIMA: {e}")
        
        # 5. Suavizamiento exponencial
        try:
            # Holt-Winters
            hw_model = ExponentialSmoothing(
                ts, 
                trend='add' if len(ts) > 24 else None,
                seasonal='add' if len(ts) > 24 else None,
                seasonal_periods=12 if len(ts) > 24 else None
            )
            hw_fitted = hw_model.fit()
            
            results['holt_winters'] = {
                'model': hw_fitted,
                'aic': hw_fitted.aic,
                'forecast': hw_fitted.forecast(steps=30)
            }
            
            print(f"✅ Holt-Winters: AIC = {hw_fitted.aic:.2f}")
            
        except Exception as e:
            print(f"⚠️  Error en Holt-Winters: {e}")
        
        return results

    def bayesian_analysis(self, data: Dict[str, np.ndarray], 
                         model_type: str = 'regression') -> Dict[str, any]:
        """
        Análisis bayesiano usando PyMC.
        
        Args:
            data: Dict con 'X' (predictores) y 'y' (objetivo)
            model_type: Tipo de modelo ('regression', 'classification')
            
        Returns:
            Dict con resultados del análisis bayesiano
        """
        
        if not BAYESIAN_AVAILABLE:
            return {"error": "PyMC no está disponible"}
        
        X = data['X']
        y = data['y']
        
        print("🔄 Iniciando análisis bayesiano...")
        
        try:
            with pm.Model() as model:
                if model_type == 'regression':
                    # Modelo de regresión lineal bayesiana
                    
                    # Priors para los coeficientes
                    alpha = pm.Normal('alpha', mu=0, sigma=10)  # Intercepto
                    beta = pm.Normal('beta', mu=0, sigma=10, shape=X.shape[1])  # Coeficientes
                    sigma = pm.HalfCauchy('sigma', beta=1)  # Error estándar
                    
                    # Likelihood (verosimilitud)
                    mu = alpha + pm.math.dot(X, beta)
                    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
                    
                    # Sampling
                    print("🔄 Realizando muestreo MCMC...")
                    trace = pm.sample(2000, tune=1000, random_seed=42, 
                                    target_accept=0.9, return_inferencedata=True)
                    
                    # Diagnósticos
                    print("📊 Calculando diagnósticos...")
                    summary = az.summary(trace)
                    rhat = az.rhat(trace)
                    ess = az.ess(trace)
                    
                    # Predicciones posteriores
                    with model:
                        posterior_pred = pm.sample_posterior_predictive(trace, random_seed=42)
                    
                    results = {
                        'model': model,
                        'trace': trace,
                        'summary': summary,
                        'rhat': rhat,
                        'ess': ess,
                        'posterior_predictive': posterior_pred,
                        'diagnostics': {
                            'converged': bool(rhat.max() < 1.1),
                            'effective_samples': ess.min()
                        }
                    }
                    
                    print("✅ Análisis bayesiano completado")
                    return results
                    
        except Exception as e:
            print(f"❌ Error en análisis bayesiano: {e}")
            return {"error": str(e)}

    def create_advanced_visualizations(self, data: pd.DataFrame, 
                                     analysis_results: Dict = None) -> None:
        """
        Crea visualizaciones avanzadas e interactivas.
        
        Args:
            data: DataFrame con los datos
            analysis_results: Resultados de análisis previos
        """
        
        # Configurar subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Distribuciones', 'Correlaciones', 'Box Plots', 
                          'Scatter Matrix', 'Series Temporal', 'Residuos'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "box"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Obtener columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns[:4]  # Máximo 4 columnas
        
        if len(numeric_cols) == 0:
            print("⚠️  No hay columnas numéricas para visualizar")
            return
        
        # 1. Histogramas de distribución
        for i, col in enumerate(numeric_cols):
            if i < 2:  # Solo 2 histogramas
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, nbinsx=30, opacity=0.7),
                    row=1, col=1
                )
        
        # 2. Mapa de calor de correlaciones
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu',
                    zmin=-1, zmax=1
                ),
                row=1, col=2
            )
        
        # 3. Box plots
        for i, col in enumerate(numeric_cols):
            if i < 2:
                fig.add_trace(
                    go.Box(y=data[col], name=col),
                    row=2, col=1
                )
        
        # 4. Scatter plot
        if len(numeric_cols) >= 2:
            fig.add_trace(
                go.Scatter(
                    x=data[numeric_cols[0]], 
                    y=data[numeric_cols[1]],
                    mode='markers',
                    name=f'{numeric_cols[0]} vs {numeric_cols[1]}',
                    marker=dict(size=8, opacity=0.6)
                ),
                row=2, col=2
            )
        
        # 5. Serie temporal (si hay columna de fecha)
        date_cols = data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            fig.add_trace(
                go.Scatter(
                    x=data[date_cols[0]], 
                    y=data[numeric_cols[0]],
                    mode='lines+markers',
                    name='Serie Temporal'
                ),
                row=3, col=1
            )
        
        # 6. Residuos (si hay resultados de regresión)
        if analysis_results and 'regression_results' in analysis_results:
            reg_results = analysis_results['regression_results']
            best_model = max(reg_results.items(), key=lambda x: x[1]['r2'])
            predictions = best_model[1]['predictions']
            
            if len(predictions) == len(data):
                residuals = data[numeric_cols[0]][:len(predictions)] - predictions
                fig.add_trace(
                    go.Scatter(
                        x=predictions,
                        y=residuals,
                        mode='markers',
                        name='Residuos vs Predicciones',
                        marker=dict(size=6)
                    ),
                    row=3, col=2
                )
        
        # Actualizar layout
        fig.update_layout(
            height=1200,
            title_text="Dashboard Avanzado de Análisis de Datos",
            showlegend=True,
            template="plotly_white"
        )
        
        # Mostrar gráfico
        fig.show()
        
        # Crear gráfico de correlaciones 3D si hay suficientes variables
        if len(numeric_cols) >= 3:
            fig_3d = go.Figure(data=go.Scatter3d(
                x=data[numeric_cols[0]],
                y=data[numeric_cols[1]],
                z=data[numeric_cols[2]],
                mode='markers',
                marker=dict(
                    size=5,
                    color=data[numeric_cols[0]],
                    colorscale='Viridis',
                    showscale=True,
                    opacity=0.8
                )
            ))
            
            fig_3d.update_layout(
                title='Análisis 3D de Variables',
                scene=dict(
                    xaxis_title=numeric_cols[0],
                    yaxis_title=numeric_cols[1],
                    zaxis_title=numeric_cols[2]
                ),
                width=800,
                height=600
            )
            
            fig_3d.show()

    def generate_comprehensive_report(self, results: Dict[str, any], 
                                    data: pd.DataFrame) -> str:
        """
        Genera un reporte completo en formato markdown.
        
        Args:
            results: Dict con todos los resultados de análisis
            data: DataFrame original
            
        Returns:
            String con el reporte en formato markdown
        """
        
        report = f"""
# 📊 REPORTE AVANZADO DE ANÁLISIS INFERENCIAL DE DATOS

**Fecha de generación:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Registros analizados:** {len(data)}
**Variables:** {len(data.columns)}

---

## 🔍 RESUMEN EJECUTIVO

### Características del Dataset
- **Dimensiones:** {data.shape[0]} filas × {data.shape[1]} columnas
- **Variables numéricas:** {len(data.select_dtypes(include=[np.number]).columns)}
- **Variables categóricas:** {len(data.select_dtypes(include=['object']).columns)}
- **Valores faltantes:** {data.isnull().sum().sum()} ({(data.isnull().sum().sum() / (data.shape[0] * data.shape[1]) * 100):.2f}%)

---

## 📈 ANÁLISIS ESTADÍSTICO DESCRIPTIVO

### Estadísticas Centrales
"""
        
        # Agregar estadísticas descriptivas
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            report += "\n| Variable | Media | Mediana | Desv. Std | Min | Max |\n"
            report += "|----------|-------|---------|-----------|-----|-----|\n"
            
            for col in numeric_data.columns:
                stats = numeric_data[col].describe()
                report += f"| {col} | {stats['mean']:.3f} | {numeric_data[col].median():.3f} | "
                report += f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |\n"

        # Análisis de normalidad si está disponible
        if 'normality_tests' in results:
            report += "\n---\n\n## 🧮 PRUEBAS DE NORMALIDAD\n\n"
            norm_results = results['normality_tests']
            
            report += "| Prueba | Estadístico | p-valor | Interpretación |\n"
            report += "|--------|-------------|---------|----------------|\n"
            
            for test_name, test_result in norm_results.items():
                if hasattr(test_result, 'test_statistic'):
                    p_val = f"{test_result.p_value:.6f}" if test_result.p_value else "N/A"
                    report += f"| {test_name.title()} | {test_result.test_statistic:.4f} | "
                    report += f"{p_val} | {test_result.interpretation} |\n"

        # Análisis de correlación si está disponible
        if 'correlation_analysis' in results:
            report += "\n---\n\n## 🔗 ANÁLISIS DE CORRELACIONES\n\n"
            corr_results = results['correlation_analysis']
            
            for method, corr_matrix in corr_results.items():
                if isinstance(corr_matrix, pd.DataFrame):
                    # Encontrar las correlaciones más altas (excluyendo diagonal)
                    corr_abs = corr_matrix.abs()
                    np.fill_diagonal(corr_abs.values, 0)
                    
                    if corr_abs.max().max() > 0:
                        max_corr = corr_abs.max().max()
                        max_pair = corr_abs.stack().idxmax()
                        actual_corr = corr_matrix.loc[max_pair[0], max_pair[1]]
                        
                        report += f"### Correlación {method.title()}\n"
                        report += f"- **Correlación más alta:** {max_pair[0]} ↔ {max_pair[1]} "
                        report += f"(r = {actual_corr:.4f})\n"
                        report += f"- **Correlación promedio:** {corr_abs.mean().mean():.4f}\n\n"

        # Resultados de regresión si están disponibles
        if 'regression_results' in results:
            report += "\n---\n\n## 🎯 ANÁLISIS DE REGRESIÓN\n\n"
            reg_results = results['regression_results']
            
            report += "| Modelo | R² | RMSE | MAE | CV Score |\n"
            report += "|--------|----|----|-----|----------|\n"
            
            # Ordenar modelos por R²
            sorted_models = sorted(reg_results.items(), key=lambda x: x[1]['r2'], reverse=True)
            
            for model_name, metrics in sorted_models:
                cv_score = f"{metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}"
                report += f"| {model_name.title()} | {metrics['r2']:.4f} | "
                report += f"{metrics['rmse']:.4f} | {metrics['mae']:.4f} | {cv_score} |\n"
            
            # Destacar el mejor modelo
            best_model = sorted_models[0]
            report += f"\n**🏆 Mejor modelo:** {best_model[0].title()} "
            report += f"(R² = {best_model[1]['r2']:.4f})\n"

        # Análisis de series temporales si está disponible
        if 'time_series_analysis' in results:
            report += "\n---\n\n## ⏰ ANÁLISIS DE SERIES TEMPORALES\n\n"
            ts_results = results['time_series_analysis']
            
            # Test de estacionariedad
            if 'adf_test' in ts_results:
                adf = ts_results['adf_test']
                report += f"### Test de Estacionariedad\n"
                report += f"- **ADF Test:** {adf.interpretation} (p-valor: {adf.p_value:.6f})\n"
                
            if 'kpss_test' in ts_results:
                kpss = ts_results['kpss_test']
                report += f"- **KPSS Test:** {kpss.interpretation} (p-valor: {kpss.p_value:.6f})\n"
            
            # Modelo ARIMA
            if 'arima_model' in ts_results:
                arima = ts_results['arima_model']
                report += f"\n### Modelo ARIMA\n"
                report += f"- **Mejores parámetros:** ARIMA{arima['params']}\n"
                report += f"- **AIC:** {arima['aic']:.2f}\n"
                report += f"- **BIC:** {arima['bic']:.2f}\n"

        # Análisis bayesiano si está disponible
        if 'bayesian_analysis' in results:
            report += "\n---\n\n## 🎲 ANÁLISIS BAYESIANO\n\n"
            bayes_results = results['bayesian_analysis']
            
            if 'diagnostics' in bayes_results:
                diag = bayes_results['diagnostics']
                convergence = "✅ Convergencia alcanzada" if diag['converged'] else "❌ No convergencia"
                report += f"- **Convergencia:** {convergence}\n"
                report += f"- **Muestras efectivas mínimas:** {diag['effective_samples']:.0f}\n"

        # Conclusiones y recomendaciones
        report += "\n---\n\n## 💡 CONCLUSIONES Y RECOMENDACIONES\n\n"
        
        conclusions = []
        
        # Conclusiones basadas en normalidad
        if 'normality_tests' in results:
            normal_count = sum(1 for test in results['normality_tests'].values() 
                             if hasattr(test, 'interpretation') and 'Normal' in test.interpretation)
            total_tests = len(results['normality_tests'])
            
            if normal_count / total_tests > 0.6:
                conclusions.append("📊 La mayoría de variables siguen distribución normal - apropiado usar métodos paramétricos")
            else:
                conclusions.append("📊 Variables no siguen distribución normal - considerar métodos no paramétricos")
        
        # Conclusiones basadas en correlaciones
        if 'correlation_analysis' in results:
            corr_results = results['correlation_analysis']
            if 'pearson' in corr_results:
                max_corr = corr_results['pearson'].abs().max().max()
                if max_corr > 0.8:
                    conclusions.append("🔗 Correlaciones muy altas detectadas - posible multicolinealidad")
                elif max_corr > 0.5:
                    conclusions.append("🔗 Correlaciones moderadas encontradas - relaciones lineales presentes")
        
        # Conclusiones basadas en regresión
        if 'regression_results' in results:
            best_r2 = max(results['regression_results'].values(), key=lambda x: x['r2'])['r2']
            if best_r2 > 0.8:
                conclusions.append("🎯 Modelos predictivos excelentes - alta capacidad explicativa")
            elif best_r2 > 0.6:
                conclusions.append("🎯 Modelos predictivos buenos - capacidad explicativa moderada")
            else:
                conclusions.append("🎯 Modelos predictivos limitados - considerar más variables o transformaciones")
        
        if conclusions:
            for i, conclusion in enumerate(conclusions, 1):
                report += f"{i}. {conclusion}\n"
        else:
            report += "- Realizar análisis más específicos según objetivos del negocio\n"
            report += "- Considerar transformaciones de variables si es necesario\n"
            report += "- Validar resultados con datos adicionales\n"

        # Próximos pasos
        report += "\n### 🚀 Próximos Pasos Recomendados\n\n"
        report += "1. **Validación cruzada:** Confirmar resultados con conjunto de datos independiente\n"
        report += "2. **Feature engineering:** Crear nuevas variables derivadas\n"
        report += "3. **Análisis de outliers:** Identificar y tratar valores atípicos\n"
        report += "4. **Interpretabilidad:** Usar modelos explicables para insights de negocio\n"
        report += "5. **Monitoreo:** Implementar seguimiento de performance en producción\n"

        report += f"\n---\n\n*Reporte generado automáticamente por AdvancedDataAnalyzer v3.0*\n"
        report += f"*Configuración: Nivel de confianza {self.confidence_level*100}%, α = {self.alpha}*"
        
        return report

# ============================================================================
# FUNCIÓN PRINCIPAL DE DEMOSTRACIÓN
# ============================================================================

def run_comprehensive_analysis_demo():
    """
    Función de demostración que ejecuta todos los análisis con datos sintéticos.
    Muestra el uso completo del framework de análisis avanzado.
    """
    
    print("=" * 80)
    print("🚀 INICIANDO DEMOSTRACIÓN DEL SISTEMA AVANZADO DE ANÁLISIS")
    print("=" * 80)
    
    # Configurar semilla para reproducibilidad
    np.random.seed(42)
    
    # 1. GENERAR DATOS SINTÉTICOS REALISTAS
    print("\n📊 Generando dataset sintético...")
    
    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Variables base con diferentes distribuciones
    x1 = np.random.normal(100, 15, n_samples)  # Normal
    x2 = np.random.exponential(2, n_samples)   # Exponencial
    x3 = np.random.lognormal(0, 0.5, n_samples)  # Log-normal
    x4 = np.random.uniform(0, 10, n_samples)   # Uniforme
    
    # Variable objetivo con relación no-lineal y ruido
    noise = np.random.normal(0, 5, n_samples)
    y = 2.5 * x1 + 1.8 * np.log(x2 + 1) + 0.3 * x3**0.5 + 0.5 * x4**2 + noise
    
    # Añadir componente estacional para serie temporal
    seasonal_component = 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25)
    y_temporal = y + seasonal_component
    
    # Variable categórica
    categories = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.35, 0.25])
    
    # Crear DataFrame
    data = pd.DataFrame({
        'fecha': dates,
        'variable_normal': x1,
        'variable_exponencial': x2,
        'variable_lognormal': x3,
        'variable_uniforme': x4,
        'objetivo': y,
        'objetivo_temporal': y_temporal,
        'categoria': categories
    })
    
    # Introducir algunos valores faltantes para realismo
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data.loc[missing_indices, 'variable_normal'] = np.nan
    
    print(f"✅ Dataset generado: {data.shape[0]} filas × {data.shape[1]} columnas")
    print(f"📊 Vista previa:")
    print(data.head())
    
    # 2. INICIALIZAR ANALIZADOR
    print("\n🔧 Inicializando AdvancedDataAnalyzer...")
    analyzer = AdvancedDataAnalyzer(confidence_level=0.95)
    
    # Diccionario para almacenar todos los resultados
    all_results = {}
    
    # 3. ANÁLISIS DE NORMALIDAD
    print("\n🧮 Ejecutando pruebas de normalidad...")
    normality_results = {}
    
    numeric_columns = ['variable_normal', 'variable_exponencial', 'variable_lognormal', 'objetivo']
    for col in numeric_columns:
        print(f"   🔍 Analizando: {col}")
        clean_data = data[col].dropna().values
        try:
            norm_test = analyzer.comprehensive_normality_test(clean_data)
            normality_results[col] = norm_test
            
            # Resumen de normalidad
            normal_count = sum(1 for test in norm_test.values() 
                             if hasattr(test, 'p_value') and test.p_value and test.p_value > 0.05)
            total_tests = len([test for test in norm_test.values() if hasattr(test, 'p_value') and test.p_value])
            
            print(f"      📈 {normal_count}/{total_tests} tests indican normalidad")
            
        except Exception as e:
            print(f"      ❌ Error en {col}: {e}")
    
    all_results['normality_tests'] = normality_results
    
    # 4. ANÁLISIS DE CORRELACIÓN
    print("\n🔗 Ejecutando análisis de correlación...")
    try:
        correlation_results = analyzer.advanced_correlation_analysis(data)
        all_results['correlation_analysis'] = correlation_results
        
        # Mostrar correlación más alta
        pearson_corr = correlation_results['pearson'].abs()
        np.fill_diagonal(pearson_corr.values, 0)
        max_corr = pearson_corr.max().max()
        max_pair = pearson_corr.stack().idxmax()
        
        print(f"   🔍 Correlación más alta: {max_pair[0]} ↔ {max_pair[1]} (r = {max_corr:.4f})")
        
    except Exception as e:
        print(f"   ❌ Error en análisis de correlación: {e}")
    
    # 5. ANÁLISIS DE REGRESIÓN ENSEMBLE
    print("\n🎯 Ejecutando análisis de regresión ensemble...")
    try:
        # Preparar datos para regresión
        X_features = data[['variable_normal', 'variable_exponencial', 'variable_lognormal', 'variable_uniforme']].fillna(data.mean())
        y_target = data['objetivo']
        
        regression_results = analyzer.ensemble_regression_analysis(X_features, y_target, test_size=0.25)
        all_results['regression_results'] = regression_results
        
        # Mostrar resumen
        if regression_results:
            best_model = max(regression_results.items(), key=lambda x: x[1]['r2'])
            print(f"   🏆 Mejor modelo: {best_model[0]} (R² = {best_model[1]['r2']:.4f})")
        
    except Exception as e:
        print(f"   ❌ Error en regresión ensemble: {e}")
    
    # 6. ANÁLISIS DE SERIES TEMPORALES
    print("\n⏰ Ejecutando análisis de series temporales...")
    try:
        # Crear serie temporal
        ts_data = data.set_index('fecha')['objetivo_temporal']
        
        ts_results = analyzer.time_series_analysis(ts_data, freq='D')
        all_results['time_series_analysis'] = ts_results
        
        # Mostrar resultados de estacionariedad
        if 'adf_test' in ts_results:
            adf_result = ts_results['adf_test']
            print(f"   📈 ADF Test: {adf_result.interpretation} (p = {adf_result.p_value:.6f})")
        
        if 'arima_model' in ts_results:
            arima_info = ts_results['arima_model']
            print(f"   🎯 Mejor ARIMA: {arima_info['params']} (AIC = {arima_info['aic']:.2f})")
        
    except Exception as e:
        print(f"   ❌ Error en análisis temporal: {e}")
    
    # 7. ANÁLISIS BAYESIANO (si está disponible)
    if BAYESIAN_AVAILABLE:
        print("\n🎲 Ejecutando análisis bayesiano...")
        try:
            # Preparar datos para análisis bayesiano
            X_bayes = X_features.values
            y_bayes = y_target.values
            
            bayes_data = {'X': X_bayes, 'y': y_bayes}
            bayes_results = analyzer.bayesian_analysis(bayes_data, model_type='regression')
            
            if 'error' not in bayes_results:
                all_results['bayesian_analysis'] = bayes_results
                
                if 'diagnostics' in bayes_results:
                    convergence = "✅ Sí" if bayes_results['diagnostics']['converged'] else "❌ No"
                    print(f"   🔍 Convergencia: {convergence}")
                    print(f"   📊 Muestras efectivas: {bayes_results['diagnostics']['effective_samples']:.0f}")
            else:
                print(f"   ❌ {bayes_results['error']}")
                
        except Exception as e:
            print(f"   ❌ Error en análisis bayesiano: {e}")
    
    # 8. GENERAR VISUALIZACIONES
    print("\n📊 Generando visualizaciones avanzadas...")
    try:
        analyzer.create_advanced_visualizations(data, all_results)
        print("   ✅ Visualizaciones generadas exitosamente")
    except Exception as e:
        print(f"   ❌ Error en visualizaciones: {e}")
    
    # 9. GENERAR REPORTE COMPLETO
    print("\n📋 Generando reporte comprehensive...")
    try:
        comprehensive_report = analyzer.generate_comprehensive_report(all_results, data)
        
        # Guardar reporte
        with open('reporte_analisis_avanzado.md', 'w', encoding='utf-8') as f:
            f.write(comprehensive_report)
        
        print("   ✅ Reporte guardado como 'reporte_analisis_avanzado.md'")
        
        # Mostrar extracto del reporte
        print("\n" + "="*50)
        print("📄 EXTRACTO DEL REPORTE:")
        print("="*50)
        print(comprehensive_report[:1500] + "\n...\n[Reporte completo en archivo .md]")
        
    except Exception as e:
        print(f"   ❌ Error generando reporte: {e}")
    
    # 10. RESUMEN FINAL
    print("\n" + "="*80)
    print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("="*80)
    print(f"📊 Análisis ejecutados: {len(all_results)}")
    print(f"🔍 Variables analizadas: {len(numeric_columns)}")
    print(f"📈 Modelos evaluados: {len(regression_results) if 'regression_results' in all_results else 0}")
    print(f"💾 Resultados almacenados en: analyzer.results_cache")
    print(f"📋 Reporte disponible en: reporte_analisis_avanzado.md")
    
    return analyzer, data, all_results

# ============================================================================
# EJECUCIÓN DE LA DEMOSTRACIÓN
# ============================================================================

if __name__ == "__main__":
    # Ejecutar demostración completa
    try:
        analyzer, dataset, results = run_comprehensive_analysis_demo()
        
        print(f"\n🔧 Uso posterior del analizador:")
        print(f"   - analyzer.comprehensive_normality_test(data)")
        print(f"   - analyzer.advanced_correlation_analysis(df)")
        print(f"   - analyzer.ensemble_regression_analysis(X, y)")
        print(f"   - analyzer.time_series_analysis(timeseries)")
        print(f"   - analyzer.create_advanced_visualizations(df)")
        
    except Exception as e:
        print(f"❌ Error en la demostración: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n💡 Framework listo para análisis de datos profesionales!")