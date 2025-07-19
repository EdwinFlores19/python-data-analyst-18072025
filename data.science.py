"""
SISTEMA AVANZADO DE INFERENCIA ESTADÍSTICA Y MACHINE LEARNING
================================================================

Este código demuestra técnicas avanzadas de inferencia estadística utilizando
múltiples librerías de data science para análisis bayesiano, inferencia causal,
modelado probabilístico, y machine learning con incertidumbre.

Librerías utilizadas:
- PyMC: Modelado probabilístico bayesiano
- CausalInference: Inferencia causal
- Scipy.stats: Estadística inferencial clásica
- Sklearn: Machine learning con intervalos de confianza
- Statsmodels: Modelos estadísticos avanzados
- Arviz: Visualización y diagnóstico bayesiano
- Pandas/Numpy: Manipulación de datos
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Librerías de inferencia estadística y modelado
import scipy.stats as stats
from scipy.stats import norm, t, chi2, f
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Machine Learning con inferencia
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Librerías avanzadas de inferencia bayesiana
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    print("PyMC no está disponible. Instalarlo con: pip install pymc arviz")
    PYMC_AVAILABLE = False

# Para inferencia causal
try:
    from causalinference import CausalModel
    CAUSAL_AVAILABLE = True
except ImportError:
    print("CausalInference no disponible. Instalarlo con: pip install CausalInference")
    CAUSAL_AVAILABLE = False

class AdvancedInferenceSystem:
    """
    Sistema avanzado que combina múltiples técnicas de inferencia estadística:
    1. Inferencia bayesiana con PyMC
    2. Inferencia causal
    3. Modelos probabilísticos con incertidumbre
    4. Tests estadísticos avanzados
    5. Machine learning con intervalos de confianza
    """
    
    def __init__(self, random_state=42):
        """
        Inicializa el sistema de inferencia avanzado
        
        Parameters:
        -----------
        random_state : int
            Semilla para reproducibilidad
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.results = {}
        
    def generate_complex_dataset(self, n_samples=1000):
        """
        Genera un dataset complejo con múltiples variables y relaciones causales
        para demostrar técnicas de inferencia avanzada
        
        Parameters:
        -----------
        n_samples : int
            Número de observaciones a generar
            
        Returns:
        --------
        pd.DataFrame : Dataset generado con variables correlacionadas
        """
        print("🔧 Generando dataset complejo con relaciones causales...")
        
        # Variables exógenas (causas raíz)
        np.random.seed(self.random_state)
        
        # Variable latente no observada (confounder)
        latent_confounder = np.random.normal(0, 1, n_samples)
        
        # Variables demográficas
        age = np.random.gamma(2, 20, n_samples)  # Edad con distribución gamma
        gender = np.random.binomial(1, 0.5, n_samples)  # Género binario
        education = np.random.poisson(12, n_samples)  # Años de educación
        
        # Variable de tratamiento (influenciada por demografía y confounder)
        treatment_prob = stats.logistic.cdf(
            0.1 * age - 0.5 * gender + 0.3 * education + 0.4 * latent_confounder
        )
        treatment = np.random.binomial(1, treatment_prob, n_samples)
        
        # Variables mediadoras
        stress_level = (0.3 * treatment + 0.2 * age/50 - 0.1 * education + 
                       0.4 * latent_confounder + np.random.normal(0, 0.5, n_samples))
        
        social_support = (-0.2 * stress_level + 0.15 * education + 
                         0.1 * gender + np.random.normal(0, 0.3, n_samples))
        
        # Variable de resultado (outcome) con efectos directos e indirectos
        # Efecto causal complejo: directo del tratamiento + mediado por stress y soporte
        outcome = (1.5 * treatment +  # Efecto directo del tratamiento
                  -0.8 * stress_level +  # Efecto del estrés
                  0.6 * social_support +  # Efecto del soporte social
                  0.1 * age/50 +  # Efecto de la edad
                  0.2 * education +  # Efecto de la educación
                  0.3 * latent_confounder +  # Confounder no observado
                  np.random.normal(0, 0.4, n_samples))  # Ruido
        
        # Variables adicionales para análisis multivariado
        income = (30000 + 2000 * education + 15000 * treatment + 
                 1000 * outcome + np.random.normal(0, 5000, n_samples))
        
        health_score = (70 + 10 * outcome - 5 * stress_level + 
                       3 * social_support + np.random.normal(0, 8, n_samples))
        
        # Crear DataFrame
        data = pd.DataFrame({
            'age': age,
            'gender': gender,
            'education': education,
            'treatment': treatment,
            'stress_level': stress_level,
            'social_support': social_support,
            'outcome': outcome,
            'income': income,
            'health_score': health_score,
            'latent_confounder': latent_confounder  # Normalmente no observable
        })
        
        print(f"✅ Dataset generado: {data.shape[0]} observaciones, {data.shape[1]} variables")
        return data
    
    def comprehensive_statistical_tests(self, data):
        """
        Realiza una batería completa de tests estadísticos inferenciales
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset para análisis
        """
        print("\n📊 REALIZANDO TESTS ESTADÍSTICOS AVANZADOS")
        print("=" * 50)
        
        tests_results = {}
        
        # 1. TEST DE NORMALIDAD (Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling)
        print("🔍 1. Tests de Normalidad:")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limitamos a 5 variables por claridad
            # Shapiro-Wilk (mejor para muestras pequeñas)
            shapiro_stat, shapiro_p = stats.shapiro(data[col].dropna()[:5000])  # Máximo 5000
            
            # Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(data[col].dropna(), 'norm',
                                       args=(data[col].mean(), data[col].std()))
            
            # Anderson-Darling
            anderson_result = stats.anderson(data[col].dropna(), dist='norm')
            
            tests_results[f'{col}_normality'] = {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'ks': {'statistic': ks_stat, 'p_value': ks_p},
                'anderson': {'statistic': anderson_result.statistic, 
                           'critical_values': anderson_result.critical_values}
            }
            
            print(f"   {col}:")
            print(f"     Shapiro-Wilk: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}")
            print(f"     Kolmogorov-Smirnov: stat={ks_stat:.4f}, p={ks_p:.4f}")
        
        # 2. TESTS DE HOMOGENEIDAD DE VARIANZA
        print("\n🔍 2. Tests de Homogeneidad de Varianza:")
        
        # Levene's test (robusto a no-normalidad)
        groups = [data[data['treatment'] == i]['outcome'].dropna() for i in [0, 1]]
        levene_stat, levene_p = stats.levene(*groups, center='median')
        
        # Bartlett's test (asume normalidad)
        bartlett_stat, bartlett_p = stats.bartlett(*groups)
        
        tests_results['variance_homogeneity'] = {
            'levene': {'statistic': levene_stat, 'p_value': levene_p},
            'bartlett': {'statistic': bartlett_stat, 'p_value': bartlett_p}
        }
        
        print(f"   Levene (robusto): stat={levene_stat:.4f}, p={levene_p:.4f}")
        print(f"   Bartlett (normalidad): stat={bartlett_stat:.4f}, p={bartlett_p:.4f}")
        
        # 3. TESTS DE INDEPENDENCIA Y ASOCIACIÓN
        print("\n🔍 3. Tests de Independencia:")
        
        # Chi-cuadrado de independencia
        contingency_table = pd.crosstab(data['treatment'], data['gender'])
        chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(contingency_table)
        
        # V de Cramér (medida de asociación)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        tests_results['independence'] = {
            'chi2': {'statistic': chi2_stat, 'p_value': chi2_p, 'dof': dof},
            'cramers_v': cramers_v
        }
        
        print(f"   Chi-cuadrado: stat={chi2_stat:.4f}, p={chi2_p:.4f}, dof={dof}")
        print(f"   V de Cramér: {cramers_v:.4f}")
        
        # 4. TESTS NO PARAMÉTRICOS
        print("\n🔍 4. Tests No Paramétricos:")
        
        # Mann-Whitney U (alternativa a t-test)
        mw_stat, mw_p = stats.mannwhitneyu(
            data[data['treatment'] == 0]['outcome'].dropna(),
            data[data['treatment'] == 1]['outcome'].dropna(),
            alternative='two-sided'
        )
        
        # Wilcoxon signed-rank test (datos pareados)
        # Creamos datos pareados artificialmente para demostración
        paired_before = data['outcome'][:500].values
        paired_after = paired_before + np.random.normal(0.5, 0.2, 500)  # Simulamos mejora
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(paired_before, paired_after)
        
        # Kruskal-Wallis (ANOVA no paramétrica)
        education_groups = [
            data[data['education'] <= 10]['outcome'].dropna(),
            data[(data['education'] > 10) & (data['education'] <= 14)]['outcome'].dropna(),
            data[data['education'] > 14]['outcome'].dropna()
        ]
        kw_stat, kw_p = stats.kruskal(*education_groups)
        
        tests_results['nonparametric'] = {
            'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p},
            'wilcoxon': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
            'kruskal_wallis': {'statistic': kw_stat, 'p_value': kw_p}
        }
        
        print(f"   Mann-Whitney U: stat={mw_stat:.4f}, p={mw_p:.4f}")
        print(f"   Wilcoxon: stat={wilcoxon_stat:.4f}, p={wilcoxon_p:.4f}")
        print(f"   Kruskal-Wallis: stat={kw_stat:.4f}, p={kw_p:.4f}")
        
        # 5. ANÁLISIS DE CORRELACIONES CON INTERVALOS DE CONFIANZA
        print("\n🔍 5. Correlaciones con Intervalos de Confianza:")
        
        # Correlación de Pearson con intervalo de confianza
        r_pearson, p_pearson = stats.pearsonr(data['stress_level'], data['outcome'])
        
        # Intervalo de confianza para correlación (transformación Fisher)
        z_fisher = np.arctanh(r_pearson)  # Transformación Fisher
        se_z = 1 / np.sqrt(len(data) - 3)  # Error estándar
        z_lower = z_fisher - 1.96 * se_z
        z_upper = z_fisher + 1.96 * se_z
        r_lower = np.tanh(z_lower)  # Transformación inversa
        r_upper = np.tanh(z_upper)
        
        # Correlación de Spearman (no paramétrica)
        r_spearman, p_spearman = stats.spearmanr(data['stress_level'], data['outcome'])
        
        # Correlación de Kendall (tau)
        tau_kendall, p_kendall = stats.kendalltau(data['stress_level'], data['outcome'])
        
        tests_results['correlations'] = {
            'pearson': {'r': r_pearson, 'p_value': p_pearson, 'ci': (r_lower, r_upper)},
            'spearman': {'r': r_spearman, 'p_value': p_spearman},
            'kendall': {'tau': tau_kendall, 'p_value': p_kendall}
        }
        
        print(f"   Pearson: r={r_pearson:.4f}, p={p_pearson:.4f}, CI=({r_lower:.4f}, {r_upper:.4f})")
        print(f"   Spearman: ρ={r_spearman:.4f}, p={p_spearman:.4f}")
        print(f"   Kendall: τ={tau_kendall:.4f}, p={p_kendall:.4f}")
        
        self.results['statistical_tests'] = tests_results
        return tests_results
    
    def bayesian_inference_analysis(self, data):
        """
        Realiza análisis de inferencia bayesiana usando PyMC
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset para análisis bayesiano
        """
        if not PYMC_AVAILABLE:
            print("\n❌ PyMC no está disponible para análisis bayesiano")
            return None
            
        print("\n🎯 ANÁLISIS DE INFERENCIA BAYESIANA")
        print("=" * 40)
        
        # Preparar datos
        y = data['outcome'].values
        X = data[['treatment', 'stress_level', 'social_support', 'age', 'education']].values
        X_scaled = StandardScaler().fit_transform(X)
        
        try:
            # Modelo bayesiano de regresión lineal con priors informativos
            with pm.Model() as bayesian_model:
                print("🔨 Construyendo modelo bayesiano...")
                
                # Priors para los coeficientes (informativos basados en conocimiento previo)
                # Intercepto
                alpha = pm.Normal('alpha', mu=0, sigma=1)
                
                # Coeficientes con priors específicos
                beta_treatment = pm.Normal('beta_treatment', mu=1.5, sigma=0.5)  # Esperamos efecto positivo
                beta_stress = pm.Normal('beta_stress', mu=-0.8, sigma=0.3)  # Esperamos efecto negativo
                beta_support = pm.Normal('beta_support', mu=0.6, sigma=0.3)  # Esperamos efecto positivo
                beta_age = pm.Normal('beta_age', mu=0.1, sigma=0.2)  # Efecto menor
                beta_education = pm.Normal('beta_education', mu=0.2, sigma=0.2)  # Efecto menor
                
                betas = pm.math.stack([beta_treatment, beta_stress, beta_support, beta_age, beta_education])
                
                # Prior para la varianza (distribución gamma inversa)
                sigma = pm.HalfNormal('sigma', sigma=1)
                
                # Likelihood (verosimilitud)
                mu = alpha + pm.math.dot(X_scaled, betas)
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
                
                print("🔄 Ejecutando muestreo MCMC...")
                # Muestreo MCMC
                trace = pm.sample(2000, tune=1000, return_inferencedata=True, 
                                chains=4, random_seed=self.random_state)
                
                # Diagnósticos de convergencia
                print("📈 Diagnósticos de convergencia:")
                
                # R-hat (debe ser < 1.1 para buena convergencia)
                rhat = az.rhat(trace)
                print(f"   R-hat máximo: {max(rhat.to_array().values):.4f}")
                
                # Effective Sample Size
                ess = az.ess(trace)
                print(f"   ESS mínimo: {min(ess.to_array().values):.0f}")
                
                # Resumen posterior
                summary = az.summary(trace)
                print("\n📊 Resumen Posterior:")
                print(summary)
                
                # Intervalos de credibilidad al 95%
                hdi = az.hdi(trace, hdi_prob=0.95)
                print("\n🎯 Intervalos de Credibilidad al 95%:")
                for var in hdi.data_vars:
                    if var != 'y_obs':  # Excluir observaciones
                        values = hdi[var].values
                        if values.ndim == 0:  # Escalar
                            print(f"   {var}: [{values:.4f}]")
                        else:  # Vector
                            print(f"   {var}: [{values[0]:.4f}, {values[1]:.4f}]")
                
                # Probabilidades posteriores de efectos específicos
                posterior_samples = trace.posterior
                
                # Probabilidad de que el tratamiento tenga efecto positivo
                prob_treatment_positive = (posterior_samples['beta_treatment'] > 0).mean().values
                print(f"\n🎲 Probabilidad de efecto positivo del tratamiento: {prob_treatment_positive:.3f}")
                
                # Probabilidad de que el estrés tenga efecto negativo
                prob_stress_negative = (posterior_samples['beta_stress'] < 0).mean().values
                print(f"🎲 Probabilidad de efecto negativo del estrés: {prob_stress_negative:.3f}")
                
                # Comparación de modelos usando WAIC (Widely Applicable Information Criterion)
                waic = az.waic(trace)
                print(f"\n📏 WAIC del modelo: {waic.waic:.2f} ± {waic.waic_se:.2f}")
                
                self.results['bayesian_analysis'] = {
                    'trace': trace,
                    'summary': summary,
                    'rhat_max': max(rhat.to_array().values),
                    'ess_min': min(ess.to_array().values),
                    'prob_treatment_positive': prob_treatment_positive,
                    'prob_stress_negative': prob_stress_negative,
                    'waic': waic.waic
                }
                
                return trace
                
        except Exception as e:
            print(f"❌ Error en análisis bayesiano: {e}")
            return None
    
    def causal_inference_analysis(self, data):
        """
        Realiza análisis de inferencia causal usando diferentes métodos
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset para análisis causal
        """
        print("\n🎯 ANÁLISIS DE INFERENCIA CAUSAL")
        print("=" * 35)
        
        causal_results = {}
        
        # 1. ESTIMACIÓN NAIVE DEL EFECTO CAUSAL (sin ajustar confounders)
        print("🔍 1. Estimación Naive (sin controles):")
        
        treated_mean = data[data['treatment'] == 1]['outcome'].mean()
        control_mean = data[data['treatment'] == 0]['outcome'].mean()
        naive_ate = treated_mean - control_mean
        
        # Test t para la diferencia
        treated_group = data[data['treatment'] == 1]['outcome']
        control_group = data[data['treatment'] == 0]['outcome']
        t_stat, t_pvalue = stats.ttest_ind(treated_group, control_group)
        
        print(f"   ATE naive: {naive_ate:.4f}")
        print(f"   t-statistic: {t_stat:.4f}, p-value: {t_pvalue:.4f}")
        
        causal_results['naive_ate'] = naive_ate
        
        # 2. PROPENSITY SCORE MATCHING (usando regresión logística)
        print("\n🔍 2. Propensity Score Matching:")
        
        # Variables para calcular propensity score (excluyendo outcome y treatment)
        ps_vars = ['age', 'gender', 'education', 'stress_level', 'social_support']
        X_ps = data[ps_vars]
        y_treatment = data['treatment']
        
        # Modelo de propensity score
        from sklearn.linear_model import LogisticRegression
        ps_model = LogisticRegression(random_state=self.random_state)
        ps_model.fit(X_ps, y_treatment)
        
        # Calcular propensity scores
        propensity_scores = ps_model.predict_proba(X_ps)[:, 1]
        data_ps = data.copy()
        data_ps['propensity_score'] = propensity_scores
        
        # Matching simple por nearest neighbor
        def simple_matching(data_with_ps, caliper=0.01):
            treated = data_with_ps[data_with_ps['treatment'] == 1].copy()
            control = data_with_ps[data_with_ps['treatment'] == 0].copy()
            
            matched_pairs = []
            used_controls = set()
            
            for _, treated_unit in treated.iterrows():
                # Encontrar el control más cercano en propensity score
                distances = np.abs(control['propensity_score'] - treated_unit['propensity_score'])
                
                # Filtrar por caliper y unidades no usadas
                valid_matches = control[
                    (distances <= caliper) & 
                    (~control.index.isin(used_controls))
                ]
                
                if len(valid_matches) > 0:
                    best_match_idx = distances[valid_matches.index].idxmin()
                    matched_pairs.append((treated_unit['outcome'], control.loc[best_match_idx, 'outcome']))
                    used_controls.add(best_match_idx)
            
            return matched_pairs
        
        # Realizar matching
        matched_pairs = simple_matching(data_ps)
        
        if matched_pairs:
            treated_outcomes = [pair[0] for pair in matched_pairs]
            matched_control_outcomes = [pair[1] for pair in matched_pairs]
            
            psm_ate = np.mean(treated_outcomes) - np.mean(matched_control_outcomes)
            
            # Test t pareado para el efecto
            t_stat_psm, t_pvalue_psm = stats.ttest_rel(treated_outcomes, matched_control_outcomes)
            
            print(f"   Pares encontrados: {len(matched_pairs)}")
            print(f"   ATE con PSM: {psm_ate:.4f}")
            print(f"   t-statistic: {t_stat_psm:.4f}, p-value: {t_pvalue_psm:.4f}")
            
            causal_results['psm_ate'] = psm_ate
            causal_results['psm_pairs'] = len(matched_pairs)
        else:
            print("   ❌ No se encontraron pares válidos para matching")
        
        # 3. REGRESIÓN CON CONTROLES (Conditional Average Treatment Effect)
        print("\n🔍 3. Regresión con Controles:")
        
        # Modelo con controles
        control_vars = ['treatment', 'age', 'gender', 'education', 'stress_level', 'social_support']
        X_control = sm.add_constant(data[control_vars])
        y_outcome = data['outcome']
        
        ols_model = sm.OLS(y_outcome, X_control).fit()
        
        # Efecto del tratamiento es el coeficiente
        treatment_coeff = ols_model.params['treatment']
        treatment_se = ols_model.bse['treatment']
        treatment_pvalue = ols_model.pvalues['treatment']
        treatment_ci = ols_model.conf_int().loc['treatment']
        
        print(f"   ATE con controles: {treatment_coeff:.4f} ± {treatment_se:.4f}")
        print(f"   p-value: {treatment_pvalue:.4f}")
        print(f"   IC 95%: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]")
        print(f"   R-squared: {ols_model.rsquared:.4f}")
        
        causal_results['regression_ate'] = treatment_coeff
        causal_results['regression_se'] = treatment_se
        causal_results['regression_r2'] = ols_model.rsquared
        
        # 4. ANÁLISIS DE MEDIACIÓN
        print("\n🔍 4. Análisis de Mediación (Efecto Indirecto):")
        
        # Efecto del tratamiento en el mediador (stress_level)
        mediator_model = sm.OLS(data['stress_level'], 
                               sm.add_constant(data[['treatment', 'age', 'education']])).fit()
        a_path = mediator_model.params['treatment']  # Efecto treatment -> mediador
        
        # Efecto del mediador en el outcome (controlando por treatment)
        outcome_model = sm.OLS(data['outcome'], 
                              sm.add_constant(data[['treatment', 'stress_level', 
                                                   'age', 'education']])).fit()
        b_path = outcome_model.params['stress_level']  # Efecto mediador -> outcome
        c_prime = outcome_model.params['treatment']  # Efecto directo
        
        # Efecto indirecto (mediado)
        indirect_effect = a_path * b_path
        
        # Test de Sobel para significancia del efecto indirecto
        se_a = mediator_model.bse['treatment']
        se_b = outcome_model.bse['stress_level']
        se_indirect = np.sqrt(b_path**2 * se_a**2 + a_path**2 * se_b**2)
        z_sobel = indirect_effect / se_indirect
        p_sobel = 2 * (1 - stats.norm.cdf(abs(z_sobel)))
        
        print(f"   Efecto directo (c'): {c_prime:.4f}")
        print(f"   Efecto indirecto (a*b): {indirect_effect:.4f}")
        print(f"   Test de Sobel: z={z_sobel:.4f}, p={p_sobel:.4f}")
        
        causal_results['mediation'] = {
            'direct_effect': c_prime,
            'indirect_effect': indirect_effect,
            'sobel_z': z_sobel,
            'sobel_p': p_sobel
        }
        
        # 5. INSTRUMENTAL VARIABLES (simulado)
        print("\n🔍 5. Variables Instrumentales (simulado):")
        
        # Creamos un instrumento artificial correlacionado con treatment pero no con outcome
        # (excepto a través del treatment)
        instrument = (0.7 * data['age'] / 50 + 0.3 * data['education'] / 15 + 
                     np.random.normal(0, 0.5, len(data))) > 0.5
        
        # First stage: predecir treatment usando instrumento
        first_stage = sm.OLS(data['treatment'], 
                            sm.add_constant(pd.DataFrame({'instrument': instrument}))).fit()
        
        # F-statistic para weak instrument test
        f_stat_iv = first_stage.fvalue
        
        # Second stage: usar predicted treatment
        predicted_treatment = first_stage.fittedvalues
        second_stage = sm.OLS(data['outcome'], 
                             sm.add_constant(predicted_treatment)).fit()
        
        iv_ate = second_stage.params['const']  # Coeficiente del treatment predicho
        
        print(f"   F-stat primer stage: {f_stat_iv:.4f} (>10 indica instrumento fuerte)")
        print(f"   ATE con IV: {iv_ate:.4f}")
        
        causal_results['iv_ate'] = iv_ate
        causal_results['iv_f_stat'] = f_stat_iv
        
        self.results['causal_analysis'] = causal_results
        return causal_results
    
    def probabilistic_ml_with_uncertainty(self, data):
        """
        Implementa modelos de ML que capturan incertidumbre epistémica y aleatórica
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset para modelado probabilístico
        """
        print("\n🤖 MACHINE LEARNING PROBABILÍSTICO CON INCERTIDUMBRE")
        print("=" * 55)
        
        # Preparar datos
        feature_cols = ['age', 'gender', 'education', 'treatment', 'stress_level', 'social_support']
        X = data[feature_cols].values
        y = data['outcome'].values
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Estandarizar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ml_results = {}
        
        # 1. BAYESIAN RIDGE REGRESSION (Regresión Bayesiana)
        print("🔍 1. Bayesian Ridge Regression:")
        
        # Modelo con priors automáticos
        bayesian_ridge = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,  # Priors para precisión del ruido
            lambda_1=1e-6, lambda_2=1e-6,  # Priors para precisión de los pesos
            compute_score=True,
            fit_intercept=True
        )
        
        bayesian_ridge.fit(X_train_scaled, y_train)
        
        # Predicciones con incertidumbre
        y_pred_br, y_std_br = bayesian_ridge.predict(X_test_scaled, return_std=True)
        
        # Métricas
        mse_br = mean_squared_error(y_test, y_pred_br)
        r2_br = r2_score(y_test, y_pred_br)
        
        # Intervalos de confianza al 95%
        y_lower_br = y_pred_br - 1.96 * y_std_br
        y_upper_br = y_pred_br + 1.96 * y_std_br
        
        # Coverage probability (proporción de valores reales dentro del IC)
        coverage_br = np.mean((y_test >= y_lower_br) & (y_test <= y_upper_br))
        
        print(f"   MSE: {mse_br:.4f}")
        print(f"   R²: {r2_br:.4f}")
        print(f"   Cobertura IC 95%: {coverage_br:.3f} (ideal: 0.95)")
        print(f"   Incertidumbre promedio: ±{np.mean(y_std_br):.4f}")
        
        ml_results['bayesian_ridge'] = {
            'mse': mse_br, 'r2': r2_br, 'coverage': coverage_br,
            'uncertainty_mean': np.mean(y_std_br)
        }
        
        # 2. AUTOMATIC RELEVANCE DETERMINATION (ARD)
        print("\n🔍 2. Automatic Relevance Determination:")
        
        ard_model = ARDRegression(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            threshold_lambda=1e4,  # Threshold para feature selection
            compute_score=True
        )
        
        ard_model.fit(X_train_scaled, y_train)
        
        # Predicciones con incertidumbre
        y_pred_ard, y_std_ard = ard_model.predict(X_test_scaled, return_std=True)
        
        # Relevancia de features (lambda_ más alto = menos relevante)
        feature_relevance = 1.0 / ard_model.lambda_
        relevant_features = feature_relevance > np.median(feature_relevance)
        
        mse_ard = mean_squared_error(y_test, y_pred_ard)
        r2_ard = r2_score(y_test, y_pred_ard)
        coverage_ard = np.mean((y_test >= y_pred_ard - 1.96 * y_std_ard) & 
                              (y_test <= y_pred_ard + 1.96 * y_std_ard))
        
        print(f"   MSE: {mse_ard:.4f}")
        print(f"   R²: {r2_ard:.4f}")
        print(f"   Cobertura IC 95%: {coverage_ard:.3f}")
        print(f"   Features relevantes: {np.sum(relevant_features)}/{len(feature_cols)}")
        
        # Mostrar relevancia por feature
        for i, (feature, relevance) in enumerate(zip(feature_cols, feature_relevance)):
            print(f"     {feature}: {relevance:.2e} {'✓' if relevant_features[i] else '✗'}")
        
        ml_results['ard'] = {
            'mse': mse_ard, 'r2': r2_ard, 'coverage': coverage_ard,
            'relevant_features': np.sum(relevant_features),
            'feature_relevance': dict(zip(feature_cols, feature_relevance))
        }
        
        # 3. GAUSSIAN PROCESS REGRESSION
        print("\n🔍 3. Gaussian Process Regression:")
        
        # Kernel compuesto: RBF + Ruido blanco + Matern
        kernel = (1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + 
                 WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1)) +
                 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5))
        
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,  # Regularización numérica
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=3,
            normalize_y=True,
            copy_X_train=True,
            random_state=self.random_state
        )
        
        # Entrenar (usar subset para eficiencia)
        train_subset_size = min(500, len(X_train_scaled))  # GP es O(n³)
        idx_subset = np.random.choice(len(X_train_scaled), train_subset_size, replace=False)
        
        gp_model.fit(X_train_scaled[idx_subset], y_train[idx_subset])
        
        # Predicciones con incertidumbre epistémica
        y_pred_gp, y_std_gp = gp_model.predict(X_test_scaled, return_std=True)
        
        mse_gp = mean_squared_error(y_test, y_pred_gp)
        r2_gp = r2_score(y_test, y_pred_gp)
        coverage_gp = np.mean((y_test >= y_pred_gp - 1.96 * y_std_gp) & 
                             (y_test <= y_pred_gp + 1.96 * y_std_gp))
        
        print(f"   MSE: {mse_gp:.4f}")
        print(f"   R²: {r2_gp:.4f}")
        print(f"   Cobertura IC 95%: {coverage_gp:.3f}")
        print(f"   Log-marginal likelihood: {gp_model.log_marginal_likelihood():.2f}")
        print(f"   Kernel optimizado: {gp_model.kernel_}")
        
        ml_results['gaussian_process'] = {
            'mse': mse_gp, 'r2': r2_gp, 'coverage': coverage_gp,
            'log_marginal_likelihood': gp_model.log_marginal_likelihood(),
            'kernel': str(gp_model.kernel_)
        }
        
        # 4. ENSEMBLE METHODS CON CUANTIFICACIÓN DE INCERTIDUMBRE
        print("\n🔍 4. Random Forest con Intervalos de Predicción:")
        
        # Random Forest con múltiples estimadores para incertidumbre
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            bootstrap=True,
            oob_score=True,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predicciones de todos los árboles para estimar incertidumbre
        tree_predictions = np.array([
            tree.predict(X_test_scaled) for tree in rf_model.estimators_
        ])
        
        y_pred_rf = np.mean(tree_predictions, axis=0)
        y_std_rf = np.std(tree_predictions, axis=0)  # Incertidumbre del ensemble
        
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        coverage_rf = np.mean((y_test >= y_pred_rf - 1.96 * y_std_rf) & 
                             (y_test <= y_pred_rf + 1.96 * y_std_rf))
        
        # Feature importance con intervalos de confianza
        feature_importances = rf_model.feature_importances_
        
        print(f"   MSE: {mse_rf:.4f}")
        print(f"   R²: {r2_rf:.4f}")
        print(f"   OOB Score: {rf_model.oob_score_:.4f}")
        print(f"   Cobertura IC 95%: {coverage_rf:.3f}")
        print("   Feature Importance:")
        
        for feature, importance in zip(feature_cols, feature_importances):
            print(f"     {feature}: {importance:.4f}")
        
        ml_results['random_forest'] = {
            'mse': mse_rf, 'r2': r2_rf, 'oob_score': rf_model.oob_score_,
            'coverage': coverage_rf, 
            'feature_importance': dict(zip(feature_cols, feature_importances))
        }
        
        # 5. GRADIENT BOOSTING CON PÉRDIDA CUANTIL
        print("\n🔍 5. Gradient Boosting con Intervalos de Predicción:")
        
        # Modelos para diferentes cuantiles
        quantiles = [0.025, 0.5, 0.975]  # 2.5%, mediana, 97.5%
        gb_models = {}
        
        for q in quantiles:
            gb_models[q] = GradientBoostingRegressor(
                loss='quantile',
                alpha=q,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                random_state=self.random_state
            )
            gb_models[q].fit(X_train_scaled, y_train)
        
        # Predicciones para cada cuantil
        predictions_q = {q: gb_models[q].predict(X_test_scaled) for q in quantiles}
        
        y_pred_gb = predictions_q[0.5]  # Mediana como predicción central
        y_lower_gb = predictions_q[0.025]
        y_upper_gb = predictions_q[0.975]
        
        mse_gb = mean_squared_error(y_test, y_pred_gb)
        r2_gb = r2_score(y_test, y_pred_gb)
        coverage_gb = np.mean((y_test >= y_lower_gb) & (y_test <= y_upper_gb))
        
        # Ancho promedio del intervalo
        interval_width = np.mean(y_upper_gb - y_lower_gb)
        
        print(f"   MSE: {mse_gb:.4f}")
        print(f"   R²: {r2_gb:.4f}")
        print(f"   Cobertura IC 95%: {coverage_gb:.3f}")
        print(f"   Ancho promedio intervalo: {interval_width:.4f}")
        
        ml_results['gradient_boosting'] = {
            'mse': mse_gb, 'r2': r2_gb, 'coverage': coverage_gb,
            'interval_width': interval_width
        }
        
        # 6. CROSS-VALIDATION CON INTERVALOS DE CONFIANZA
        print("\n🔍 6. Validación Cruzada con Intervalos de Confianza:")
        
        # Validación cruzada estratificada para varios modelos
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # Convertir y a categórico para StratifiedKFold
        y_categorical = pd.cut(y, bins=5, labels=False)  # 5 bins
        
        models_cv = {
            'Bayesian Ridge': BayesianRidge(),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=self.random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=self.random_state)
        }
        
        cv_results = {}
        
        for name, model in models_cv.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Intervalo de confianza para la media (t-distribution)
            t_critical = stats.t.ppf(0.975, df=len(scores)-1)
            margin_error = t_critical * std_score / np.sqrt(len(scores))
            ci_lower = mean_score - margin_error
            ci_upper = mean_score + margin_error
            
            print(f"   {name}:")
            print(f"     R² promedio: {mean_score:.4f} ± {std_score:.4f}")
            print(f"     IC 95%: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            cv_results[name] = {
                'mean_r2': mean_score,
                'std_r2': std_score,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        ml_results['cross_validation'] = cv_results
        
        self.results['probabilistic_ml'] = ml_results
        return ml_results
    
    def advanced_time_series_inference(self, data):
        """
        Análisis avanzado de series temporales con inferencia estadística
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset con componente temporal
        """
        print("\n📈 ANÁLISIS DE SERIES TEMPORALES CON INFERENCIA")
        print("=" * 45)
        
        # Crear una serie temporal artificial basada en los datos
        np.random.seed(self.random_state)
        
        # Generar serie temporal con tendencia, estacionalidad y ruido
        n_periods = 100
        time_index = pd.date_range(start='2020-01-01', periods=n_periods, freq='M')
        
        # Componentes de la serie
        trend = 0.05 * np.arange(n_periods)  # Tendencia linear
        seasonal = 2 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Estacionalidad anual
        noise = np.random.normal(0, 0.5, n_periods)
        
        # Serie base
        ts_base = 10 + trend + seasonal + noise
        
        # Añadir efectos de las variables del dataset original
        treatment_effect = np.random.choice([0, 1.5], n_periods, p=[0.7, 0.3])
        stress_effect = -0.3 * np.random.normal(0.5, 0.2, n_periods)
        
        # Serie temporal final
        ts_values = ts_base + treatment_effect + stress_effect
        
        ts_data = pd.DataFrame({
            'date': time_index,
            'value': ts_values,
            'treatment_effect': treatment_effect,
            'stress_component': stress_effect
        })
        ts_data.set_index('date', inplace=True)
        
        ts_results = {}
        
        # 1. TESTS DE ESTACIONARIEDAD
        print("🔍 1. Tests de Estacionariedad:")
        
        # Augmented Dickey-Fuller Test
        adf_result = adfuller(ts_data['value'], autolag='AIC')
        adf_statistic, adf_pvalue = adf_result[0], adf_result[1]
        adf_critical_values = adf_result[4]
        
        print(f"   Augmented Dickey-Fuller:")
        print(f"     Estadístico: {adf_statistic:.4f}")
        print(f"     p-value: {adf_pvalue:.4f}")
        print(f"     Valores críticos: {adf_critical_values}")
        
        # KPSS Test (null: estacionaria)
        kpss_result = kpss(ts_data['value'], regression='ct')  # con tendencia y constante
        kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]
        kpss_critical_values = kpss_result[3]
        
        print(f"   KPSS Test:")
        print(f"     Estadístico: {kpss_statistic:.4f}")
        print(f"     p-value: {kpss_pvalue:.4f}")
        print(f"     Valores críticos: {kpss_critical_values}")
        
        # Interpretación conjunta
        if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
            stationarity = "Estacionaria"
        elif adf_pvalue > 0.05 and kpss_pvalue < 0.05:
            stationarity = "No estacionaria"
        else:
            stationarity = "Inconcluso"
        
        print(f"   Conclusión: {stationarity}")
        
        ts_results['stationarity'] = {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue,
            'conclusion': stationarity
        }
        
        # 2. DESCOMPOSICIÓN DE SERIES TEMPORALES
        print("\n🔍 2. Descomposición de Series Temporales:")
        
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Descomposición STL (Seasonal and Trend decomposition using Loess)
        decomposition = seasonal_decompose(
            ts_data['value'], 
            model='additive',  # modelo aditivo
            period=12,  # período estacional (mensual)
            extrapolate_trend='freq'
        )
        
        # Extraer componentes
        trend_component = decomposition.trend
        seasonal_component = decomposition.seasonal
        residual_component = decomposition.resid
        
        # Estadísticas de los componentes
        trend_variance = np.var(trend_component.dropna())
        seasonal_variance = np.var(seasonal_component.dropna())
        residual_variance = np.var(residual_component.dropna())
        total_variance = trend_variance + seasonal_variance + residual_variance
        
        print(f"   Varianza explicada por componentes:")
        print(f"     Tendencia: {trend_variance/total_variance:.3f}")
        print(f"     Estacionalidad: {seasonal_variance/total_variance:.3f}")
        print(f"     Residuos: {residual_variance/total_variance:.3f}")
        
        # Test de aleatoriedad en residuos (Ljung-Box)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box_result = acorr_ljungbox(residual_component.dropna(), lags=10, return_df=True)
        
        print(f"   Test Ljung-Box (residuos aleatorios):")
        print(f"     p-value promedio: {ljung_box_result['lb_pvalue'].mean():.4f}")
        
        ts_results['decomposition'] = {
            'trend_variance_prop': trend_variance/total_variance,
            'seasonal_variance_prop': seasonal_variance/total_variance,
            'residual_variance_prop': residual_variance/total_variance,
            'ljung_box_pvalue': ljung_box_result['lb_pvalue'].mean()
        }
        
        # 3. MODELOS ARIMA CON SELECCIÓN AUTOMÁTICA
        print("\n🔍 3. Modelado ARIMA con Selección Automática:")
        
        from statsmodels.tsa.arima.model import ARIMA
        from itertools import product
        
        # Grid search para mejores parámetros ARIMA
        p_values = range(0, 3)  # AR order
        d_values = range(0, 2)  # Differencing
        q_values = range(0, 3)  # MA order
        
        best_aic = np.inf
        best_params = None
        best_model = None
        
        print("   Buscando mejores parámetros ARIMA...")
        
        for p, d, q in product(p_values, d_values, q_values):
            try:
                model = ARIMA(ts_data['value'], order=(p, d, q))
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_params = (p, d, q)
                    best_model = fitted_model
                    
            except:
                continue
        
        if best_model is not None:
            print(f"   Mejor modelo: ARIMA{best_params}")
            print(f"   AIC: {best_aic:.2f}")
            print(f"   BIC: {best_model.bic:.2f}")
            print(f"   Log-likelihood: {best_model.llf:.2f}")
            
            # Diagnósticos del modelo
            residuals = best_model.resid
            
            # Test de normalidad de residuos
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            
            # Test de autocorrelación en residuos
            ljung_box_resid = acorr_ljungbox(residuals, lags=10, return_df=True)
            
            print(f"   Diagnósticos:")
            print(f"     Normalidad residuos (Shapiro): p={shapiro_p:.4f}")
            print(f"     Autocorrelación residuos (Ljung-Box): p={ljung_box_resid['lb_pvalue'].iloc[-1]:.4f}")
            
            # Predicción con intervalos de confianza
            forecast_steps = 12
            forecast = best_model.forecast(steps=forecast_steps)
            forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
            
            print(f"   Predicción próximos {forecast_steps} períodos:")
            print(f"     Promedio: {forecast.mean():.4f}")
            print(f"     Rango IC 95%: [{forecast_ci.iloc[:, 0].mean():.4f}, {forecast_ci.iloc[:, 1].mean():.4f}]")
            
            ts_results['arima'] = {
                'best_params': best_params,
                'aic': best_aic,
                'bic': best_model.bic,
                'residuals_normality_p': shapiro_p,
                'residuals_autocorr_p': ljung_box_resid['lb_pvalue'].iloc[-1],
                'forecast_mean': forecast.mean(),
                'forecast_ci_width': (forecast_ci.iloc[:, 1] - forecast_ci.iloc[:, 0]).mean()
            }
        
        # 4. ANÁLISIS DE INTERVENCIÓN (CAMBIO ESTRUCTURAL)
        print("\n🔍 4. Análisis de Cambio Estructural:")
        
        # Test de Chow para cambio estructural en el punto medio
        from statsmodels.stats.diagnostic import breaks_cusumolsresid
        
        # Preparar datos para regresión
        ts_regression_data = ts_data.copy()
        ts_regression_data['time_trend'] = np.arange(len(ts_regression_data))
        ts_regression_data['seasonal_cos'] = np.cos(2 * np.pi * ts_regression_data.index.month / 12)
        ts_regression_data['seasonal_sin'] = np.sin(2 * np.pi * ts_regression_data.index.month / 12)
        
        # Modelo base
        X_ts = sm.add_constant(ts_regression_data[['time_trend', 'seasonal_cos', 'seasonal_sin']])
        y_ts = ts_regression_data['value']
        
        ts_model = sm.OLS(y_ts, X_ts).fit()
        
        # Test CUSUM para estabilidad de parámetros
        cusum_result = breaks_cusumolsresid(ts_model.resid)
        
        print(f"   Test CUSUM de estabilidad:")
        print(f"     Estadístico: {cusum_result[0]:.4f}")
        print(f"     p-value: {cusum_result[1]:.4f}")
        
        # Detección de puntos de cambio usando método de suma acumulativa
        def detect_changepoints(series, threshold=2.0):
            """Detecta puntos de cambio usando CUSUM"""
            n = len(series)
            mean_series = np.mean(series)
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)
            
            for i in range(1, n):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (series[i] - mean_series))
                cusum_neg[i] = min(0, cusum_neg[i-1] + (series[i] - mean_series))
            
            # Puntos donde CUSUM excede el threshold
            changepoints = []
            for i in range(n):
                if abs(cusum_pos[i]) > threshold or abs(cusum_neg[i]) > threshold:
                    changepoints.append(i)
            
            return changepoints
        
        changepoints = detect_changepoints(ts_data['value'].values)
        
        if changepoints:
            print(f"   Puntos de cambio detectados: {len(changepoints)}")
            print(f"   Fechas aproximadas: {[ts_data.index[cp].strftime('%Y-%m') for cp in changepoints[:3]]}")
        else:
            print("   No se detectaron puntos de cambio significativos")
        
        ts_results['structural_change'] = {
            'cusum_statistic': cusum_result[0],
            'cusum_pvalue': cusum_result[1],
            'changepoints_detected': len(changepoints)
        }
        
        self.results['time_series_analysis'] = ts_results
        return ts_results
    
    def comprehensive_model_comparison(self):
        """
        Compara todos los métodos de inferencia implementados
        """
        print("\n🏆 COMPARACIÓN INTEGRAL DE MÉTODOS DE INFERENCIA")
        print("=" * 50)
        
        if not hasattr(self, 'results') or not self.results:
            print("❌ No hay resultados para comparar. Ejecute primero los análisis.")
            return
        
        comparison = {}
        
        # 1. COMPARACIÓN DE ESTIMACIONES DE EFECTOS CAUSALES
        if 'causal_analysis' in self.results:
            causal = self.results['causal_analysis']
            
            print("📊 Estimaciones del Efecto Causal del Tratamiento:")
            print(f"   Naive (sin controles): {causal['naive_ate']:.4f}")
            
            if 'psm_ate' in causal:
                print(f"   Propensity Score Matching: {causal['psm_ate']:.4f}")
            
            print(f"   Regresión con controles: {causal['regression_ate']:.4f}")
            
            if 'iv_ate' in causal:
                print(f"   Variables instrumentales: {causal['iv_ate']:.4f}")
            
            comparison['causal_estimates'] = causal
        
        # 2. COMPARACIÓN DE MODELOS ML
        if 'probabilistic_ml' in self.results:
            ml = self.results['probabilistic_ml']
            
            print("\n📊 Comparación de Modelos de Machine Learning:")
            print("   Modelo                  | R²      | Cobertura | Incertidumbre")
            print("   " + "-"*60)
            
            models_to_compare = [
                ('Bayesian Ridge', 'bayesian_ridge'),
                ('ARD Regression', 'ard'),
                ('Gaussian Process', 'gaussian_process'),
                ('Random Forest', 'random_forest'),
                                ('Gradient Boosting', 'gradient_boosting')
            ]

            for model_name, key in models_to_compare:
                if key in ml:
                    r2 = ml[key].get('r2', np.nan)
                    coverage = ml[key].get('coverage', np.nan)
                    uncertainty = ml[key].get('uncertainty', np.nan)
                    print(f"   {model_name:<24} | {r2:>7.4f} | {coverage:>9.2%} | {uncertainty:>13.4f}")
            
            comparison['ml_models'] = ml

        # 3. COMPARACIÓN DE ANÁLISIS DE SERIES TEMPORALES
        if 'time_series_analysis' in self.results:
            ts = self.results['time_series_analysis']
            
            print("\n📊 Análisis de Series Temporales:")
            print(f"   Estadístico CUSUM: {ts['structural_change']['cusum_statistic']:.4f}")
            print(f"   p-value CUSUM: {ts['structural_change']['cusum_pvalue']:.4f}")
            print(f"   Puntos de cambio detectados: {ts['structural_change']['changepoints_detected']}")
            
            comparison['time_series'] = ts

        print("\n✅ Comparación completada.")
        return comparison
