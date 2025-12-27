
# In[0]: Instalações de Pacotes

# pip install pandas
# pip install numpy
# pip install statsmodels --upgrade
# pip install scikit-learn
# pip install scipy
# pip install tqdm
# pip install bcb  # ou python-bcb (são o mesmo pacote)
# pip install matplotlib
# pip install seaborn --upgrade
# pip install patsy
# pip install pandas pyarrow


# In[1]: Importação das Bibliotecas
import pandas as pd
import numpy as np
import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf
# from statsmodels.formula.api import mixedlm
# from statsmodels.stats.diagnostic import acorr_ljungbox
# from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
# from statsmodels.stats.stattools import jarque_bera
# from statsmodels.stats.diagnostic import het_breuschpagan
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# import statsmodels.formula.api as smf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy.stats.mstats import winsorize
# from tqdm import tqdm
from bcb import sgs
import calendar
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.dates as mdates
# from patsy import dmatrices
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
import joblib

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# In[2]: Carregamento e Pré-processamento dos Dados
print("--- Etapa 2: Carregamento e Pré-processamento dos Dados ---")
df = pd.read_csv('banco_historico_novo.csv', sep=';', decimal=',')
df.to_parquet("dados_tcc_processados.parquet", engine="pyarrow")


df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')

numeric_cols = ['investimento', 'impressoes', 'cliques', 'visitas', 'pedidos']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df.fillna(0, inplace=True)

for col in numeric_cols:
    df[col] = df[col].apply(lambda x: max(0, x))
    
df = df[df['orientacao'].isin(['consideracao', 'conversao'])]

df = pd.get_dummies(df, columns=['orientacao'], drop_first=True, prefix='orient')

print("Dados carregados e limpos.\n")

# In[3]: Engenharia de Variáveis (com Fatores Macro-econômicos)

print("--- Etapa 3: Engenharia de Variáveis ---")

# Cria a nossa principal variável de agrupamento
df['midia_tipo'] = df['midia'].astype(str) + '_' + df['tipo_midia'].astype(str)

# Cria variável mês como categórica
df['mes'] = df['data'].dt.month.apply(lambda x: calendar.month_name[x]).astype('category')

# Garanta que a coluna 'data' está no formato datetime
df['data'] = pd.to_datetime(df['data'])

# --- Etapa 1: Definir os dias do mês que iniciam o "evento de pagamento" ---
# Estes são os dias mais comuns para recebimento de salários e vales.
dias_de_pagamento = [5, 15, 20, 30]

# --- Etapa 2: Definir a "janela" do evento ---
# Com base no seu insight de "crescente, pico, decrescente".
# Vamos considerar o dia anterior, o dia do evento, e os dois dias seguintes.
# Isso cria uma janela de 4 dias para cada evento.
dias_antes = 1
dias_depois = 2

# --- Etapa 3: Criar a "flag" (a nova variável) ---
# Crie uma flag para identificar Sábados (5) e Domingos (6)
df['flag_fim_de_semana'] = df['data'].dt.dayofweek.isin([5, 6])

# --- BUSCA CONSOLIDADA DE DADOS ECONÔMICOS DO BANCO CENTRAL ---
print("Buscando dados econômicos do Banco Central (IPCA e IBC-Br)...")

# Define as datas de início e fim para a consulta
start_date_data = df['data'].min()
end_date_today = pd.Timestamp.now()

# Dicionário com TODOS os códigos que precisamos
codigos_bcb = {
    'ipca': 433        # IPCA - para ajuste de inflação
    # 'ibc_br': 24369     # IBC-Br - Índice de Atividade Econômica
}

try:
    # Faz uma única chamada para buscar todas as séries de uma vez
    dados_economicos = sgs.get(codigos_bcb, start=start_date_data, end=end_date_today)
    if dados_economicos.empty: raise ValueError("Nenhum dado retornado pelo webservice do BCB.")
    
    # --- 1. TRATAMENTO DO IPCA PARA AJUSTE DE INVESTIMENTO ---
    # (Esta parte do seu código já estava perfeita)
    ipca_df = dados_economicos[['ipca']].copy()
    ipca_df['ipca'] = ipca_df['ipca'] / 100
    ipca_df['price_index'] = (1 + ipca_df['ipca']).iloc[::-1].cumprod().iloc[::-1]
    last_known_index = ipca_df['price_index'].iloc[-1]
    ipca_df['price_index'] = ipca_df['price_index'] / last_known_index
    
    df['mes_ano'] = df['data'].dt.to_period('M')
    ipca_df['mes_ano'] = ipca_df.index.to_period('M')
    
    if 'price_index' in df.columns: df = df.drop(columns=['price_index'])
    df = pd.merge(df, ipca_df[['mes_ano', 'price_index']], on='mes_ano', how='left')
    df['price_index'].fillna(method='ffill', inplace=True)
    df['price_index'].fillna(method='bfill', inplace=True)
    df['investimento_real'] = df['investimento'] / df['price_index']
    print("Investimento ajustado pela inflação (IPCA).")

    # # --- 2. TRATAMENTO DO IBC-Br (FATOR MACRO-ECONÔMICO) ---
    # ibc_br_df = dados_economicos[['ibc_br']].copy()
    
    # # Juntar os dados do IBC-Br de forma robusta
    # # Usamos merge_asof para alinhar os dados mensais com o seu dataframe diário
    # df = df.sort_values(by='data') # Garante a ordenação correta para o merge
    # df = pd.merge_asof(df, ibc_br_df, on='data', direction='backward')
    
    # # Preenche quaisquer valores nulos no início com o primeiro valor válido
    # df['ibc_br'].fillna(method='bfill', inplace=True)
    # print("Dados macro-econômicos (IBC-Br) integrados.")

except Exception as e:
    print(f"AVISO: Falha ao buscar ou processar dados econômicos: {e}. A análise continuará sem eles.")
    # Garante que as colunas existam mesmo em caso de falha
    if 'investimento_real' not in df.columns:
        df['investimento_real'] = df['investimento']
    if 'ibc_br' not in df.columns:
        df['ibc_br'] = 0

# --- CRIAÇÃO DAS VARIÁVEIS PREDITORAS FINAIS ---
df['log_investimento_real'] = np.log1p(df['investimento_real'])
df['log_impressoes'] = np.log1p(df['impressoes'])
df['log_cliques'] = np.log1p(df['cliques'])
df['log_visitas'] = np.log1p(df['visitas'])

# Geração de Termos de Fourier para modelar a sazonalidade (MANTIDO COMENTADO)
import re
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier
df_temp_fourier = df.drop_duplicates(subset=['data']).set_index('data').sort_index()
fourier_annual = CalendarFourier("A", 4)
# fourier_weekly = CalendarFourier("W", 3)
dp = DeterministicProcess(index=df_temp_fourier.index, order=0, seasonal=False, additional_terms=[fourier_annual])
fourier_terms = dp.in_sample()
clean_names = {col: re.sub(r'[(),=]', '', col).replace(' ', '_').replace('-', '_') for col in fourier_terms.columns}
fourier_terms.rename(columns=clean_names, inplace=True)
fourier_cols = list(fourier_terms.columns)
df = pd.merge(df, fourier_terms, on='data', how='left')
print("Termos de Fourier criados.")

# Criar vetor autocorrelacionado
df = df.sort_values(by=['midia_tipo', 'campanha', 'orient_conversao', 'data'])
df['pedidos_lag_1'] = df.groupby(['midia_tipo', 'campanha','orient_conversao'])['pedidos'].shift(1).fillna(0)
df['visitas_lag_1'] = df.groupby(['midia_tipo', 'campanha','orient_conversao'])['visitas'].shift(1).fillna(0)

# #CRIANDO MEDIA MOVEL 15
# print("Criando a média móvel de 30 dias para 'pedidos'...")
# df['pedidos_media_movel_30d'] = df.groupby('midia_tipo')['pedidos'].transform(
#     lambda x: x.rolling(window=30, min_periods=1).mean()
# )
# df['pedidos_media_movel_30d_lag_1'] = df.groupby('midia_tipo')['pedidos_media_movel_30d'].transform(
#     lambda x: x.shift(1)
# ).fillna(0)

# print("Variável 'pedidos_media_movel_30d_lag_1' criada.")

print("\nEngenharia de variáveis concluída.\n")

# In[4]: Modelagem de Ad Stock e Efeito Hill (Granular por Mídia-Tipo)
print("--- Etapa 4: Modelagem de Ad Stock e Efeito Hill ---")

def adstock(series, decay):
    """ Modela o efeito de memória/carryover da publicidade. """
    series = series.reset_index(drop=True)
    adstock_series = np.zeros_like(series, dtype=float)
    if not series.empty:
        adstock_series[0] = series.iloc[0]
        for i in range(1, len(series)):
            adstock_series[i] = series.iloc[i] + decay * adstock_series[i-1]
    return adstock_series

def hill(series, alpha, theta):
    """ Modela o efeito de saturação/retornos decrescentes. """
    theta = theta if theta > 0 else 1e-6
    return series**alpha / (series**alpha + theta**alpha)

# Definindo os hiperparâmetros para o grid search
lambda_grid = np.arange(0.00, 1.00, 0.05)   #[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
alpha_grid = np.arange(0.25, 5.00, 0.25) #[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
theta_grid_percentiles = np.arange(0.10, 1.00, 0.05) #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
best_params = {}
media_tipo_channels = df['midia_tipo'].unique()
df_transformed = df.copy()
df_transformed['hill_adstock'] = 0.00

# Loop para otimizar os parâmetros para cada canal individualmente
for channel in media_tipo_channels:
    print(f"Otimizando parâmetros para o canal: {channel}")
    media_mask = df['midia_tipo'] == channel
    # Garante que os dados do canal estão em ordem cronológica para o cálculo
    media_df = df[media_mask].copy().sort_values(by='data')
    
    investment_col = 'investimento_real'
    best_channel_params = {}
    if not (media_df.empty or media_df[investment_col].sum() == 0 or media_df[investment_col].nunique() <= 1 or media_df['pedidos'].nunique() <= 1):
        best_corr = -1
        temp_adstock = adstock(media_df[investment_col], 0.5)
        valid_adstock = temp_adstock[temp_adstock > 0]
        theta_grid = np.quantile(valid_adstock, theta_grid_percentiles) if len(valid_adstock) > 0 else [1]
        for l in lambda_grid:
            media_df['adstock'] = adstock(media_df[investment_col], l)
            for a in alpha_grid:
                for t in np.unique(theta_grid):
                    if t <= 0: continue
                    media_df['hill_adstock_temp'] = hill(media_df['adstock'], a, t)
                    correlation = np.corrcoef(media_df['hill_adstock_temp'], np.log1p(media_df['pedidos']))[0, 1]
                    if pd.notna(correlation) and correlation > best_corr:
                        best_corr = correlation
                        best_channel_params = {'lambda': l, 'alpha': a, 'theta': t}

    best_params[channel] = best_channel_params if best_channel_params else {'lambda': 0, 'alpha': 1, 'theta': 1}
    print(f"  - Melhores parâmetros: {best_params[channel]}")
    
    # Aplica os melhores parâmetros para criar a variável final
    final_adstock_values = adstock(media_df[investment_col], best_params[channel]['lambda'])
    final_hill_values = hill(final_adstock_values, best_params[channel]['alpha'], best_params[channel]['theta'])
    
    # Usa o índice para garantir que os valores voltem para as linhas corretas
    hill_series_to_assign = pd.Series(final_hill_values, index=media_df.index)
    df_transformed['hill_adstock'].update(hill_series_to_assign)
print("\nParâmetros de Ad Stock e Hill otimizados.\n")

# In[5]: Preparação Final dos Dados (com Divisão Cronológica)
print("--- Etapa 5: Preparação Final dos Dados ---")
df_model = df_transformed.copy()

# Preditores contínuos que serão escalados
continuous_predictors = ['hill_adstock'
                         , 'log_investimento_real'
                         , 'log_cliques'
                         , 'log_impressoes'
                         , 'log_visitas'
                         , 'pedidos_lag_1'
                         , 'visitas_lag_1']
# Winsorização para tratar outliers extremos sem removê-los
for col in continuous_predictors:
    df_model[col] = winsorize(df_model[col], limits=[0, 0.01])

# --- DIVISÃO TREINO-TESTE CRONOLÓGICA ---
# Esta é a única forma metodologicamente correta para modelos de série temporal,
# garantindo que o modelo aprenda com o passado para prever o futuro.
df_model = df_model.sort_values(by=['data', 'midia_tipo', 'campanha', 'orient_conversao']).reset_index(drop=True)
cutoff_date = df_model['data'].quantile(0.8, interpolation='nearest')
train_df = df_model[df_model['data'] <= cutoff_date].copy()
test_df = df_model[df_model['data'] > cutoff_date].copy()
print(f"Divisão Treino-Teste feita com data de corte em: {cutoff_date.date()}")

# Transformação Box-Cox para normalizar a variável resposta e estabilizar a variância.
pedidos_treino_pos = train_df['pedidos'] + 1
transformed_pedidos_treino, best_lambda = boxcox(pedidos_treino_pos)
train_df['pedidos_boxcox'] = transformed_pedidos_treino
test_df['pedidos_boxcox'] = boxcox(test_df['pedidos'] + 1, lmbda=best_lambda)
response_var = 'pedidos_boxcox'

# Lista final de preditores para o modelo
fixed_effects = ['hill_adstock:investimento_real'
                 , 'hill_adstock'
                 , 'flag_fim_de_semana'
                 , 'log_investimento_real'
                 , 'log_impressoes'
                 , 'log_cliques'
                 , 'orient_conversao'
                 , 'orient_conversao:hill_adstock'
                 , 'log_visitas'
                 , 'pedidos_lag_1'
                 , 'visitas_lag_1'] + fourier_cols

# Escalonamento dos dados para que as variáveis tenham escalas comparáveis.
# RobustScaler é usado por ser resistente a outliers.
scaler = RobustScaler()
train_df[continuous_predictors] = scaler.fit_transform(train_df[continuous_predictors])
test_df[continuous_predictors] = scaler.transform(test_df[continuous_predictors])
print("Dados divididos e preditores finalizados.\n")



# In[6]: Construção, Avaliação e Calibração de Viés do Modelo Final

print("--- Etapa 6: Construção, Avaliação e Calibração do Modelo Final ---")

# --- 1. TREINAMENTO DO MODELO FINAL ---
formula = f"{response_var} ~ {' + '.join(fixed_effects)}"
groups = train_df['midia_tipo'].astype(str) + "_" + train_df['mes'].astype(str)
vc_formula = {"campanha": "0 + C(campanha)"}
model = sm.MixedLM.from_formula(formula,
                                train_df,
                                groups=groups,
                                re_formula='~ hill_adstock',
                                vc_formula=vc_formula).fit()

print("\n--- Resumo do Modelo ---")
print(model.summary())

# --- 2. CÁLCULO DO ICC (Versão Final e Robusta) ---
try:
    vcomp_values = model.vcomp
    variancia_entre_grupos = vcomp_values[-1]
    variancia_dentro_grupos = model.scale
    icc = variancia_entre_grupos / (variancia_entre_grupos + variancia_dentro_grupos)
    print("\n--- Coeficiente de Correlação Intraclasse (ICC) ---")
    print(f"Variância Entre Grupos (campanha Var): {variancia_entre_grupos:.4f}")
    print(f"Variância Residual (scale):           {variancia_dentro_grupos:.4f}")
    print(f"ICC = {variancia_entre_grupos:.4f} / ({variancia_entre_grupos:.4f} + {variancia_dentro_grupos:.4f}) = {icc:.4f}")
    print("\n---------------------------------------------------------------------------------")
    print(f"Interpretação: {icc:.1%} da variabilidade total nos pedidos é explicada pelas diferenças inerentes entre as CAMPANHAS.")
    print("---------------------------------------------------------------------------------")

except (IndexError, AttributeError, KeyError) as e:
    print(f"\n--- ICC não pôde ser calculado. Erro: {e} ---")
    print("--- Verifique se o modelo convergiu e se os componentes de variância foram estimados corretamente. ---")

# --- 3. AVALIAÇÃO E CALIBRAÇÃO NO CONJUNTO DE TESTE ---
if not test_df.empty:
    y_true_raw = test_df['pedidos']

    # 3.1 Previsão Bruta (sem calibração)
    y_pred_transformed = model.predict(test_df)
    y_pred_uncalibrated = inv_boxcox(y_pred_transformed, best_lambda) - 1
    y_pred_uncalibrated.clip(0, inplace=True)
    
    print("\n--- Métricas de Avaliação (ANTES de Qualquer Calibração) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true_raw, y_pred_uncalibrated)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_raw, y_pred_uncalibrated):.4f}")
    print(f"R²: {r2_score(y_true_raw, y_pred_uncalibrated):.4f}")

    # 3.2 Calibração Flat (Multiplicativa) - Mantida para comparação
    print("\n--- Comparativo: Calibração Multiplicativa (Flat) ---")
    y_pred_train_transformed = model.predict(train_df)
    y_pred_train_raw = inv_boxcox(y_pred_train_transformed, best_lambda) - 1
    multiplicative_factor = train_df['pedidos'].sum() / y_pred_train_raw.clip(0).sum()
    y_pred_flat_calibrated = y_pred_uncalibrated * multiplicative_factor
    print(f"Fator de Correção Multiplicativo: {multiplicative_factor:.4f}")
    print(f"R² (Após Calibração Flat): {r2_score(y_true_raw, y_pred_flat_calibrated):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true_raw, y_pred_flat_calibrated)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_raw, y_pred_flat_calibrated):.4f}")

    # --- 3.3 Calibração Contínua com Regressão Isotônica (Método Otimizado) ---
    print("\n--- Calibração Contínua com Regressão Isotônica (Método Otimizado) ---")

    # 1. Preparar os dados para o *treinamento* do modelo de calibração.
    #    A Regressão Isotônica exige que os dados de treino ('x') estejam ordenados.
    df_calibracao_fit = pd.DataFrame({
        'real': y_true_raw,
        'previsto': y_pred_uncalibrated
    }).sort_values(by='previsto')

    # 2. Treinar o modelo de calibração.
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(df_calibracao_fit['previsto'], df_calibracao_fit['real'])
    print("Modelo de calibração (Regressão Isotônica) treinado com sucesso.")

    # 3. Aplicar o modelo treinado para corrigir as previsões *originais* (brutas e não ordenadas).
    y_pred_dynamic_calibrated = iso_reg.predict(y_pred_uncalibrated)
    
    print("\n--- Métricas de Avaliação (APÓS Calibração Isotônica) ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true_raw, y_pred_dynamic_calibrated)):.4f}")
    print(f"MAE: {mean_absolute_error(y_true_raw, y_pred_dynamic_calibrated):.4f}")
    print(f"R²: {r2_score(y_true_raw, y_pred_dynamic_calibrated):.4f}")

    # --- 4. DEFINIÇÃO DA PREVISÃO FINAL ---
    # A variável y_pred_raw conterá as previsões finais que serão usadas nas próximas etapas
    y_pred_raw = y_pred_dynamic_calibrated
    print("\n-------------------------------------------------------------------------")
    print(">>> As previsões foram finalizadas usando a CALIBRAÇÃO ISOTÔNICA OTIMIZADA. <<<")
    print(">>> O restante do script utilizará estes valores calibrados. <<<")
    print("-------------------------------------------------------------------------")

else:
    print("\nConjunto de teste vazio. Nenhuma métrica de avaliação para calcular.")


# In[6.1]: Salvando os Artefatos do Modelo

print("\n--- Etapa 6.1: Salvando os artefatos do modelo treinado ---")

# Garante que a pasta de artefatos exista
import os
os.makedirs('artifacts', exist_ok=True)

# Define os nomes dos arquivos
model_filename = 'artifacts/mmm_model.pkl'
scaler_filename = 'artifacts/mmm_scaler.pkl'
lambda_filename = 'artifacts/mmm_best_lambda.pkl'
params_filename = 'artifacts/mmm_adstock_hill_params.pkl'
data_filename = 'artifacts/dados_tcc_processados.parquet' # Vamos salvar os dados também

# Salva cada objeto em um arquivo
joblib.dump(model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(best_lambda, lambda_filename)
joblib.dump(best_params, params_filename)
df_transformed.to_parquet(data_filename) # Salva o dataframe processado

print(f"Modelo salvo em: {model_filename}")
print(f"Scaler salvo em: {scaler_filename}")
print(f"Lambda salvo em: {lambda_filename}")
print(f"Parâmetros Adstock/Hill salvos em: {params_filename}")
print(f"Dados processados salvos em: {data_filename}")
print("Treinamento concluído e artefatos salvos com sucesso.")



# # In[7]: Decomposição de Pedidos - Análise Agregada e Mensal (Versão Ajustada)

# # Este código assume que o pipeline completo (Etapas 1-6) já foi executado.

# # =============================================================================
# #                 FILTRO DE PERÍODO (OPCIONAL)
# start_date_filter = '2024-01-01'
# end_date_filter = '2025-05-31'

# try:
#     if start_date_filter and end_date_filter:
#         print(f"--- Iniciando Decomposição de Resultados para o período de {start_date_filter} a {end_date_filter} ---")
#         df_decomp = df_transformed[
#             (df_transformed['data'] >= start_date_filter) & 
#             (df_transformed['data'] <= end_date_filter)
#         ].copy()
#         period_title = f"({pd.to_datetime(start_date_filter).strftime('%b %Y')} a {pd.to_datetime(end_date_filter).strftime('%b %Y')})"
#     else:
#         raise NameError
# except NameError:
#     print("--- Iniciando Decomposição de Resultados para o PERÍODO COMPLETO ---")
#     df_decomp = df_transformed.copy()
#     period_title = "(Período Completo)"

# if df_decomp.empty:
#     print("\nERRO: Nenhum dado encontrado para o período especificado. Por favor, verifique as datas.")
# else:
#     # Passo 1: Calcular a Previsão Total com todas as mídias ativas
#     print("Calculando a previsão total para o período...")
#     df_decomp_scaled = df_decomp.copy()
#     df_decomp_scaled[continuous_predictors] = scaler.transform(df_decomp_scaled[continuous_predictors])
#     pred_total_transformed = model.predict(df_decomp_scaled)
#     pred_total_raw = inv_boxcox(pred_total_transformed, best_lambda) - 1
#     df_decomp['pred_total'] = pred_total_raw.clip(0)

#     # Passo 2: Calcular a contribuição incremental de cada canal
#     contributions = {}
#     # >> AJUSTE: Criamos um dataframe para guardar as contribuições diárias <<
#     daily_contributions_df = pd.DataFrame(index=df_decomp.index)
    
#     channels = df_decomp['midia_tipo'].unique()

#     for channel in tqdm(channels, desc="Calculando contribuição por canal"):
#         df_scenario = df_decomp.copy()
#         mask = df_scenario['midia_tipo'] == channel
#         df_scenario.loc[mask, 'log_investimento_real'] = 0
#         df_scenario.loc[mask, 'hill_adstock'] = 0
        
#         df_scenario_scaled = df_scenario.copy()
#         df_scenario_scaled[continuous_predictors] = scaler.transform(df_scenario_scaled[continuous_predictors])
        
#         pred_scenario_transformed = model.predict(df_scenario_scaled)
#         pred_scenario_raw = inv_boxcox(pred_scenario_transformed, best_lambda) - 1
        
#         contribution = (df_decomp['pred_total'] - pred_scenario_raw.clip(0)).clip(0)
        
#         # >> AJUSTE: Guardamos a contribuição diária E a agregada <<
#         daily_contributions_df[channel] = contribution # Guarda a série diária
#         contributions[channel] = contribution.sum()     # Mantém sua lógica original

#     # O resto do seu código original continua EXATAMENTE IGUAL
#     contribution_df = pd.DataFrame.from_dict(contributions, orient='index', columns=['Pedidos Incrementais'])
#     contribution_df = contribution_df.sort_values(by='Pedidos Incrementais', ascending=False)

#     total_pedidos_reais = df_decomp['pedidos'].sum()
#     total_incremental = contribution_df['Pedidos Incrementais'].sum()
#     baseline_pedidos = total_pedidos_reais - total_incremental
#     contribution_df.loc['Baseline'] = baseline_pedidos
    
#     if total_pedidos_reais > 0:
#         contribution_df['Porcentagem (%)'] = (contribution_df['Pedidos Incrementais'] / total_pedidos_reais) * 100
#     else:
#         contribution_df['Porcentagem (%)'] = 0

#     print(f"\n\n--- Tabela de Decomposição de Pedidos {period_title} ---")
#     print(contribution_df.round(2))

#     print("\nGerando gráfico de decomposição...")
#     plot_df = contribution_df.sort_values(by='Pedidos Incrementais', ascending=True)
#     fig, ax = plt.subplots(figsize=(12, 8))
#     plot_df['Pedidos Incrementais'].plot(kind='barh', ax=ax)
#     title_text = f"Decomposição de Pedidos {period_title}"
#     ax.set_title(title_text, fontsize=16)
#     ax.set_xlabel('Total de Pedidos Gerados no Período', fontsize=12)
#     ax.set_ylabel('Componente', fontsize=12)
#     for index, value in enumerate(plot_df['Pedidos Incrementais']):
#         ax.text(value, index, f' {value:,.0f}\n ({plot_df["Porcentagem (%)"].iloc[index]:.1f}%)', va='center')
#     plt.tight_layout()
#     plt.show()

#     # =============================================================================
#     # >> NOVO GRÁFICO: DECOMPOSIÇÃO MENSAL (100%) <<
#     # =============================================================================
#     print("\nGerando gráfico de decomposição mensal (100%)...")
    
#     # Calculamos o baseline diário a partir das contribuições diárias que guardamos
#     total_daily_incremental = daily_contributions_df.sum(axis=1)
#     daily_baseline = (df_decomp['pedidos'] - total_daily_incremental).clip(0)
#     daily_contributions_df['Baseline'] = daily_baseline
    
#     daily_contributions_df['data'] = df_decomp['data']
#     monthly_decomp = daily_contributions_df.groupby(pd.Grouper(key='data', freq='M')).sum()
#     monthly_totals = monthly_decomp.sum(axis=1)
#     # Evitar divisão por zero em meses sem pedidos
#     monthly_totals[monthly_totals == 0] = 1 
#     monthly_decomp_pct = monthly_decomp.divide(monthly_totals, axis=0) * 100

#     if 'Baseline' in monthly_decomp_pct.columns:
#         cols = ['Baseline'] + [col for col in monthly_decomp_pct.columns if col != 'Baseline' and col != 'data']
#         monthly_decomp_pct = monthly_decomp_pct[cols]

#     fig, ax = plt.subplots(figsize=(18, 9))
#     monthly_decomp_pct.plot(kind='bar', stacked=True, ax=ax, width=0.8, colormap='viridis')
    
#     ax.set_title(f'Decomposição Mensal de Contribuição (%) para Pedidos {period_title}', fontsize=18)
#     ax.set_ylabel('Contribuição Percentual (%)', fontsize=12)
#     ax.set_xlabel('Mês', fontsize=12)
#     ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
#     ax.set_xticklabels([d.strftime('%b %Y') for d in monthly_decomp_pct.index], rotation=45, ha='right')
#     ax.legend(title='Componentes', bbox_to_anchor=(1.02, 1), loc='upper left')
    
#     plt.tight_layout()
#     plt.show()

# # In[8]: PLOTS PARA ANALISE



# ##################################################
# # Variáveis Explicativas vs. Variável Objetivo####
# ##################################################

# # Selecionar as principais variáveis preditoras contínuas
# main_predictors = continuous_predictors

# # Criar um "pairplot" para visualizar todas as relações de uma vez
# # Usamos uma amostra dos dados para o gráfico não ficar muito pesado
# g = sns.pairplot(
#     train_df.sample(n=1000, random_state=1), 
#     x_vars=main_predictors,
#     y_vars=[response_var],
#     kind='reg', # 'reg' adiciona a linha de regressão
#     height=4,
#     plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.3}}
# )

# g.fig.suptitle('Relação entre Preditores e a Variável Resposta', y=1.03, fontsize=16)
# plt.show()


# # In[8.1]: Plotagem Real x Previsto

# ###########################
# ####  LINHA TEMPORAL   ####
# ###########################

# print("--- Análise Visual: Pedidos Reais vs. Pedidos Previstos (Gráfico de Linha Temporal) ---")

# # Criar um dataframe para a plotagem
# df_plot = pd.DataFrame({
#     'Data': test_df['data'],
#     'Pedidos Reais': y_true_raw,
#     'Pedidos Previstos': y_pred_raw
# })

# # Garantir que o dataframe está estritamente ordenado por data antes de plotar
# df_plot = df_plot.sort_values(by='Data')

# plt.figure(figsize=(18, 8))
# plt.plot(df_plot['Data'], df_plot['Pedidos Reais'], label='Pedidos Reais', color='royalblue', linewidth=2, alpha=0.8)
# plt.plot(df_plot['Data'], df_plot['Pedidos Previstos'], label='Pedidos Previstos pelo Modelo', color='red', linestyle='--', linewidth=2)

# plt.title('Comparação Temporal: Pedidos Reais vs. Previstos', fontsize=18)
# plt.xlabel('Data', fontsize=12)
# plt.ylabel('Número de Pedidos', fontsize=12)
# plt.legend(loc='upper left')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# ###########################
# # GRAFICO DE DISPERSAO ####
# ###########################

# print("\n--- Análise Visual: Pedidos Reais vs. Pedidos Previstos (Gráfico de Dispersão) ---")

# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=y_true_raw, y=y_pred_raw, alpha=0.5)
# plt.title('Diagnóstico de Precisão: Previsto vs. Real', fontsize=16)
# plt.xlabel('Pedidos Reais', fontsize=12)
# plt.ylabel('Pedidos Previstos pelo Modelo', fontsize=12)

# # Adiciona a linha de 45 graus (y=x), que representa a previsão perfeita
# max_val = max(y_true_raw.max(), y_pred_raw.max())
# plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Previsão Perfeita')

# plt.legend()
# plt.grid(True)
# plt.axis('equal') # Garante que as escalas dos eixos X e Y sejam as mesmas
# plt.tight_layout()
# plt.show()

# # In[8.2]: Teste de Resíduos

# print("\n--- Análise Temporal dos Resíduos do Modelo ---")

# # Calcular os resíduos
# residuos = y_true_raw - y_pred_raw


# # 1. Plotar o Gráfico ACF
# fig, ax = plt.subplots(figsize=(12, 6))
# plot_acf(residuos, ax=ax, lags=60) # Analisando até 40 dias de lag
# ax.set_title('Função de Autocorrelação (ACF) dos Resíduos')
# plt.show()

# # 2. Realizar o Teste de Ljung-Box -
# ljung_box_result = acorr_ljungbox(residuos, lags=[5, 10, 15, 20, 25, 30]) # Testando para lags de 5, 10, 15, 20, 25, 30 dias


# # Histograma dos resíduos
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# sns.histplot(residuos, kde=True, ax=axes[0], color='royalblue')
# axes[0].set_title('Histograma dos Resíduos do Modelo Otimizado')
# axes[0].set_xlim(-300, 300)

# # Q-Q Plot
# sm.qqplot(residuos, line='s', ax=axes[1])
# axes[1].set_title('Q-Q Plot dos Resíduos')

# plt.tight_layout()
# plt.show()

# print("\n--- Teste de Ljung-Box para Autocorrelação nos Resíduos ---")
# print(ljung_box_result)
# print("\nInterpretação:")
# print("A hipótese nula (H₀) é que não há autocorrelação.")
# print("Se os p-valores ('lb_pvalue') forem menores que 0.05, rejeitamos a H₀.")
# print("Isso indica que existem padrões nos erros que o modelo não capturou.")




# # In[9]: Comparação com Modelo Nulo
# # --- Treinando o Modelo Nulo (Apenas Intercepto) ---
# # Ele precisa da mesma estrutura de grupo e variância para ser comparável
# null_formula = f"{response_var} ~ 1" # "1" significa "apenas o intercepto"


# null_model = sm.MixedLM.from_formula(null_formula, 
#                                 train_df, 
#                                 groups=groups, 
#                                 re_formula='1',
#                                 vc_formula= vc_formula).fit()

# # --- Treinando novamente o nosso Modelo Vencedor (para garantir que temos o objeto 'model') ---
# # Supondo que 'fixed_effects' é a lista de preditores do modelo final
# winner_formula = f"{response_var} ~ {' + '.join(fixed_effects)}"
# winner_model = sm.MixedLM.from_formula(formula, 
#                                 train_df, 
#                                 groups=groups, 
#                                 re_formula='~ hill_adstock',
#                                 vc_formula= vc_formula).fit()


# # --- Comparação ---
# loglik_null = null_model.llf
# loglik_winner = winner_model.llf

# print("--- Comparação de Ajuste do Modelo (Log-Likelihood) ---")
# print(f"Log-Likelihood do Modelo Nulo (só a média): {loglik_null:,.4f}")
# print(f"Log-Likelihood do Modelo Vencedor: {loglik_winner:,.4f}")
# print("\n")

# if loglik_winner > loglik_null:
#     print("Conclusão: O Modelo Vencedor tem um ajuste aos dados significativamente superior.")
#     print("Um valor de Log-Likelihood mais alto (mais próximo de zero) é melhor.")
# else:
#     print("Atenção: O modelo vencedor não apresentou um ajuste melhor que o modelo nulo.")
    
# df_llf = pd.DataFrame({'modelo':['OLS Nulo','HLM2 Final'],
#                       'loglik':[null_model.llf,winner_model.llf]})

# fig, ax = plt.subplots(figsize=(15,15))

# c = ['dimgray','darkslategray']

# ax1 = ax.barh(df_llf.modelo,df_llf.loglik, color = c)
# ax.bar_label(ax1, label_type='center', color='white', fontsize=40)
# ax.set_ylabel("Modelo Proposto", fontsize=24)
# ax.set_xlabel("LogLik", fontsize=24)
# ax.tick_params(axis='y', labelsize=20)
# ax.tick_params(axis='x', labelsize=20)
# plt.show()



# # In[10]: ADSTOCK E SATURACAO - ANALISES GERAIS


# ######################
# ##     CURVAS S     ##
# ######################
   
# # Função Hill que usamos no modelo
# def hill(series, alpha, theta):
#     return series**alpha / (series**alpha + theta**alpha)

# # Definir os canais que queremos plotar (excluindo os que não tiveram dados)
# channels_to_plot = {k: v for k, v in best_params.items() if v.get('theta', 1) > 1}

# # Criar uma faixa de valores de Ad Stock para simulação
# max_adstock = max([v['theta'] for v in channels_to_plot.values()]) * 2
# adstock_range = np.linspace(0, max_adstock, 500)

# plt.figure(figsize=(15, 10))
# sns.set_style("whitegrid")

# for channel, params in channels_to_plot.items():
#     alpha = params.get('alpha', 1)
#     theta = params.get('theta', 1)
    
#     # Calcular a resposta da curva de Hill
#     response = hill(adstock_range, alpha, theta)
    
#     plt.plot(adstock_range, response, label=f"{channel}\n(α={alpha:.2f}, θ={theta:,.0f})")

# plt.title('Curvas de Saturação (Função Hill) por Canal', fontsize=18)
# plt.xlabel('Nível de Ad Stock Acumulado', fontsize=12)
# plt.ylabel('Resposta de Marketing (Saturação)', fontsize=12)
# plt.legend(title='Canal (Mídia-Tipo)', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# # In[10.1]: ADSTOCK E SATURACAO -  AVALIAÇÃO INDIVIDUAL

# # =============================================================================
# #           AVALIAÇÃO INDIVIDUAL - CONSULTA
# # =============================================================================
# # Escolha o canal (midia_tipo) que você quer analisar aqui
# canal_para_analisar = 'midia1_tipo1' # Ex: 'bing_search' (alpha>1) ou 'google_search' (alpha<1)
# # =============================================================================

# try:
#     # --- CÁLCULO DO PONTO DE INFLEXÃO PARA TODOS OS CANAIS ---
#     inflection_points = {}
#     print("--- Cálculo do Ponto de Inflexão (Máximo Retorno Marginal) ---")
#     for channel, params in best_params.items():
#         alpha = params.get('alpha', 1)
#         theta = params.get('theta', 1)
        
#         if alpha > 1:
#             inflection_adstock = theta * ((alpha - 1) / (alpha + 1))**(1 / alpha)
#             inflection_points[channel] = inflection_adstock
#             print(f"Canal: {channel:<25} Ponto de Inflexão em Ad Stock = {inflection_adstock:,.2f} (Sweet Spot)")
#         else:
#             inflection_points[channel] = 0
#             print(f"Canal: {channel:<25} (α <= 1) Máxima eficiência marginal no início.")
#     print("-" * 70)


#     # --- GRÁFICO 1: CURVA DE SATURAÇÃO (S-CURVE) ---
    
#     params = best_params[canal_para_analisar]
#     lam, alpha, theta = params.get('lambda', 0), params.get('alpha', 1), params.get('theta', 1)
#     inflection_point = inflection_points.get(canal_para_analisar, 0)

#     investment_series = df_transformed[df_transformed['midia_tipo'] == canal_para_analisar]['investimento_real']
#     historical_adstock = adstock(investment_series, lam)
#     historical_hill = df_transformed[df_transformed['midia_tipo'] == canal_para_analisar]['hill_adstock']
#     historical_dates = df_transformed[df_transformed['midia_tipo'] == canal_para_analisar]['data']
    
#     global_min_date_num = mdates.date2num(df_transformed['data'].min())
#     global_max_date_num = mdates.date2num(df_transformed['data'].max())
#     adstock_range = np.linspace(0, max(theta * 2, np.max(historical_adstock) * 1.1), 500)
#     response_curve = hill(adstock_range, alpha, theta)

#     fig1, ax1 = plt.subplots(figsize=(15, 8))
#     ax1.plot(adstock_range, response_curve, color='black', linestyle='--', label='Curva de Saturação Teórica')
#     scatter = ax1.scatter(historical_adstock, historical_hill, c=mdates.date2num(historical_dates), 
#                            cmap='viridis', alpha=0.6, label='Posição Histórica Diária',
#                            vmin=global_min_date_num, vmax=global_max_date_num)
#     ax1.axvline(x=theta, color='red', linestyle='-.', linewidth=2, label=f'Theta (θ) = {theta:,.0f}\n(Ponto de 50% Saturação)')
    
#     if alpha > 1:
#         ax1.axvline(x=inflection_point, color='green', linestyle=':', linewidth=2.5, label=f'Ponto de Inflexão = {inflection_point:,.0f}\n(Máximo Retorno Marginal)')

#     ax1.set_title(f'Análise de Saturação e Eficiência para: {canal_para_analisar}', fontsize=18, pad=20)
#     ax1.set_xlabel('Ad Stock Acumulado', fontsize=12)
#     ax1.set_ylabel('Resposta de Marketing (Saturação)', fontsize=12)
#     ax1.legend(loc='best')

#     cbar = fig1.colorbar(scatter)
#     cbar.set_label('Período (do mais antigo ao mais recente)')
#     tick_dates_dt = pd.date_range(start=df_transformed['data'].min(), end=df_transformed['data'].max(), periods=5)
#     cbar.set_ticks(mdates.date2num(tick_dates_dt))
#     cbar.set_ticklabels([d.strftime('%b %Y') for d in tick_dates_dt])
#     fig1.tight_layout()


#     # --- GRÁFICO 2: EVOLUÇÃO DO AD STOCK NO TEMPO (COM INFLEXÃO) ---

#     channel_df = df_transformed[df_transformed['midia_tipo'] == canal_para_analisar].copy().sort_values(by='data')
#     channel_df['historical_adstock'] = adstock(channel_df['investimento_real'], lam)
    
#     fig2, ax1_ts = plt.subplots(figsize=(18, 8))
    
#     color_adstock = 'royalblue'
#     ax1_ts.set_xlabel('Data', fontsize=12)
#     ax1_ts.set_ylabel('Ad Stock Acumulado', color=color_adstock, fontsize=12)
#     ax1_ts.plot(channel_df['data'], channel_df['historical_adstock'], color=color_adstock, label='Ad Stock Acumulado')
#     ax1_ts.tick_params(axis='y', labelcolor=color_adstock)
    
#     ax1_ts.axhline(y=theta, color='red', linestyle='-.', linewidth=2, label=f'Theta (θ) = {theta:,.0f} (Ponto de 50% Saturação)')
    
#     # >>> LINHA ADICIONADA AQUI <<<
#     # Adicionar a linha do Ponto de Inflexão se for relevante (alpha > 1)
#     if alpha > 1:
#         ax1_ts.axhline(y=inflection_point, color='green', linestyle=':', linewidth=2.5, label=f'Ponto de Inflexão = {inflection_point:,.0f} (Máx. Eficiência Marginal)')
    
#     ax2_ts = ax1_ts.twinx()
#     color_invest = 'grey'
#     ax2_ts.set_ylabel('Investimento Real Diário (R$)', color=color_invest, fontsize=12)
#     ax2_ts.bar(channel_df['data'], channel_df['investimento_real'], color=color_invest, alpha=0.3, label='Investimento Real Diário')
#     ax2_ts.tick_params(axis='y', labelcolor=color_invest)
#     ax2_ts.grid(False)

#     fig2.suptitle(f'Evolução do Ad Stock vs. Investimento para: {canal_para_analisar}', fontsize=18)
#     lines, labels = ax1_ts.get_legend_handles_labels()
#     lines2, labels2 = ax2_ts.get_legend_handles_labels()
#     ax2_ts.legend(lines + lines2, labels + labels2, loc='upper left')
#     fig2.tight_layout(rect=[0, 0, 1, 0.96])

#     # Exibe os dois gráficos gerados
#     plt.show()

# except KeyError:
#     print(f"\nERRO: O canal '{canal_para_analisar}' não foi encontrado.")
#     print("Canais disponíveis:", df_transformed['midia_tipo'].unique())
# except NameError as e:
#     print(f"\nERRO: Uma variável necessária não foi encontrada: {e}")
#     print("Por favor, certifique-se de que as Etapas 1-4 foram executadas.")
    
# # In[11]: DIAGNÓSTICO FINAL

    
# # --- PREPARAÇÃO: CALCULAR OS RESÍDUOS ---
# # Usaremos os resíduos do conjunto de TREINO, pois é com eles que o modelo foi construído.
# y_pred_train_transformed = model.predict(train_df)
# residuals_transformed = train_df[response_var] - y_pred_train_transformed

# # --- TESTE 1: NORMALIDADE DOS RESÍDUOS ---
# print("--- Teste 1: Normalidade dos Resíduos ---")
# # Histograma dos resíduos
# fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# sns.histplot(residuals_transformed, kde=True, ax=axes[0], color='royalblue')
# axes[0].set_title('Histograma dos Resíduos do Modelo Otimizado - Conjunto Treino')

# # Q-Q Plot
# sm.qqplot(residuals_transformed, line='s', ax=axes[1])
# axes[1].set_title('Q-Q Plot dos Resíduos')

# plt.tight_layout()
# plt.show()

# # Teste de Jarque-Bera
# jb_stat, jb_pvalue, _, _ = jarque_bera(residuals_transformed)
# print(f"Estatística de Jarque-Bera: {jb_stat:.2f}")
# print(f"P-valor: {jb_pvalue:.4f}")
# if jb_pvalue > 0.05:
#     print("Resultado: Os resíduos parecem ser normalmente distribuídos (p > 0.05). Bom sinal!")
# else:
#     print("Resultado: Os resíduos NÃO parecem ser normalmente distribuídos (p <= 0.05).")
    

# # --- TESTE 2: HOMOCEDASTICIDADE ---
# print("\n--- Teste 2: Homocedasticidade dos Resíduos ---")
# # Gráfico de Resíduos vs. Previstos
# plt.scatter(y_pred_train_transformed, residuals_transformed, alpha=0.3)
# plt.axhline(0, color='red', linestyle='--')
# plt.title('Resíduos vs. Valores Previstos')
# plt.xlabel('Valores Previstos (Transformados)')
# plt.ylabel('Resíduos (Transformados)')
# plt.show()

# # Teste de Breusch-Pagan
# # Precisamos da matriz de design (X) do modelo para o teste
# y_bp, X_bp = dmatrices(formula, data=train_df, return_type='dataframe')
# bp_test = het_breuschpagan(residuals_transformed, X_bp)
# print(f"Estatística de Breusch-Pagan: {bp_test[0]:.2f}")
# print(f"P-valor: {bp_test[1]:.4f}")
# if bp_test[1] > 0.05:
#     print("Resultado: Os resíduos parecem ser homocedásticos (p > 0.05). Ótimo sinal!")
# else:
#     print("Resultado: Os resíduos são heterocedásticos (p <= 0.05). O erro varia com a previsão.")


# # --- TESTE 3: MULTICOLINEARIDADE (VIF) ---
# print("\n--- Teste 3: Multicolinearidade (VIF) ---")

# # 1. Criar a matriz de design (X) a partir da fórmula do modelo.
# #    É crucial usar a mesma fórmula e os mesmos dados do modelo final.
# #    O VIF precisa de um intercepto, e dmatrices adiciona um por padrão.
# print("Criando matriz de design para o cálculo do VIF...")
# y, X = dmatrices(formula, data=train_df, return_type='dataframe')

# # 2. Calcular o VIF para cada variável preditora em X.
# vif_data = pd.DataFrame()
# # Usamos X.columns para pegar o nome de todas as variáveis preditoras
# vif_data["feature"] = X.columns
# vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# print(vif_data.sort_values(by='VIF', ascending=False))
# print("\nInterpretação: Geralmente, VIF > 5 é um sinal de alerta e VIF > 10 é problemático.")


# # --- APRIMORAMENTO DO GRÁFICO ---

# # 3. Preparar dados para o plot (remover o intercepto, que sempre tem VIF alto)
# vif_plot = vif_data[vif_data["feature"] != 'Intercept'].sort_values('VIF', ascending=False)

# # 4. Aplicar um estilo visual limpo
# sns.set_style("whitegrid")
# plt.style.use('seaborn-v0_8-whitegrid')

# # 5. Criar a figura e os eixos
# fig, ax = plt.subplots(figsize=(12, 8))

# # 6. Criar o gráfico de barras horizontal
# #    Usamos 'feature' no eixo y, que é o nome correto da coluna
# barplot = sns.barplot(data=vif_plot, x='VIF', y='feature', palette='mako', ax=ax)

# # 7. Adicionar linhas de referência e anotações
# ax.axvline(5, color='red', linestyle='--', linewidth=1.5, label='Alerta de Multicolinearidade (VIF = 5)')
# ax.axvline(10, color='darkred', linestyle=':', linewidth=2, label='Multicolinearidade Problemática (VIF = 10)')

# # Adicionar os valores exatos do VIF no final de cada barra
# for i in barplot.patches:
#     ax.text(i.get_width() + 0.1, i.get_y() + 0.5, 
#             f'{i.get_width():.2f}', 
#             ha='left', va='center', fontsize=10)

# # 8. Melhorar Títulos e Legendas
# ax.set_title('Análise de Multicolinearidade (VIF) dos Preditores', fontsize=18, fontweight='bold')
# ax.set_xlabel('Fator de Inflação da Variância (VIF)', fontsize=12)
# ax.set_ylabel('Variáveis Preditoras', fontsize=12)
# ax.tick_params(axis='both', which='major', labelsize=10)
# ax.legend()

# # 9. Limpar o visual
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.set_xlim(0, max(vif_plot['VIF'].max() * 1.15, 11)) # Ajustar o limite do eixo x para caber as anotações

# # 10. Finalizar e mostrar
# plt.tight_layout()
# plt.show()




# # In[Analise Exploratória]: Séries Temporais

# df['investimento'].describe()

# # Preparando as séries temporais
# bd_investimento_ts = df.groupby('data')['investimento'].sum()
# bd_impressoes_ts = df.groupby('data')['impressoes'].sum()
# bd_cliques_ts = df.groupby('data')['cliques'].sum()
# bd_visitas_ts = df.groupby('data')['visitas'].sum()
# bd_pedidos_ts = df.groupby('data')['pedidos'].sum()
# bd_ad_stock_ts = df_transformed.groupby('data')['hill_adstock'].mean()

# # Função auxiliar para plotar série + linha de tendência
# def plot_com_tendencia(ax, serie, titulo, ylabel):
#     x = np.arange(len(serie))
#     y = serie.values
#     coef = np.polyfit(x, y, 1)
#     tendencia = np.poly1d(coef)
#     ax.plot(serie.index, y, label='Valor real')
#     ax.plot(serie.index, tendencia(x), color='red', linestyle='--', label='Tendência')
#     ax.set_title(titulo)
#     ax.set_xlabel('Data')
#     ax.set_ylabel(ylabel)
#     ax.legend()

# # Criando subplots 2x2
# fig, axs = plt.subplots(2, 3, figsize=(14, 10))
# fig.suptitle('Análise de Variáveis com Tendência', fontsize=16)

# plot_com_tendencia(axs[0, 0], bd_investimento_ts, 'Investimento Diário', 'Investimento')
# plot_com_tendencia(axs[0, 1], bd_impressoes_ts, 'Impressões Diárias', 'Impressões')
# plot_com_tendencia(axs[0, 2], bd_cliques_ts, 'Cliques Diários', 'Cliques')
# plot_com_tendencia(axs[1, 0], bd_visitas_ts, 'Visitas DIárias', 'Visitas')
# plot_com_tendencia(axs[1, 1], bd_pedidos_ts, 'Pedidos Diários', 'Pedidos')
# plot_com_tendencia(axs[1, 2], bd_ad_stock_ts, 'Capacidade de Mídia', 'hill_adstock')


# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# # In[Analise Exploratória]: Representatividade de Mídia


# #GRAFICOS POR MIDIA NORMALIZADOS
# # Agrupa e calcula média
# agg = df.groupby('midia_tipo').mean(numeric_only=True).reset_index()

# # Normaliza para percentual por métrica
# metrics = ['investimento', 'impressoes', 'cliques', 'pedidos']
# for m in metrics:
#     agg[m + '_pct'] = 100 * agg[m] / agg[m].sum()

# # Plot 2x2 para as métricas percentuais
# fig, axs = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle('Participação Percentual Média por Mídia', fontsize=18)

# sns.barplot(x='investimento_pct', y='midia_tipo', data=agg, ax=axs[0, 0], color='skyblue')
# axs[0, 0].set_title('Investimento (%)')
# axs[0, 0].set_xlabel('Percentual (%)')

# sns.barplot(x='impressoes_pct', y='midia_tipo', data=agg, ax=axs[0, 1], color='salmon')
# axs[0, 1].set_title('Impressões (%)')
# axs[0, 1].set_xlabel('Percentual (%)')

# sns.barplot(x='cliques_pct', y='midia_tipo', data=agg, ax=axs[1, 0], color='mediumseagreen')
# axs[1, 0].set_title('Cliques (%)')
# axs[1, 0].set_xlabel('Percentual (%)')

# sns.barplot(x='pedidos_pct', y='midia_tipo', data=agg, ax=axs[1, 1], color='orchid')
# axs[1, 1].set_title('Pedidos (%)')
# axs[1, 1].set_xlabel('Percentual (%)')

# for ax in axs.flat:
#     ax.set_ylabel('')
#     ax.tick_params(labelsize=11)

# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


# # In[Analise Exploratória]: BoxPlot

# fig, axs = plt.subplots(2, 2, figsize=(18, 14))
# fig.suptitle('Boxplot das principais variáveis', fontsize=24, fontweight='bold')

# # Visitas
# sns.boxplot(data=df_transformed, y='visitas', ax=axs[0,0],
#             linewidth=2, color='deepskyblue', orient='v')
# sns.stripplot(data=df_transformed, y='visitas', ax=axs[0,0],
#               color='darkorange', jitter=0.1, size=8, alpha=0.5)
# axs[0,0].set_ylabel('Visitas', fontsize=18, fontweight='semibold')
# axs[0,0].set_title('Visitas', fontsize=20, fontweight='semibold')
# axs[0,0].tick_params(axis='y', labelsize=14)

# # Pedidos
# sns.boxplot(data=df_transformed, y='pedidos', ax=axs[0,1],
#             linewidth=2, color='deepskyblue', orient='v')
# sns.stripplot(data=df_transformed, y='pedidos', ax=axs[0,1],
#               color='darkorange', jitter=0.1, size=8, alpha=0.5)
# axs[0,1].set_ylabel('Total de Pedidos', fontsize=18, fontweight='semibold')
# axs[0,1].set_title('Total de Pedidos', fontsize=20, fontweight='semibold')
# axs[0,1].tick_params(axis='y', labelsize=14)

# # Impressões
# sns.boxplot(data=df_transformed, y='impressoes', ax=axs[1,0],
#             linewidth=2, color='deepskyblue', orient='v')
# sns.stripplot(data=df_transformed, y='impressoes', ax=axs[1,0],
#               color='darkorange', jitter=0.1, size=8, alpha=0.5)
# axs[1,0].set_ylabel('Impressões', fontsize=18, fontweight='semibold')
# axs[1,0].set_title('Impressões', fontsize=20, fontweight='semibold')
# axs[1,0].tick_params(axis='y', labelsize=14)

# # Investimento
# sns.boxplot(data=df_transformed, y='investimento', ax=axs[1,1],
#             linewidth=2, color='deepskyblue', orient='v')
# sns.stripplot(data=df_transformed, y='investimento', ax=axs[1,1],
#               color='darkorange', jitter=0.1, size=8, alpha=0.5)
# axs[1,1].set_ylabel('Investimento', fontsize=18, fontweight='semibold')
# axs[1,1].set_title('Investimento', fontsize=20, fontweight='semibold')
# axs[1,1].tick_params(axis='y', labelsize=14)

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()


# # In[Analise Exploratória]: Correlação entre variáveis independentes

# vars_exp = ['investimento', 'impressoes', 'cliques', 'visitas', 'pedidos', 'hill_adstock']

# # Seleciona as variáveis e remove NaN
# df_corr = df_transformed[vars_exp].dropna()

# # Calcula a matriz de correlação
# corr_matrix = df_corr.corr()

# # Exibe a matriz
# print(corr_matrix)

# plt.figure(figsize=(10, 8))
# sns.set(style="whitegrid")

# # Heatmap com melhorias estéticas
# sns.heatmap(
#     corr_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap='coolwarm',
#     linewidths=0.5,
#     linecolor='white',
#     annot_kws={"size": 12},
#     cbar_kws={'shrink': 0.8, 'label': 'Correlação'}
# )

# plt.xticks(fontsize=12, rotation=45, ha='right')
# plt.yticks(fontsize=12)
# plt.title('Matriz de Correlação entre Variáveis Contínuas', fontsize=16, pad=20)
# plt.tight_layout()
# plt.show()


# target = 'pedidos'

# df_corr = df_transformed[vars_exp + [target]].dropna()

# correlations = df_corr.corr()[target].drop(target)

# print(correlations)

# # In[Analise Exploratória]: Estatisticas Descrivivas

# # --- Tabela de Estatísticas Descritivas ---

# # 1. Lista das colunas que queremos analisar
# colunas_para_analise = ['investimento', 'impressoes', 'cliques', 'visitas', 'pedidos']

# # 2. Verificar se todas as colunas existem no DataFrame
# colunas_existentes = [col for col in colunas_para_analise if col in df.columns]

# if colunas_existentes:
#     # 3. Aplicar o método .describe()
#     tabela_descritiva = df[colunas_existentes].describe()

#     # 4. Imprimir a tabela diretamente (sem a formatação .style)
#     print("--- Tabela de Estatísticas Descritivas ---")
#     print(tabela_descritiva)
# else:
#     print("Nenhuma das colunas especificadas foi encontrada no DataFrame.")


# In[Analise Exploratória]: Export Model Summary
# =============================================================================
# print("\n--- Gerando imagens com os resultados do modelo ---")

# try:
#     # --- Passo 1: Extrair a tabela principal do sumário ---
#     results_summary = model.summary()
    
#     if len(results_summary.tables) < 2:
#         raise ValueError("O sumário do modelo não tem a estrutura de tabelas esperada.")

#     # A tabela principal (índice 1) contém TUDO: efeitos fixos e aleatórios
#     main_df = results_summary.tables[1].copy()
    
#     # Limpar os nomes das colunas e o índice
#     main_df.columns = ['Valor', 'Std.Err.', 'z', 'P>|z|', '[0.025', '0.975]']
#     main_df.index.name = 'Parâmetro'

#     # --- Passo 2: Separar de forma inteligente os Efeitos Fixos e Aleatórios ---
    
#     # Efeitos Aleatórios (variâncias) são as linhas cujo nome contém " Var" ou " Cov"
#     is_random_effect = main_df.index.str.contains(' Var| Cov', na=False)
    
#     random_effects_df = main_df[is_random_effect]
#     fixed_effects_df = main_df[~is_random_effect] # O '~' inverte a seleção

#     # Renomear colunas da tabela de efeitos aleatórios para clareza
#     random_effects_df = random_effects_df[['Valor', 'Std.Err.']]

#     # --- Passo 3: Função para converter um DataFrame em uma imagem de tabela ---
#     # (Esta função não precisa de alterações)
#     def criar_imagem_tabela(df, titulo, nome_arquivo):
#         df_plot = df.round(3)
#         fig, ax = plt.subplots(figsize=(10, len(df_plot) * 0.4 + 1.2)) # Aumentar espaço para o título
#         ax.axis('off')
#         tabela = ax.table(cellText=df_plot.values,
#                           colLabels=df_plot.columns,
#                           rowLabels=df_plot.index,
#                           cellLoc='center',
#                           loc='center')
#         tabela.auto_set_font_size(False)
#         tabela.set_fontsize(10)
#         tabela.scale(1.2, 1.2)
#         for (row, col), cell in tabela.get_celld().items():
#             if row == 0 or col == -1:
#                 cell.set_text_props(weight='bold')
#         plt.title(titulo, weight='bold', size=16, pad=20) # Adicionar 'pad' para espaçamento
#         plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight', pad_inches=0.2)
#         plt.close()
#         print(f"Tabela de resultados salva com sucesso como '{nome_arquivo}'")

#     # --- Passo 4: Gerar e salvar as imagens ---
#     criar_imagem_tabela(fixed_effects_df, 
#                         titulo='Tabela 1: Resultados do Modelo - Efeitos Fixos', 
#                         nome_arquivo='tabela_resultados_efeitos_fixos.png')

#     criar_imagem_tabela(random_effects_df, 
#                         titulo='Tabela 2: Resultados do Modelo - Efeitos Aleatórios', 
#                         nome_arquivo='tabela_resultados_efeitos_aleatorios.png')

# except Exception as e:
#     print(f"\nAVISO: Não foi possível gerar as imagens da tabela: {e}")
    
# In[Analise Exploratória]: Investigação Resíduos

# print("--- Investigando os Dias com Maiores Erros do Modelo ---")

# # --- Passo 1 e 2: Criar um DataFrame de análise com os resíduos e as datas ---
# df_residuos = pd.DataFrame({
#     'data': train_df['data'],
#     'pedidos_reais': train_df['pedidos'],
#     'pedidos_previstos_transf': model.fittedvalues,
#     'residuos_transf': model.resid
# })

# # Calcula o valor absoluto do resíduo para ranquear os maiores erros
# df_residuos['residuos_abs_transf'] = np.abs(df_residuos['residuos_transf'])

# # Ordena o DataFrame para mostrar os maiores erros primeiro
# df_residuos = df_residuos.sort_values(by='residuos_abs_transf', ascending=False)


# # --- Passo 3: Identificar e listar os dias com os maiores erros ---
# N_DIAS_PARA_ANALISAR = 60
# top_erros = df_residuos.head(N_DIAS_PARA_ANALISAR)

# print(f"\n--- TOP {N_DIAS_PARA_ANALISAR} DIAS COM MAIORES ERROS (RESÍDUOS) ---")
# print(top_erros[['data', 'pedidos_reais','pedidos_previstos_transf', 'residuos_transf']])


# # --- Passo 4: Analisar as características desses dias para encontrar o padrão ---
# print(f"\n--- ANÁLISE DE PADRÕES NOS TOP {N_DIAS_PARA_ANALISAR} DIAS DE ERRO ---")

# # Analisa a frequência por dia da semana
# print("\nFrequência por Dia da Semana:")
# print(top_erros['data'].dt.day_name().value_counts())

# # Analisa a frequência por mês
# print("\nFrequência por Mês:")
# print(top_erros['data'].dt.month_name().value_counts())

# # Analisa a frequência por dia do mês
# print("\nFrequência por Dia do Mês:")
# print(top_erros['data'].dt.day.value_counts())


# # # Crie uma flag para identificar Terca (1) e Quarta (2)
# # df_model['flag_terca_quarta'] = df_model['data'].dt.dayofweek.isin([1, 2])

# # # Verifique se funcionou
# # print("Média de pedidos por tipo de dia:")
# # print(df_model.groupby('flag_terca_quarta')['pedidos'].mean())
# In[X]: Módulo de Otimização de Orçamento
# print("--- Etapa 7: Módulo de Otimização de Orçamento ---")

# def predict_total_orders(investments_nominal, channels_to_optimize, last_adstocks, best_params, scaler, model, boxcox_lambda, latest_price_index, dp):
#     avg_metrics = df[['log_impressoes', 'log_cliques', 'log_visitas']].mean().to_dict()
    
#     future_date = pd.Timestamp.now()
#     pred_df = pd.DataFrame(index=range(len(channels_to_optimize)))
#     pred_df['data'] = future_date
#     pred_df['midia_tipo'] = channels_to_optimize
#     pred_df['campanha'] = 'optimization_scenario'
#     pred_df['orient_conversao'] = 1
    
#     for k, v in avg_metrics.items():
#         pred_df[k] = v
        
#     future_fourier_terms = dp.out_of_sample(steps=1, index=[future_date])
#     clean_names = {}
#     for col in future_fourier_terms.columns:
#         new_name = re.sub(r'[(),=]', '', col).replace(' ', '_').replace('-', '_')
#         clean_names[col] = new_name
#     future_fourier_terms.rename(columns=clean_names, inplace=True)

#     for col in future_fourier_terms.columns:
#         pred_df[col] = future_fourier_terms[col].values[0]

#     hill_values = []
#     investments_real_log = []
#     for i, channel in enumerate(channels_to_optimize):
#         investment_nominal = investments_nominal[i]
#         investment_real = investment_nominal / latest_price_index
#         investments_real_log.append(np.log1p(investment_real))
#         params = best_params.get(channel, {})
#         last_adstock = last_adstocks.get(channel, 0)
#         new_adstock = investment_real + params.get('lambda', 0) * last_adstock
#         new_hill = hill(new_adstock, params.get('alpha', 1), params.get('theta', 1))
#         hill_values.append(new_hill)
        
#     pred_df['hill_adstock'] = hill_values
#     pred_df['log_investimento_real'] = investments_real_log

#     pred_df_scaled = pred_df.copy()
#     pred_df_scaled[continuous_predictors] = scaler.transform(pred_df_scaled[continuous_predictors])
#     for col in continuous_ predictors:
#         pred_df_scaled[col] = winsorize(pred_df_scaled[col], limits=[0, 0.01])
    
#     transformed_predictions = model.predict(pred_df_scaled)
#     raw_predictions = inv_boxcox(transformed_predictions, boxcox_lambda) - 1
#     raw_predictions[raw_predictions < 0] = 0
#     return np.sum(raw_predictions)

# def optimization_objective(investments, *args):
#     return -predict_total_orders(investments, *args)

# def optimize_budget(total_budget, channels_to_optimize, last_adstocks, best_params, scaler, model, boxcox_lambda, latest_price_index, dp):
#     num_channels = len(channels_to_optimize)
#     constraints = ({'type': 'eq', 'fun': lambda x: total_budget - np.sum(x)})
#     bounds = [(0, total_budget) for _ in range(num_channels)]
#     initial_guess = [total_budget / num_channels] * num_channels
#     args_for_optimizer = (channels_to_optimize, last_adstocks, best_params, scaler, model, boxcox_lambda, latest_price_index, dp)
#     result = minimize(optimization_objective, initial_guess, args=args_for_optimizer, method='SLSQP', bounds=bounds, constraints=constraints)
#     if result.success:
#         return result.x, -result.fun
#     else:
#         raise ValueError("Otimização falhou: " + result.message)

# # --- Exemplo de Uso do Otimizador ---
# channels_to_optimize = df['midia_tipo'].unique()
# last_month_start = df['data'].max() - pd.DateOffset(days=30)
# current_allocation_df = df[df['data'] > last_month_start].groupby('midia_tipo')['investimento'].sum()
# total_budget_example = current_allocation_df.sum()
# print(f"\nExemplo de Otimização com Orçamento Total de R$ {total_budget_example:,.2f} entre {len(channels_to_optimize)} canais")
# try:
#     optimal_investments, max_orders = optimize_budget(
#         total_budget_example, channels_to_optimize, last_adstocks, 
#         best_params, scaler, model, best_lambda, latest_price_index_value, dp
#     )
#     optimal_allocation = pd.Series(optimal_investments, index=channels_to_optimize)
#     current_investments_array = current_allocation_df.reindex(channels_to_optimize).fillna(0).values
#     current_predicted_orders = predict_total_orders(
#         current_investments_array, channels_to_optimize, last_adstocks, 
#         best_params, scaler, model, best_lambda, latest_price_index_value, dp
#     )
#     comparison_df = pd.DataFrame({
#         "Alocação Atual (R$)": current_allocation_df,
#         "Alocação Ótima (R$)": optimal_allocation
#     })
#     comparison_df.fillna(0, inplace=True)
#     comparison_df['Diferença (R$)'] = comparison_df['Alocação Ótima (R$)'] - comparison_df['Alocação Atual (R$)']
#     print("\n--- Tabela Comparativa de Alocação de Orçamento ---")
#     print(comparison_df.round(2))
#     print("\n--- Resultados da Otimização ---")
#     print(f"Pedidos previstos com alocação ATUAL: {current_predicted_orders:,.0f}")
#     print(f"Pedidos previstos com alocação ÓTIMA: {max_orders:,.0f}")
#     gain = max_orders - current_predicted_orders
#     gain_percent = (gain / current_predicted_orders) * 100 if current_predicted_orders > 0 else 0
#     print(f"Ganho estimado em pedidos: {gain:,.0f} (+{gain_percent:.2f}%)")
# except (ValueError, np.linalg.LinAlgError, KeyError) as e:
#     print(f"\nNão foi possível completar a otimização: {e}")
    
#%%  ELASTICIDADE - REFINAR - pois utilizava serie de FOURIER

# # =============================================================================
# #           CONFIGURAÇÕES PRINCIPAIS
# # =============================================================================
# canal_para_analisar = 'facebook_socialads'
# # Defina a margem de lucro média por pedido para calcular o break-even
# margem_de_lucro_por_pedido = 50 # Exemplo: R$ 50 de lucro por pedido
# # =============================================================================

# def predict_orders_for_simulation(df_scenario, scaler, model, boxcox_lambda):
#     """Função auxiliar para prever pedidos para um dataframe de cenário."""
#     df_scenario_scaled = df_scenario.copy()
    
#     predictors_in_scaler = [col for col in continuous_predictors if col in df_scenario_scaled.columns]
#     df_scenario_scaled[predictors_in_scaler] = scaler.transform(df_scenario_scaled[predictors_in_scaler])
    
#     pred_transformed = model.predict(df_scenario_scaled)
#     pred_raw = inv_boxcox(pred_transformed, boxcox_lambda) - 1
#     return pred_raw.clip(0)

# def calculate_mroi_timeseries(channel_data, params, scaler, model, boxcox_lambda, dp):
#     """Calcula o Retorno Marginal sobre o Investimento (mROI) para cada dia."""
#     mroi_list = []
    
#     lam = params.get('lambda', 0)
#     channel_data = channel_data.sort_values('data').reset_index(drop=True)
#     channel_data['adstock'] = adstock(channel_data['investimento_real'], lam)
    
#     channel_data_clean = channel_data.copy()
#     all_fourier_cols_original = [col for col in dp.in_sample().columns]
#     clean_names = {}
#     for col in all_fourier_cols_original:
#         new_name = re.sub(r'[(),=]', '', col).replace(' ', '_').replace('-', '_')
#         clean_names[col] = new_name
#     channel_data_clean.rename(columns=clean_names, inplace=True)
    
#     # Pegamos o nome do canal ANTES do loop para usar na descrição
#     channel_name = channel_data['midia_tipo'].iloc[0] if not channel_data.empty else "canal desconhecido"
    
#     last_adstock = 0
#     # Usamos a variável 'channel_name' que acabamos de criar
#     for index, row in tqdm(channel_data_clean.iterrows(), total=len(channel_data), desc=f"Calculando mROI para {channel_name}"):
#         scenario_base = pd.DataFrame([row])
        
#         scenario_plus1 = scenario_base.copy()
#         investment_plus1_real = row['investimento_real'] + 1
#         scenario_plus1['log_investimento_real'] = np.log1p(investment_plus1_real)
        
#         new_adstock_plus1 = investment_plus1_real + lam * last_adstock
#         scenario_plus1['hill_adstock'] = hill(new_adstock_plus1, params['alpha'], params['theta'])
        
#         orders_base = predict_orders_for_simulation(scenario_base, scaler, model, boxcox_lambda)
#         orders_plus1 = predict_orders_for_simulation(scenario_plus1, scaler, model, boxcox_lambda)
        
#         mroi = orders_plus1.iloc[0] - orders_base.iloc[0]
#         mroi_list.append(mroi)
        
#         last_adstock = row['adstock']
        
#     return mroi_list


# # --- Execução Principal ---
# try:
#     print(f"Preparando dados para análise de elasticidade temporal de '{canal_para_analisar}'...")
#     channel_df_full = df_transformed[df_transformed['midia_tipo'] == canal_para_analisar].copy()
#     params = best_params[canal_para_analisar]

#     channel_df_full['mROI'] = calculate_mroi_timeseries(channel_df_full, params, scaler, model, best_lambda, dp)
    
#     break_even_mroi = 1 / margem_de_lucro_por_pedido if margem_de_lucro_por_pedido > 0 else 0

#     fig, ax = plt.subplots(figsize=(18, 8))
#     ax.plot(channel_df_full['data'], channel_df_full['mROI'], label='Elasticidade Diária (mROI)', color='royalblue')
    
#     ax.axhline(y=break_even_mroi, color='red', linestyle='-.', linewidth=2, label=f'Ponto de Equilíbrio (Break-even) = {break_even_mroi:.4f}')
    
#     ax.set_title(f'Elasticidade Temporal (Retorno Marginal) para: {canal_para_analisar}', fontsize=18)
#     ax.set_xlabel('Data', fontsize=12)
#     ax.set_ylabel('Pedidos Adicionais por R$ 1,00 de Investimento', fontsize=12)
#     ax.legend(loc='best')
#     ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
#     plt.text(channel_df_full['data'].iloc[0], break_even_mroi * 1.1, 'Zona Lucrativa (Elástica)', color='green', weight='bold')
#     plt.text(channel_df_full['data'].iloc[0], break_even_mroi * 0.9, 'Zona Não-Lucrativa (Inelástica)', color='darkred', weight='bold', va='top')
    
#     plt.tight_layout()
#     plt.show()

# except NameError as e:
#     print(f"\nERRO: Uma variável necessária não foi encontrada: {e}")
#     print("Por favor, certifique-se de que o pipeline completo do modelo (Etapas 1-6) foi executado.")
# except KeyError:
#     print(f"\nERRO: Canal '{canal_para_analisar}' não encontrado.")