# ==============================================================================
# In[1]. IMPORTS
# ==============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
from scipy.special import inv_boxcox # Precisamos para reverter a previsão

# ==============================================================================
# In[2]. CARREGAMENTO DOS ARTEFATOS (COM CACHE)
# ==============================================================================
st.set_page_config(layout="wide")

# Uma função de cache para cada artefato
@st.cache_resource
def load_model():
    return joblib.load('artifacts/mmm_model.pkl')

@st.cache_resource
def load_scaler():
    return joblib.load('artifacts/mmm_scaler.pkl')

@st.cache_data
def load_params():
    return joblib.load('artifacts/mmm_adstock_hill_params.pkl')
    
@st.cache_data
def load_lambda():
    return joblib.load('artifacts/mmm_best_lambda.pkl')

@st.cache_data
def load_data():
    return pd.read_parquet('artifacts/dados_tcc_processados.parquet')

# Carrega tudo
model = load_model()
scaler = load_scaler()
best_params = load_params()
best_lambda = load_lambda()
df = load_data()

# ==============================================================================
# In[3]. TÍTULO E INTRODUÇÃO
# ==============================================================================

st.title("🚀 Budget Optimization Simulator (Marketing Mix Modeling)")

st.markdown("""
This interactive application is a demonstration of my MBA thesis in Data Science & Analytics (**USP**). 
The goal is to transform complex **Marketing Mix Modeling (MMM)** outputs into an actionable tool for business stakeholders.
""")

# ==============================================================================
# In[4]. ANÁLISE HISTÓRICA: HISTORICO DE INVESTIMENTO
# ==============================================================================

st.header("Historical Analysis: Media Investment Spend")

# Agrupando os dados por mês para criar a visualização
df_monthly = df.copy()

# Para o app, vamos simplificar a decomposição. Vamos assumir que 
# a contribuição é proporcional ao investimento, só para fins de visualização.
# Em um projeto real, usaríamos a saída exata do modelo.
df_monthly['mes_ano'] = df_monthly['data'].dt.to_period('M').astype(str)

# Agrupamos o investimento por canal e mês
contribution_monthly = df_monthly.groupby(['mes_ano', 'midia_tipo'])['investimento'].sum().unstack(fill_value=0)

# # Simplificação: Adicionando um "Baseline" fake para o gráfico ficar mais realista
# total_investment = contribution_monthly.sum(axis=1)
# contribution_monthly['Baseline'] = total_investment * np.random.uniform(0.3, 0.5, size=len(total_investment)) # Baseline entre 30-50%

# Criando o gráfico de barras empilhadas com Plotly
fig_decomp = px.bar(
    contribution_monthly, 
    x=contribution_monthly.index, 
    y=contribution_monthly.columns,
    title="Investment History",
    labels={'x': 'Mês', 'y': 'Total Invested'},
    template='plotly_white'
)

# Melhorando a aparência do gráfico
fig_decomp.update_layout(
    barmode='stack',
    legend_title_text='Media Channels',
    yaxis_title="Sales Volume (Simulated)",
    xaxis_title="Period"
)

# Exibindo o gráfico no Streamlit, usando a largura total do container
st.plotly_chart(fig_decomp, use_container_width=True)

st.info("This chart illustrates the estimated contribution of each media channel and the baseline (organic sales) over time. This is a core MMM output for strategic performance analysis.")

# ==============================================================================
# In[5] SIMULADOR INTERATIVO INTELIGENTE v2.0
# ==============================================================================
st.divider()
st.header("💡 Media Control Panel")
st.markdown("Use the tools below to diagnose saturation and marginal costs for each channel.")

# --- DADOS DE REFERÊNCIA E VARIÁVEIS GLOBAIS ---
ultima_data = df['data'].max()
st.info(f"Diagnosis based on data up to: **{ultima_data.strftime('%B %d, %Y')}**")

CONTINUOUS_PREDICTORS = ['hill_adstock', 'log_investimento_real', 'log_cliques', 'log_impressoes', 'log_visitas', 'pedidos_lag_1', 'visitas_lag_1']

# --- FUNÇÕES DE SIMULAÇÃO (DEFINIDAS PRIMEIRO) ---
def adstock_calc(series, decay):
    res = np.zeros_like(series, dtype=float)
    if len(series) > 0:
        res[0] = series[0]
        for i in range(1, len(series)): res[i] = series[i] + decay * res[i-1]
    return res

@st.cache_data
def simular_metricas_avancadas(channel_name, budget, _model, _best_params, _best_lambda, _df):
    params = _best_params.get(channel_name, {})
    lam, alpha, theta = params.get('lambda', 0), params.get('alpha', 1), params.get('theta', 1)
    if theta == 0: return 0, 0, 0, np.inf

    df_channel = _df[_df['midia_tipo'] == channel_name]
    adstock_historico = adstock_calc(df_channel['investimento_real'].values, lam)
    last_real_adstock = adstock_historico[-1] if len(adstock_historico) > 0 else 0

    adstock_simulado = last_real_adstock
    if budget > 0:
        daily_budget = budget / 30
        for _ in range(30): adstock_simulado = daily_budget + lam * adstock_simulado
    else:
        adstock_simulado = last_real_adstock * (lam ** 30)

    saturacao = (adstock_simulado**alpha) / (adstock_simulado**alpha + theta**alpha) if adstock_simulado > 0 else 0
    elasticidade = alpha * (1 - saturacao)

    adstock_plus_1 = last_real_adstock
    daily_budget_plus_1 = (budget + 1) / 30
    for _ in range(30): adstock_plus_1 = daily_budget_plus_1 + lam * adstock_plus_1
    saturacao_plus_1 = (adstock_plus_1**alpha) / (adstock_plus_1**alpha + theta**alpha) if adstock_plus_1 > 0 else 0
    
    delta_saturacao = saturacao_plus_1 - saturacao
    coef_potencia = _model.params.get("hill_adstock", 0)
    delta_boxcox = delta_saturacao * coef_potencia
    pred_base_boxcox = _model.params.get("Intercept", 0)
    
    pedidos_base = inv_boxcox(pred_base_boxcox + (saturacao * coef_potencia), _best_lambda) - 1
    pedidos_plus1 = inv_boxcox(pred_base_boxcox + (saturacao * coef_potencia) + delta_boxcox, _best_lambda) - 1
    
    retorno_marginal = max(0, pedidos_plus1 - pedidos_base)
    cpa_marginal = 1 / retorno_marginal if retorno_marginal > 0 else np.inf
    
    return saturacao * 100, elasticidade, retorno_marginal, cpa_marginal

# --- ESTRUTURA DA INTERFACE ---
col_config, col_resultado = st.columns([1, 2])

with col_config:
    st.subheader("1. Set Total Budget")
    # O INPUT DO ORÇAMENTO TOTAL ESTÁ DE VOLTA
    total_budget = st.number_input(
        "Remaining Monthly Budget (USD)", 10000, 5000000, 500000, 10000, format="%d", key="total_budget_input"
    )

    st.subheader("2. Budget Allocation")
    channels_to_sim = list(best_params.keys())
    
    budget_simulado = {}
    for i, channel in enumerate(channels_to_sim):
        channel_name_clean = channel.replace('_', ' ').title()
        # O valor máximo de cada slider agora é o orçamento total
        budget_simulado[channel] = st.slider(
            channel_name_clean, 0, total_budget, 10000, 1000, key=f'slider_{i}'
        )
    
    current_allocated = sum(budget_simulado.values())
    unallocated = total_budget - current_allocated
    
    # Exibe o status do orçamento
    st.metric("Total Allocated in Sliders", f"USD {current_allocated:,.0f}")
    if unallocated < 0:
        st.error(f"Remaining Budget: USD {unallocated:,.0f} (Overspent!)")
    else:
        st.success(f"Remaining Budget: USD {unallocated:,.0f}")

    def limpar_orcamentos():
        for i in range(len(channels_to_sim)):
            st.session_state[f'slider_{i}'] = 0
    st.button("🗑️ Clean Allocation", on_click=limpar_orcamentos, use_container_width=True)


# --- CÁLCULO E RESULTADOS ---
with col_resultado:
    st.subheader("3. Diagnosis and Results")
    
    resultados = []
    with st.spinner('Calculating...'):
        for channel, budget in budget_simulado.items():
            saturacao, elasticidade, ret_marginal, cpa_marg = simular_metricas_avancadas(
                channel, budget, model, best_params, best_lambda, df
            )
            resultados.append({
                'Channel': channel.replace('_', ' ').title(),
                'Investment (USD)': budget,
                'Saturation (%)': saturacao,
                'Elasticity': elasticidade,
                'Marginal Return (Orders/USD)': ret_marginal,
                'Marginal CPA (USD)': cpa_marg
            })

    if resultados:
        df_resultados = pd.DataFrame(resultados).sort_values(by='Marginal CPA (USD)', ascending=True)
        st.dataframe(
            df_resultados.style.format({
                'Investment (USD)': "$ {:,.0f}", 'Saturation (%)': "{:.1f}%",
                'Elasticity': "{:.3f}", 'Marginal Return (Orders/USD)': "{:.4f}",
                'Marginal CPA (USD)': "$ {:,.2f}"
            }).background_gradient(cmap='Reds', subset=['Saturation (%)'])
            .background_gradient(cmap='Greens_r', subset=['Marginal CPA (USD)']),
            use_container_width=True, hide_index=True
        )
        
        st.subheader("Efficiency Diagnosis: Cost of the Next Sale")
        st.info("The Marginal CPA indicates **how much it will cost to generate the next sale** for each channel. Channels with a lower Marginal CPA represent the best opportunities for additional investment.")
        
        fig_cpa_m = px.bar(
            df_resultados.sort_values(by='Marginal CPA (USD)', ascending=False),
            y='Channel', x='Marginal CPA (USD)', color='Marginal CPA (USD)',
            color_continuous_scale='Greens_r', orientation='h',
            title='Marginal Cost per Acquisition (Marginal CPA)'
        )
        fig_cpa_m.update_layout(xaxis_title="Cost to Generate +1 Order (USD)", yaxis_title="")
        st.plotly_chart(fig_cpa_m, use_container_width=True)
        
st.divider()

# ==============================================================================
# In[6] ANÁLISE HISTÓRICA: DECOMPOSIÇÃO CORRETA (BASELINE vs. INCREMENTAL)
# ==============================================================================
st.header("📊 Historical Sales Decomposition")
st.markdown("This analysis breaks down total sales into **Baseline** (organic sales) and **Incremental** (sales driven by media effort), using the model's decomposition methodology.")

# --- LÓGICA DE DECOMPOSIÇÃO CORRETA (ADAPTADA DO TCC) ---

@st.cache_data # CRÍTICO: O cache vai rodar essa função pesada apenas uma vez.
def calcular_decomposicao_historica_correta():
    df_decomp = df.copy()
    
    # Define os preditores contínuos para escalonamento
    continuous_predictors = ['hill_adstock', 'log_investimento_real', 'log_cliques', 'log_impressoes', 'log_visitas', 'pedidos_lag_1', 'visitas_lag_1']

    # Passo 1: Calcular a Previsão Total com todas as mídias ativas
    df_decomp_scaled = df_decomp.copy()
    df_decomp_scaled[continuous_predictors] = scaler.transform(df_decomp_scaled[continuous_predictors])
    
    pred_total_transformed = model.predict(df_decomp_scaled)
    pred_total_raw = inv_boxcox(pred_total_transformed, best_lambda) - 1
    df_decomp['pred_total'] = pred_total_raw.clip(0)

    # Passo 2: Calcular a contribuição incremental de cada canal (iterativamente)
    daily_contributions_df = pd.DataFrame(index=df_decomp.index)
    channels = df_decomp['midia_tipo'].unique()

    for channel in channels:
        # Cria um cenário onde o canal atual é "desligado"
        df_scenario = df_decomp.copy()
        mask = df_scenario['midia_tipo'] == channel
        df_scenario.loc[mask, 'log_investimento_real'] = 0
        df_scenario.loc[mask, 'hill_adstock'] = 0
        
        # Escala o cenário e faz a previsão
        df_scenario_scaled = df_scenario.copy()
        df_scenario_scaled[continuous_predictors] = scaler.transform(df_scenario_scaled[continuous_predictors])
        
        pred_scenario_transformed = model.predict(df_scenario_scaled)
        pred_scenario_raw = inv_boxcox(pred_scenario_transformed, best_lambda) - 1
        
        # A contribuição é a diferença entre a previsão total e a previsão do cenário
        contribution = (df_decomp['pred_total'] - pred_scenario_raw.clip(0)).clip(0)
        daily_contributions_df[channel] = contribution

    # Passo 3: Calcular o Baseline
    # O total incremental diário é a soma das contribuições de todos os canais
    total_daily_incremental = daily_contributions_df.sum(axis=1)
    # O baseline diário é a diferença entre o total REAL e o incremental total
    daily_baseline = (df_decomp['pedidos'] - total_daily_incremental).clip(0)
    daily_contributions_df['Baseline'] = daily_baseline
    
    # Adiciona a coluna de data para o agrupamento
    daily_contributions_df['data'] = df_decomp['data']
    
    # Passo 4: Agrupar por mês
    monthly_decomp_final = daily_contributions_df.groupby(pd.Grouper(key='data', freq='ME')).sum()
    
    return monthly_decomp_final

# Executa a função de cálculo (graças ao cache, só roda na primeira vez)
monthly_decomp_df = calcular_decomposicao_historica_correta()

# --- CÁLCULO DAS PORCENTAGENS E PLOT (IGUAL A ANTES) ---
monthly_totals = monthly_decomp_df.sum(axis=1)
# Evita divisão por zero
monthly_totals[monthly_totals == 0] = 1
monthly_decomp_pct = monthly_decomp_df.divide(monthly_totals, axis=0).multiply(100)

# Reordena colunas para o Baseline vir primeiro
if 'Baseline' in monthly_decomp_pct.columns:
    cols_ordered = ['Baseline'] + [col for col in monthly_decomp_pct.columns if col != 'Baseline' and col != 'data']
    monthly_decomp_pct = monthly_decomp_pct[cols_ordered]

# --- CRIAÇÃO DO GRÁFICO COM PLOTLY ---
fig_decomp_percent = px.bar(
    monthly_decomp_pct,
    x=monthly_decomp_pct.index,
    y=monthly_decomp_pct.columns,
    title='Monthly Percentage Decomposition (Baseline vs. Incremental)',
    labels={'x': 'Month', 'value': 'Percentage Contribution (%)'},
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Vivid # Um esquema de cores com bom contraste
)

fig_decomp_percent.update_layout(
    barmode='stack',
    yaxis_ticksuffix='%',
    legend_title_text='Components',
    yaxis_title="Contribution (%)"
)
fig_decomp_percent.update_xaxes(tickangle=45)

# Exibe o gráfico no Streamlit
st.plotly_chart(fig_decomp_percent, use_container_width=True)