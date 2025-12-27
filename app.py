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

st.title("🚀 Simulador de Otimização de Budget (Marketing Mix Model)")

st.markdown("""
Esta aplicação interativa é uma demonstração do meu trabalho de conclusão de curso do MBA em Data Science & Analytics (USP/ESALQ). 
O objetivo é transformar os resultados de um complexo Modelo Hierárquico de Marketing Mix (MMM) em uma ferramenta acionável para stakeholders de negócio.
""")

# ==============================================================================
# In[4]. ANÁLISE HISTÓRICA: HISTORICO DE INVESTIMENTO
# ==============================================================================

st.header("Análise Histórica: Investimento Histórico por Mídia")

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
    title="Histórico de Investimento",
    labels={'x': 'Mês', 'y': 'Total Investido'},
    template='plotly_white'
)

# Melhorando a aparência do gráfico
fig_decomp.update_layout(
    barmode='stack',
    legend_title_text='Canais de Mídia',
    yaxis_title="Volume de Vendas (Simulado)",
    xaxis_title="Período"
)

# Exibindo o gráfico no Streamlit, usando a largura total do container
st.plotly_chart(fig_decomp, width='stretch')

st.info("Este gráfico demonstra a contribuição estimada de cada canal de mídia e do baseline (vendas orgânicas) ao longo do tempo. É uma das principais saídas do modelo MMM, permitindo a análise estratégica da performance de cada canal.")

# ==============================================================================
# In[5] SIMULADOR INTERATIVO INTELIGENTE v2.0
# ==============================================================================
st.divider()
st.header("💡 Painel de Controle de Mídia")
st.markdown("Use as ferramentas abaixo para diagnosticar a saturação e o custo marginal de cada canal.")

# --- DADOS DE REFERÊNCIA E VARIÁVEIS GLOBAIS ---
ultima_data = df['data'].max()
st.info(f"Diagnóstico baseado em dados até: **{ultima_data.strftime('%d de %B de %Y')}**")

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
    st.subheader("1. Defina o Orçamento Total")
    # O INPUT DO ORÇAMENTO TOTAL ESTÁ DE VOLTA
    total_budget = st.number_input(
        "Orçamento Mensal Remanescente (R$)", 10000, 5000000, 500000, 10000, format="%d", key="total_budget_input"
    )

    st.subheader("2. Distribua o Orçamento")
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
    st.metric("Total Alocado nos Sliders", f"R$ {current_allocated:,.0f}")
    if unallocated < 0:
        st.error(f"Orçamento Restante: R$ {unallocated:,.0f} (Atenção: excedido!)")
    else:
        st.success(f"Orçamento Restante: R$ {unallocated:,.0f}")

    def limpar_orcamentos():
        for i in range(len(channels_to_sim)):
            st.session_state[f'slider_{i}'] = 0
    st.button("🗑️ Limpar Orçamentos", on_click=limpar_orcamentos, width='stretch')


# --- CÁLCULO E RESULTADOS ---
with col_resultado:
    st.subheader("3. Diagnóstico e Resultados")
    
    resultados = []
    with st.spinner('Calculando métricas...'):
        for channel, budget in budget_simulado.items():
            saturacao, elasticidade, ret_marginal, cpa_marg = simular_metricas_avancadas(
                channel, budget, model, best_params, best_lambda, df
            )
            resultados.append({
                'Canal': channel.replace('_', ' ').title(),
                'Investimento (R$)': budget,
                'Saturação (%)': saturacao,
                'Elasticidade': elasticidade,
                'Retorno Marginal (Pedidos/R$)': ret_marginal,
                'CPA Marginal (R$)': cpa_marg
            })

    if resultados:
        df_resultados = pd.DataFrame(resultados).sort_values(by='CPA Marginal (R$)', ascending=True)
        st.dataframe(
            df_resultados.style.format({
                'Investimento (R$)': "R$ {:,.0f}", 'Saturação (%)': "{:.1f}%",
                'Elasticidade': "{:.3f}", 'Retorno Marginal (Pedidos/R$)': "{:.4f}",
                'CPA Marginal (R$)': "R$ {:,.2f}"
            }).background_gradient(cmap='Reds', subset=['Saturação (%)'])
            .background_gradient(cmap='Greens_r', subset=['CPA Marginal (R$)']),
            width='stretch', hide_index=True
        )
        
        st.subheader("Diagnóstico de Oportunidade: Custo da Próxima Venda")
        st.info("O CPA Marginal indica **quanto custará para gerar a próxima venda** em cada canal. Canais com menor CPA Marginal são as melhores oportunidades para investimento adicional.")
        
        fig_cpa_m = px.bar(
            df_resultados.sort_values(by='CPA Marginal (R$)', ascending=False), 
            y='Canal', x='CPA Marginal (R$)', color='CPA Marginal (R$)',
            color_continuous_scale='Greens_r', orientation='h',
            title='Custo por Aquisição Marginal (CPA Marginal)'
        )
        fig_cpa_m.update_layout(xaxis_title="Custo para Gerar +1 Pedido (R$)", yaxis_title="")
        st.plotly_chart(fig_cpa_m, width='stretch')
        
st.divider()

# ==============================================================================
# In[6] ANÁLISE HISTÓRICA: DECOMPOSIÇÃO CORRETA (BASELINE vs. INCREMENTAL)
# ==============================================================================
st.header("📊 Análise Histórica: Decomposição das Vendas")
st.markdown("Esta análise separa as vendas totais em **Baseline** (vendas orgânicas) e **Incremental** (vendas geradas pelo esforço de cada canal), usando a metodologia de decomposição do modelo.")

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
    title='Decomposição Percentual Mensal de Vendas (Baseline vs. Incremental)',
    labels={'x': 'Mês', 'value': 'Contribuição Percentual (%)'},
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Vivid # Um esquema de cores com bom contraste
)

fig_decomp_percent.update_layout(
    barmode='stack',
    yaxis_ticksuffix='%',
    legend_title_text='Componentes',
    yaxis_title="Contribuição (%)"
)
fig_decomp_percent.update_xaxes(tickangle=45)

# Exibe o gráfico no Streamlit
st.plotly_chart(fig_decomp_percent, width='stretch')