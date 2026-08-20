import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# DATASET
# 1. Padronização das informações baseada no dataset

df = pd.read_excel("data/dataset.xlsx")

df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].str.strip().str.lower().str.replace(" ", "_", regex=False)

df = df.rename(columns={"idade_veículo": "idade_veiculo", "reclamações": "reclamacoes"})

df['reclamacoes'] = pd.to_numeric(df['reclamacoes'])

df['recall'] = df['recall'].astype(str).str.lower().str.strip()
df['recall'] = df['recall'].map({'sim': 1, 'não': 0, 'nao': 0, '1': 1, '0': 0}).fillna(0)

df['idade_veiculo'] = pd.to_numeric(df['idade_veiculo'])
df['km'] = pd.to_numeric(df['km'])

# 2. Título de configração da página
st.set_page_config(page_title="Dashboard de Risco", layout="wide")
st.title("Dashboard de Análise de Recall de Veículos")

# 3. Indicadores (KPIs)
st.markdown("### 📊 Indicadores Gerais")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total de Veículos", len(df))

col2.metric("Reclamações Registradas", int(df['reclamacoes'].sum()))
col3.metric("Total de Recalls", df['recall'].sum())
col4.metric("Média de Idade dos Carros (Anos)", round(df['idade_veiculo'].mean(), 1))

st.divider()

# 4. Gráficos
col_esq, col_meio, col_dir = st.columns(3)

with col_esq:
    st.markdown("### Taxa de Recall por Modelo")
    
    df_modelo = df.groupby('modelo').agg(
        total_veiculos=('modelo', 'count'),
        total_recalls=('recall', 'sum')
    ).reset_index()
    
    df_modelo['taxa_recall_pct'] = (df_modelo['total_recalls'] / df_modelo['total_veiculos']) * 100
    df_modelo["modelo"] = df_modelo["modelo"].str.title()
    
    fig_recall = px.bar(df_modelo, x='modelo', y='taxa_recall_pct', 
                        color='taxa_recall_pct', color_continuous_scale='Reds', 
                        #title="Taxa de Recall (%) por Modelo",
                        labels={'taxa_recall_pct': 'Taxa de Recall (%)', 'modelo': 'Modelo'})
    
    fig_recall.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
    st.plotly_chart(fig_recall, use_container_width=True)


with col_meio:
    st.markdown("### Evolução da Taxa de Recall por Ano")
    
    df['ano_fabricacao'] = 2026 - df['idade_veiculo']
    
    df_ano = df.groupby('ano_fabricacao').agg(
        total_veiculos=('ano_fabricacao', 'count'),
        total_recalls=('recall', 'sum')
    ).reset_index()
    
    df_ano['taxa_recall_pct'] = (df_ano['total_recalls'] / df_ano['total_veiculos']) * 100
    
    df_ano = df_ano.sort_values('ano_fabricacao', ascending=True)
    
    fig_evolucao = px.line(df_ano, x='ano_fabricacao', y='taxa_recall_pct', 
                           markers=True,
                           labels={'taxa_recall_pct': 'Taxa de Recall (%)', 'ano_fabricacao': 'Ano do Veículo'})
    
    fig_evolucao.update_layout(xaxis=dict(type='category'))
    
    st.plotly_chart(fig_evolucao, use_container_width=True)


with col_dir:
    st.markdown("### Simulador de Risco")
    
    st.caption("ℹ️ Modelo preditivo treinado e exportado durante o desafio.")
    
    lista_modelos = df['modelo'].str.title().unique()
    input_modelo = st.selectbox("Modelo do Veículo", lista_modelos)
    
    lista_anos = list(range(2026, 1999, -1))
    input_ano = st.selectbox("Ano de Fabricação", lista_anos)
    
    input_km = st.number_input("Quilometragem", min_value=0, value=50000, step=1000)
    input_reclamacoes = st.number_input("Qtd. de Reclamações", min_value=0, value=0, step=1)
    
    idade_calculada = 2026 - input_ano

    if st.button("Calcular Probabilidade", use_container_width=True):
        try:
            pipeline = joblib.load('modelo_recall.pkl')
            
            dados_entrada = pd.DataFrame({
                'idade_veiculo': [idade_calculada],
                'reclamacoes': [input_reclamacoes]
            })
            
            prob_recall = pipeline.predict_proba(dados_entrada)[0][1]
            
            st.markdown("#### Resultado da Análise:")
            
            if prob_recall >= 0.6:
                st.error("🚨 **Risco Alto:** Altas chances de Recall.")
            elif prob_recall >= 0.47:
                st.warning("⚠️ **Risco Médio:** Existem consideráveis chances de Recall.")
            else:
                st.success("✅ **Risco Controlado:** Baixa chance de Recall.")
            
        except Exception as e:
            st.error(f"Erro ao calcular: {e}")
            st.info("Dica: Verifique se as variáveis enviadas para o modelo são exatamente as mesmas da lista 'features' do treinamento.")


st.divider()

# 5. Ranking (Tabela interativa)
st.markdown("### Ranking de Segmentos Críticos (Modelo + Ano)")

df_segmentos = df.groupby(['modelo', 'ano_fabricacao']).agg(
    volume_veiculos=('modelo', 'count'),          
    total_recalls=('recall', 'sum'),              
    media_reclamacoes=('reclamacoes', 'mean')     
).reset_index()

df_segmentos['modelo'] = df_segmentos['modelo'].str.title()

df_segmentos['taxa_recall_pct'] = ((df_segmentos['total_recalls'] / df_segmentos['volume_veiculos']) * 100).round(1)
df_segmentos['media_reclamacoes'] = df_segmentos['media_reclamacoes'].round(1)
df_ranking = df_segmentos.sort_values(by=['taxa_recall_pct', 'media_reclamacoes'], ascending=[False, False])
df_ranking = df_ranking.reset_index(drop=True)

df_ranking = df_ranking.rename(columns={
    'modelo': 'Modelo',
    'ano_fabricacao': 'Ano de Fabricação',
    'volume_veiculos': 'Volume na Base',
    'total_recalls': 'Total de Recalls (Real)',
    'taxa_recall_pct': 'Taxa de Recall (%)',
    'media_reclamacoes': 'Média de Reclamações'
})

st.dataframe(
    df_ranking, 
    use_container_width=True, 
    hide_index=True,
    column_config={
        "Taxa de Recall (%)": st.column_config.NumberColumn(
            "Taxa de Recall (%)",
            format="%.1f" 
        ),
        "Média de Reclamações": st.column_config.NumberColumn(
            "Média de Reclamações",
            format="%.1f"
        )
    }
)