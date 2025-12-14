import warnings; warnings.filterwarnings("ignore"); 
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns; 
from sklearn.model_selection import train_test_split; 
from sklearn.ensemble import RandomForestClassifier; 
from sklearn.metrics import accuracy_score, confusion_matrix; # Adicionado confusion_matrix
from sklearn.preprocessing import MinMaxScaler; 

# --- 1. CONFIGURAÇÕES E CRIAÇÃO DO DATASET ---
num_jogadores = 250; 
TAMANHO_PISTA_PADRAO_KM = 5.0 
NOMES_TESTE = ["A. Senna", "M. Schumacher", "L. Hamilton", "M. Verstappen", "F. Alonso", "C. Leclerc", "S. Perez", "G. Russell", "L. Norris", "V. Bottas"]
features = ["velocidade_media", "arranque", "tempo_por_corrida", "tempo_jogo", "porcentagem_vitorias", "desvio_padrao_tempo_volta", "erros_por_corrida", "razao_vitoria_derrota", "tamanho_pista_km"]

# Geração de dados de treino dinâmicos
partidas_total = np.random.randint(50, 500, num_jogadores)
partidas_vencidas = np.round(partidas_total * np.clip(np.random.normal(0.20, 0.15, num_jogadores), 0.05, 0.50))
partidas_perdidas = partidas_total - partidas_vencidas
razao_vitoria_derrota = partidas_vencidas / (partidas_perdidas + 1); 
dados = {
    "velocidade_media": np.random.normal(180, 15, num_jogadores), "arranque": np.random.normal(3.5, 0.4, num_jogadores), "tempo_por_corrida": np.random.normal(3.8, 0.5, num_jogadores), "tempo_jogo": np.random.normal(250, 60, num_jogadores), "porcentagem_vitorias": np.clip(np.random.normal(0.15, 0.10, num_jogadores), 0.0, 0.5), 
    "desvio_padrao_tempo_volta": np.random.normal(0.20, 0.05, num_jogadores), "erros_por_corrida": np.random.poisson(2, num_jogadores), "razao_vitoria_derrota": razao_vitoria_derrota,
    "tamanho_pista_km": np.full(num_jogadores, TAMANHO_PISTA_PADRAO_KM) 
}; 
df = pd.DataFrame(dados); 
# Fórmula consistente para ranking real
desempenho = (df["velocidade_media"] * 0.45 - df["arranque"] * 12 - df["tempo_por_corrida"] * 8 + df["tempo_jogo"] * 0.15 + df["porcentagem_vitorias"] * 80 - df["desvio_padrao_tempo_volta"] * 50 - df["erros_por_corrida"] * 4 + df["razao_vitoria_derrota"] * 30 - df["tamanho_pista_km"] * 0.5 + np.random.normal(0, 5, num_jogadores)); 
df["ranking_real"] = pd.qcut(desempenho, 10, labels=False, duplicates='drop') + 1; 

# Treinamento da IA
X, y = df[features], df["ranking_real"]; 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y); 
scaler = MinMaxScaler(); 
X_train_scaled = scaler.fit_transform(X_train); 
X_test_scaled = scaler.transform(X_test); 

print("\n" + "="*70)
print("     ⚡ INICIANDO TREINAMENTO RÁPIDO DO MODELO (n_estimators=100)")
print("="*70)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_scaled, y_train); 
y_pred = modelo_rf.predict(X_test_scaled); 
acuracia = accuracy_score(y_test, y_pred); 

# Geração e Previsão dos Jogadores Base
num_test = 10
partidas_total_test = np.random.randint(100, 400, num_test)
partidas_vencidas_test = np.round(partidas_total_test * np.clip(np.random.normal(0.25, 0.12, num_test), 0.05, 0.45))
partidas_perdidas_test = partidas_total_test - partidas_vencidas_test
razao_vitoria_derrota_test = partidas_vencidas_test / (partidas_perdidas_test + 1);
jogadores_teste_base = pd.DataFrame({
    "Jogador": NOMES_TESTE, "velocidade_media": np.random.normal(185, 12, num_test), "arranque": np.random.normal(3.4, 0.3, num_test), "tempo_por_corrida": np.random.normal(3.7, 0.4, num_test), "tempo_jogo": np.random.normal(270, 50, num_test),
    "porcentagem_vitorias": np.clip(np.random.normal(0.20, 0.10, num_test), 0.0, 0.5), "desvio_padrao_tempo_volta": np.random.normal(0.18, 0.04, num_test), "erros_por_corrida": np.random.poisson(1.5, num_test),
    "razao_vitoria_derrota": razao_vitoria_derrota_test, "tamanho_pista_km": np.full(num_test, TAMANHO_PISTA_PADRAO_KM) 
});

df_scaled_base = scaler.transform(jogadores_teste_base[features])
jogadores_teste_base["ranking_previsto"] = modelo_rf.predict(df_scaled_base)
ranking_base = jogadores_teste_base.sort_values("ranking_previsto").reset_index(drop=True)


# --- 2. EXIBIÇÃO DE RESULTADOS EM TEXTO ---
print("\n" + "="*70 + f"\n             🎯 Acurácia do Modelo: {acuracia:.2%}")
print("="*70 + "\n"); 

# RELATÓRIO DE IMPORTÂNCIA
importancia = pd.DataFrame({"Atributo": features, "Importância": modelo_rf.feature_importances_}).sort_values("Importância", ascending=False); 
print("📝 RELATÓRIO DE IMPORTÂNCIA DOS ATRIBUTOS:\n", importancia.to_string(index=False)); 

# RANKING BASE
print("\n" + "="*70 + "\n🏆 RANKING PREVISTO DOS 10 COMPETIDORES BASE (1 = Melhor):\n", ranking_base[["Jogador", "ranking_previsto"]].to_string(index=False)); 
print("="*70)

# --- 3. GRÁFICOS DETALHADOS (Com tratamento de erro para estabilidade) ---

# GRÁFICO 1: IMPORTÂNCIA DE ATRIBUTOS
try:
    plt.figure(figsize=(12, 6)); 
    sns.barplot(x="Importância", y="Atributo", data=importancia, palette="viridis"); 
    plt.title("1. Impacto Relativo dos Atributos no Resultado do Ranking", fontsize=16); 
    plt.xlabel("Importância (Peso no Modelo)"); plt.ylabel("Atributo"); plt.grid(axis='x'); plt.show(); 
    plt.close()
except Exception as e:
    print(f"\n❌ Falha ao gerar o Gráfico 1 (Importância). Erro: {e}")

# GRÁFICO 2: DISTRIBUIÇÃO DOS COMPETIDORES BASE
try:
    plt.figure(figsize=(12, 6));
    # Usa a função kdeplot para mostrar a distribuição de densidade dos rankings
    sns.kdeplot(ranking_base["ranking_previsto"], fill=True, color="#4daf4a", linewidth=2);
    plt.axvline(ranking_base["ranking_previsto"].mean(), color='r', linestyle='--', label=f'Média: {ranking_base["ranking_previsto"].mean():.1f}');
    plt.title("2. Distribuição da Previsão de Ranking dos 10 Competidores Base", fontsize=16);
    plt.xlabel("Ranking Previsto (1 = Melhor)"); plt.ylabel("Densidade");
    plt.xticks(range(1, 11)); plt.legend(); plt.grid(axis='y'); plt.show();
    plt.close()
except Exception as e:
    print(f"\n❌ Falha ao gerar o Gráfico 2 (Distribuição). Erro: {e}")


# GRÁFICO 3: MATRIZ DE CONFUSÃO (Análise de Desempenho do Modelo)
try:
    plt.figure(figsize=(10, 8));
    # Gera a matriz de confusão com base no set de teste (X_test, y_test)
    cm = confusion_matrix(y_test, y_pred);
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=modelo_rf.classes_, yticklabels=modelo_rf.classes_);
    plt.title(f"3. Matriz de Confusão do Modelo (Acurácia: {acuracia:.2%})", fontsize=16);
    plt.ylabel("Ranking Real"); plt.xlabel("Ranking Previsto");
    plt.show();
    plt.close()
    print("\n💡 A Matriz de Confusão mostra: Os números na diagonal (azul escuro) são as previsões corretas.")
except Exception as e:
    print(f"\n❌ Falha ao gerar o Gráfico 3 (Matriz de Confusão). Erro: {e}")

# --- FIM DO CÓDIGO ---
