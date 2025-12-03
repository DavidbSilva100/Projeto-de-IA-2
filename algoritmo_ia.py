import warnings; warnings.filterwarnings("ignore"); import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns; from sklearn.model_selection import train_test_split; from sklearn.ensemble import RandomForestClassifier; from sklearn.metrics import accuracy_score, confusion_matrix; from sklearn.preprocessing import MinMaxScaler; 
np.random.seed(42); num_jogadores = 250; 
# --- DEFINIÇÃO DO PADRÃO ---
TAMANHO_PISTA_PADRAO_KM = 5.0 # Padrão: 5.0 KM para todos os competidores
NOMES_TESTE = ["A. Senna", "M. Schumacher", "L. Hamilton", "M. Verstappen", "F. Alonso", "C. Leclerc", "S. Perez", "G. Russell", "L. Norris", "V. Bottas"]
# -----------------------------

# --- CRIAÇÃO DOS DADOS DE TREINAMENTO (8 FEATURES) ---
partidas_total = np.random.randint(50, 500, num_jogadores)
partidas_vencidas = np.round(partidas_total * np.clip(np.random.normal(0.20, 0.15, num_jogadores), 0.05, 0.50))
partidas_perdidas = partidas_total - partidas_vencidas
razao_vitoria_derrota = partidas_vencidas / (partidas_perdidas + 1); 

dados = {
    "velocidade_media": np.random.normal(180, 15, num_jogadores),
    "arranque": np.random.normal(3.5, 0.4, num_jogadores),
    "tempo_por_corrida": np.random.normal(3.8, 0.5, num_jogadores),
    "tempo_jogo": np.random.normal(250, 60, num_jogadores),
    "porcentagem_vitorias": np.clip(np.random.normal(0.15, 0.10, num_jogadores), 0.0, 0.5), 
    "desvio_padrao_tempo_volta": np.random.normal(0.20, 0.05, num_jogadores), 
    "erros_por_corrida": np.random.poisson(2, num_jogadores),
    "razao_vitoria_derrota": razao_vitoria_derrota,
    "tamanho_pista_km": np.full(num_jogadores, TAMANHO_PISTA_PADRAO_KM) 
}; 
df = pd.DataFrame(dados); 
desempenho = (
    df["velocidade_media"] * 0.45 
    - df["arranque"] * 12 
    - df["tempo_por_corrida"] * 8 
    + df["tempo_jogo"] * 0.15
    + df["porcentagem_vitorias"] * 80 
    - df["desvio_padrao_tempo_volta"] * 50
    - df["erros_por_corrida"] * 4
    + df["razao_vitoria_derrota"] * 30
    - df["tamanho_pista_km"] * 0.5
    + np.random.normal(0, 5, num_jogadores)
); 

df["ranking_real"] = pd.qcut(desempenho, 10, labels=False, duplicates='drop') + 1; 
features = ["velocidade_media", "arranque", "tempo_por_corrida", "tempo_jogo", "porcentagem_vitorias", "desvio_padrao_tempo_volta", "erros_por_corrida", "razao_vitoria_derrota", "tamanho_pista_km"]; 
X, y = df[features], df["ranking_real"]; 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y); 
scaler = MinMaxScaler(); 
X_train_scaled = scaler.fit_transform(X_train); 
X_test_scaled = scaler.transform(X_test); 
modelo_rf = RandomForestClassifier(n_estimators=200, random_state=42); 
modelo_rf.fit(X_train_scaled, y_train); 
y_pred = modelo_rf.predict(X_test_scaled); 
acuracia = accuracy_score(y_test, y_pred); 

print("\n" + "="*70 + f"\n             🎯 Acurácia do Modelo (Random Forest): {acuracia:.2%}" + "\n" + "="*70 + "\n"); 

# --- FUNÇÃO PARA COLETAR OS DADOS DO USUÁRIO (O 11º JOGADOR) ---
def coletar_dados_usuario():
    dados_piloto = {}
    print("\n" + "="*70)
    print("     INÍCIO DA COLETA DE DADOS: VOCÊ É O 11º COMPETIDOR")
    print("="*70)
    
    # 1. Coleta do Nome
    nome_usuario = input("Digite seu nome ou apelido para o Ranking: ")
    
    # 2. Coleta das features numéricas
    features_input = [f for f in features if f != "tamanho_pista_km"]
    
    prompts = {
        "velocidade_media": "1. Velocidade Média (km/h, ex: 185.5): ",
        "arranque": "2. Tempo de Arranque 0-100 (segundos, ex: 3.2): ",
        "tempo_por_corrida": "3. Tempo Médio por Corrida (minutos, ex: 3.5): ",
        "tempo_jogo": "4. Total de Horas Jogadas (ex: 300): ",
        "porcentagem_vitorias": "5. Porcentagem de Corridas Vencidas (ex: 0.25 para 25%): ",
        "desvio_padrao_tempo_volta": "6. Desvio Padrão do Tempo de Volta (Consistência, ex: 0.15): ",
        "erros_por_corrida": "7. Erros/Batidas por Corrida (ex: 2): ",
        "razao_vitoria_derrota": "8. Razão Vitória/Derrota (Vitórias / Derrotas + 1, ex: 0.5): ",
    }

    for feature in features_input:
        try:
            valor = float(input(prompts[feature]))
            dados_piloto[feature] = valor
        except ValueError:
            raise ValueError("Entrada inválida. Por favor, use apenas números para todos os campos.")
    
    # Adiciona o valor padrão da pista e o nome
    dados_piloto["tamanho_pista_km"] = TAMANHO_PISTA_PADRAO_KM 
    dados_piloto["Jogador"] = nome_usuario
    return dados_piloto

# --- GERAÇÃO DOS 10 JOGADORES BASE ---
num_test = 10
partidas_total_test = np.random.randint(100, 400, num_test)
partidas_vencidas_test = np.round(partidas_total_test * np.clip(np.random.normal(0.25, 0.12, num_test), 0.05, 0.45))
partidas_perdidas_test = partidas_total_test - partidas_vencidas_test
razao_vitoria_derrota_test = partidas_vencidas_test / (partidas_perdidas_test + 1);

jogadores_teste_base = pd.DataFrame({
    "Jogador": NOMES_TESTE, # Nomes personalizados!
    "velocidade_media": np.random.normal(185, 12, num_test), 
    "arranque": np.random.normal(3.4, 0.3, num_test), 
    "tempo_por_corrida": np.random.normal(3.7, 0.4, num_test), 
    "tempo_jogo": np.random.normal(270, 50, num_test),
    "porcentagem_vitorias": np.clip(np.random.normal(0.20, 0.10, num_test), 0.0, 0.5),
    "desvio_padrao_tempo_volta": np.random.normal(0.18, 0.04, num_test),
    "erros_por_corrida": np.random.poisson(1.5, num_test),
    "razao_vitoria_derrota": razao_vitoria_derrota_test,
    "tamanho_pista_km": np.full(num_test, TAMANHO_PISTA_PADRAO_KM) 
});

# --- COLETA DE DADOS E COMBINAÇÃO ---
try:
    dados_usuario = coletar_dados_usuario()
    df_usuario = pd.DataFrame([dados_usuario], columns=["Jogador"] + features)

    df_comparacao = pd.concat([jogadores_teste_base, df_usuario], ignore_index=True)
    
    df_features_to_scale = df_comparacao[features]
    df_scaled = scaler.transform(df_features_to_scale)
    
    df_comparacao["ranking_previsto"] = modelo_rf.predict(df_scaled)
    ranking_final = df_comparacao.sort_values("ranking_previsto").reset_index(drop=True)
    
    print("\n" + "="*70)
    print("🏆 RANKING COMPARATIVO DOS 11 COMPETIDORES (1 = Melhor)")
    print("="*70)
    print(ranking_final[["Jogador", "ranking_previsto"]].to_string(index=False))

    # GRÁFICO 1: RANKING COMPARATIVO (Destacando o Usuário)
    plt.figure(figsize=(12, 7))
    cores = ['red' if j == dados_usuario["Jogador"] else 'darkgreen' for j in ranking_final['Jogador']]
    sns.barplot(x="Jogador", y="ranking_previsto", data=ranking_final, palette=cores, order=ranking_final["Jogador"])
    plt.title(f"Sua Posição de Ranking Comparada com a Elite (Pista: {TAMANHO_PISTA_PADRAO_KM} KM)", fontsize=16)
    plt.xlabel("Competidor")
    plt.ylabel("Ranking (1 = Melhor, 10 = Pior)")
    plt.yticks(range(1, 11))
    plt.gca().invert_yaxis()
    plt.grid(axis='y')
    plt.show()

except ValueError as e:
    print(f"\nERRO: {e}")

# --- RELATÓRIOS E GRÁFICOS DE AVALIAÇÃO DO MODELO (Finais) ---
importancia = pd.DataFrame({"Atributo": features, "Importância": modelo_rf.feature_importances_}).sort_values("Importância", ascending=False); 
print("\n" + "="*70 + "\n                 📝 RELATÓRIO DE IMPORTÂNCIA DOS ATRIBUTOS" + "\n" + "="*70 + "\n", importancia.to_string(index=False)); 
plt.figure(figsize=(10, 5)); sns.barplot(x="Importância", y="Atributo", data=importancia, palette="magma"); 
plt.title("Impacto Relativo dos Atributos no Resultado do Ranking", fontsize=16); 
plt.xlabel("Importância (Peso no Modelo)"); plt.ylabel("Atributo"); plt.show(); 
cm = confusion_matrix(y_test, y_pred); 
plt.figure(figsize=(8, 7)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=modelo_rf.classes_, yticklabels=modelo_rf.classes_); 
plt.title('Matriz de Confusão', fontsize=16); plt.xlabel('Ranking Previsto'); plt.ylabel('Ranking Real'); plt.show()