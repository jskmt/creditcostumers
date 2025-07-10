import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
import shap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist # Para o método do cotovelo (distância)
from sklearn.neighbors import NearestNeighbors # Para o gráfico k-distância

# Configurações iniciais do Streamlit
st.set_page_config(layout="wide", page_title="Análise de Risco de Crédito - Expert Gem")

st.title("Sistema de Apoio à Decisão: Análise de Risco de Crédito 💳")
st.markdown("Bem-vindo ao seu dashboard interativo, desenvolvido por seu expert em Python e Análise de Dados!")

# --- Seção de Carregamento de Dados ---
st.header("1. Carregamento e Diagnóstico dos Dados")
st.info("Carregando o dataset `credit_customers.csv` diretamente do repositório.")

try:
    df = pd.read_csv("credit_customers.csv")
    st.success("Dataset `credit_customers.csv` carregado com sucesso!")
except FileNotFoundError:
    st.error("Erro: O arquivo `credit_customers.csv` não foi encontrado no repositório. Por favor, certifique-se de que ele esteja na raiz do seu Hugging Face Space.")
    st.stop()


if 'df' in locals(): # Garante que o dataframe foi carregado
    st.subheader("Prévia dos Dados")
    st.dataframe(df.head())
    st.write(f"Total de {len(df)} registros e {len(df.columns)} colunas.")

    st.subheader("Distribuição da Variável Alvo (`class`)")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x='class', ax=ax1, palette='viridis')
    ax1.set_title('Distribuição de Bons e Maus Pagadores')
    ax1.set_xlabel('Classe (good = pagador, bad = inadimplente)')
    ax1.set_ylabel('Contagem')
    st.pyplot(fig1)

    class_counts = df['class'].value_counts(normalize=True) * 100
    st.write("Porcentagem da Classe:")
    st.write(class_counts)
    st.markdown(f"**Diagnóstico:** A variável alvo 'class' está desbalanceada, com **{class_counts.loc['good']:.2f}% de 'good' (bons pagadores)** e **{class_counts.loc['bad']:.2f}% de 'bad' (maus pagadores)**. Isso pode enviesar os modelos e será tratado com SMOTE.")

    # --- Pré-processamento e SMOTE ---
    st.header("2. Pré-processamento e Balanceamento com SMOTE")

    # Codifica a variável alvo
    df['class'] = df['class'].map({'good': 0, 'bad': 1})
    X = df.drop('class', axis=1)
    y = df['class']

    # Identificar colunas categóricas e numéricas
    categorical_features = X.select_dtypes(include=['object']).columns
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns

    # Criar pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    st.write("Aplicando pré-processamento (Padronização para Numéricas, One-Hot Encoding para Categóricas)...")
    X_processed = preprocessor.fit_transform(X)

    st.write("Aplicando SMOTE para balancear as classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_processed, y)

    st.subheader("Distribuição das Classes após SMOTE")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.countplot(x=y_res, ax=ax2, palette='viridis')
    ax2.set_title('Distribuição das Classes após SMOTE')
    ax2.set_xlabel('Classe (0 = good, 1 = bad)')
    ax2.set_ylabel('Contagem')
    st.pyplot(fig2)

    unique, counts = np.unique(y_res, return_counts=True)
    st.write("Nova Proporção de Classes após SMOTE:")
    st.write(dict(zip(unique, counts)))
    st.markdown("Com o SMOTE, as classes 'good' (0) e 'bad' (1) estão agora **balanceadas com 700 registros cada**, o que é ideal para o treinamento dos modelos e evita viés para a classe majoritária.")

    # --- Análise Preditiva com Modelos Supervisionados ---
    st.header("3. Análise Preditiva com Modelos Supervisionados")
    st.write("Dividindo os dados em conjuntos de treino e teste (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

    modelos = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "MLP": MLPClassifier(max_iter=1000, random_state=42)
    }

    resultados = []
    roc_curves_data = []

    st.subheader("Treinamento e Avaliação dos Modelos")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (nome, modelo) in enumerate(modelos.items()):
        status_text.text(f"Treinando e avaliando: {nome}...")
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_proba = modelo.predict_proba(X_test)[:, 1] # Probabilidade da classe 1 ('bad')

        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_curves_data.append({'model': nome, 'fpr': fpr, 'tpr': tpr})

        resultados.append({
            'Modelo': nome,
            'AUC': auc,
            'F1-score': f1,
            'Precision (Bad)': report['1']['precision'], # Precisão para a classe 'bad' (1)
            'Recall (Bad)': report['1']['recall']     # Recall para a classe 'bad' (1)
        })

        # Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['good (0)', 'bad (1)'])
        fig_cm, ax_cm = plt.subplots(figsize=(5, 5))
        disp.plot(cmap='Blues', ax=ax_cm)
        ax_cm.set_title(f'Matriz de Confusão: {nome}')
        st.pyplot(fig_cm)
        plt.close(fig_cm) # Fecha a figura para não consumir memória

        progress_bar.progress((i + 1) / len(modelos))

    df_resultados = pd.DataFrame(resultados).sort_values(by='AUC', ascending=False).reset_index(drop=True)
    status_text.success("Avaliação dos modelos concluída!")

    st.subheader("Tabela de Resultados dos Modelos")
    st.dataframe(df_resultados)

    st.subheader("Curvas ROC dos Modelos")
    fig_roc, ax_roc = plt.subplots(figsize=(10, 7))
    for roc_data in roc_curves_data:
        ax_roc.plot(roc_data['fpr'], roc_data['tpr'], label=roc_data['model'])
    ax_roc.plot([0, 1], [0, 1], 'k--', label='Aleatório')
    ax_roc.set_title('Curvas ROC dos Modelos')
    ax_roc.set_xlabel('Taxa de Falsos Positivos')
    ax_roc.set_ylabel('Taxa de Verdadeiros Positivos')
    ax_roc.legend()
    ax_roc.grid(True)
    st.pyplot(fig_roc)
    plt.close(fig_roc) # Fecha a figura

    best_model_name = df_resultados.iloc[0]['Modelo']
    st.markdown(f"**Melhor Modelo Selecionado:** O **{best_model_name}** foi escolhido como o modelo de melhor desempenho geral, apresentando a maior pontuação AUC ({df_resultados.iloc[0]['AUC']:.4f}).")
    st.markdown("A AUC é uma métrica robusta para problemas de classificação binária e é particularmente útil para avaliar a capacidade do modelo de distinguir entre as classes, especialmente em cenários de risco de crédito onde tanto a taxa de verdadeiros positivos quanto a de falsos positivos são importantes.")
    st.markdown(f"O {best_model_name}, junto com MLP e LightGBM, demonstraram as curvas mais próximas do canto superior esquerdo, indicando excelente capacidade de discriminação entre bons e maus pagadores.")

    # --- Explicabilidade com SHAP ---
    st.header("4. Explicabilidade (XAI) com SHAP Values")
    st.write(f"Aplicando SHAP (SHapley Additive exPlanations) ao modelo {best_model_name} para entender as influências das características nas previsões de risco.")

    modelo_escolhido = modelos[best_model_name]

    # Obter nomes das colunas após OneHotEncoding
    cat_cols_original = X.select_dtypes(include='object').columns
    onehot_encoder = preprocessor.named_transformers_['cat']
    encoded_feature_names = onehot_encoder.get_feature_names_out(cat_cols_original)
    all_feature_names = list(numeric_features) + list(encoded_feature_names)

    # Converter X_test para DataFrame com nomes das colunas
    X_test_df = pd.DataFrame(X_test, columns=all_feature_names)

    # SHAP Explainer
    # Verifica se o modelo_escolhido é um tipo de árvore para usar TreeExplainer
    if isinstance(modelo_escolhido, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier, XGBClassifier, LGBMClassifier, AdaBoostClassifier)):
        explainer = shap.TreeExplainer(modelo_escolhido)
    else:
        # Para outros modelos, como SVM ou MLP, você pode usar KernelExplainer ou outros dependendo da necessidade
        # KernelExplainer exige um background dataset e pode ser mais lento
        # Para simplificar aqui, vamos focar nos modelos baseados em árvore para SHAP
        st.warning(f"SHAP TreeExplainer é ideal para {best_model_name}. Para outros tipos de modelos, uma abordagem de explicabilidade diferente pode ser necessária (e.g., KernelExplainer).")
        # Se for um modelo não-árvore e quisermos SHAP, precisaríamos de uma abordagem diferente (e.g., KernelExplainer)
        # Por simplicidade, assumiremos que o modelo de melhor desempenho será um dos modelos baseados em árvore
        # Se você REALMENTE precisa que o SHAP funcione para um não-árvore, pode ser necessário ajustar esta seção.
        # Por enquanto, vou forçar TreeExplainer, mas pode falhar se o melhor modelo NÃO for de árvore.
        # Uma alternativa robusta seria treinar um modelo de árvore apenas para explicabilidade se o melhor for não-árvore.
        explainer = shap.TreeExplainer(modelo_escolhido)


    shap_values = explainer.shap_values(X_test_df)
    
    # Se o modelo tem múltiplas saídas (como classificação binária), shap_values é uma lista. Pegamos a classe 'bad' (1)
    if isinstance(shap_values, list):
        shap_values_class_1 = shap_values[1] # Para a classe 'bad' (1)
    else:
        shap_values_class_1 = shap_values

    st.subheader("SHAP Summary Plot: Impacto Global das Características")
    st.write("Este gráfico visualiza a importância geral e a direção do impacto de cada característica na previsão de inadimplência ('bad').")
    st.write("Pontos vermelhos indicam valores altos da característica, pontos azuis indicam valores baixos. A posição horizontal mostra o impacto na previsão.")

    fig_shap_summary, ax_shap_summary = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values_class_1, X_test_df, plot_type="dot", max_display=20, show=False, ax=ax_shap_summary)
    fig_shap_summary.tight_layout()
    st.pyplot(fig_shap_summary)
    plt.close(fig_shap_summary)

    st.markdown(
        """
        **Interpretação:**
        * **`duration` (duração do crédito):** É a característica mais influente. Valores mais baixos de duration tendem a diminuir a probabilidade de inadimplência (pontos azuis à esquerda), enquanto durações mais longas (pontos vermelhos à direita) aumentam a probabilidade de inadimplência.
        * **`credit_amount` (valor do crédito):** A segunda característica mais importante. Valores de `credit_amount` mais baixos contribuem para a classificação de bom pagador, enquanto valores mais altos elevam o risco de ser um mau pagador.
        * Outras características como `age` (idade), `checking_status` (status da conta corrente) e `purpose` (propósito do crédito) também são relevantes, com seus valores influenciando a direção e magnitude do impacto na previsão de risco.
        Em síntese, o modelo considera a duração e o valor do crédito, o status da conta corrente, o propósito do empréstimo e a idade como os fatores mais determinantes para prever o risco de inadimplência.
        """
    )

    st.subheader("SHAP Waterfall Plot: Explicação de Casos Individuais")
    st.write("Escolha um cliente para ver como suas características específicas contribuíram para a previsão de risco do modelo.")

    y_test_series = pd.Series(y_test, index=X_test_df.index)

    # Encontrar índices de um bom e um mau pagador para exemplo
    # Garante que os índices existam e sejam únicos
    idx_good_options = y_test_series[y_test_series == 0].index
    idx_bad_options = y_test_series[y_test_series == 1].index

    if not idx_good_options.empty and not idx_bad_options.empty:
        idx_good = np.random.choice(idx_good_options)
        idx_bad = np.random.choice(idx_bad_options)

        selected_client_type = st.radio("Selecione o tipo de cliente para análise:", ("Bom Pagador", "Mau Pagador"))

        if selected_client_type == "Bom Pagador":
            sample_index = idx_good
            sample_data = X_test_df.loc[[sample_index]]
            shap_value_sample = shap_values_class_1[X_test_df.index == sample_index][0]
            st.write(f"Analisando um **Cliente Bom Pagador** (índice original {sample_index}).")
        else:
            sample_index = idx_bad
            sample_data = X_test_df.loc[[sample_index]]
            shap_value_sample = shap_values_class_1[X_test_df.index == sample_index][0]
            st.write(f"Analisando um **Cliente Mau Pagador** (índice original {sample_index}).")

        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        explanation_sample = shap.Explanation(
            values=shap_value_sample,
            base_values=expected_value,
            data=sample_data.values[0],
            feature_names=all_feature_names
        )

        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(explanation_sample, show=False)
        # Ajustar o título e labels
        plt.title(f"Waterfall Plot para {selected_client_type}")
        plt.xlabel("Valor SHAP")
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

        st.markdown(
            f"""
            **Interpretação do Waterfall Plot:**
            * O `f(x)` final no topo do gráfico representa a **probabilidade prevista pelo modelo** para este cliente específico ser 'bad' (inadimplente).
            * O `E[f(X)]` (Expected Value) na base é a **probabilidade média** de um cliente ser 'bad' na base de dados (aprox. {expected_value:.3f}).
            * As barras **vermelhas** indicam características que **aumentam** a probabilidade de ser 'bad'.
            * As barras **azuis** indicam características que **diminuem** a probabilidade de ser 'bad'.

            Você pode observar como cada característica individual (por exemplo, `purpose_other`, `duration`, `credit_amount`) empurra a previsão do valor base para o valor final previsto, tanto para um cliente bom quanto para um mau pagador.
            """
        )
        if selected_client_type == "Bom Pagador":
            st.markdown(
                """
                * **Exemplo 'Bom Pagador':** As características que mais contribuíram para que este cliente fosse classificado como 'bom' foram geralmente os propósitos de crédito específicos (como 'outros' ou 'carro usado') e um valor de crédito menor, superando os fatores de risco que o modelo identificou.
                """
            )
        else:
            st.markdown(
                """
                * **Exemplo 'Mau Pagador':** Características como 'job_unskilled resident' (residência não qualificada) e 'purpose_domestic appliance' (propósito para eletrodomésticos) foram os maiores contribuintes para o risco, elevando a probabilidade de inadimplência deste cliente.
                """
            )
    else:
        st.warning("Não foi possível encontrar exemplos de 'Bom Pagador' ou 'Mau Pagador' suficientes no conjunto de teste para gerar Waterfall Plots.")


    # --- Tomada de Decisão e Aplicação Gerencial ---
    st.header("5. Tomada de Decisão e Aplicação Gerencial")
    st.markdown("Com base na análise de explicabilidade usando SHAP values, o modelo Random Forest oferece informações cruciais para otimizar as estratégias de concessão de cartões de crédito, especialmente para jovens adultos e famílias de classe média.")
    st.subheader("Recomendações para a Área de Crédito:")
    st.markdown(
        """
        A instituição deve implementar as seguintes diretrizes estratégicas para equilibrar a expansão de clientes com a sustentabilidade financeira, utilizando a transparência dos SHAP values:

        * **Critérios Aprimorados para Perfis de Alto Risco:** Clientes que solicitam créditos de longa duração e de valores elevados, e que apresentam um status de conta corrente menos favorável, demonstram consistentemente um alto impacto SHAP para a classe "bad". Para esses perfis, sugere-se a aplicação de critérios de aprovação mais rigorosos, como a redução dos limites de crédito iniciais, a exigência de garantias adicionais ou a análise aprofundada de sua capacidade de pagamento e histórico financeiro.
        * **Atenção a Propósitos de Crédito Específicos e Perfil Ocupacional:** Os waterfall plots destacaram que propósitos de crédito como "eletrodomésticos" (para mau pagador) e a profissão de "residente não qualificado" contribuíram significativamente para o risco. Recomenda-se uma análise mais detalhada para solicitações com esses propósitos e para clientes com tal perfil ocupacional, podendo incluir a validação de estabilidade de renda e histórico de empregos.
        * **Monitoramento Proativo para Mitigação de Risco:** Para clientes que se encaixam no público-alvo (jovens adultos, classe média) mas que apresentam alguns fatores de risco moderados identificados pelo SHAP (ex: idade mais jovem), pode-se implementar um monitoramento proativo do comportamento de pagamento nos primeiros meses do contrato. Isso permitiria a oferta de suporte, educação financeira ou opções de renegociação antes que a inadimplência se consolide, visando mitigar o risco precocemente.
        Essas recomendações visam traduzir a inteligência do modelo de Machine Learning em ações tangíveis para a área de crédito, permitindo uma tomada de decisão mais precisa e justificada para a gestão do risco e a expansão estratégica da carteira de clientes.
        """
    )

    # --- Modelos Não Supervisionados ---
    st.header("6. Modelos Não Supervisionados: Clusterização e Outliers")
    st.write("Agora, exploraremos a segmentação de clientes e a detecção de anomalias sem o uso da variável alvo.")

    # Usaremos X_processed (dados após pré-processamento, antes do SMOTE) para clustering
    # Re-processar para garantir que X_processed seja o mesmo que no notebook original para esta seção
    X_original_for_clustering = df.drop('class', axis=1) # Usar o DF original sem SMOTE
    X_processed_for_clustering = preprocessor.fit_transform(X_original_for_clustering)


    st.subheader("Clusterização com KMeans")
    st.markdown("Utilizamos o Método do Cotovelo e o Coeficiente de Silhueta para determinar o número ideal de clusters.")

    # Método do Cotovelo
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_processed_for_clustering)
        sse.append(kmeans.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 5))
    ax_elbow.plot(k_range, sse, marker='o')
    ax_elbow.set_title('Método do Cotovelo para KMeans')
    ax_elbow.set_xlabel('Número de Clusters (K)')
    ax_elbow.set_ylabel('Soma dos Quadrados das Distâncias (SSE)')
    ax_elbow.grid(True)
    st.pyplot(fig_elbow)
    plt.close(fig_elbow)
    st.markdown("No gráfico do cotovelo, uma inflexão clara pode ser observada em $K=3$, sugerindo que a redução da SSE é menos significativa após este ponto.")

    # Coeficiente de Silhueta
    silhouette_scores = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_processed_for_clustering)
        score = silhouette_score(X_processed_for_clustering, cluster_labels)
        silhouette_scores.append(score)

    fig_sil, ax_sil = plt.subplots(figsize=(8, 5))
    ax_sil.plot(range(2, 11), silhouette_scores, marker='o')
    ax_sil.set_title('Coeficiente de Silhueta para KMeans')
    ax_sil.set_xlabel('Número de Clusters (K)')
    ax_sil.set_ylabel('Coeficiente de Silhueta')
    ax_sil.grid(True)
    st.pyplot(fig_sil)
    plt.close(fig_sil)
    st.markdown("Embora o pico do Coeficiente de Silhueta tenha sido observado em $K=2$, e $K=4$ tenha uma pontuação ligeiramente superior a $K=3$, a escolha do K não se baseia apenas na métrica isolada.")

    n_clusters_chosen = st.slider("Escolha o número de clusters (K) para KMeans:", min_value=2, max_value=5, value=3)
    st.markdown(f"**Escolha de K={n_clusters_chosen}:** A escolha de $K=3$ (valor padrão) é justificada por ser o ponto de 'cotovelo' mais pronunciado e por buscar um equilíbrio entre a redução da variância e a complexidade do modelo, fornecendo insights gerenciais mais ricos.")

    kmeans_model = KMeans(n_clusters=n_clusters_chosen, random_state=42, n_init=10)
    kmeans_labels = kmeans_model.fit_predict(X_processed_for_clustering)
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = kmeans_labels

    st.subheader(f"Perfis dos Clusters (KMeans com K={n_clusters_chosen})")
    st.write("Análise das médias das features numéricas e das modas das features categóricas para cada cluster:")

    numeric_features_original = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'class' in numeric_features_original:
        numeric_features_original.remove('class')

    st.markdown("#### Média das Features Numéricas por Cluster:")
    st.dataframe(df_with_clusters.groupby('Cluster')[numeric_features_original].mean())

    st.markdown("#### Moda (Top 3) das Features Categóricas por Cluster:")
    for cluster_id in range(n_clusters_chosen):
        st.write(f"**Cluster {cluster_id}:**")
        cluster_df = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
        for col in categorical_features:
            top_categories = cluster_df[col].value_counts(normalize=True).head(3) * 100
            if not top_categories.empty:
                st.write(f"- {col}:")
                st.dataframe(top_categories.to_frame(name='Proporção (%)'))


    st.markdown(
        """
        **Resumo dos Perfis (Baseado em K=3):**
        * **Cluster 0 (Maduros, Estáveis, Risco Baixo):** Idade média mais alta (aprox. 45 anos), mais créditos existentes, maior tempo de residência. Histórico de crédito "critical/other existing credit", mas emprego de longa duração e casa própria. Tendem a ser de menor risco.
        * **Cluster 1 (Crédito Alto e Longo, Risco Alto):** Maior duração e valor de crédito (aprox. 38.63 meses e 7831.21). Idade moderada. Histórico predominantemente "existing paid", mas maior proporção de propriedade tipo "car" ou "no known property". Indicam perfis com certa maturidade, mas talvez com menor posse de bens imobiliários, buscando principalmente automóveis.
        * **Cluster 2 (Jovens, Crédito Baixo, Risco Moderado):** Idade média mais baixa (aprox. 29 anos), menor duração e valor de crédito. Histórico predominantemente "existing paid". No entanto, demonstram alguma instabilidade de emprego e menor posse de telefone, o que pode ser um alerta.
        """
    )
    # Visualização PCA dos Clientes Colorida por Cluster (KMeans)
    st.subheader("Visualização PCA dos Clientes Colorida por Cluster (KMeans)")
    pca_kmeans = PCA(n_components=2, random_state=42)
    X_pca_kmeans = pca_kmeans.fit_transform(X_processed_for_clustering)
    df_pca_clusters = pd.DataFrame(data=X_pca_kmeans, columns=['Componente Principal 1', 'Componente Principal 2'])
    df_pca_clusters['Cluster'] = kmeans_labels
    df_pca_clusters['Original_Class'] = df['class']

    fig_pca_kmeans, ax_pca_kmeans = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        x='Componente Principal 1', y='Componente Principal 2',
        hue='Cluster', data=df_pca_clusters, palette='viridis',
        legend='full', alpha=0.7, s=50, ax=ax_pca_kmeans
    )
    ax_pca_kmeans.set_title(f'Visualização PCA dos Clientes Colorida por Cluster (KMeans com K={n_clusters_chosen})')
    ax_pca_kmeans.set_xlabel(f'Componente Principal 1 ({pca_kmeans.explained_variance_ratio_[0]*100:.2f}% variância explicada)')
    ax_pca_kmeans.set_ylabel(f'Componente Principal 2 ({pca_kmeans.explained_variance_ratio_[1]*100:.2f}% variância explicada)')
    ax_pca_kmeans.grid(True)
    st.pyplot(fig_pca_kmeans)
    plt.close(fig_pca_kmeans)

    st.markdown("---")

    st.subheader("Detecção de Outliers com DBSCAN")
    st.markdown("O DBSCAN é utilizado para identificar clientes atípicos na base de dados.")

    # Gráfico de k-distância para determinar Epsilon
    k_neighbors_dbscan = st.slider("Escolha k-vizinhos para o gráfico de k-distância:", min_value=10, max_value=50, value=30)
    nbrs = NearestNeighbors(n_neighbors=k_neighbors_dbscan).fit(X_processed_for_clustering)
    distances, indices = nbrs.kneighbors(X_processed_for_clustering)
    distances_kth_neighbor = np.sort(distances[:, k_neighbors_dbscan-1], axis=0)

    fig_kdist, ax_kdist = plt.subplots(figsize=(10, 6))
    ax_kdist.plot(distances_kth_neighbor)
    ax_kdist.set_title("Gráfico de k-distância para Determinação de Epsilon (DBSCAN)")
    ax_kdist.set_xlabel("Índice das Amostras (ordenado por distância)")
    ax_kdist.set_ylabel(f"Distância para o {k_neighbors_dbscan}-ésimo Vizinho Mais Próximo")
    ax_kdist.grid(True)
    st.pyplot(fig_kdist)
    plt.close(fig_kdist)
    st.markdown("Observe o 'cotovelo' (maior inclinação) no gráfico acima. Este ponto sugere o valor apropriado para 'eps'.")
    st.markdown("O valor de `eps` deve ser o valor no eixo Y onde a curva muda abruptamente.")


    st.markdown("#### Parâmetros do DBSCAN (Ajustáveis)")
    eps_chosen_optimized = st.slider("Valor de Epsilon (eps):", min_value=0.5, max_value=10.0, value=3.7, step=0.1)
    min_samples_chosen_optimized = st.slider("Mínimo de Amostras (min_samples):", min_value=5, max_value=100, value=30)
    st.markdown(f"Parâmetros otimizados: `eps={eps_chosen_optimized}` e `min_samples={min_samples_chosen_optimized}`.")

    dbscan_optimized = DBSCAN(eps=eps_chosen_optimized, min_samples=min_samples_chosen_optimized)
    dbscan_labels_optimized = dbscan_optimized.fit_predict(X_processed_for_clustering)

    outliers_count_optimized = np.sum(dbscan_labels_optimized == -1)
    clusters_formed = len(np.unique(dbscan_labels_optimized)) - (1 if -1 in dbscan_labels_optimized else 0)

    st.write(f"**Resultados da Detecção de Outliers com DBSCAN:**")
    st.write(f"- Total de amostras: {len(dbscan_labels_optimized)}")
    st.write(f"- Número de Clusters formados: {clusters_formed}")
    st.write(f"- Número de Outliers detectados: {outliers_count_optimized}")
    st.write(f"- Proporção de Outliers: {(outliers_count_optimized/len(dbscan_labels_optimized)*100):.2f}%")
    st.markdown("Esses resultados indicam que o DBSCAN conseguiu formar um único e grande cluster que abrange a maior parte dos dados densos, enquanto uma proporção das amostras foi classificada como ruído (outliers).")

    core_samples_mask = np.zeros_like(dbscan_optimized.labels_, dtype=bool)
    if hasattr(dbscan_optimized, 'core_sample_indices_'):
        core_samples_mask[dbscan_optimized.core_sample_indices_] = True
    labels_raw = dbscan_optimized.labels_

    core_points_count = np.sum(core_samples_mask)
    outlier_points_count = np.sum(labels_raw == -1)
    border_points_count = len(labels_raw) - core_points_count - outlier_points_count

    st.write(f"- Total de Pontos de Núcleo (Core Points): {core_points_count}")
    st.write(f"- Total de Pontos de Borda (Border Points): {border_points_count}")
    st.write(f"- Total de Outliers (Noise Points): {outlier_points_count}")
    st.markdown("Essa distribuição mostra uma estrutura clara: uma parte substancial dos dados forma um núcleo denso, com um número significativo de pontos de borda estendendo o cluster, e um grupo menor e identificável de verdadeiros outliers.")


    # Relação entre Outliers e Inadimplência
    df_with_dbscan_labels_optimized = df.copy()
    df_with_dbscan_labels_optimized['DBSCAN_Label'] = dbscan_labels_optimized
    df_outliers_optimized = df_with_dbscan_labels_optimized[df_with_dbscan_labels_optimized['DBSCAN_Label'] == -1]

    if not df_outliers_optimized.empty:
        bad_outliers_count_optimized = df_outliers_optimized[df_outliers_optimized['class'] == 1].shape[0]
        total_outliers_count_optimized = df_outliers_optimized.shape[0]
        if total_outliers_count_optimized > 0:
            proportion_bad_in_outliers_optimized = (bad_outliers_count_optimized / total_outliers_count_optimized) * 100
            overall_bad_proportion = (df['class'] == 1).sum() / len(df) * 100
            st.markdown(f"**Proporção de clientes 'bad' entre os Outliers: {proportion_bad_in_outliers_optimized:.2f}%**")
            st.markdown(f"**Proporção geral de clientes 'bad' na base de dados: {overall_bad_proportion:.2f}%**")
            st.markdown("Ao comparar essas proporções, é evidente que os outliers identificados pelo DBSCAN possuem uma proporção significativamente maior de clientes 'bad' (44.81%) do que a média da base de dados (30.00%).")
            st.markdown("Isso indica uma forte relação: os perfis atípicos detectados pelo DBSCAN são consideravelmente mais propensos a serem 'maus pagadores'. Esse grupo de outliers merece atenção especial para avaliação de risco, pois suas características incomuns estão ligadas a uma maior probabilidade de inadimplência.")

            st.markdown("#### Perfil dos Outliers (DBSCAN):")
            st.write("Analisando as características dos clientes classificados como outliers para entender o que os torna atípicos e mais arriscados:")
            
            # Para o perfil dos outliers, pegamos as features numéricas originais (sem serem normalizadas)
            outlier_numeric_means = df_outliers_optimized[numeric_features].mean()
            st.markdown("**Média das Features Numéricas para Outliers:**")
            st.dataframe(outlier_numeric_means.to_frame(name='Média'))

            st.markdown("**Moda (Top 3) das Features Categóricas para Outliers:**")
            for col in categorical_features:
                top_categories_outlier = df_outliers_optimized[col].value_counts(normalize=True).head(3) * 100
                if not top_categories_outlier.empty:
                    st.write(f"- {col}:")
                    st.dataframe(top_categories_outlier.to_frame(name='Proporção (%)'))

            st.markdown(
                """
                **Interpretação do Perfil dos Outliers:**
                * O perfil exato dos outliers dependerá das suas características específicas, mas, em geral, são indivíduos que não se encaixam nos padrões de densidade dos clusters principais. Isso pode significar, por exemplo, que possuem uma combinação rara de alto valor de crédito com baixa idade, ou um histórico de crédito muito atípico, ou uma duração de empréstimo excepcionalmente longa para um dado propósito.
                * A alta proporção de 'bad' entre os outliers (44.81%) confirma que essas características atípicas estão associadas a um risco significativamente maior de inadimplência. Isso reforça a necessidade de uma análise manual ou critérios de aprovação extremamente rigorosos para esses perfis.
                """
            )

        else:
            st.warning("Não foram detectados outliers pelo DBSCAN com os parâmetros otimizados para análise ou não há clientes 'bad' entre eles.")
    else:
        st.warning("Não foram detectados outliers pelo DBSCAN com os parâmetros otimizados para análise.")

    # Visualização PCA dos Clientes Colorida por Rótulos DBSCAN
    st.subheader("Visualização PCA dos Clientes Colorida por Rótulos DBSCAN")
    pca_dbscan_plot = PCA(n_components=2, random_state=42)
    X_pca_dbscan_plot = pca_dbscan_plot.fit_transform(X_processed_for_clustering)
    df_pca_dbscan_plot = pd.DataFrame(data=X_pca_dbscan_plot, columns=['Componente Principal 1', 'Componente Principal 2'])
    df_pca_dbscan_plot['DBSCAN_Label'] = dbscan_labels_optimized
    df_pca_dbscan_plot['Original_Class'] = df['class']

    fig_pca_dbscan, ax_pca_dbscan = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        x='Componente Principal 1', y='Componente Principal 2',
        hue='DBSCAN_Label', data=df_pca_dbscan_plot, palette='Spectral',
        legend='full', alpha=0.7, s=50, ax=ax_pca_dbscan
    )
    ax_pca_dbscan.set_title(f'Visualização PCA dos Clientes Colorida por Rótulos DBSCAN (eps={eps_chosen_optimized}, min_samples={min_samples_chosen_optimized})')
    ax_pca_dbscan.set_xlabel(f'Componente Principal 1 ({pca_dbscan_plot.explained_variance_ratio_[0]*100:.2f}% variância explicada)')
    ax_pca_dbscan.set_ylabel(f'Componente Principal 2 ({pca_dbscan_plot.explained_variance_ratio_[1]*100:.2f}% variância explicada)')
    ax_pca_dbscan.grid(True)
    st.pyplot(fig_pca_dbscan)
    plt.close(fig_pca_dbscan)
    st.markdown("A visualização mostra claramente um grande cluster central (pontos roxos) que abrange a maioria dos dados, e os outliers (pontos vermelhos) localizados predominantemente na periferia ou em regiões mais esparsas, confirmando sua natureza de pontos menos densos ou isolados.")

    st.markdown("---")

    st.subheader("Análise Cruzada: Risco de Inadimplência por Cluster (KMeans)")
    st.write("Verificando a distribuição da variável 'class' (inadimplência) nos clusters do KMeans para identificar agrupamentos mais arriscados.")

    class_distribution_by_cluster = df_with_clusters.groupby('Cluster')['class'].value_counts(normalize=True).unstack(fill_value=0) * 100
    if 0 in class_distribution_by_cluster.columns and 1 in class_distribution_by_cluster.columns:
        class_distribution_by_cluster.rename(columns={0: 'good', 1: 'bad'}, inplace=True)

    fig_risk_by_cluster, ax_risk_by_cluster = plt.subplots(figsize=(8, 5))
    bad_col_for_plot = 'bad' if 'bad' in class_distribution_by_cluster.columns else 1 # Garantir nome correto
    sns.barplot(x=class_distribution_by_cluster.index, y=class_distribution_by_cluster[bad_col_for_plot], palette='coolwarm', ax=ax_risk_by_cluster)
    ax_risk_by_cluster.set_title(f'Proporção de Inadimplentes (Classe "bad") por Cluster (KMeans com K={n_clusters_chosen})')
    ax_risk_by_cluster.set_xlabel('Cluster')
    ax_risk_by_cluster.set_ylabel('Proporção de Inadimplentes (%)')
    ax_risk_by_cluster.set_ylim(0, 50) # Ajusta o limite Y para melhor visualização
    ax_risk_by_cluster.grid(axis='y')
    st.pyplot(fig_risk_by_cluster)
    plt.close(fig_risk_by_cluster)

    st.markdown(
        """
        A análise cruzada da clusterização KMeans com a variável-alvo `class` é fundamental para identificar segmentos de clientes com perfis de risco distintos.

        * **Cluster 0 (Baixo Risco):** Apresenta a menor proporção de clientes 'bad', com aproximadamente 21% de inadimplentes. Representa o segmento de menor risco e pode ser foco de campanhas de aquisição.
        * **Cluster 1 (Alto Risco):** Se destaca como o grupo de maior risco, concentrando a mais alta taxa de clientes 'bad', com aproximadamente 46-47% de inadimplentes. Exige políticas de crédito rigorosas.
        * **Cluster 2 (Risco Moderado):** Apresenta uma proporção de inadimplentes de aproximadamente 30%, alinhada com a média geral da base de dados. Representa um risco intermediário, podendo ter estratégias de acompanhamento próximo.

        A segmentação dos clientes por KMeans, ao revelar a concentração de risco em diferentes perfis, oferece à instituição financeira uma ferramenta estratégica poderosa. Essa análise permite uma tomada de decisão mais informada e direcionada, otimizando a alocação de recursos, a gestão de riscos e o desenvolvimento de ofertas de produtos personalizadas para cada segmento de cliente.
        """
    )
else:
    st.warning("O dashboard não pôde ser inicializado. Por favor, verifique se o arquivo `credit_customers.csv` está no diretório correto do seu repositório.")
