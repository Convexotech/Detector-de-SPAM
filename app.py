"""
Detector de Spam em E-mails - Streamlit + TensorFlow
----------------------------------------------------
Aplicação interna para classificar mensagens de e-mail como SPAM ou LEGÍTIMA.

Como funciona (visão geral):
1. O texto é convertido em números (TextVectorization).
2. Cada palavra vira um vetor (Embedding) e os vetores são resumidos (média).
3. Camadas densas aprendem o padrão e devolvem uma probabilidade de spam (0 a 1).

Executar localmente:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Detector de Spam | Marketing",
    page_icon="📧",
    layout="centered",
)

# ---------------------------------------------------------------------------
# 2. DADOS DE TREINO INICIAIS
# Base pequena, apenas para demonstração. Para uso real, carregue uma base
# maior pela barra lateral (CSV com colunas: texto, rotulo).
# rotulo: 1 = spam | 0 = legítima
# ---------------------------------------------------------------------------
DADOS_BASE = [
    # --- SPAM (1) ---
    ("Parabéns! Você ganhou um prêmio de R$ 10.000. Clique aqui agora para resgatar", 1),
    ("URGENTE: sua conta será bloqueada. Confirme seus dados neste link imediatamente", 1),
    ("Ganhe dinheiro fácil trabalhando de casa, sem experiência. Cadastre-se grátis", 1),
    ("Oferta imperdível! 90% de desconto só hoje, compre já e receba brinde grátis", 1),
    ("Você foi selecionado para receber um cartão de crédito sem consulta ao SPC", 1),
    ("Emagreça 10 kg em 7 dias com este método secreto. Clique e garanta o seu", 1),
    ("Seu empréstimo foi pré-aprovado! Dinheiro na conta em 5 minutos, sem burocracia", 1),
    ("Atenção: atualize sua senha do banco agora ou perderá o acesso à conta", 1),
    ("Remédio sem receita com preço baixo, entrega rápida, compre agora", 1),
    ("Ganhe seguidores e curtidas de graça, clique no link e receba hoje", 1),
    ("Último aviso: você tem uma herança de milhões esperando, responda este e-mail", 1),
    ("Promoção exclusiva, clique para ganhar um iPhone grátis, vagas limitadas", 1),
    ("Congratulations! You won a free gift card. Click here to claim your prize now", 1),
    ("Urgent: verify your account password immediately or it will be suspended", 1),
    ("Make money fast from home, no experience needed, limited offer, sign up free", 1),
    ("Cheap meds online without prescription, buy now and save 80 percent", 1),
    ("You have been selected for a free cruise, call now to confirm your winnings", 1),
    ("Lowest price guaranteed, act now, this deal expires today, click the link", 1),
    ("Dinheiro rápido, investimento com lucro garantido de 300% ao mês, entre agora", 1),
    ("Seu pacote está retido, pague a taxa neste link para liberar a entrega", 1),
    # --- LEGÍTIMAS (0) ---
    ("Olá equipe, segue a ata da reunião de ontem com os próximos passos do projeto", 0),
    ("Bom dia! Podemos agendar a revisão da campanha para quinta-feira às 14h?", 0),
    ("Anexei o relatório mensal de desempenho das redes sociais para sua avaliação", 0),
    ("Lembrete: o treinamento de ferramentas de análise acontece amanhã às 10h", 0),
    ("Obrigado pelo retorno. Vou ajustar o orçamento conforme conversamos e reenviar", 0),
    ("Segue o contrato revisado para assinatura. Qualquer dúvida estou à disposição", 0),
    ("Pessoal, a apresentação para o cliente está pronta. Podem revisar até sexta?", 0),
    ("Confirmo o recebimento da nota fiscal. O pagamento será feito no prazo combinado", 0),
    ("Atualização do cronograma: a entrega do layout foi movida para a próxima semana", 0),
    ("Oi, tudo bem? Gostaria de marcar uma conversa sobre o planejamento do trimestre", 0),
    ("Prezados, encaminho em anexo a pauta da reunião de alinhamento de segunda-feira", 0),
    ("Parabéns pelo excelente resultado da campanha, o time está de parabéns", 0),
    ("Hi team, please find attached the agenda for tomorrow's planning meeting", 0),
    ("Thanks for your feedback. I will update the proposal and send it by Friday", 0),
    ("Reminder: the quarterly report is due next week, let me know if you need help", 0),
    ("Could we reschedule our call to Thursday afternoon? I have a conflict today", 0),
    ("Attached is the invoice for last month's services. Please confirm receipt", 0),
    ("Good morning, the client approved the creative brief. We can start production", 0),
    ("Segue o link da reunião de hoje às 15h pelo Google Meet, até já", 0),
    ("Poderia me enviar os dados de conversão do último mês para o relatório?", 0),
]

TAMANHO_VOCABULARIO = 5000
TAMANHO_SEQUENCIA = 60
SEMENTE = 42


# ---------------------------------------------------------------------------
# 3. MODELO (treinamento com cache)
# O @st.cache_resource garante que o modelo é treinado UMA vez e reutilizado
# em todas as interações, em vez de retreinar a cada clique.
# ---------------------------------------------------------------------------
def construir_modelo(textos: list[str]) -> tf.keras.Model:
    """Cria o modelo: texto bruto -> probabilidade de spam."""
    # Converte texto em sequência de números inteiros (um por palavra).
    vetorizador = tf.keras.layers.TextVectorization(
        max_tokens=TAMANHO_VOCABULARIO,
        output_sequence_length=TAMANHO_SEQUENCIA,
        standardize="lower_and_strip_punctuation",
    )
    vetorizador.adapt(tf.constant(textos))  # aprende o vocabulário

    modelo = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(1,), dtype=tf.string),
            vetorizador,
            # Cada palavra vira um vetor de 16 números que o modelo ajusta.
            tf.keras.layers.Embedding(TAMANHO_VOCABULARIO, 16),
            # Resume a mensagem inteira em um único vetor (média das palavras).
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dropout(0.3),  # reduz decoreba (overfitting)
            # Saída: valor entre 0 (legítima) e 1 (spam).
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


@st.cache_resource(show_spinner="Treinando o modelo de identificação de spam...")
def treinar_modelo(textos: tuple, rotulos: tuple):
    """Treina o modelo e devolve (modelo, métricas)."""
    tf.keras.utils.set_random_seed(SEMENTE)

    textos = list(textos)
    rotulos = np.array(rotulos, dtype="float32")

    # Embaralha os dados para a separação treino/teste não ficar viciada.
    indices = np.random.default_rng(SEMENTE).permutation(len(textos))
    textos = [textos[i] for i in indices]
    rotulos = rotulos[indices]

    # Com base grande, reservamos 20% para medir a qualidade em dados "novos".
    usar_teste = len(textos) >= 100
    corte = int(len(textos) * 0.8) if usar_teste else len(textos)

    x_treino = tf.constant([[t] for t in textos[:corte]])
    y_treino = rotulos[:corte]

    modelo = construir_modelo(textos[:corte])
    modelo.fit(x_treino, y_treino, epochs=150, batch_size=8, verbose=0)

    metricas = {"base_treino": corte, "acuracia_teste": None}
    if usar_teste:
        x_teste = tf.constant([[t] for t in textos[corte:]])
        _, acc = modelo.evaluate(x_teste, rotulos[corte:], verbose=0)
        metricas["acuracia_teste"] = float(acc)

    return modelo, metricas


def prever_spam(modelo: tf.keras.Model, mensagem: str) -> float:
    """Retorna a probabilidade (0 a 1) de a mensagem ser spam."""
    entrada = tf.constant([[mensagem]])
    return float(modelo.predict(entrada, verbose=0)[0][0])


# ---------------------------------------------------------------------------
# 4. BARRA LATERAL (configurações)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configurações")

    limite = st.slider(
        "Sensibilidade (limite de spam)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Acima deste valor a mensagem é classificada como SPAM. "
        "Valores menores tornam o filtro mais rigoroso.",
    )

    st.divider()
    st.subheader("Base de treinamento")
    arquivo = st.file_uploader(
        "Enviar CSV próprio (opcional)",
        type="csv",
        help="O arquivo deve ter as colunas: texto, rotulo (1 = spam, 0 = legítima).",
    )

# Define qual base usar: a enviada pelo usuário ou a base de demonstração.
df = pd.DataFrame(DADOS_BASE, columns=["texto", "rotulo"])
origem_dados = "Base de demonstração"

if arquivo is not None:
    try:
        df_up = pd.read_csv(arquivo)
        if not {"texto", "rotulo"}.issubset(df_up.columns):
            raise ValueError("Colunas obrigatórias: texto, rotulo")
        df_up = df_up[["texto", "rotulo"]].dropna()
        df_up["texto"] = df_up["texto"].astype(str)
        df_up["rotulo"] = df_up["rotulo"].astype(int)
        if df_up["rotulo"].nunique() < 2 or len(df_up) < 20:
            raise ValueError("Envie ao menos 20 linhas contendo spam e não spam.")
        df, origem_dados = df_up, f"CSV enviado ({len(df_up)} mensagens)"
    except Exception as erro:
        st.sidebar.error(f"Arquivo inválido: {erro}")

modelo, metricas = treinar_modelo(tuple(df["texto"]), tuple(df["rotulo"]))

with st.sidebar:
    st.caption(f"Origem: {origem_dados}")
    if metricas["acuracia_teste"] is not None:
        st.metric("Acurácia em dados de teste", f"{metricas['acuracia_teste']:.0%}")
    else:
        st.caption("Base pequena: sem separação de teste. Use um CSV maior para medir a acurácia.")

# ---------------------------------------------------------------------------
# 5. INTERFACE PRINCIPAL
# ---------------------------------------------------------------------------
st.title("📧 Detector de Spam")
st.write("Cole o conteúdo da mensagem abaixo para verificar se ela é spam.")

mensagem = st.text_area(
    "Mensagem do e-mail",
    height=200,
    placeholder="Cole aqui o assunto e o corpo do e-mail...",
)

if st.button("Analisar mensagem", type="primary", use_container_width=True):
    if not mensagem.strip():
        st.warning("Digite ou cole uma mensagem para análise.")
    else:
        prob = prever_spam(modelo, mensagem.strip())
        eh_spam = prob >= limite

        st.divider()
        if eh_spam:
            st.error("🚫 **SPAM** — esta mensagem tem características de spam.")
            confianca = prob
        else:
            st.success("✅ **NÃO É SPAM** — a mensagem parece legítima.")
            confianca = 1 - prob

        col1, col2 = st.columns(2)
        col1.metric("Probabilidade de spam", f"{prob:.1%}")
        col2.metric("Confiança da classificação", f"{confianca:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        st.caption(
            "Resultado gerado por modelo de IA. Use como apoio à decisão "
            "e revise casos próximos do limite."
        )
