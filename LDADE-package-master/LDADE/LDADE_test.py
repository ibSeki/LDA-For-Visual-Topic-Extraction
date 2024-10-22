# -*- coding: utf-8 -*-
import os
import numpy as np
import operator
import codecs
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from collections import Counter
from LDADE import LDADE, UserTestConfig

# Downloads necessários do NLTK
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Carregar o modelo de linguagem do spaCy para português
try:
    nlp_pt = spacy.load('pt_core_news_sm')
except OSError:
    spacy.cli.download('pt_core_news_sm')
    nlp_pt = spacy.load('pt_core_news_sm')

from nltk.stem import WordNetLemmatizer

def read_file(filename='', language='english'):
    documents = []

    # Abrir o arquivo com a codificação UTF-8
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                processed_line = preprocess_text(line.strip(), language)
                documents.append(processed_line)

    return documents  # Retorna lista de documentos

def preprocess_text(text, language='english'):
    if language == 'portuguese':
        stop_words = set(stopwords.words('portuguese'))
        doc = nlp_pt(text.lower())
        processed = [token.lemma_ for token in doc if token.lemma_ not in stop_words and token.is_alpha]
    else:
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        words = nltk.word_tokenize(text.lower())
        processed = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(processed)

# Função para identificar possíveis TGWs (sem remover)
def identify_possible_tgws(topics, threshold=1):
    word_topic_count = Counter()

    # Contabilizar em quantos tópicos cada palavra aparece
    for topic in topics:
        unique_words_in_topic = set(topic)
        for word in unique_words_in_topic:
            word_topic_count[word] += 1

    # Selecionar palavras que aparecem em mais de 'threshold' tópicos
    possible_tgws = [word for word, count in word_topic_count.items() if count > threshold]
    return possible_tgws

# Função para execução do LDA
def perform_lda(data_samples, num_words, random_state, n_components, doc_topic_prior, topic_word_prior, language, manual_tgws, threshold=1):
    stop_words = stopwords.words(language)

    # Adicionar TGWs manuais à lista de stop words
    stop_words.extend(manual_tgws)

    # Vetorização do texto
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words=stop_words, token_pattern=r'(?u)\b\w+\b')
    tf = tf_vectorizer.fit_transform(data_samples)
    
    # Configurar e executar o LDA
    lda = LDA(n_components=n_components, doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior, random_state=random_state)
    lda.fit(tf)

    tf_feature_names = tf_vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        topics.append([tf_feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]])

    # Identificar possíveis TGWs (sem removê-las)
    possible_tgws = identify_possible_tgws(topics, threshold=threshold)
    print(f"Possíveis TGWs encontradas (aparecem em mais de {threshold} tópicos): {possible_tgws}")

    # Exibir TGWs removidas manualmente
    print(f"TGWs manuais removidas: {manual_tgws}")

    # Criar um dicionário Gensim a partir dos dados pré-processados
    dictionary = Dictionary([doc.split() for doc in data_samples])

    # Avaliar a coerência dos tópicos
    coherence_model_lda = CoherenceModel(topics=topics, texts=[doc.split() for doc in data_samples], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coerência dos tópicos: {coherence_lda}")

    return topics

# Função principal para demonstrar o LDA com TGWs manuais e detecção de TGWs automáticas
def demo(filename, num_words, n_topics, language='english'):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    data_path = os.path.join(data_dir, filename)

    if not os.path.exists(data_path):
        print(f"Arquivo não encontrado: {data_path}")
        return
    else:
        print(f"Arquivo encontrado: {data_path}")

    # Ler e pré-processar os dados
    data = read_file(data_path, language)
    what = UserTestConfig()
    what["data_samples"] = data

    # Obter parâmetros otimizados do LDADE
    val = LDADE(what)
    print("Parâmetros Otimizados e Pontuação de Fitness:")
    print(val)

    # Inserção manual de TGWs
    manual_tgws = ['gente', 'aqui', 'então', 'agora', 'beleza', 'poder', 'pegar', 'outro', 'ir', 'pessoal', 'onde', 'uh', 'say', 'q', 'cara', 'cima', 'algo', 'aceitar', 'colocar', 'aspa', 'assim', 'coloco', 'tentar',
                   'ainda', 'sobre', 'preciso', 'bom', 'ideia', 'querer', 'tô', 'olhar', 'receber', 'bom', 'som', 'fazer', 'né', 'ficar', 'dar', 'dizer', 'lá', 'ficar', 'ó', 'exemplo', 'deixar', 'algum',
                   'né', 'aí', 'dia', 'pouco', 'falar', 'tá', 'ver', 'assim', 'porque', 'seguir', 'conseguir', 'tentar', 'tanto', 'vez', 'vaso', 'botor', 'novamente', 'último', 'galerar', 'voltar', 'vejo',
                   'rei', 'bem', 'botar', 'joão', 'tudo', 'ainda', 'lado', 'sentido', 'hudson', 'dois', 'igual', 'parecer', 'saber', 'táxi', 'vídeo', 'mateus', 'basicamente', 'todo', 'hoje', 'converso',
                   'claro', 'cada', 'chegar', 'parte', 'correr', 'tio', 'via', 'regra', 'idade', 'canal', 'sempre', 'like', 'valer', 'bacano', 'precinho', 'sim', 'mestre', 'entender', 'interessante', 'direto',
                   'fácil', 'errar', 'precisar', 'ok', 'vários', 'play', 'fim', 'questão', 'matheu', 'tapete', 'meio', 'deus', 'comentar', 'ponto', 'bor', 'bota', 'beck', 'sol', 'inscrever', 'cidade', 'normalmente',
                   'criei', 'corretamente', 'vale', 'pessoa', 'acontecer', 'prático', 'volta', 'reclamar', 'acabar', 'pe', 'importante', 'profissão', 'hobby', 'hobbies', 'continuar', 'impacto', 'boto',
                   'quase', 'vila', 'palco', 'cá', 'carro', 'bt', 'indicar', 'topo', 'exatamente', 'direito', 'começar', 'direita', 'maiúsculo', 'creme', 'oh', 'ml', 'levar', 'ter', 'coisa', 'fizemos',
                   'gravamos', 'aceder', 'oi', 'and', 'logo', 'médico', 'dever', 'três', 'dois', 'um', 'vir', 'metro', 'zero', 'parar', 'partir', 'momento', 'ambos', 'repar', 'básico', 'grude', 'quatro',
                   'mandar', 'bastar', 'mar', 'tchau', 'ente', 'passo', 'tal', 'quanto', 'certo', 'fiz', 'jogar', 'nada', 'inscrito', 'professor', 'crio', 'ali', 'desculpa', 'ponho', 'apostador',
                   'ah', 'cemto', 'colar', 'ex', 'verdinho', 'stephanie', 'quê', 'segundo', 'bosta', 'certinho', 'caixinha', 'porquê', 'pra', 'tar', 'audi', 'legal', 'cento', 'pequeno', 'oitenta',
                   'playzinho', 'após', 'presente', 'pouquinho', 'realmente', 'cinco', 'hein', 'garantido', 'aposta', 'intuito', 'primeiro', 'y', 'seguida', 'l', 'tão', 'diante', 'frango', 'além', 'amiga',
                   'esc', 'cê', 'mês', 'cerveja', 'incrível', 'feliz', 'terminar', 'rack', 'zé', 'jardim', 'jesus', 'frete', 'basquete', 'zinho', 'pé', 'usar', 'absoluto', 'fael', 'testir', 'gatinha',
                   'léia', 'apenas', 'sete', 'demais', 'quanta', 'trás', 'informei', 'nunca', 'pouquíssima', 'preocupar', 'mai', 'antes', 'marco', 'bitencourt', 'senão', 'invés', 'verer',
                   'coder', 'the', 'i', 'of', 'which', 'so', 'to', 'see', 'let', 'it', 'thi', 'we', 'that', 'here', 'is', 'an', 'but', 'really', 'from',
                   'are', 'from', 'be', 'in', 'if', 'because', 'going', 'one', 'into', 'my', 'now', 'want', 'can', 'gonna', 'go', 'on',
                   'or', 'you', 'make', 'just', 'thing', 'only', 'also', 'went', 'doing', 'what', 'will', 'give', 'being', 'well', 'with', 'they',
                   'us', 'out', 'there', 'have', 'v', 'hello', 'these', 'then', 'when', 'your', 'our', 'doe', 'john', 'again',
                   'guys', 'okay', 'at', 'once', 'up', 'ahead', 'has', 'herir', 'two', 'b', 'f', 'green', 'way', 'd', 'how', 'than', 'by', 'per'
                   , 'comer', 'guy', 'four', 'today', 'goes', 'was', 'same', 'any', 'g', 'ot', 'too', 'am', 'need', 'five', 'seven', 'six', 'eight',
                   'did', 'yeah', 'car', 'using', 'many', 'very', 'all', 'egg', 'thar', 'white', 'adar']  # TGWs manuais

    # Realizar análise LDA
    doc_topic_prior = val[0]['doc_topic_prior']
    topic_word_prior = val[0]['topic_word_prior']

    topics = perform_lda(data_samples=what["data_samples"], num_words=num_words, random_state=what['random_state'], n_components=n_topics, doc_topic_prior=doc_topic_prior, topic_word_prior=topic_word_prior, language=language, manual_tgws=manual_tgws, threshold=1)

    print("\nTópicos Descobertos:")
    for i, topic in enumerate(topics):
        print(f"Tópico {i+1}: {', '.join(topic)}")

if __name__ == "__main__":
    filename = 'JacksonIG.txt'  # Arquivo TXT para análise
    num_words = 10  # QTD de palavras por tópico
    n_topics = 5  # QTD de tópicos
    language = 'portuguese'  # Defina o idioma como 'english' ou 'portuguese'
    demo(filename=filename, num_words=num_words, n_topics=n_topics, language=language)
