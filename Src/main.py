import numpy as np
import tensorflow as tf
import nltk
import tkinter as tk
from tkinter import scrolledtext
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from gensim import corpora
from gensim.models import LdaModel, Word2Vec
from tensorflow import keras
from keras.backend import clear_session
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
def create_word2vec_model(text):
    sentences = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)
    return model
# 生成句子
def generate_sentence(model, tokenizer, keywords, max_length):
        seed_text = ' '.join(keywords)
        generated_text = seed_text  # 初始化生成的文本为种子文本
        for _ in range(max_length):
            seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
            seed_seq = keras.preprocessing.sequence.pad_sequences([seed_seq], maxlen=max_length)
            predicted_index = np.argmax(model.predict(seed_seq), axis=-1)
            predicted_word = tokenizer.index_word.get(str(predicted_index[0]), '')  # 获取预测的词
            if not predicted_word or predicted_word == '':  # 如果预测的词为空或空字符串，则退出循环
                break
            generated_text += " " + predicted_word
            seed_text = generated_text  # 更新种子文本为生成的文本
        return generated_text
# 修改print_generated_sentences函数，避免函数内部调用自身
def print_generated_sentences(model, tokenizer, keywords, max_length, num_sentences):
    for _ in range(num_sentences):
        generated_sentence = generate_sentence(model, tokenizer, keywords, max_length)
        result_text.config(state="normal")
        result_text.insert("end", f"Generated Sentence {_ + 1}:\n{generated_sentence}\n\n")
        result_text.config(state="disabled")
def process_text():
    text = text_input.get("1.0", "end-1c")
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    fdist = FreqDist(lemmatized_words)
    top_words = fdist.most_common(5)
    result_text.config(state="normal")
    result_text.delete("1.0", "end")
    result_text.insert("insert", "Sentences:\n")
    for sentence in sentences:
        result_text.insert("end", sentence + "\n")
    result_text.insert("end", "\nWords:\n")
    for word in words:
        result_text.insert("end", word + "\n")
    result_text.insert("end", "\n过滤后的单词:\n")
    for word in filtered_words:
        result_text.insert("end", word + "\n")
    result_text.insert("end", "\nLemmatized Words:\n")
    for word in lemmatized_words:
        result_text.insert("end", word + "\n")
    result_text.insert("end", "\n权重排行前五的单词:\n")
    for word, freq in top_words:
        result_text.insert("end", f"{word}: {freq}\n")
    result_text.config(state="disabled")
    dictionary = corpora.Dictionary([lemmatized_words])
    corpus = [dictionary.doc2bow(lemmatized_words)]
    lda_model = LdaModel(corpus, num_topics=2, id2word=dictionary, passes=30)
    topics = lda_model.print_topics(num_words=20)
    sorted_topics = sorted(topics, key=lambda x: x[0])
    result_text.config(state="normal")
    result_text.insert("end", "\nLDA Topics:\n")
    for topic in sorted_topics:
        result_text.insert("end", f"Topic {topic[0]}: {topic[1]}\n")
    result_text.config(state="disabled")
    w2v_model = create_word2vec_model(text)
    similar_words = w2v_model.wv.most_similar("island", topn=10)
    similar_words_weights = {word: freq * similarity for word, similarity in similar_words}
    result_text.config(state="normal")
    result_text.insert("end", "\n与 “island” 相似的单词及其重量:\n")
    for word, weight in similar_words_weights.items():
        result_text.insert("end", f"{word}: {weight}\n")
    result_text.config(state="disabled")
#文本生成部分
 #1. 提取LDA主题中的关键词
    keywords = []
    for topic in sorted_topics:
        words = topic[1].split("+")
        for word in words:
            keyword = word.split("*")[1].replace('"', '').strip()
            keywords.append(keyword)
 # 2.创建训练数据，将句子拆分成一对一对的形式
    train_data = []
    for i in range(len(sentences) - 1):
        train_data.append([sentences[i], sentences[i + 1]])
    # 创建一个空列表，用于存储训练数据的输入和目标
    train_X = []
    train_y = []
    # 遍历训练数据
    for data in train_data:
        # 将当前句子作为输入
        train_X.append(data[0])
        # 将下一句子作为目标
        train_y.append(data[1])
    # 创建词汇表
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_X)
    # 清除默认计算图
    clear_session()
    tf.compat.v1.reset_default_graph()
    # Create a Tokenizer object and fit it on the sentences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    train_X_seq = tokenizer.texts_to_sequences(train_X)
    train_y_seq = tokenizer.texts_to_sequences(train_y)
    max_len = max(max(len(seq) for seq in train_X_seq), max(len(seq) for seq in train_y_seq))
    train_X_seq = tf.keras.preprocessing.sequence.pad_sequences(train_X_seq, maxlen=max_len)
    train_y_seq = tf.keras.preprocessing.sequence.pad_sequences(train_y_seq, maxlen=max_len)
    # Convert train_y_seq to one-hot encoded format
    train_y_seq_one_hot = np.array([to_categorical(i, num_classes=len(tokenizer.word_index) + 1) for i in train_y_seq])
    # Clear the session and reset the default graph
    tf.keras.backend.clear_session()
    # Create and compile the model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len))
    model.add(LSTM(100, return_sequences=True))
    model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    model.fit(train_X_seq, train_y_seq_one_hot, epochs=10, batch_size=16)
    # 种子文本应该在生成句子之前就确定
    seed_text = text_input.get("1.0", "end-1c")
    # 设定生成句子的数量
    num_sentences = 1
    # 在模型训练完毕后，调用print_generated_sentences函数生成文本
    print_generated_sentences(model, tokenizer, keywords, max_len, num_sentences)
    result_text.config(state="normal")
    result_text.insert("end", "\nGenerated Sentences:\n")
    result_text.config(state="disabled")

root = tk.Tk()
root.title("自然语言处理(文本总结工具)")
# 样式
root.geometry("800x600")  # 设置窗口大小
root.configure(bg='#f0f0f0')  # 设置背景颜色
# 输入的文本框
text_label = tk.Label(root, text="Enter Text:", bg='#f0f0f0', font=("Helvetica", 14))
text_label.pack(pady=(20, 5))
text_input = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=10)
text_input.pack()
# 流程按钮
process_button = tk.Button(root, text="Process Text", command=process_text, font=("Helvetica", 12), bg='#4caf50', fg='white')
process_button.pack(pady=10)
# 结果文本
result_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=80, height=20)
result_text.pack()
result_text.config(state="disabled")
root.mainloop()