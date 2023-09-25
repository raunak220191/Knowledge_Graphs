import PyPDF2
import gensim
from gensim import corpora
import networkx as nx
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        for page_num in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_num)
            text += page.extractText()
    return text

# Function to perform LDA topic modeling
def perform_lda(text, num_topics=5):
    # Tokenize and preprocess the text
    text = [token.lower() for token in gensim.utils.simple_preprocess(text) if len(token) > 3]

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary([text])
    corpus = [dictionary.doc2bow(text)]

    # Perform LDA
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # Print topics and their keywords
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    return lda_model

# Function to create a knowledge graph from LDA topics
def create_graph_from_lda_topics(lda_model):
    G = nx.Graph()
    topics = lda_model.show_topics(formatted=False)

    for topic_id, topic in topics:
        topic_name = f"Topic {topic_id}"
        G.add_node(topic_name, entity_type="Topic")

        for word, word_score in topic:
            G.add_node(word, entity_type="Word")
            G.add_edge(topic_name, word, weight=word_score)

    return G

# Function to visualize the knowledge graph
def visualize_graph(graph):
    pos = nx.spring_layout(graph, seed=42)
    labels = {node: node for node in graph.nodes()}

    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=1000, font_size=10, font_color="black")
    plt.title("Knowledge Graph")
    plt.show()

# Main script
if __name__ == "__main__":
    pdf_path = "your_pdf_file.pdf"  # Replace with the path to your PDF file
    pdf_text = extract_text_from_pdf(pdf_path)

    lda_model = perform_lda(pdf_text, num_topics=5)

    knowledge_graph = create_graph_from_lda_topics(lda_model)
    
    visualize_graph(knowledge_graph)
