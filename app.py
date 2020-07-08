import streamlit as st

# NLP pkgs
import spacy
from textblob import TextBlob
from gensim.summarization.summarizer import summarize

# sumy pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# sumy function
def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence)for sentence in summary]
    result = ' '.join(summary_list)
    return result


def text_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)

    #tokens = [token.text for token in docx]
    allData = [('"Tokens":{},\n"Lemma":{}'.format(
        token.text, token.lemma_))for token in docx]
    return allData


def entity_analyzer(my_text):
    nlp = spacy.load('en')
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_)for entity in docx.ents]
    return entities


# pkgs


def main():
    """ NLP APP WITH STREAMLIT"""
    st.title("NLP with streamlit")
    st.subheader("Natural Language Processing on the Go")

    # tokenization
    if st.checkbox("show tokens and lemma", False):
        st.markdown("Tokenize your Text")
        message = st.text_area("Enter your text", "Type Here")
        if st.button("Analyze"):
            nlp_result = text_analyzer(message)
            st.json(nlp_result)

    # named entity
    if st.checkbox("show Named Entities", False):
        st.markdown("Extract entities from your Text")
        message = st.text_area("Enter ur text", "Type Here")
        if st.button("Extract"):
            nlp_result = entity_analyzer(message)
            st.json(nlp_result)
    # sentiment analysis
    if st.checkbox("show Sentiment Analysis", False):
        st.markdown("Sentiment of your Text")
        message = st.text_area("Enter text", "Type Here")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # text summarization
    if st.checkbox("show Text Summarization", False):
        st.markdown("Summarize of your Text")
        message = st.text_area("Enter text", "Type Here")
        summary_options = st.selectbox(
            "Choose your summarizer", ("gensim", "sumy"))
        if st.button("Summarize"):
            if summary_options == 'sumy':
                st.text("Using sumy...")
                summary_result = sumy_summarizer(message)
            elif summary_options == 'gensim':
                st.text("Using gensim summarizer")
                summary_result = summarize(message)
            else:
                st.warning("Using default summarizer")
                st.text("Using Gensim")
                summary_result = summarize(message)

            st.success(summary_result)


if __name__ == '__main__':
    main()
