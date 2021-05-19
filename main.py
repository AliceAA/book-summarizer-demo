import streamlit as st
from text_prepro import *
from text_models import summarize_with_lsa, summarize_with_tf_idf
from summarize import TextSummarizer

st.title('Extractive Text Summarization')

text = st.text_area('Provide the text to summarize:')

prepro_choice = ['Convert to lowercase', 'Stemming', 'Lemmatization']
prepro_pipe = st.multiselect('Select text preprocessing techniques:', prepro_choice)

lowercase = False
stem = False
lemm = ''

if 'Convert to lowercase' in prepro_pipe:
    lowercase = True
if 'Stemming' in prepro_pipe:
    stem = True
if 'Lemmatization' in prepro_pipe:
    lemm = 'spacy'


def prepro_review(input_text, max_rows=5):
    preview_sent = text2sentences(input_text)
    n = min(max_rows, len(preview_sent))
    preview_text = ' '.join(preview_sent[:n])
    prepro_output = normalize_text(preview_text, lowercase=lowercase, stemming=stem, lemmatize_method=lemm)
    st.markdown('<br>'.join(prepro_output), unsafe_allow_html=True)


if st.button('Preview preprocessing'):
    prepro_review(text)

model = st.selectbox('Select the summarization model', ('TextRank', 'other'))

summary_size = st.slider('Select summary size with respect to the initial text (%): ',
                         min_value=1, max_value=99, step=1)

def run_summarizer(input_text):
    ts = TextSummarizer(input_text, summary_size)
    if model == 'TextRank':
        model_output = ts.summarize(lowercase=lowercase, stemming=stem, lemmatize_method=lemm)
        st.markdown(model_output, unsafe_allow_html=True)
    else:
        st.markdown('Hello')


if st.button('Submit'):
    run_summarizer(text)

# # if model == 'LSA':
# #     _num_sentences = 5
# #
# #     col1 = st.beta_columns(1)
# #     _num_sentences = col1[0].number_input("_num_sentences", value=_num_sentences)
# # elif model == 'TF-idf':
# #     _num_sentences = 5
# #
# #     col1 = st.beta_columns(1)
# #     _num_sentences = col1[0].number_input("_num_sentences", value=_num_sentences)
# def run_model(input_text, _num_sentences=5):

    # if model == 'LSA':
    #     output = summarize_with_lsa(str(input_text), _num_sentences)
    #     st.write('Summary')
    #     st.success(output)
    #
    # if model == 'TF-idf':
    #     output = summarize_with_tf_idf(str(input_text))
    #     st.write('Summary')
    #     st.success(output)


# if st.button('Submit'):
#     run_model(text)
