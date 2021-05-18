import streamlit as st

from models import summarize_with_lsa, summarize_with_tf_idf

st.title('Text Summarization Demo')
st.markdown('Using BART and T5 transformer model')

model = st.selectbox('Select the model', ('LSA', 'TF-idf'))
if model == 'LSA':
    _num_sentences = 5

    col1 = st.beta_columns(1)
    _num_sentences = col1[0].number_input("_num_sentences", value=_num_sentences)
elif model == 'TF-idf':
    _num_sentences = 5

    col1 = st.beta_columns(1)
    _num_sentences = col1[0].number_input("_num_sentences", value=_num_sentences)


text = st.text_area('Text Input')


def run_model(input_text, _num_sentences):
    if model == 'LSA':
        output = summarize_with_lsa(str(input_text), _num_sentences)
        st.write('Summary')
        st.success(output)

    if model == 'TF-idf':
        output = summarize_with_tf_idf(str(input_text))
        st.write('Summary')
        st.success(output)


if st.button('Submit'):
    run_model(text, _num_sentences)