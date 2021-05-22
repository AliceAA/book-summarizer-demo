# book-summarizer-demo
This repo contains implementation of the book summarization approaches prepared as an ML course project in the UCU.
## Authors
- Antypova Alisa
- Vladyslava Potapova
- Yura Budzyn
## Technical summary
Implemented solution contains the following text summarization approaches:
- TextRank
- LexRank
- Luhn Summarizer
- Latent Semantic Analysis (LSA)
- KLSum 

Additionally, user can choose form three preprocessing options:
- converting text to lower case
- lemmatization
- stemming

All of the above options are available on the web-version of the demo.
Web version was implemented using Streamlit framework
## Deployment
In order to use this repository locally, follow the next steps:

1. Clone the repo and setup virtual environment from the project root folder
    ```buildoutcfg
    # create "book_summarizer" environment
    python3 -m venv .virtualenvs/book_summarizer
    
    # activate environment
    source .virtualenvs/experiments/bin/activate
    ```

2.  Install dependencies 
    ```buildoutcfg
    # load required packages
    pip install -r requirements.txt
    ```
    
3. To run the demo use the following command In the terminal 
    ```buildoutcfg
    streamlit run src/main.py 
    ```
   
After that demo will be available in your default browser.
