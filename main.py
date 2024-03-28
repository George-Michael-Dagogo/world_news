from newsapi.newsapi_client import NewsApiClient
import pandas as pd
from newspaper import Article, Config
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import shutup; shutup.please()


# Init
newsapi = NewsApiClient(api_key='ff4373852c2343a98303951439854f8c')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(
                                          category='general',
                                          language='en',
                                          page_size=90,
                                          page=1
                                        )

# source = https://newsapi.org/docs/client-libraries/python
articles = top_headlines.get('articles', [])

    # Create a DataFrame with specific columns
df = pd.DataFrame(articles, columns=['source','title','publishedAt','author','url'])

# Data Cleaning
df['source'] = df['source'].apply(lambda x: x['name'] if pd.notna(x) and 'name' in x else None)

from newspaper import Article, Config

def full_content(url):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
    config.request_timeout = 10
    page = Article(url, config=config)
    
    try:
        page.download()
        page.parse()
        return page.text
    except Exception as e:
        print(f"Error retrieving content from {url}: {e}")
        return 'couldnt retrieve'




df['content'] = df['url'].apply(full_content)
df['content'] = df['content'].str.replace('\n', ' ')

df = df[df['content'] != 'couldnt retrieve']



import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Function to count words without stopwords
def count_words_without_stopwords(text):
    if isinstance(text, (str, bytes)):
        words = nltk.word_tokenize(str(text))
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return len(filtered_words)
    else:
        return 0

# Apply the function to the 'content' column using .loc for assignment
df['WordCount'] = df['content'].apply(count_words_without_stopwords)



import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Create a SentimentIntensityAnalyzer object
sid = SentimentIntensityAnalyzer()


# Function to get sentiment scores and label
def get_sentiment(row):
    sentiment_scores = sid.polarity_scores(row)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = 'Positive'
    elif compound_score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, compound_score

# Apply sentiment analysis to each row of the DataFrame
df[['Sentiment', 'Compound_Score']] = df['content'].astype(str).apply(lambda x: pd.Series(get_sentiment(x)))
print(df)