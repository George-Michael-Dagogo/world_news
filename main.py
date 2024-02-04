from newsapi import NewsApiClient
import pandas as pd
from newspaper import Article, Config
import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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

def full_content(url):
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    config = Config()
    config.browser_user_agent = user_agent
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

# Download the stopwords dataset
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
# Function to count words without stopwords
def count_words_without_stopwords(text):
    # Check if the value is a string or bytes-like object
    if isinstance(text, (str, bytes)):
        # Tokenize the text
        words = nltk.word_tokenize(str(text))
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Return the count of non-stopwords
        return len(filtered_words)
    else:
        # If the value is not a string or bytes-like, return 0
        return 0
# Apply the function to the 'Text' column and create a new column 'WordCount'
df['WordCount'] = df['content'].apply(count_words_without_stopwords)

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
df[['Sentiment', 'Compound_Score']] = df['content'].apply(lambda x: pd.Series(get_sentiment(x)))
df

