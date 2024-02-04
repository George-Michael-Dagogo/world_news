from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key='ff4373852c2343a98303951439854f8c')

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(
                                          category='general',
                                          language='en',
                                        )



