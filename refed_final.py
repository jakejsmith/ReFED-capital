# -*- coding: utf-8 -*-
"""ReFED Final 03022025.ipynb

# Enhancing, Expanding, and Analyzing ReFED's Capital Tracker
"""

!pip install -r requirements.txt

import numpy as np
import pandas as pd
import requests
import json
import glob
import datetime
import time
import re
import selenium
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import plotly.express as px

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler

from sentence_transformers import SentenceTransformer

"""## 1. Scraping the ReFED Capital Tracker
##### We begin by grabbing the data from [this table](https://insights-engine.refed.org/capital-tracker/list?dateFrom=2012-01-01&dateTo=2025-02-19&list.page=1&list.searchScope[]=funder_name,funder_desc,recipient_name,recipient_desc&list.sortBy=name&list.view=investments). This involves setting up a scraper with an operation to page through all 58 pages of the table; the `selenium` package works well for this.

##### Perform the scrape
"""

# Initialize empty list to be populated
x = []

# Positions of the pagination buttons to be "clicked" in sequential order
clicks = sum([list(range(0, 9)), [7] * 42, list(range(8, 14))], [])

# Basic scraper setup
driver = webdriver.Firefox()
raw = driver.get("https://insights-engine.refed.org/capital-tracker/list?dateFrom=2012-01-01&dateTo=2025-02-19&list.page=1&list.searchScope[]=funder_name,funder_desc,recipient_name,recipient_desc&list.sortBy=name&list.view=investments")

for i in clicks:

    # Automating selenium to click the appropriate button to go to the next page of the table
    pages = driver.find_elements(By.CLASS_NAME, 'pagination__item')
    driver.execute_script("arguments[0].click();", pages[i])

    # Pause to allow page to load before scraping data
    time.sleep(3)

    # Find all rows on a given page of the table
    rows = driver.find_elements(By.CLASS_NAME, "table2--row")

    # For each row...
    for row in rows:

        rowdata = []

        # Find all cells in the row and add the data to rowdata
        cells = row.find_elements(By.CLASS_NAME, "table2--cell")
        for cell in cells:
            text = cell.get_attribute('innerText')
            rowdata.append(text)

        # add the full row of data to the master list
        x.append(rowdata)

driver.close()

# master list to df
df = pd.DataFrame(x)

"""##### Clean the scraped data"""

# Rename columns using header row
df.columns = df.iloc[1]

# Exclude rows with null dates and duplicates of the header row
df = df.loc[(df['DATE'] != 'DATE') & (df['DATE'] != '')].dropna(subset = ['DATE'])

# Exclude null columns
df = df[df.columns[~df.columns.isnull()]].reset_index()

df.shape

df.head()

"""##### Take a quick look at the Solution column"""

print(df['SOLUTION'].value_counts())

print('Number of Categories: ' + str(len(df['SOLUTION'].value_counts()) - 1))

"""## 2. Predicting Missing `Solution` Categories

##### The table we just scraped has a useful and fairly detailed 'Solution' column which classifies investments in 46 discrete categories. However, this field is missing for approximately two-thirds of the investments.

##### Next we'll apply `BERTopic`, the topic modeling framework based on a prominent language model (BERT), to predict the missing `Solution` values. Specifically, we will deploy BERTopic as a supervised model which we will train on `Company Description`, a field in our table that contains unstructured text describing the recipient of each investment.
"""

df = pd.read_parquet('refed.parquet')

"""##### Define training and response data"""

df_reduced = df[['COMPANY DESCRIPTION', 'SOLUTION']].loc[(df['SOLUTION'] != '') & (df['SOLUTION'].notna()) & (df['COMPANY DESCRIPTION'] != '') & (df['COMPANY DESCRIPTION'].notna())]

df_reduced = df_reduced.drop_duplicates()

df_reduced.shape

train = df_reduced[['COMPANY DESCRIPTION']]
y = df_reduced[['SOLUTION']]

# Convert the string labels in 'y' to numerical labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Convert string labels to numerical labels

"""##### Evaluate imbalance of Solution categories
This shows a pretty severe imbalance, with more than a dozen categories that have only 1 sample, while the largest groups have 20+. We will need to address this.
"""

px.histogram(y, x = 'SOLUTION')

y['SOLUTION'].value_counts()

"""##### Oversample to address sparsity

To mitigate bias from the severe imbalance we found above, we apply an oversampling technique. While a method that applies a nearest-neighbors approach (e.g., SMOTE) is typically more efficient, such methods require each category to have at least 2 samples. Because that criteria is not fulfilled, we resort to a "naive" random oversampling strategy.
"""

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_resample(train, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state = 2525, test_size=0.2)

"""##### Pre-calculate embeddings

Because BERT and BERTopic rely on document-based embeddings (rather than simple word embeddings), it is not necessary to conduct most of the typical NLP pre-processing (e.g., stemming, lemmatizing, tokenizing, etc.) In fact, [BERTopic documentation warns that these steps can actually undermine the efficacy of the model](https://maartengr.github.io/BERTopic/faq.html#should-i-preprocess-the-data).

However, because determining the document embeddings is cost-intensive, and since we are going to be testing different hyperparameters iteratively, calculating the embeddings ahead of time will drastically speed things up.
"""

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(list(X_train))

"""##### Test supervised BERTopic model, iterating over various hyperparameter combinations

Because we are using BERTopic to do supervised training on a pre-established set of Solutions, the various parameters related to number of topics (e.g., nr_topics) or samples per topic (e.g., min_topic_size) are not relevant.

These are the hyperparameters that we will experiment with:

- N-Gram Range: The range of the number of discrete words that BERTopic will evaluate as a single token

- Top N Words: The max number of words that will be used to construct each topic.
"""

X_test_newindex = X_test.reset_index(names = 'old_index')

empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

for n_gram_range_val in [(1, 1), (1, 2), (1, 3)]:
    for top_n_words_val in [5, 10, 15]:

        # Create a supervised BERTopic instance
        topic_model= BERTopic(
                embedding_model = embedding_model,
                umap_model = empty_dimensionality_model,
                hdbscan_model = clf,
                ctfidf_model = ctfidf_model,
                low_memory = True,
                top_n_words = top_n_words_val,
                n_gram_range = n_gram_range_val
        )

        # Train model for each iteration
        topics, probs = topic_model.fit_transform(X_train['COMPANY DESCRIPTION'], y = y_train)

        topic_model_out = topic_model.get_topic_info()

        # Generate and save predicted topic numbers
        all_preds = []

        for i in range(len(X_test_newindex)):
            topic = re.sub(r'\W','',str(topic_model.transform(X_test_newindex['COMPANY DESCRIPTION'].iloc[i])[0]))
            all_preds.append(topic)

        # Put old and new topic numbers in data frame
        z = pd.DataFrame()
        z['Predicted'] = all_preds
        z['Original'] = y_test

        # Create dictionaries to convert topic numbers back to topic names -- note that the predictions and original topics have different encodings, and therefore require separate dictionaries to map back to text
        topic_output = topic_model.get_topic_info()

        topic_names = []

        # Defining dictionary for predicted solutions
        for i in range(len(topic_output['Representative_Docs'])):
            doc_clean = re.sub(r'\[|\]','',topic_output['Representative_Docs'][i][1])[0:50]
            temp_df = df[['SOLUTION']].loc[df['COMPANY DESCRIPTION'].str.contains(doc_clean, regex = False)].reset_index()

            if temp_df['SOLUTION'][0] != '':
                sol_name = temp_df['SOLUTION'][0]
            else:
                sol_name = temp_df['SOLUTION'][1]

            topic_names.append(sol_name)

        topic_dictionary_pred = dict(zip(topic_output['Topic'], topic_names))

        z['Predicted Name'] = z['Predicted'].astype(int).map(topic_dictionary_pred)

        # Defining dictionary for original solutions
        topic_dictionary_orig = dict(zip(y_encoded, y['SOLUTION']))

        z['Original Name'] = z['Original'].astype(int).map(topic_dictionary_orig)

        z['Match'] = z['Original Name'] == z['Predicted Name']

        print('N-Gram Range: ' + str(n_gram_range_val) + '; Top N Words ' + str(top_n_words_val))

        # accuracy
        acc = z['Match'].value_counts()[0] / len(z)
        print('Accuracy Rate: ' + str(acc))

        all_recalls = []
        all_precs = []
        all_f1s = []

        for p in z['Predicted Name'].unique():

            # True positives
            tp = z['Match'].loc[(z['Match'] == True) & (z['Predicted Name'] == p)].value_counts()[0]

            # False negatives
            if len(z['Match'].loc[(z['Match'] == False) & (z['Original Name'] == p)].value_counts()) == 0:
                fn = 0
            else:
                fn = z['Match'].loc[(z['Match'] == False) & (z['Original Name'] == p)].value_counts()[0]

            if len(z['Match'].loc[(z['Match'] == False) & (z['Predicted Name'] == p)].value_counts()) == 0:
                fp = 0
            else:
                fp = z['Match'].loc[(z['Match'] == False) & (z['Predicted Name'] == p)].value_counts()[0]

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1 = 2 * (precision * recall) / (precision + recall)

            all_recalls.append(recall)
            all_precs.append(precision)
            all_f1s.append(f1)
            print('F1 Score, ' + p + ' : ' + str(f1))

        print('Average Recall: ' + str(np.mean(all_recalls)))
        print('Average Precision: ' + str(np.mean(all_precs)))
        print('Average F1 Score: ' + str(np.mean(all_f1s)))
        print('')

"""All of the models performed identically, achieving 94.3% accuracy and an F1 score of .94 out-of-sample.

##### Select and train final model
In this case, all of the models performed identically on the test set. That suggests the exact selection of hyperparameters has little effect; we will therefore revert to the defaults.
"""

topic_model = BERTopic(
                umap_model = empty_dimensionality_model,
                hdbscan_model = clf,
                ctfidf_model = ctfidf_model,
                low_memory = True
        )

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(list(X_resampled))

# Train model
topics, probs = topic_model.fit_transform(X_resampled['COMPANY DESCRIPTION'], y = y_resampled)

"""##### Create dictionary to re-map encoded solutions back to names

In order to which BERTopic-determined topic numbers correspond to which actual topic names, we have to reverse engineer them.
"""

topic_output = topic_model.get_topic_info()

topic_names = []

for i in range(len(topic_output['Representative_Docs'])):
    doc_clean = re.sub(r'\[|\]','',topic_output['Representative_Docs'][i][1])[0:50]
    temp_df = df[['SOLUTION']].loc[df['COMPANY DESCRIPTION'].str.contains(doc_clean, regex = False)].reset_index()

    if temp_df['SOLUTION'][0] != '':
        sol_name = temp_df['SOLUTION'][0]
    else:
        sol_name = temp_df['SOLUTION'][1]

    topic_names.append(sol_name)

topic_dictionary = dict(zip(topic_output['Topic'], topic_names))

"""##### Predict solution categories for full dataset"""

# Take subset with non-missing Company Description
df_new = df.loc[(df['COMPANY DESCRIPTION'].notna())].reset_index()

solution = df['SOLUTION'].loc[(df['COMPANY DESCRIPTION'].notna())]
sol_encoded = le.fit_transform(solution)

# Init empty list of predictions
all_preds = []

for i in range(len(df_new)):
    topic = re.sub(r'\W','',str(topic_model.transform(df_new['COMPANY DESCRIPTION'][i])[0]))
    all_preds.append(topic)

df_new['Predicted Solution Number'] = all_preds

# Map topics using dictionary
df_new['Predicted Solution'] = df_new['Predicted Solution Number'].astype(int).map(topic_dictionary)

df_new.head()

"""##### Evaluate in-sample performance"""

df_new.columns

# Generate match dummy
df_new['MATCH'] = (df_new['SOLUTION'] == df_new['Predicted Solution'])

# Overall accuracy
# Generate match dummy
df_new['MATCH'] = (df_new['SOLUTION'] == df_new['Predicted Solution'])

# Overall accuracy
match_results = df_new['MATCH'].loc[(df_new['COMPANY DESCRIPTION'].notna()) & (df_new['COMPANY DESCRIPTION'] != '') & (df_new['SOLUTION'].notna()) & (df_new['SOLUTION'] != '')].value_counts()

print('Overall In-Sample Accuracy Rate: ' + str(match_results[0] / (match_results[0] + match_results[1])))

all_recalls = []
all_precs = []
all_f1s = []

for p in df_new['Predicted Solution'].unique():

    # True positives
    tp = df_new['MATCH'].loc[(df_new['MATCH'] == True) & (df_new['Predicted Solution'] == p)].value_counts()[0]

    # False negatives
    if len(df_new['MATCH'].loc[(df_new['MATCH'] == False) & (df_new['SOLUTION'] == p)].value_counts()) == 0:
        fn = 0
    else:
        fn = df_new['MATCH'].loc[(df_new['MATCH'] == False) & (df_new['SOLUTION'] == p)].value_counts()[0]

    if len(df_new['MATCH'].loc[(df_new['MATCH'] == False) & (df_new['Predicted Solution'] == p)].value_counts()) == 0:
        fp = 0
    else:
        fp = df_new['MATCH'].loc[(df_new['MATCH'] == False) & (df_new['Predicted Solution'] == p)].value_counts()[0]

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)

    all_recalls.append(recall)
    all_precs.append(precision)
    all_f1s.append(f1)

print('Average Recall: ' + str(np.mean(all_recalls)))
print('Average Precision: ' + str(np.mean(all_precs)))
print('Average F1 Score: ' + str(np.mean(all_f1s)))

df_new[['RECIPIENT', 'COMPANY DESCRIPTION', 'SOLUTION', 'Predicted Solution']].to_csv('df_new.csv')

"""##### Visualize the topic embeddings in a 2D space
This shows substantial overlap among many of the topics, which is likely driving much of the mislassification we see.
"""

topic_model.visualize_topics()

"""## 3. Plotting deal size by `Solution` category

##### Now we will use Solution category (the real one where available, or the predicted where none was originally listed) to create a visualization plotting investment size against solution type.
"""

# Convert deal size to integer
df_new['DEAL_INT'] = df_new['DEAL SIZE'].str.replace(',', '').str.extract(r'([0-9]+)', expand = False).astype(float)

# Coalesce to actual Solution category where available; otherwise use predicted
df_new['SOLUTION_COALESCE'] = df_new['Predicted Solution'].combine_first(df_new['SOLUTION'])

# Replace empties with 'Unknown'
df_new['SOLUTION_COALESCE'].loc[df_new['SOLUTION_COALESCE'] == ''] = 'Unknown'

df_new.SOLUTION_COALESCE.value_counts()

df_new.head()

df_new['IS_ORIG'] = np.where(df_new['SOLUTION'] == '', 'Predicted', 'Actual')

df_new['DEAL_INT'] = df_new['DEAL SIZE'].str.replace(',', '').str.extract(r'([0-9]+)', expand = False).astype(float)

fig = px.strip(df_new,
               x='DEAL_INT',
               y='SOLUTION_COALESCE',
               log_x = True,
               stripmode='group',
               custom_data = ['RECIPIENT', 'IS_ORIG'],
               color = 'SOLUTION_COALESCE',
               color_discrete_sequence = px.colors.qualitative.Safe
               )

fig.update_traces(hovertemplate = "<br>".join(["Deal Size: %{x}",
                                               "Recipient: %{customdata[0]}",
                                               "Solution Type: %{y}",
                                               "Solution Type Source: %{customdata[1]}",
    ]),
                  jitter = 1.0,
                  marker = {'size': 10,
                            'line' : {'width': 1,
                                      'color': 'rgba(128, 128, 128, 1)'}}
                  )

fig.update_layout(
                title={
                    'text': 'Deal Size by Solution Category',
                    'x': 0.5,
                    'xanchor': 'center',
                    'y': 1,
                    'yanchor': 'top',
                    'font': {
                        'family': 'Arial',
                        'size': 24,
                        'color': 'grey'
                    }
                },
                  xaxis_title = 'Deal Size ($)',
                  yaxis_title = '',
                  paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  bargap = 1,
                  showlegend = False)

fig.write_html('fig.html')

fig.show()

"""## 4. Searching for Similar Capital Flows

##### Finally, we will build a semantic similarity model that looks for texts that are similar ReFED's descriptions of companies and investments. We'll deploy this model against against data from APIs and RSS feeds from various news/PR sources, in order to find news stories that may describe similar investments that might be of interest to ReFED.
"""

# Init empty lists
all_content = []
all_headlines = []
all_links = []

"""##### Define a function that calls and rolls up APIs/RSS feeds from several news/press release services
- NewsAPI: This is a news aggregator service that pulls stories and metadata from thousands of international sources and aggregates them into its API. (Note: The current code uses the free version, which limits the number of results that can be pulled via API.)
- PR Newswire: This is a newswire service that publishes press releases from companies, governments, academic institutions, nonprofits, etc. It maintains separate RSS feeds for several topic areas of potential interest -- we will need to call them all separately.

(Note that if a tool like this would be worthwhile to ReFED, the organization might consider paid subscriptions to additional feeds/APIs from organizations that specialize in food/ag science/biotech. The present APIs were included mainly to provide proof of concept.)
"""

def get_news():

    # NewsAPI
    headers = {'x-api-key': 'b2b12189dc8443bebddee191ee64d95c'}
    response_newsapi = requests.get("https://newsapi.org/v2/everything?q=(agriculture OR biotech OR food OR farm) AND (rescue OR waste) AND (venture OR investment OR acquires OR acquired OR 'seed funding' OR funds OR merger OR 'angel investor' OR incubator OR accelerator VC OR buyout)&language=en&sortBy=publishedAt",
                            headers=headers)

    # For each article, grab content, headline, and URL
    for i in range(len(response_newsapi.json()['articles'])):

        content = response_newsapi.json()['articles'][i]['content']
        all_content.append(content)

        headline = response_newsapi.json()['articles'][i]['title']
        all_headlines.append(headline)

        url = response_newsapi.json()['articles'][i]['url']
        all_links.append(url)

    # List of PR Newswire's topic-specific RSS feeds
    response_env = requests.get("https://www.prnewswire.com/rss/environment-latest-news/environment-latest-news-list.rss")
    response_health = requests.get("https://www.prnewswire.com/rss/health-latest-news/health-latest-news-list.rss")
    response_policy = requests.get("https://www.prnewswire.com/rss/policy-public-interest-latest-news/policy-public-interest-latest-news-list.rss")
    response_tech = requests.get("https://www.prnewswire.com/rss/consumer-technology-latest-news/consumer-technology-latest-news-list.rss")

    responses = [response_env, response_health, response_policy, response_tech]

    # For each press release, grab content, headline, and URL
    for r in responses:
        soup = BeautifulSoup(r.content)

        items = soup.find_all('description')

        for item in items[1:21]: # Capturing the 20 news items that appear in each feed, while ignoring the first item which is a header
            content = item.text
            all_content.append(content)

        headlines = soup.find_all('title')

        for hd in headlines[1:21]:
            headline = hd.text
            all_headlines.append(headline)

        links = soup.find_all('guid')

        for u in links:
            url = u.text
            all_links.append(url)

    # Aggregate all items from all news/PR feeds
    newsfeeds_new = pd.DataFrame({'Headline':all_headlines, 'Link':all_links, 'Content':all_content})

    # Save to parquet
    newsfeeds_new.to_parquet('newsfeeds' + str(datetime.datetime.now()) + '.parquet')

    return newsfeeds_new

"""##### Automate hourly pulls from all of the news feeds"""

# import schedule
# schedule.every(1).hour.do(get_news)

# while True:
#    schedule.run_pending()
#    time.sleep(1)

"""##### Define the transformer model that we'll use to encode"""

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")

"""##### Pull and aggregate all of the files from previous API/RSS pulls"""

filelist = []

newsfeeds = get_news()

for file in glob.glob("prnewswire_rss*.parquet|newsfeeds*.parquet"):
    temp = pd.read_parquet(file)
    newsfeeds = pd.concat([newsfeeds, temp]).drop_duplicates

"""##### Calculate semantic similarity for each news article/press release"""

# Create column containing concatenation of `COMPANY DESCRIPTION` and `DEAL DESCRIPTION`
df['FULL_EMBED'] = df['COMPANY DESCRIPTION'] + '' + df['DEAL DESCRIPTION']

# Encode the concatenated column and the news articles
desc_embed = model.encode(df['FULL_EMBED'])
articles_embed = model.encode(newsfeeds['Content'].to_numpy())

# Calculate semantic similarity (using cosine similarity)
similarity = util.cos_sim(articles_embed, desc_embed)

# Append scores to df of news/press releases
scores = []
for i in range(newsfeeds.shape[0]):
    scores.append(abs(similarity[i]).max())

newsfeeds['Similarity_Score'] = scores

newsfeeds.shape

# Sort so that highest-similarity articles appear first
newsfeeds.sort_values(['Similarity_Score'], ascending = False).drop_duplicates('Headline').head(20)

newsfeeds.sort_values(['Similarity_Score'], ascending = False).drop_duplicates('Headline').to_csv('scores.csv')
