from __future__ import print_function

import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
from pprint import pprint
import re, spacy, gensim

from joblib import Parallel, delayed, cpu_count
import pyLDAvis
import pyLDAvis.sklearn
import gensim

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from pprint import pprint
from scipy import sparse
import boto3

import time
from time import time as timer
from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

print('Loading spacy')
nlp = spacy.load('en', disable=['parser', 'ner']) 

# === TERM FREQUENCY FUNCTIONS ===
lambda_step = 0.01
n_jobs = -1
R = 100
sort_topics = True

def _find_relevance(log_ttd, log_lift, R, lambda_):
    relevance = lambda_ * log_ttd + (1 - lambda_) * log_lift
    return relevance.T.apply(lambda s: s.sort_values(ascending=False).index).head(R)


def _find_relevance_chunks(log_ttd, log_lift, R, lambda_seq):
    return pd.concat([_find_relevance(log_ttd, log_lift, R, l) for l in lambda_seq])

def _chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def _job_chunks(l, n_jobs):
    n_chunks = n_jobs
    if n_jobs < 0:
        # so, have n chunks if we are using all n cores/cpus
        n_chunks = cpu_count() + 1 - n_jobs

    return _chunks(l, n_chunks)

def _topic_info(topic_term_dists, topic_proportion, term_frequency, term_topic_freq,
                vocab, lambda_step, R, n_jobs):
    # marginal distribution over terms (width of blue bars)
    term_proportion = term_frequency / term_frequency.sum()

    # compute the distinctiveness and saliency of the terms:
    # this determines the R terms that are displayed when no topic is selected
    topic_given_term = topic_term_dists / topic_term_dists.sum()
    kernel = (topic_given_term * np.log((topic_given_term.T / topic_proportion).T))
    distinctiveness = kernel.sum()
    saliency = term_proportion * distinctiveness
    # Order the terms for the "default" view by decreasing saliency:
    default_term_info = pd.DataFrame({
        'saliency': saliency,
        'Term': vocab,
        'Freq': term_frequency,
        'Total': term_frequency,
        'Category': 'Default'})
    default_term_info = default_term_info.sort_values(
        by='saliency', ascending=False).head(R).drop('saliency', 1)
    # Rounding Freq and Total to integer values to match LDAvis code:
    default_term_info['Freq'] = np.floor(default_term_info['Freq'])
    default_term_info['Total'] = np.floor(default_term_info['Total'])
    ranks = np.arange(R, 0, -1)
    default_term_info['logprob'] = default_term_info['loglift'] = ranks

    # compute relevance and top terms for each topic
    log_lift = np.log(topic_term_dists / term_proportion)
    log_ttd = np.log(topic_term_dists)
    lambda_seq = np.arange(0, 1 + lambda_step, lambda_step)

    def topic_top_term_df(tup):
        new_topic_id, (original_topic_id, topic_terms) = tup
        term_ix = topic_terms.unique()
        return pd.DataFrame({'Term': vocab[term_ix],
                             'Freq': term_topic_freq.loc[original_topic_id, term_ix],
                             'Total': term_frequency[term_ix],
                             'logprob': log_ttd.loc[original_topic_id, term_ix].round(4),
                             'loglift': log_lift.loc[original_topic_id, term_ix].round(4),
                             'Category': 'Topic%d' % new_topic_id})

    top_terms = pd.concat(Parallel(n_jobs=n_jobs)
                          (delayed(_find_relevance_chunks)(log_ttd, log_lift, R, ls)
                          for ls in _job_chunks(lambda_seq, n_jobs)))
    topic_dfs = map(topic_top_term_df, enumerate(top_terms.T.iterrows(), 1))
    return pd.concat([default_term_info] + list(topic_dfs), sort=True)


def _token_table(topic_info, term_topic_freq, vocab, term_frequency):
    # last, to compute the areas of the circles when a term is highlighted
    # we must gather all unique terms that could show up (for every combination
    # of topic and value of lambda) and compute its distribution over topics.

    # term-topic frequency table of unique terms across all topics and all values of lambda
    term_ix = topic_info.index.unique()
    term_ix = np.sort(term_ix)

    top_topic_terms_freq = term_topic_freq[term_ix]
    # use the new ordering for the topics
    K = len(term_topic_freq)
    top_topic_terms_freq.index = range(1, K + 1)
    top_topic_terms_freq.index.name = 'Topic'

    # we filter to Freq >= 0.5 to avoid sending too much data to the browser
    token_table = pd.DataFrame({'Freq': top_topic_terms_freq.unstack()})\
        .reset_index().set_index('term').query('Freq >= 0.5')

    token_table['Freq'] = token_table['Freq'].round()
    token_table['Term'] = vocab[token_table.index.values].values
    # Normalize token frequencies:
    token_table['Freq'] = token_table.Freq / term_frequency[token_table.index]
    return token_table.sort_values(by=['Term', 'Topic'])

def _get_doc_lengths(dtm):
    return dtm.sum(axis=1).getA1()


def _get_term_freqs(dtm):
    return dtm.sum(axis=0).getA1()


def _get_vocab(vectorizer):
    return vectorizer.get_feature_names()


def _row_norm(dists):
    # row normalization function required
    # for doc_topic_dists and topic_term_dists
    return dists / dists.sum(axis=1)[:, None]


def _get_doc_topic_dists(lda_model, dtm):
    return _row_norm(lda_model.transform(dtm))


def _get_topic_term_dists(lda_model):
    return _row_norm(lda_model.components_)

def _df_with_names(data, index_name, columns_name):
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)
    

def _get_doc_topic_dists(lda_model, dtm):
    return _row_norm(lda_model.transform(dtm))


def _input_check(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency):
    ttds = topic_term_dists.shape
    dtds = doc_topic_dists.shape
    errors = []

    def err(msg):
        errors.append(msg)

    if dtds[1] != ttds[0]:
        err_msg = ('Number of rows of topic_term_dists does not match number of columns of '
                   'doc_topic_dists; both should be equal to the number of topics in the model.')
        err(err_msg)

    if len(doc_lengths) != dtds[0]:
        err_msg = ('Length of doc_lengths not equal to the number of rows in doc_topic_dists;'
                   'both should be equal to the number of documents in the data.')
        err(err_msg)

    W = len(vocab)
    if ttds[1] != W:
        err_msg = ('Number of terms in vocabulary does not match the number of columns of '
                   'topic_term_dists (where each row of topic_term_dists is a probability '
                   'distribution of terms for a given topic)')
        err(err_msg)
    if len(term_frequency) != W:
        err_msg = ('Length of term_frequency not equal to the number of terms in the '
                   'number of terms in the vocabulary (len of vocab)')
        err(err_msg)

    if __num_dist_rows__(topic_term_dists) != ttds[0]:
        err('Not all rows (distributions) in topic_term_dists sum to 1.')

    if __num_dist_rows__(doc_topic_dists) != dtds[0]:
        err('Not all rows (distributions) in doc_topic_dists sum to 1.')

    if len(errors) > 0:
        return errors


def _input_validate(*args):
    res = _input_check(*args)
    if res:
        raise ValidationError('\n' + '\n'.join([' * ' + s for s in res]))
        

def __num_dist_rows__(array, ndigits=2):
    return array.shape[0] - int((pd.DataFrame(array).sum(axis=1) < 0.999).sum())


class ValidationError(ValueError):
    pass

def build_term_frequency(lda_model, sparse_matrix, vectorizer):
    vocab = _get_vocab(vectorizer)
    doc_lengths = _get_doc_lengths(sparse_matrix)
    term_freqs = _get_term_freqs(sparse_matrix)
    topic_term_dists = _get_topic_term_dists(lda_model)
    doc_topic_dists = _get_doc_topic_dists(lda_model, sparse_matrix)

    topic_term_dists = _df_with_names(topic_term_dists, 'topic', 'term')
    doc_topic_dists = _df_with_names(doc_topic_dists, 'doc', 'topic')
    term_frequency = _series_with_name(term_freqs, 'term_frequency')
    doc_lengths = _series_with_name(doc_lengths, 'doc_length')
    vocab = _series_with_name(vocab, 'vocab')

    _input_validate(topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency)
    
    t0 = timer()
    topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
    # topic_freq       = np.dot(doc_topic_dists.T, doc_lengths)
    if (sort_topics):
        topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
    else:
        topic_proportion = (topic_freq / topic_freq.sum())

    topic_order = topic_proportion.index
    # reorder all data based on new ordering of topics
    topic_freq = topic_freq[topic_order]
    topic_term_dists = topic_term_dists.iloc[topic_order]
    doc_topic_dists = doc_topic_dists[topic_order]

    # token counts for each term-topic combination (widths of red bars)
    term_topic_freq = (topic_term_dists.T * topic_freq).T
    # Quick fix for red bar width bug.  We calculate the
    # term frequencies internally, using the topic term distributions and the
    # topic frequencies, rather than using the user-supplied term frequencies.
    # For a detailed discussion, see: https://github.com/cpsievert/LDAvis/pull/41
    term_frequency = np.sum(term_topic_freq, axis=0)

    topic_info = _topic_info(topic_term_dists, topic_proportion,
                             term_frequency, term_topic_freq, vocab, lambda_step, R, n_jobs)
    token_table = _token_table(topic_info, term_topic_freq, vocab, term_frequency)

    print("done in %0.3fs." % (timer() - t0))
    return { 'topic_info': topic_info, 'token_table': token_table }

def preprocess(data):
    print("Processing {} of data".format(len(data)))
    # Remove Emails
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]

    # Remove new line characters
    data = [re.sub(r'\s+', ' ', sent) for sent in data]

    # Remove distracting single quotes
    data = [re.sub(r"\'", "", sent) for sent in data]

    def sent_to_words(sentences):
        for sentence in sentences:
            #yield(gensim.utils.tokenize(str(sentence), deacc=True))
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(data))
    
    #print ("=== DATA WORDS ===")
    #print(data_words)
    
    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
        return texts_out

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # Run in terminal: python3 -m spacy download en
    nlp = spacy.load('en', disable=['parser','ner'])

    # Do lemmatization keeping only Noun, Adj, Verb, Adverb
    data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    
    #print ("=== LEMMATIZED ===")
    #print(data_lemmatized)
    vectorizer = CountVectorizer(analyzer='word',       
                             min_df=2,                        # minimum reqd occurences of a word 
                             stop_words='english',             # remove stop words
                             lowercase=False,                   # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{5,}',  # num chars > 3
                             max_features=50000,             # max number of uniq words
                            )

    
    data_vectorized = vectorizer.fit_transform(data_lemmatized)
    return vectorizer, data_vectorized

def get_words(vectorizer, lda_model):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        topic_keyword_location = (-topic_weights).argsort()[:20]
        topic_keywords.append(keywords.take(topic_keyword_location))
    return topic_keywords

def build_LDA(data_vectorized, n_components=10):
    # Build LDA Model
    print("Data Vectorized Length: ", data_vectorized.shape)
    lda_model = LatentDirichletAllocation(n_components=n_components,               # Number of topics
                                          max_iter=10,               # Max learning iterations
                                          learning_method='online',   
                                          random_state=100,          # Random state
                                          batch_size=128,            # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                         )
    lda_output = lda_model.fit_transform(data_vectorized)
                                                                                        
    print(lda_model)  # Model attributes
    return lda_model, lda_output
                                                                                                                            
def build_KMeans(lda_output, n_clusters=10):
    # Construct the k-means clusters
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=100)
    clusters = kmeans.fit_predict(lda_output)

    # Build the Singular Value Decomposition(SVD) model
    svd_model = TruncatedSVD(n_components=2)  # 2 components
    lda_output_svd = svd_model.fit_transform(lda_output)

    # X and Y axes of the plot using SVD decomposition
    # x = lda_output_svd[:, 0]
    # y = lda_output_svd[:, 1]

    # Weights for the 15 columns of lda_output, for each component
    # print("Component's weights: \n", np.round(svd_model.components_, 2))

    # Percentage of total information in 'lda_output' explained by the two components
    # print("Perc of Variance Explained: \n", np.round(svd_model.explained_variance_ratio_, 2))
    return kmeans, clusters

def build_pyLDAvis(lda_model, sparse_matrix, vectorizer):
    panel = pyLDAvis.sklearn.prepare(lda_model, sparse_matrix, vectorizer, mds='tsne')
    return panel

def get_sector_names(sector_panels):
    sector_panel_token_table = sector_panels['token_table']
    mostFrequentTerm = sector_panel_token_table.loc[[sector_panel_token_table.Freq.idxmax()]]
    print(mostFrequentTerm['Term'].max())
    return mostFrequentTerm['Term'].max()

def get_industry_names(industry_panels, sectorIndex):
    industry_names = []
    for i in range(len(industry_panels)):
        print ("Getting panel ", i)
        industry_panel_token_table = industry_panels[i]['token_table']
        mostFrequentTerm = industry_panel_token_table.loc[[industry_panel_token_table.Freq.idxmax()]]
        print(mostFrequentTerm['Term'].max())
        industry_names.append([mostFrequentTerm['Term'].max(), i, sectorIndex])
    industry_names_df = pd.DataFrame(industry_names,columns=['Industry Names', 'Industry Index', 'Sector Index'])
    return industry_names_df



def train(data):
    t0 = timer()
    countVectorizer, data_vectorized = preprocess(data.Article)
    word_vectorized_df = pd.DataFrame(data_vectorized.toarray())
    word_vectorized_df.head()
    print("done in %0.3fs." % (timer() - t0))


    t0 = timer()
    lda_model, lda_output = build_LDA(word_vectorized_df)
    kmeans, clusters = build_KMeans(lda_output)
    print("done in %0.3fs." % (timer() - t0))

    clusters_df = pd.DataFrame(clusters, columns=['Cluster Index'])
    sectors_df = pd.concat([data, clusters_df], axis=1)

    # Now that it's grouped let's iterate each cluster and scope LDA to just those words
    t0 = timer()
    for i in range(kmeans.n_clusters):
    #for i in range(1):
        industry_panels = []
        
        print ("Processing Sector Cluster ", i)
        cluster_n = sectors_df[sectors_df['Cluster Index'] == i]
        print ("Data Length: ", len(cluster_n))
        sector_word_vectorizer, sector_data_vectorized = preprocess(cluster_n.Article)
        print(cluster_n.head())
        
        print ("=== Building LDA for Sector ===")
        cluster_n_lda_model, cluster_n_lda_output = build_LDA(sector_data_vectorized, 10)
        
        print ("=== Building K Means for Industries ===")
        industry_kmeans, industry_clusters = build_KMeans(cluster_n_lda_output, 5)
        
        # Zip the given sector index (key) with the industry index and set the industry index back to the original frame
        combined = zip(cluster_n.index.values, industry_clusters)
        for sectorIndex, industryIndex in combined:
            sectors_df.at[sectorIndex, 'Industry Index'] = industryIndex
        
        for industry_index in range(industry_kmeans.n_clusters):
        #for industry_index in range(1):
            print ("Processing Industry Cluster ", industry_index)
            filteredCompaniesByIndustryCluster = sectors_df[sectors_df['Industry Index'] == industry_index]
            industry_word_vectorizer, industry_data_vectorized = preprocess(filteredCompaniesByIndustryCluster.Article)       
            industry_n_lda_model, industry_n_lda_output = build_LDA(industry_data_vectorized, 10)
            
            # Now get the best topic and keyword frequency to get industsy name
            print ("=== Building Term Frequency for Industry ===")
            industry_panel = build_term_frequency(industry_n_lda_model, industry_data_vectorized, industry_word_vectorizer)
            industry_panels.append(industry_panel)
        
        industry_names_df = get_industry_names(industry_panels, i)
        
        # Set the industry names back to the original dataset
        for index, row in industry_names_df.iterrows():
            indexKeys = sectors_df[(sectors_df['Cluster Index'] == row['Sector Index']) 
                            & (sectors_df['Industry Index'] == row['Industry Index'])].index.values
            for key in indexKeys:
                sectors_df.at[key, 'Industry Names'] = row['Industry Names']
        
        
        print ("===  LDA Components ===", cluster_n_lda_model.components_)
        print ("=== Building Term Frequency for Sector ===")
        sector_panel = build_term_frequency(cluster_n_lda_model, sector_data_vectorized, sector_word_vectorizer)
        sector_name = get_sector_names(sector_panel)
        # Set the sector name back to the original dataset TODO: Maybe create a function
        
        indexKeys = sectors_df[(sectors_df['Cluster Index'] == i)].index.values
        for key in indexKeys:
            sectors_df.at[key, 'Sector Names'] = sector_name
    print("done in %0.3fs." % (timer() - t0))
    return kmeans, sectors_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas dataframe
    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    print(input_files)
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    
    raw_data = [ pd.read_csv(
        file) for file in input_files ]
    concat_data = pd.concat(raw_data)
    
    print("Got raw data")
    
    print(concat_data.head())
    
    print("Length of data ", len(concat_data))


    data = concat_data
    model, sectors_df = train(concat_data)
    sectors_df.to_csv('output.csv')
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    prefix = 'output'
    s3 = boto3.resource('s3')
    bucket = 'mdas-pipeline-data'
    key = "{}/{}.csv".format(prefix, timestr)
    print ("Uploading to s3")
    url = 's3://{}/{}'.format(bucket, key)
    s3.Bucket(bucket).Object(key).upload_file('output.csv')
    print('Done writing to {}'.format(url))


    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print('Done writing model')

    
def input_fn(input_data, content_type):
    """Parse input data payload
    
    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    # if content_type == 'text/csv':
    #     # Read the raw input data as CSV.
    #     df = pd.read_csv(StringIO(input_data), 
    #                      header=None)
        
    #     if len(df.columns) == len(feature_columns_names) + 1:
    #         # This is a labelled example, includes the ring label
    #         df.columns = feature_columns_names + [label_column]
    #     elif len(df.columns) == len(feature_columns_names):
    #         # This is an unlabelled example.
    #         df.columns = feature_columns_names
            
    #     return df
    # else:
    #raise ValueError("Not yet supported by script!")
    return "Not yet supported by script"
        

def output_fn(prediction, accept):
    """Format prediction output
    
    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    return worker.Response("Not yet supported by script")
    #raise ValueError("Not yet supported by script!")
    # if accept == "application/json":
    #     instances = []
    #     for row in prediction.tolist():
    #         instances.append({"features": row})

    #     json_output = {"instances": instances}

    #     return worker.Response(json.dumps(json_output), mimetype=accept)
    # elif accept == 'text/csv':
    #     return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    # else:
    #     raise RuntimeException("{} accept type is not supported by this script.".format(accept))


def predict_fn(input_data, model):
    """Preprocess input data
    
    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:
    
        rest of features either one hot encoded or standardized
    """
    return "Not yet supported by script"
    #raise ValueError("Predict is not supported by script!")
    # features = model.transform(input_data)
    
    # if label_column in input_data:
    #     # Return the label (as the first column) and the set of features.
    #     return np.insert(features, 0, input_data[label_column], axis=1)
    # else:
    #     # Return only the set of features
    #     return features
    

def model_fn(model_dir):
    """Deserialize fitted model
    """
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model