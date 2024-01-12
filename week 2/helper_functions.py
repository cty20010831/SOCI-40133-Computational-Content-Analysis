# This file contains the helper functions needed for Week 2 homework

import requests
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import scipy
import seaborn as sns
import sklearn.manifold
import json
import spacy
import nltk

nlp = spacy.load("en_core_web_sm")

# This function retrives github files
def getGithubFiles(target, maxFiles = 100):
    #We are setting a max so our examples don't take too long to run
    #For converting to a DataFrame
    releasesDict = {
        'name' : [], #The name of the file
        'text' : [], #The text of the file, watch out for binary files
        'path' : [], #The path in the git repo to the file
        'html_url' : [], #The url to see the file on Github
        'download_url' : [], #The url to download the file
    }

    #Get the directory information from Github
    r = requests.get(target)
    filesLst = json.loads(r.text)

    for fileDict in filesLst[:maxFiles]:
        #These are provided by the directory
        releasesDict['name'].append(fileDict['name'])
        releasesDict['path'].append(fileDict['path'])
        releasesDict['html_url'].append(fileDict['html_url'])
        releasesDict['download_url'].append(fileDict['download_url'])

        #We need to download the text though
        text = requests.get(fileDict['download_url']).text
        releasesDict['text'].append(text)

    return pd.DataFrame(releasesDict)

# This function is used to (roughly) clean raw texts 
def clean_raw_text_updated(raw_texts):
    clean_texts_lst = []
    for text in raw_texts:
        try:
            clean_text = text.replace(" \'m", "'m").replace(" \'ll", "'ll").replace(" \'re", "'re").replace(" \'s", "'s").replace(" \'re", "'re").replace(" n\'t", "n't").replace(" \'ve", "'ve").replace(" /'d", "'d")
            clean_texts_lst.append(clean_text)
        except AttributeError:
            continue
        except UnicodeDecodeError:
            continue
    return clean_texts_lst


# This function tokenizes a list of words
def word_tokenize(word_list):
    tokenized = []
    # pass word list through language model.
    doc = nlp(word_list)
    for token in doc:
        if not token.is_punct and len(token.text.strip()) > 0:
            tokenized.append(token.text)
    return tokenized

# This function normalizes tokens
def normalizeTokens(word_list, extra_stop=[]):
    #We can use a generator here as we just need to iterate over it
    normalized = []
    if type(word_list) == list and len(word_list) == 1:
        word_list = word_list[0]

    if type(word_list) == list:
        word_list = ' '.join([str(elem) for elem in word_list])

    doc = nlp(word_list.lower())

    # add the property of stop word to words considered as stop words
    if len(extra_stop) > 0:
        for stopword in extra_stop:
            lexeme = nlp.vocab[stopword]
            lexeme.is_stop = True

    for w in doc:
        # if it's not a stop word or punctuation mark, add it to our article
        if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text.strip()) > 0:
            # we add the lematized version of the word
            normalized.append(str(w.lemma_))

    return normalized

# This function tokenizes into sentences
def sent_tokenize(word_list):
    doc = nlp(word_list)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

# This function tags NLTK pos on sentences
def tag_sents_pos(sentences):
    """
    function which replicates NLTK pos tagging on sentences.
    """
    new_sents = []
    for sentence in sentences:
        new_sent = ' '.join(sentence)
        new_sents.append(new_sent)
    final_string = ' '.join(new_sents)
    doc = nlp(final_string)

    pos_sents = []
    for sent in doc.sents:
        pos_sent = []
        for token in sent:
            pos_sent.append((token.text, token.tag_))
        pos_sents.append(pos_sent)

    return pos_sents

# This function finds the frequency distribution (list of words) associated with specified POS tags
def find_POS(df, pos_tag, top_n):
    counts = {}

    for entry in df['POS_Sentence']:
        for sentence in entry:
            for token, kind in sentence:
                 if kind == pos_tag:
                      counts[token] = counts.get(token, 0) + 1
    
    freq_dist = nltk.FreqDist(counts)

    # Convert to a DataFrame for Seaborn
    df_freq = pd.DataFrame(freq_dist.most_common(top_n), columns=['Token', 'Frequency'])

    # Plotting with Seaborn
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Token', data=df_freq, palette="Blues_d", orient='h')
    plt.title(f'Frequency Distribution of Top {top_n} "{pos_tag}" Tokens')
    plt.xlabel('Frequency')
    plt.ylabel('Tokens')
    plt.show()

    return freq_dist

# Define a function to find conditional associations of tokens with two different POS tags
def find_conditional_associations(df, pos_tag_tuple):
    pos_tag_1, pos_tag_2 = pos_tag_tuple

    # Create a dictionary to count conditional associations 
    conditional_association = {}

    for entry in df['POS_Sentence']:
        for sentence in entry:
            for (ent1, kind1),(ent2, kind2) in zip(sentence[:-1], sentence[1:]):
                if kind1 == pos_tag_1 and kind2 == pos_tag_2:
                    token_tuple = ent1, ent2
                    conditional_association[token_tuple] = conditional_association.get(token_tuple, 0) + 1
    
    # Sort the dictionary based on count
    conditional_association = sorted(conditional_association.items(), key=lambda x: x[1], reverse=True)

    return conditional_association 

# This function replicates NLTK ner tagging on sentences.
def tag_sents_ner(sentences):
    new_sents = []
    for sentence in sentences:
        new_sent = ' '.join(sentence)
        new_sents.append(new_sent)
    final_string = ' '.join(new_sents)
    doc = nlp(final_string)

    pos_sents = []
    for sent in doc.sents:
        pos_sent = []
        for ent in sent.ents:
            pos_sent.append((ent.text, ent.label_))
        pos_sents.append(pos_sent)

    return pos_sents

# This function helps find the critical value for student t test
def find_t_critical(n_sample, alpha=0.05):
    return scipy.stats.t.ppf(q=1 - alpha / 2, df=n_sample - 1)

# This function plots the frequencies of words that follow the specified word(s) in an n-gram CFD.
def plot_ngram_frequencies(word_tuple, ngram_cfd, top_n=10):
    # Get the frequency distribution for the specified word(s)
    freq_dist = ngram_cfd[word_tuple]

    # Convert to a dictionary and then to a DataFrame
    freq_dict = dict(freq_dist)
    df = pd.DataFrame(list(freq_dict.items()), columns=['Following Word', 'Frequency'])

    # Sort the DataFrame by frequency and take the top_n items
    df = df.sort_values(by='Frequency', ascending=False).head(top_n)

    # Plot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Following Word', data=df, palette="Blues_d")
    title_word = " ".join(word_tuple) if isinstance(word_tuple, tuple) else word_tuple
    plt.title(f'Frequency of Words Following "{title_word}"')
    plt.xlabel('Frequency')
    plt.ylabel('Following Words')
    plt.show()

# This recursive function to retrieve the depth of the dependency parse tree 
def max_depth(token):
    # Base case
    if not list(token.children):
        return 1
    # Recursive case
    else:
        return 1 + max(max_depth(child) for child in token.children)

# This function helps to plot the bar charts
def plot_bar_chart(data, title, ax, top_n=10):
    value_counts = pd.Series(data).value_counts()
    value_counts.head(top_n).plot(kind='barh', ax=ax)
    ax.set_title(title)

# This function calculates KL divergence
def kl_divergence(X, Y):
    P = X.copy()
    Q = Y.copy()
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P).fillna(0)
    p = df.iloc[:,1]
    q = df.iloc[:,0]
    D_kl = scipy.stats.entropy(p, q)
    return D_kl

# This function calculates chi-square divergence
# It updates the original function that includes scaling factor and epsilon to avoid zero division
def chi2_divergence(X, Y, epsilon=1e-10):
    P = X.copy()
    Q = Y.copy()
    P.columns = ['P']
    Q.columns = ['Q']
    df = Q.join(P, how='outer').fillna(0)

    # Add smoothing to avoid division by zero
    df += epsilon

    # Calculate total counts and find scaling factor
    total_p = df['P'].sum()
    total_q = df['Q'].sum()
    scaling_factor = total_q / total_p if total_p != 0 else 1

    # Scale P to have the same total count as Q
    df['P'] *= scaling_factor

    p = df['P']
    q = df['Q']

    return scipy.stats.chisquare(p, q).statistic

# This function handle four types of distributional divergence calculation 
def Divergence(corpus1, corpus2, difference="KL"):
    """Difference parameter can equal KL, Chi2, or Wass"""
    freqP = nltk.FreqDist(corpus1)
    P = pd.DataFrame(list(freqP.values()), columns = ['frequency'], index = list(freqP.keys()))
    freqQ = nltk.FreqDist(corpus2)
    Q = pd.DataFrame(list(freqQ.values()), columns = ['frequency'], index = list(freqQ.keys()))
    if difference == "KL":
        return kl_divergence(P, Q)
    elif difference == "Chi2":
        return chi2_divergence(P, Q)
    elif difference == "KS":
        try:
            return scipy.stats.ks_2samp(P['frequency'], Q['frequency']).statistic
        except:
            return scipy.stats.ks_2samp(P['frequency'], Q['frequency'])
    elif difference == "Wasserstein":
        try:
            return scipy.stats.wasserstein_distance(P['frequency'], Q['frequency'], u_weights=None, v_weights=None).statistic
        except:
            return scipy.stats.wasserstein_distance(P['frequency'], Q['frequency'], u_weights=None, v_weights=None)
        
# This function distinguishes normalized tokens into stop and nonstop words
def split_stop_nonstop(normalized_tokens):
    stop = []
    non_stop = []
    for rt in normalized_tokens:
        if nlp.vocab[rt].is_stop:
            stop.append(rt)
        else:
            non_stop.append(rt)
    return stop, non_stop

# This function scrapes abstract of specified field of study using the semantic scholar api
def scrape_abstract(fieldsOfStudy, limit_per_field=100, url='https://api.semanticscholar.org/graph/v1/paper/search'):
    headers = {"Your Header Information": "Value"}  # Replace with actual headers

    all_data = []

    query_params = {
        'query': fieldsOfStudy,
        'fields': 'title,authors,abstract',
        'limit': limit_per_field
    }

    response = requests.get(url, headers=headers, params=query_params)

    if response.status_code == 200:
        papers = response.json().get('data', [])
        for paper in papers:
            title = paper.get('title')
            author = paper.get('authors')
            abstract = paper.get('abstract')

            if not title or not author or not abstract:
                continue

            author_names = ', '.join([a['name'] for a in author])

            all_data.append({
                "FieldOfStudy": fieldsOfStudy,
                "Title": title,
                "Author": author_names,
                "Abstract": abstract
            })
    else:
        print(f"Request failed with status code {response.status_code}")

    df = pd.DataFrame(all_data)

    # Tokenize, normalize, and split stopwords
    df['tokenized_text'] = df["Abstract"].apply(word_tokenize)
    df['normalized_tokens'] = df['tokenized_text'].apply(normalizeTokens)
    df[['stopwords', 'non_stopwords']] = df['normalized_tokens'].apply(split_stop_nonstop).apply(pd.Series)

    return df

# Define a function for plotting various measures of distributional distance between different corpora
def distributional_distance(corpora_df, measure, fieldsOfStudy, df_index):
    # Create two figures: one for heatmaps and one for MDS visualizations
    fig_heatmap, axes_heatmap = plt.subplots(len(df_index), 1, figsize=(10, 10 * len(df_index)))
    fig_mds, axes_mds = plt.subplots(len(df_index), 1, figsize=(10, 10 * len(df_index)))

    for i, index in enumerate(df_index):
        corpora = corpora_df[index]

        L = []
        for p in corpora:
            l = []
            for q in corpora:
                l.append(Divergence(p, q, difference=measure))
            L.append(l)
        M = np.array(L)

        # Plot heatmap
        div = pd.DataFrame(M, columns=fieldsOfStudy, index=fieldsOfStudy)
        if len(df_index) > 1:
            ax_heatmap = axes_heatmap[i]
        else:
            ax_heatmap = axes_heatmap
        sns.heatmap(div, ax=ax_heatmap, cbar_kws={'label': 'Divergence'})
        ax_heatmap.set_title(f"{measure} divergence measure of abstract corpora ({index})")

        # Plot MDS Visualization
        mds = sklearn.manifold.MDS()
        pos = mds.fit(M).embedding_
        x = pos[:,0]
        y = pos[:,1]
        if len(df_index) > 1:
            ax_mds = axes_mds[i]
        else:
            ax_mds = axes_mds
        ax_mds.scatter(x, y)
        for j, txt in enumerate(fieldsOfStudy):
            ax_mds.annotate(txt, (x[j], y[j]))
        ax_mds.set_title(f'MDS Plot of {index} Corpora Based on {measure} Divergence')

    plt.tight_layout()
    fig_heatmap.show()
    fig_mds.show()

    
     