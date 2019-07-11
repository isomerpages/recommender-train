# Import git
import git

# Imports for pre-processing
import sys
from bs4 import BeautifulSoup
from markdown import markdown
import re
import frontmatter
import os
import pprint
import time
import nltk
import yaml
import json
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
import shutil

from nltk.corpus import stopwords 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Imports for LSA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
from numpy import argsort

# Constants
AWS_REGION_NAME = os.environ['AWS_REGION_NAME']
AWS_DYNAMODB_ENDPOINT = os.environ['AWS_DYNAMODB_ENDPOINT']
AWS_DYNAMODB_TABLE_NAME = os.environ['AWS_DYNAMODB_TABLE_NAME']

# Import for DynamoDB
import boto3
dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION_NAME, endpoint_url=AWS_DYNAMODB_ENDPOINT)

# Constants
NUM_RELATED_POSTS = 10 # Define the number of closely related posts we want to display for each post

def markdown_to_text(content):
    """ Converts a markdown file to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(content)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.findAll(text=True))

    return text

def generate_text_array(site_url, directory):
  """ Traverse the entire directory and extract the text in markdown files into a text_array
      Returns a text_array and file_meta_array
      text_array[i] contains the text content of the file in file_meta_array[i] """

  text_array = []
  file_meta_array = []
  num_md_files = 0

  custom_stop_words = ['mr', 'mister', 'ms', 'dr', 'said', 'this', '_blank']

  # Obtain the file name
  for root, dirs, files in os.walk(directory):
    for file in files:
      if file.endswith(".md"):
        num_md_files += 1
        filename = os.path.join(root,file)
        
        print('Processing: ' + filename)

        post = frontmatter.load(filename)

        # Check that post has title and permalink, otherwise exclude it from LSA
        if ('title' not in post.keys() or 'permalink' not in post.keys()):
          print(filename + ' is excluded from LSA')
        else:
          # Run markdown_to_text
          text = markdown_to_text(post.content)
          
          # Tokenize text
          tokens = nltk.word_tokenize(text)
          
          # Stem tokens
          stemmed_text = ''
          for word in tokens:
            lower_case_word = word.lower()
            if not lower_case_word in stop_words and not lower_case_word in custom_stop_words:
                  stemmed_text += porter_stemmer.stem(word) + ' '
              
          # Append result to text_array and filename_array
          text_array.append(stemmed_text)
          file_meta_array.append({'filename': file, 'title': post['title'], 'url': site_url + post['permalink'] })

  print('Number of .md files: ', num_md_files)
  return text_array, file_meta_array

def findNRelatedPosts(n, similarity_vec):
    """ Given a similarity vector, find the n closest related posts
    
    Returns an array with the index of these n closest related posts 
    sorted in descending order of relatedness"""
   
    nRelatedPosts = argsort(-(similarity_vec))
    return nRelatedPosts[1:n+1]

def train(git_url, site_url, directory_name, table_name):
  """ Run the entire training workflow:
      download data, preprocess, LSA, and storing values in DynamoDB """

  # ============
  # Download data
  # ============

  # Delete directory if it exists
  if os.path.isdir(directory_name):
    print('Deleting dir: ' + directory_name)
    shutil.rmtree(directory_name)

  print('Cloning repo: ' + git_url + ' to ' + directory_name)
  repo = git.Repo.clone_from(git_url, directory_name, branch='master', depth=1)
  print('Repo successfully cloned: ' + git_url)

  # ============
  # Pre-processing
  # ============

  # Convert all .md files into text
  print('Start preprocessing markdown to text')
  text_array, file_meta_array = generate_text_array(site_url, directory_name)
  print('Preprocessed markdown to text')

  # ============
  # LSA
  # ============
  #
  # Tfidf vectorizer:
  #   - Strips out stop words
  #   - Filters out terms that occur in more than half of the docs (max_df=0.5)
  #   - Filters out terms that occur in only one document (min_df=2).
  #   - Selects the 10,000 most frequently occuring words in the corpus.
  #   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
  #     document length on the tf-idf values. 

  print('Starting LSA step')

  vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                               min_df=1, stop_words='english',
                               use_idf=True)

  tfidf_matrix = vectorizer.fit_transform(text_array)
  print('tf-idf matrix shape: ', tfidf_matrix.shape) 
  pprint.pprint(tfidf_matrix)

  # LSA - perform SVD on the TF-IDF matrix
  svd = TruncatedSVD(100)
  lsa = make_pipeline(svd, Normalizer(copy=False))

  lsa_result = lsa.fit_transform(tfidf_matrix)
  print('LSA matrix', lsa_result.shape)
  pprint.pprint(lsa_result)

  # ============
  # Delete tmp dir
  # ============
  if os.path.isdir(directory_name):
    shutil.rmtree(directory_name)

  # ============
  # Cosine similarity
  # ============
  # Generate a cosine similarity matrix

  # Each value represents the similarity of the terms between one document and another
  # cosine_similarity_matrix(m,n) is the similarity score between documents m and n

  print('Starting similarity measurement step')

  similarity_matrix = cosine_similarity(lsa_result, lsa_result)
  print('Cosine similarity matrix', similarity_matrix.shape)
  pprint.pprint(similarity_matrix)

  # ============
  # Save to DynamoDB
  # ============
  # Save the n-closest related posts for each post into DynamoDB
  print('Saving training results to DynamoDB')

  for idx, vec in enumerate(similarity_matrix):
    relatedIndexArray = findNRelatedPosts(NUM_RELATED_POSTS, vec)

    relatedPosts = []
    for index in relatedIndexArray:
      relatedPosts.append({'title': file_meta_array[index]['title'], 'url': file_meta_array[index]['url']})
    
    item = {
           'title': file_meta_array[idx]['title'],
           'url': file_meta_array[idx]['url'],
           'related_posts': relatedPosts
        }

    related_posts_table = dynamodb.Table(table_name)
    related_posts_table.put_item(
       Item=item
    )

  print('Training done for ' + site_url)

def main():
  with open('isomer-sites.json') as json_file:
    isomer_sites = json.load(json_file)

  for site in isomer_sites:
    print('Starting training for ', site)
    train(site['git_url'], site['site_url'], site['directory_name'], AWS_DYNAMODB_TABLE_NAME)

if __name__ == "__main__":
  main()