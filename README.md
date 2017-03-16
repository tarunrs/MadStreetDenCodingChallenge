# MadStreetDenCodingChallenge

Steps to run:
------------
- Install the required packages using:
  pip install -r requirements.txt

- Download the inverted index from https://drive.google.com/file/d/0B2NIWsIgx6EuV0VUdVNHbGxoWEU/view?usp=sharing

- Extract the tar.gz file into the project directory. It should create a folder called 'index_data' within the project directory

- Download the dataset from http://www.cs.cmu.edu/~enron/enron_mail_20150507.tgz

- Extract the data into a folder call "data" within the project directory. So the folder structure would be project_folder/data/maildir/..

- You might have to install the 'stopwords' nltk package

- Run python webserver.py

- Point your web browser to http://localhost:5000

- Perform a search in the format <OP> <keyword1> <OP> <keyword2> ... <OP> <keywordN> where OP is 'AND' or 'OR'
  Example:
  1) AND byzantine
  2) OR enron OR employess OR meeting

  Note: The first operand is not processed since it's irrelevant what the operand is. The query is processed left to right applying the operand to the previous results.

High level design:
------------------

- The index is divided into shards
- Each shard is divided into index blocks
- The index block contains the keyword and pointers to the documents it is present in
- A document can be contained only in one shard
- Each shard can contain many index blocks depending on the configured size of the index block
- A new shard is created after a configurable number of documents
- Index blocks are cached as per a LRU policy
- Number of index blocks that can be cached is configurable
- Keywords are stemmed and stop words are not indexed


Caveats:
--------
- Pharses cannot be searched for, only individual keywords
- The cache is cold when the webserver is started, so the first query will be slow
- Delete document from index has not been implemented

Benchmarks:
-----------
Test were run on a machine with 7Gb of memory. Max RAM utilization was at 80% when the cache was full(50 index blocks cached) and the metadata size was at it's maximum.
- Time taken to index all the documents in the dataset(~520,000): 7.75 Hrs
- Time taken to create a new shard 90 - 240 seconds (This can be brought down to consistently at 90 seconds by flushing only the current shards index blocks to disk, since only the current shard can have writes)
- Time to load an index block to memory 2-3 seconds
