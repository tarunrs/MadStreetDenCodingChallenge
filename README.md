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

- Perform a search in the format OP keyword1 OP keyword2 ... OP keywordN where OP is 'AND' or 'OR'
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
- Pharses cannot be searched for, only individual keywords.
- The cache is cold when the webserver is started, so the first query will be slow.
- Delete document from index has not been implemented. But it should be pretty trivial to implement.

Benchmarks:
-----------
Tests were run on a machine with 7Gb of memory. Max RAM utilization was at 80% when the cache was full(50 index blocks cached) and the metadata size was at it's maximum.
- Time taken to index all the documents in the dataset(~520,000): 7.75 Hrs
- Time taken to create a new shard 90 - 240 seconds (This can be brought down to consistently at 90 seconds by flushing only the current shards index blocks to disk, since only the current shard can have writes)
- Time taken to index a document:
 -- max = 271.060 seconds (While creating a new shard. This can be brought down to ~90 seconds if we only flush the dirty index blocks)  
 -- min = 0.0 seconds (Possible when the document was empty)
 -- mean = 0.054 seconds
 -- median = 0.002 seconds
- Time to load an index block to memory 2-3 seconds

Performance enhancements and scaling:
-------------------------------------

- Using json/masrshal/messagepack for serializing/deserializing the index blocks instead of pickle as they are known to be faster.
- Using dict instead of defaultdict and lists instead of sets can help save memory as they occupy lesser memory. But the code will get a little more complex
- Tuning index block size. Smaller size would mean it would take lesser time to save/load from/to memory.
- Flushing only the current shard (instead of the whole cache) before creating a new shard would improve index time
- Having an oplog to save all the operations and performing them FIFO would improve the user experience and guarantee the call returning within a reasonable time.
- Since a document can only be in a single shard, this is easily scalable and distributable. We can have slave processes responsible for a subset of the shards and a master process that aggregates the results from the slave processes.

