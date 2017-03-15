from nltk.tokenize import WhitespaceTokenizer
from email.parser import Parser
from collections import defaultdict
import os
try:
    import cPickle as pickle
except:
    import pickle
import sys
import operator
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
import time
from memory import asizeof
import math 
import json


class IndexCache:
    ''' Structure to save index in cache and related functionality'''
    def __init__(self):
        self._index_block_size = 100 * 1024 * 1024 #size of index block in memory
        self._index_cache_size = 50   # Number of index blocks to cache
        self._index_cache_queue = [] # LRU queue
        self._rebalance_every = 20000  # Number of writes after which we try to rebalance a block
        self._num_inserts = defaultdict(int) # To keep track of number of inserts in a block for the current shard
        self._current_index_block_num = 0
        self._current_shard = -1
        self._metadata = dict()
        self._index_cache = dict()
        if not os.path.exists("index_data"):
            os.makedirs("index_data")
        if os.path.isfile("index_data/cache.meta"):
            temp = pickle.load(open("index_data/cache.meta", "rb"))
            self._metadata = temp["metadata"]
            self._current_index_block_num = temp["current_index_block"]
            self._current_shard = temp["current_shard"]
        if self._current_shard == -1:
            self.create_new_shard()
        #self._populate_cache()

    def create_new_shard(self):
        self.flush()
        self._current_shard += 1
        path = os.path.join("index_data", str(self._current_shard))
        if not os.path.exists(path):
            os.makedirs(path)
        self._current_index_block_num = 0
        self._num_inserts = defaultdict(int)

    def get(self, query):
        index_block_nums = self._metadata.get(query)
        if index_block_nums is None:
            return 
        results = dict()
        for shard, block in enumerate(index_block_nums):
            if block is None:
                continue
            self._load_block(shard, block)
            cache_key = str(shard) + "_" + str(block)
            results.update(self._index_cache[cache_key].get(query))
        return results

    def add(self, token, key, pos):
        index_block_nums = self._metadata.get(token)
        data = [{"token": token, "key": key, "pos": set([pos])}]
        if index_block_nums and len(index_block_nums) == self._current_shard + 1:
            block = index_block_nums[self._current_shard]
            self._add_to_index_block(block, data)
        else:
            self._add_to_index_block(self._current_index_block_num, data)
    
    def _update_metadata(self, token, block_num):
        block_nums = self._metadata.get(token)
        if block_nums is None:
            start_index = 0
            self._metadata[token] = []
        else:
            start_index = len(block_nums)
        for i in range(start_index, self._current_shard + 1):
            self._metadata[token].append(None)
        self._metadata[token][self._current_shard] = block_num
        
    def _add_to_index_block(self, block_num, data):
    #''' data => [{token: <keyword token>
    #            key: <document key>
    #            pos: set[(start_position1, end_position1),... ]}....]'''
        self._load_block(self._current_shard, block_num)
        for el in data:
            token = el["token"]
            key = el["key"]
            pos = el["pos"]
            cache_key = str(self._current_shard) + "_" + str(block_num)
            if self._index_cache[cache_key].get(token):
                self._index_cache[cache_key][token][key] = self._index_cache[cache_key][token][key].union(pos)
            else:
                self._update_metadata(token, block_num)
                self._index_cache[cache_key][token] = defaultdict(set) 
                self._index_cache[cache_key][token][key] = self._index_cache[cache_key][token][key].union(pos)
            self._num_inserts[block_num] += 1 #Only current shard will have write operations
        if self._num_inserts[block_num] > self._rebalance_every:
            self._num_inserts[block_num] = 0
            self._balance_block_size(cache_key)

    def _balance_block_size(self, cache_key):
        curr_block_size = asizeof(self._index_cache[cache_key])
        if curr_block_size < self._index_block_size:
            return
        overflow_records = []
        offset_block_size = curr_block_size - self._index_block_size
        deleted_members_size = 0
        while deleted_members_size < offset_block_size:
            token, docs = self._index_cache[cache_key].popitem()
            deleted_members_size += asizeof(token)
            deleted_members_size += asizeof(docs)
            for doc in docs:
                el = dict()
                el["token"] = token
                el["key"] = doc
                el["pos"] = docs[doc]
                overflow_records.append(el)
        block_num = int(cache_key.split("_")[1])
        if block_num == self._current_index_block_num:
            self._current_index_block_num += 1
        self._add_to_index_block(self._current_index_block_num, overflow_records)

    def _load_block(self, shard, block):
        cache_key = str(shard) + "_" + str(block)
        if self._index_cache.get(cache_key):
            # Block is already in the cache. Move to top of the queue
            block_index = self._index_cache_queue.index(cache_key)
            self._index_cache_queue[:] = self._index_cache_queue[:block_index] + self._index_cache_queue[block_index+1:]
            self._index_cache_queue.insert(0, cache_key)
            return        
        self._evict_block()
        path = os.path.join("index_data", str(shard), str(block))
        if os.path.isfile(path):
            self._index_cache[cache_key] = pickle.load(open(path, "rb"))
        else:
            self._index_cache[cache_key] = dict()
        self._index_cache_queue.insert(0, cache_key)

    def _populate_cache(self):
        pass

    def _evict_block(self):
        if len(self._index_cache_queue) < self._index_cache_size:
            #Cache not full. No need to evict LRU block
            return
        lru_key = self._index_cache_queue[-1]
        lru_shard, lru_block = lru_key.split("_")
        if int(lru_shard) == self._current_shard and self._num_inserts[int(lru_block)] > 0:
            #Cache block is dirty. Save to disk
            self._flush_block(lru_shard, lru_block)
        del self._index_cache[lru_key]
        del self._index_cache_queue[-1]

    def _flush_block(self, shard, block):
        cache_key = str(shard) + "_" + str(block)
        path = os.path.join("index_data", str(shard), str(block))
        pickle.dump(self._index_cache[cache_key], open(path, "wb"))

    def flush(self):
        meta = dict()
        meta["metadata"] = self._metadata
        meta["current_index_block"] = self._current_index_block_num
        meta["current_shard"] = self._current_shard
        pickle.dump(meta, open("index_data/cache.meta", "wb"))
        for cache_key in self._index_cache:
            shard, block = cache_key.split("_")
            self._flush_block(shard, block)

class InvertedIndex:
    ''' Main Inverted-Index structure'''
    def __init__(self):
        self._tokenizer = WhitespaceTokenizer()
        self._index_cache = IndexCache()
        self._stop_words = set(stopwords.words('english'))
        self._stemmer = SnowballStemmer("english")
        self._max_documents_per_shard = 50000
        self._num_documents_in_current_shard = 0 
        if os.path.isfile("index_data/index.meta"):
            self._num_documents_in_current_shard = pickle.load(open("index_data/index.meta"))

    def search(self, query):
        combined_results = None
        ret_results = None
        for i in range(0, len(query), 2):
            op = query[i]
            keyword = self._stemmer.stem(query[i+1].strip(string.punctuation))
            keyword_results = self._search_keyword(keyword)
            if combined_results:
                if op == "AND":
                    combined_results = combined_results.intersection(set(keyword_results.keys()))
                elif op == "OR":
                    combined_results = combined_results.union(set(keyword_results.keys()))
                else:
                    return { "status": False, "message": "Malformed query"}  
                for doc in ret_results.keys():
                    if doc not in combined_results:
                        del ret_results[doc]
                    elif keyword_results.get(doc):
                        ret_results[doc].union(keyword_results[doc])
                for doc in keyword_results:
                    if doc not in ret_results:
                        ret_results[doc] = keyword_results[doc] 
            else: 
                combined_results = set(keyword_results.keys())
                ret_results = keyword_results
        result_counts = dict()
        for el in ret_results:
            result_counts[el] = len(ret_results[el])
        sorted_result_counts = sorted(result_counts.items(), key=operator.itemgetter(1), reverse=True)
        sorted_results = []
        for key, _ in sorted_result_counts:
            sorted_results.append( { "key": key, "positions" :  ret_results[key]})
        if len(sorted_results)> 0:
          ret = {"status": True, "results": sorted_results}
        else:
          ret = {"status": False, "message": "No hits"}
        return ret

    def _search_keyword(self, query):
        docs = self._index_cache.get(query)
        if not docs:
            return dict()
        return docs
    
    def add(self, key, text):
        self._num_documents_in_current_shard += 1
        if self._num_documents_in_current_shard > self._max_documents_per_shard:
            self._num_documents_in_current_shard = 0
            self._index_cache.create_new_shard()
        token_positions  = self._tokenizer.span_tokenize(text)
        for pos in token_positions:
            start_pos = pos[0]
            end_pos = pos[1]
            token = text[start_pos: end_pos].lower()
            if token in self._stop_words:
                continue
            token = token.strip(string.punctuation)
            token = self._stemmer.stem(token)
            if len(token) > 0:
                self._index_cache.add(token, key, (start_pos, end_pos))

    def delete(self, key, text):
        pass

    def save(self):
        pickle.dump(self._num_documents_in_current_shard, open("index_data/index.meta", "wb"))
        self._index_cache.flush()

class EnronInvertedIndex(InvertedIndex):

    def __init__(self):
        self.email_parser = Parser()
        InvertedIndex.__init__(self)

    def add_files(self, data_dir):
        self.traverse_and_index(data_dir)    

    def traverse_and_index(self, root_dir):
        i = 0
        for root, subdirs, files in os.walk(root_dir):
            for file in files:
                i += 1
                file_path =  os.path.join(root, file)
                with open(file_path) as f:
                    email = self.email_parser.parsestr(f.read())
                    text = email.get_payload()
                    start_time = time.time()
                    self.add(file_path, text)
                    end_time = time.time()


    def search_results(self, query, page=1, page_size=10, highlight_length=40):
        ret = dict()
        start_time = time.time()
        results = self.search(query)
        end_time = time.time()
        if not results["status"]:
            return results
        num_results = len(results["results"])
        start_index = (page - 1) * page_size
        end_index = page * page_size
        if start_index >= num_results:
            ret = {"status": False, "message": "No more results"}
        if end_index > num_results:
            end_index = num_results
        total_pages = int(math.ceil(num_results / float(page_size)))
        ret["status"] = results["status"]
        ret["total_count"] = num_results
        ret["total_pages"] = total_pages
        ret["current_page"] = page
        ret["start_index"] = start_index + 1
        ret["end_index"] = end_index
        ret["time_taken"] = end_time - start_time
        ret["results"] = []
        for result in results["results"][start_index:end_index]:
            with(open(result["key"])) as f:
                ret_el = dict()
                email = self.email_parser.parsestr(f.read())
                text = email.get_payload()
                ret_el["key"] = result["key"]
                ret_el["snippets"] = []
                for position in result["positions"]:
                    start_pos = position[0] - highlight_length if position[0] - highlight_length >= 0 else 0              
                    end_pos = position[0] + highlight_length if position[0] + highlight_length <= len(text) else len(text)
                    ret_el["snippets"] .append(text[start_pos : end_pos])
                ret["results"].append(ret_el)
        return ret
                    
