import json, collections, os, random, glob, math, string, re, torch
import numpy as np
import timeit
from tqdm import tqdm 
from torch.utils.data import TensorDataset
from transformers.models.bert.tokenization_bert import whitespace_tokenize

import logging
import math
import re
import string

from utils import *


logger = logging.getLogger(__name__)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 answers=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.answers = answers

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.start_position:
            s += ", end_position: %d" % (self.end_position)
        if self.start_position:
            s += ", is_impossible: %r" % (self.is_impossible)
        if self.start_position:
            s += ", answers: %r" % (self.answers)
        return s

def read_squad_examples(input_file, is_training):
    """Read a SQuAD json file into a list of SquadExample."""

    with open(input_file, "r", encoding='utf-8') as reader:
        source = json.load(reader)
        input_data = source["data"]
        version = source["version"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"].lower()
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                answers = []
                if is_training:
                    if version == "v2.0":
                        is_impossible = qa["is_impossible"]
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        #print(entry["title"], qas_id)
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"].lower()
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs. '%s'",
                                            actual_text, cleaned_answer_text)
                            continue
                    else: 
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                else:
                    answers = qa["answers"]

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    answers=answers)
                
                #qas_id: uit_01__08947_13_1, 
                #question_text: Dân tộc nào có nhiều nét tương đồng nhất với dân tộc Ê Đê?, 
                #doc_tokens: [Người Ê đê và người Gia Rai vốn cùng nguồn gốc từ một tộc người Orang Đê cổ được ghi chép khá nhiều trong các bia ký Champa, Khmer,...Orang Đê có thể là nhóm mà người Ê đê và Gia Rai gọi là Mdhur, trong văn hóa Mdhur có chứa đựng nhiều yếu tố văn hoá trung gian giữa người Ê đê và Gia Rai. Trong văn hóa Mdhur trước kia còn tồn tại tục hỏa táng người chết và bỏ tro trong chum, ché sau đó mới mang chôn cất trong nhà mồ, đây có thể là ảnh hưởng của đạo Hin-đu từ người Chăm. Xét về phương diện người Mdhur là cội nguồn xuất phát của người Ê đê và Gia Rai hiện đại. Trong lịch sử Orang Đê đã từng tồn tại các tiểu quốc sơ khai, với sự cai trị của các Mtao, Pơ Tao có thế lực trên một khu vực rộng lớn ở vùng người Gia Rai và Ê đê. Sự hình thành các tiểu quốc nhỏ là đặc điểm thường thấy ở các tộc người Đông Nam á:], 
                #start_position: 5, 
                #end_position: 6, 
                #is_impossible: False, 
                #answers: []

                examples.append(example)
                

                #Ranking examples
    return examples

class InputFeatures(object):
    """
    A single set of features of data.
    """
    
    def __init__(self,
                 unique_id, 
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def _improve_answer_span(doc_tokens, 
                         input_start, 
                         input_end, 
                         tokenizer,
                         orig_answer_text, add_prefix_space=False):
    """
    Returns tokenized answer spans that better match the annotated answer.
    """
    
    # print(doc_tokens)
    # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']

    # print(input_start)
    # 66
    
    # print(input_end) 
    # 69
    
    # print(tokenizer)
    # <transformers.tokenization_bert.BertTokenizer object at 0x000001C1CE9D4F98>
    
    # print(orig_answer_text)
    # in the late 1990s
    
    if add_prefix_space:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
    else:
        tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))    # print(tok_answer_text)
    # in the late 1990s

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            # print(text_span)
            # in the late 1990s
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """
    Check if this is the "max context" doc span for the token.
    """
    
    best_score = None
    best_span_index = None
    
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, 
                                 tokenizer, 
                                 max_seq_length,
                                 doc_stride,
                                 max_query_length,
                                 is_training,
                                 cls_token_at_end=False,
                                 cls_token="[CLS]", 
                                 sep_token="[SEP]", 
                                 pad_token=0,
                                 add_prefix_space=False,
                                 sequence_a_segment_id=0, 
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=0, 
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """
    Loads a data file into a list of `InputBatch`s.
    """
    unique_id = 1000000000
    
    features = []
    for (example_index, example) in enumerate(examples):
        # print(example_index)
        # 0
        
        # print(example)
        # qas_id: 56be85543aeaaa14008c9063, 
        # question_text: When did Beyonce start becoming popular?,
        # doc_tokens: [Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".], 
        # start_position: 39, 
        # end_position: 42
        
        if add_prefix_space:
            query_tokens = tokenizer.tokenize(example.question_text)
        else:
            query_tokens = tokenizer.tokenize(example.question_text)        # print(query_tokens)
        # ['when', 'did', 'beyonce', 'start', 'becoming', 'popular', '?']
    
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]
            
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        
        # `token`s are separated by whitespace; `sub_token`s are separated in a `token` by symbol
        for (i, token) in enumerate(example.doc_tokens):
            # print(i)
            # 0
            
            # print(token)
            # Beyoncé
            
            orig_to_tok_index.append(len(all_doc_tokens))
            if add_prefix_space:
                sub_tokens = tokenizer.tokenize(token)
            else:
                sub_tokens = tokenizer.tokenize(token)
            # print(sub_tokens)
            # ['beyonce']
            # ['gi', '##selle']
            # ['knowles', '-', 'carter']
            # ['(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/']
            # ...
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        
        # print(tok_to_orig_index)
        # [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 10, 11, 12, 12, 13, 13, 14, 15, 16, 17, 17, 18, 19, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 47, 47, 48, 48, 48, 49, 49, 49, 50, 50, 51, 52, 53, 54, 54, 55, 56, 56, 57, 58, 59, 60, 61, 62, 63, 63, 63, 64, 64, 64, 65, 66, 67, 68, 69, 69, 70, 71, 72, 73, 74, 75, 76, 76, 76, 77, 78, 78, 79, 80, 81, 82, 82, 82, 82, 83, 84, 85, 86, 87, 88, 89, 90, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 101, 101, 102, 103, 103, 104, 105, 105, 106, 107, 107, 108, 108, 108]
        
        # print(orig_to_tok_index)
        # [0, 1, 3, 6, 16, 23, 25, 26, 28, 30, 31, 32, 33, 35, 37, 38, 39, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 77, 80, 83, 85, 86, 87, 88, 90, 91, 93, 94, 95, 96, 97, 98, 99, 102, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 120, 121, 123, 124, 125, 126, 130, 131, 132, 133, 134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 152, 153, 155, 156, 158, 159, 161]
        
        # print(all_doc_tokens)
        # ['beyonce', 'gi', '##selle', 'knowles', '-', 'carter', '(', '/', 'bi', '##ː', '##ˈ', '##j', '##ɒ', '##nse', '##ɪ', '/', 'bee', '-', 'yo', '##n', '-', 'say', ')', '(', 'born', 'september', '4', ',', '1981', ')', 'is', 'an', 'american', 'singer', ',', 'songwriter', ',', 'record', 'producer', 'and', 'actress', '.', 'born', 'and', 'raised', 'in', 'houston', ',', 'texas', ',', 'she', 'performed', 'in', 'various', 'singing', 'and', 'dancing', 'competitions', 'as', 'a', 'child', ',', 'and', 'rose', 'to', 'fame', 'in', 'the', 'late', '1990s', 'as', 'lead', 'singer', 'of', 'r', '&', 'b', 'girl', '-', 'group', 'destiny', "'", 's', 'child', '.', 'managed', 'by', 'her', 'father', ',', 'mathew', 'knowles', ',', 'the', 'group', 'became', 'one', 'of', 'the', 'world', "'", 's', 'best', '-', 'selling', 'girl', 'groups', 'of', 'all', 'time', '.', 'their', 'hiatus', 'saw', 'the', 'release', 'of', 'beyonce', "'", 's', 'debut', 'album', ',', 'dangerously', 'in', 'love', '(', '2003', ')', ',', 'which', 'established', 'her', 'as', 'a', 'solo', 'artist', 'worldwide', ',', 'earned', 'five', 'grammy', 'awards', 'and', 'featured', 'the', 'billboard', 'hot', '100', 'number', '-', 'one', 'singles', '"', 'crazy', 'in', 'love', '"', 'and', '"', 'baby', 'boy', '"', '.']
        
        tok_start_position = None
        tok_end_position = None
        
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # print(tok_start_position)
            # 66
            
            # print(tok_end_position)
            # 69
            (tok_start_position, tok_end_position) = _improve_answer_span(all_doc_tokens, 
                                                                          tok_start_position, 
                                                                          tok_end_position, 
                                                                          tokenizer, 
                                                                          example.orig_answer_text)
            # print(tok_start_position)
            # 66
            
            # print(tok_end_position)
            # 69
            
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        
        # We can have documents that are longer than the maximum sequence length. To deal with this we do a 
        # sliding window approach, where we take chunks of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        
        while start_offset < len(all_doc_tokens):
            # print(len(all_doc_tokens))
            # 426
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            # Take an example with stride
            # 
            # print(doc_spans)
            # [DocSpan(start=0, length=373)]
            # 
            # In this case, `start` will move a `doc_strike`, 128, so the new `start` is 128  
            # And the new `length` is 426 - 128 = 298
            # 
            # [DocSpan(start=0, length=373), DocSpan(start=128, length=298)]
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)
        
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            # `p_mask`: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keeps the classification token (set to 0) (not sure why...)
            p_mask = []

            # `[CLS]` token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0
                
            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # [SEP] token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph built based on `doc_span`
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # [SEP] token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # [CLS] token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                # Index of classification token
                cls_index = len(tokens) - 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            # print(input_ids)
            # [101, 2043, 2106, 20773, 2707, 3352, 2759, 1029, 102, 20773, 21025, 19358, 22815, 1011, 5708, 1006, 1013, 12170, 23432, 29715, 3501, 29678, 12325, 29685, 1013, 10506, 1011, 10930, 2078, 1011, 2360, 1007, 1006, 2141, 2244, 1018, 1010, 3261, 1007, 2003, 2019, 2137, 3220, 1010, 6009, 1010, 2501, 3135, 1998, 3883, 1012, 2141, 1998, 2992, 1999, 5395, 1010, 3146, 1010, 2016, 2864, 1999, 2536, 4823, 1998, 5613, 6479, 2004, 1037, 2775, 1010, 1998, 3123, 2000, 4476, 1999, 1996, 2397, 4134, 2004, 2599, 3220, 1997, 1054, 1004, 1038, 2611, 1011, 2177, 10461, 1005, 1055, 2775, 1012, 3266, 2011, 2014, 2269, 1010, 25436, 22815, 1010, 1996, 2177, 2150, 2028, 1997, 1996, 2088, 1005, 1055, 2190, 1011, 4855, 2611, 2967, 1997, 2035, 2051, 1012, 2037, 14221, 2387, 1996, 2713, 1997, 20773, 1005, 1055, 2834, 2201, 1010, 20754, 1999, 2293, 1006, 2494, 1007, 1010, 2029, 2511, 2014, 2004, 1037, 3948, 3063, 4969, 1010, 3687, 2274, 8922, 2982, 1998, 2956, 1996, 4908, 2980, 2531, 2193, 1011, 2028, 3895, 1000, 4689, 1999, 2293, 1000, 1998, 1000, 3336, 2879, 1000, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # print(input_mask)
            # [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            # print(segment_ids)
            # [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            # Only `sequence_b_segment_id` is set to 1
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            
            # Get `start_position` and `end_position`
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation we throw it out, 
                # since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
           
            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible)
            )
            
            unique_id += 1

    return features

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False, model_type='bert'):

    
    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args["local_rank"] not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  
    
    # Load data features from cache or dataset file
    input_file = args["predict_file"] if evaluate else args["train_file"]
    cached_features_file = os.path.join(
        os.path.dirname(input_file), 
        'cached_{}_{}_{}_{}'.format(
            'dev' if evaluate else 'train',
            args['base_model_name'],
            str(args["max_seq_length"]),
            input_file.split('/')[-1][:-5]
        )
    )
    
    if os.path.exists(cached_features_file) and not output_examples:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", input_file)
        
        # Call `read_squad_examples()`
        examples = read_squad_examples(input_file=input_file,
                                       is_training=not evaluate)

        # Call `convert_examples_to_features()`
        features = convert_examples_to_features(examples, 
                                                tokenizer, 
                                                max_seq_length=args["max_seq_length"],
                                                doc_stride=args["doc_stride"],
                                                max_query_length=args["max_query_length"],
                                                is_training=not evaluate,
                                                cls_token_at_end=False,
                                                cls_token = '<s>' if model_type == 'roberta' else '[CLS]',
                                                sep_token = '</s>' if model_type == 'roberta' else '[SEP]',
                                                pad_token = 1 if model_type == 'roberta' else  0,
                                                add_prefix_space = True if model_type == 'roberta' else  False,
                                                sequence_a_segment_id=0, # 'pad_token': '[PAD]'
                                                sequence_b_segment_id=1,
                                                cls_token_segment_id=0, 
                                                pad_token_segment_id=0,
                                                mask_padding_with_zero=True)
        
        if args["local_rank"] in [-1, 0]:
            print("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    if args["local_rank"] == 0 and not evaluate:
        torch.distributed.barrier()  

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids,
                                all_example_index, 
                                all_cls_index, 
                                all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, 
                                all_input_mask, 
                                all_segment_ids,
                                all_start_positions, 
                                all_end_positions,
                                all_cls_index, 
                                all_p_mask)

    if output_examples:
        return dataset, examples, features

    return dataset