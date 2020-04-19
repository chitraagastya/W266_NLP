import pandas as pd
import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Load the json data
def load_json_file(name):
  """
  Load the json file and return a json object
  """
  with open(name,encoding='utf-8') as myfile:
    data = json.load(myfile)
    return data

# Convert json data object to a pandas data frame
def convert_to_pd(data):
  """
  Load the data to a pandas dataframe.
  Dataframe Columns:
    title
    para_index
    context
    q_index
    q_id
    q_isimpossible
    q_question
    q_anscount - number of answers
    q_answers - a list of object e.g [{ text: '', answer_start: 123}, ...]
  """
  result = []
  for pdata in data['data']:
    for para in pdata['paragraphs']:
      for q in para['qas']:
        result.append({
            'title' : pdata['title'],
            'context' : para['context'],
            'q_id' : q['id'],
            'q_isimpossible' : q['is_impossible'],
            'q_question' : q['question'],
            'q_anscount' : len(q['answers']),
            'q_answers' : [a for a in q['answers']],
            'q_answers_text': [a.get("text") for a in q['answers']],
            'context_lowercase': para['context'].lower(),
            'q_question_lowercase' : q['question'].lower(),
            'q_answers_text_lowercase': [a.get("text").lower() for a in q['answers']],
            
        })

  return pd.DataFrame.from_dict(result, orient='columns')

# Load the file from shareable google drive link and return a pandas dataframe
def loadDataFile(filename): 
  """
  Download a file from google drive with the shared link
  """ 
  data = load_json_file(filename)
  return convert_to_pd(data)

def get_c_q_a(dataset):
    q_id_list = []
    context_list =[]
    questions_list = []
    answers_list =[]
    q_impossible_list =[]
    for index,row in dataset.iterrows():
        q_id_list.append(row.q_id)
        context_list.append(row.context)
        questions_list.append(row.q_question)
        q_impossible_list.append(int(row.q_isimpossible))
        if len(row.q_answers_text)>0 :
            answers_list.append(row.q_answers_text[0])
        else:
            answers_list.append("")
    return [q_id_list,context_list,questions_list,q_impossible_list,answers_list]

def tokenize_c_q_a(dataset,num_words=None):
    tokenizer = Tokenizer(num_words,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'+"''",oov_token='<unk>')
    data = dataset[1]+dataset[2]+dataset[4]
    tokenizer.fit_on_texts(data)
    vocab = {}
    for word,i in tokenizer.word_index.items():
        if num_words is not None:
          if i <= num_words:
            vocab[word] = i
        else:
          vocab[word] = i
    #vocab = tokenizer.word_index
    vocab['<s>'] = len(vocab)+1
    vocab['</s>'] = len(vocab)+1
    id_vocab = {value: key for key, value in vocab.items()}
    return (tokenizer,vocab,id_vocab)
