# Matlab
#bot.py
form chatterbot import ChatBot
chatbot = ChatBot("Chatpot")
exit_conditions = (":q", "quit", "exit")
while True:
query = input("> ")
if query in exit_conditions: break
else:
print(f"🪴 {chatbot.get_response(query)}")
# bot.py
from chatterbot import ChatBot
 from chatterbot.trainers import ListTrainer
chatbot = ChatBot("Chatpot")
trainer = ListTrainer(chatbot)
 trainer.train([
    "Hi",
   "Welcome, friend 🤗",
])
trainer.train([
    "Are you a plant?",	
    "No, I'm the pot below the plant!",
])
exit_conditions = (":q", "quit", "exit")
while True:
query = input("> ")
if query in exit_conditions: break
else:
print(f"🪴 {chatbot.get_response(query)}")
#cleaner.py
import re
def remove_chat_metadata(chat_export_file):
date_time = r"(\d+\/\d+\/\d+,\s\d+:\d+)" # e.g. "9/16/22, 06:34" dash_whitespace = r"\s-\s" # " - "
username = r"([\w\s]+)" # e.g. "Martin" metadata_end = r":\s" # ": "
pattern = date_time + dash_whitespace + username + metadata_end
with open(chat_export_file, "r") as corpus_file: content = corpus_file.read()
cleaned_corpus = re.sub(pattern, "", content)
return tuple(cleaned_corpus.split("\n"))
if	name	== " main ":
print(remove_chat_metadata("chat.txt"))
# cleaner.py
def remove_non_message_text(export_text_lines):
messages = export_text_lines[1:-1]
filter_out_msgs = ("<Media omitted>",)
return tuple((msg for msg in messages if msg not in filter_out_msgs))
if	name	== " main ":
message_corpus = remove_chat_metadata("chat.txt") cleaned_corpus = remove_non_message_text(message_corpus) print(cleaned_corpus)
from gpt3 import gpt3

def chat():
prompt = """Human: Hey, how are you doing?
AI: I'm good! What would you like to chat about? Human:"""
while True:
prompt += input('You: ') answer, prompt = gpt3(prompt,
temperature=0.9, frequency_penalty=1, presence_penalty=1, start_text='\nAI:', restart_text='\nHuman: ', stop_seq=['\nHuman:', '\n'])
print('GPT-3:' + answer)
if name == ' main ': chat()
encoder_input = np.array(df.question) decoder_input = np.array(df.decoder_input) decoder_label = np.array(df.decoder_label)

n_rows = df.shape[0] print(f"{n_rows} rows")

indices = np.arange(n_rows) np.random.shuffle(indices)

encoder_input = encoder_input[indices] decoder_input = decoder_input[indices] decoder_label = decoder_label[indices]

train_size = 0.9

train_encoder_input = encoder_input[:int(n_rows*train_size)] train_decoder_input = decoder_input[:int(n_rows*train_size)] train_decoder_label = decoder_label[:int(n_rows*train_size)]

test_encoder_input = encoder_input[int(n_rows*train_size):] test_decoder_input = decoder_input[int(n_rows*train_size):] test_decoder_label = decoder_label[int(n_rows*train_size):]

print(train_encoder_input.shape) print(train_decoder_input.shape) print(train_decoder_label.shape)

print(test_encoder_input.shape) print(test_decoder_input.shape) print(test_decoder_label.shape)
 
 df['question tokens']=df['question'].apply(lambda x:len(x.split())) df['answer tokens']=df['answer'].apply(lambda x:len(x.split())) plt.style.use('fivethirtyeight') fig,ax=plt.subplots(nrows=1,ncols=2,figsize=(20,5)) sns.set_palette('Set2')
sns.histplot(x=df['question tokens'],data=df,kde=True,ax=ax[0])
sns.histplot(x=df['answer tokens'],data=df,kde=True,ax=ax[1])
sns.jointplot(x='question tokens',y='answer tokens',data=df,kind='kde',fill=True,cmap='YlGnBu')
plt.show()
Text Cleaning
def clean_text(text): text=re.sub('-',' ',text.lower())
text=re.sub('[.]',' . ',text)
text=re.sub('[1]',' 1 ',text)
text=re.sub('[2]',' 2 ',text)
text=re.sub('[3]',' 3 ',text)
 
text=re.sub('[4]',' 4 ',text)
text=re.sub('[5]',' 5 ',text)
text=re.sub('[6]',' 6 ',text)
text=re.sub('[7]',' 7 ',text)
text=re.sub('[8]',' 8 ',text)
text=re.sub('[9]',' 9 ',text)
text=re.sub('[0]',' 0 ',text)
text=re.sub('[,]',' , ',text)
text=re.sub('[?]',' ? ',text)
text=re.sub('[!]',' ! ',text)
text=re.sub('[$]',' $ ',text)
text=re.sub('[&]',' & ',text)
text=re.sub('[/]',' / ',text)
text=re.sub('[:]',' : ',text)
text=re.sub('[;]',' ; ',text)
text=re.sub('[*]',' * ',text)
text=re.sub('[\']',' \' ',text)
text=re.sub('[\"]',' \" ',text) text=re.sub('\t',' ',text) return text

df.drop(columns=['answer tokens','question tokens'],axis=1,inplace=True)
 
df['encoder_inputs']=df['question'].apply(clean_text) df['decoder_targets']=df['answer'].apply(clean_text)+' <end>' df['decoder_inputs']='<start> '+df['answer'].apply(clean_text)+' <end>' df.head(10)

After preprocessing: for example , if your birth date is january 1 2 , 1 9 8 7 , write 0 1 / 1 2 / 8 7 .
Max encoder input length: 27 Max decoder input length: 29 Max decoder target length: 28
Tokenization
vectorize_layer=TextVectorization( max_tokens=vocab_size, standardize=None, output_mode='int',
output_sequence_length=max_sequence_length

)
vectorize_layer.adapt(df['encoder_inputs']+' '+df['decoder_targets']+'
<start> <end>') vocab_size=len(vectorize_layer.get_vocabulary()) print(f'Vocab size: {len(vectorize_layer.get_vocabulary())}') print(f'{vectorize_layer.get_vocabulary()[:12]}')
 
Vocab size: 2443
['', '[UNK]', '<end>', '.', '<start>', "'", 'i', '?', 'you', ',', 'the', 'to']
def sequences2ids(sequence): return vectorize_layer(sequence)

def ids2sequences(ids): decode=''
if type(ids)==int: ids=[ids]
for id in ids: decode+=vectorize_layer.get_vocabulary()[id]+' '
return decode


x=sequences2ids(df['encoder_inputs']) yd=sequences2ids(df['decoder_inputs']) y=sequences2ids(df['decoder_targets'])

print(f'Question sentence: hi , how are you ?')
print(f'Question to tokens: {sequences2ids("hi , how are you ?")[:10]}') print(f'Encoder input shape: {x.shape}')
print(f'Decoder input shape: {yd.shape}') print(f'Decoder target shape: {y.shape}') Question sentence: hi , how are you ?
 
Question to tokens: [1971	9	45	24	8	7	0	0	0	0]
Encoder input shape: (3725, 30)
Decoder input shape: (3725, 30)
Decoder target shape: (3725, 30) print(f'Encoder input: {x[0][:12]} ...')
print(f'Decoder input: {yd[0][:12]} ...')	# shifted by one time step of the target as input to decoder is the output of the previous timestep
print(f'Decoder target: {y[0][:12]} ...')
Encoder input: [1971	9	45	24	8 194	7	0	0	0	0	0] ...

Decoder input: [ 4	6	5 38 646	3 45  41 563	7	2	0] ...
Decoder target: [ 6	5	38 646	3	45 41 563	7	2	0	0] ...
data=tf.data.Dataset.from_tensor_slices((x,yd,y)) data=data.shuffle(buffer_size)

train_data=data.take(int(.9*len(data))) train_data=train_data.cache() train_data=train_data.shuffle(buffer_size) train_data=train_data.batch(batch_size) train_data=train_data.prefetch(tf.data.AUTOTUNE) train_data_iterator=train_data.as_numpy_iterator()

val_data=data.skip(int(.9*len(data))).take(int(.1*len(data))) val_data=val_data.batch(batch_size)
 
val_data=val_data.prefetch(tf.data.AUTOTUNE)


_=train_data_iterator.next()
print(f'Number of train batches: {len(train_data)}') print(f'Number of training data: {len(train_data)*batch_size}') print(f'Number of validation batches: {len(val_data)}') print(f'Number of validation data: {len(val_data)*batch_size}') print(f'Encoder Input shape (with batches): {_[0].shape}') print(f'Decoder Input shape (with batches): {_[1].shape}') print(f'Target Output shape (with batches): {_[2].shape}') Number of train batches: 23
Number of training data: 3427 Number of validation batches: 3 Number of validation data: 447
Encoder Input shape (with batches): (149, 30) Decoder Input shape (with batches): (149, 30) Target Output shape (with batches): (149, 30)

# Install Python 3.7 Version sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3- dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev

sudo add-apt-repository ppa:deadsnakes/ppa sudo apt install python3.7
python3.7 --version # Install Pip
python3.7 -m pip install pip

# if you are getting distutils error
sudo apt-get install python3.7-distutilsCopy Code
pip install virtualenvCopy Code
python3.7 -m virtualenv venv

source venv/bin/activateCopy Code
pip install Flask

pip install ChatterBot==1.0.8
pip install chatterbot-corpus==1.2.0 pip install spacy==2.1.9
pip install nltk==3.8.1Copy Code
from chatbot import chatbot

from flask import Flask, render_template, request app = Flask(	name	)
app.static_folder = 'static'



@app.route("/") def home():
return render_template("index.html") @app.route("/get")
def get_bot_response():

userText = request.args.get('msg')
return str(chatbot.get_response(userText)) if	name	== "	main	":
app.run()Copy Code
from chatterbot import ChatBot import spacy
spacy.cli.download("en_core_web_sm") spacy.cli.download("en")


nlp = spacy.load('en_core_web_sm') chatbot = ChatBot(
'CoronaBot', storage_adapter='chatterbot.storage.SQLStorageAdapter', logic_adapters=[
'chatterbot.logic.MathematicalEvaluation',
'chatterbot.logic.TimeLogicAdapter', 'chatterbot.logic.BestMatch',
{

'import_path': 'chatterbot.logic.BestMatch',

'default_response': 'I am sorry, but I do not understand. I am still learning.', 'maximum_similarity_threshold': 0.90
}

],

database_uri='sqlite:///database.sqlite3'

)

# Training With Own Questions

from chatterbot.trainers import ListTrainer trainer = ListTrainer(chatbot)
training_data_quesans = open('training_data/ques_ans.txt').read().splitlines() training_data_personal = open('training_data/personal_ques.txt').read().splitlines() training_data = training_data_quesans + training_data_personal trainer.train(training_data)
# Training With Corpus

from chatterbot.trainers import ChatterBotCorpusTrainer
trainer_corpus = ChatterBotCorpusTrainer(chatbot) trainer_corpus.train(
'chatterbot.corpus.english'

)Copy Code
chatbot_project/

├── app.py

├── chatbot.py

├── venv

├── database.sqlite3.py

├── templates/
│	└── index.html

├── training_data/

│	├── ques_ans.txt

│	└──personal_ques.txt

└── static/

└── style.cssCopy Code
python3.7 app.pyCopy Code
