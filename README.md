# VedaLearn
Language Model on Sanskrit. <br/>
Visit the website --> 


```
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("/content/drive/MyDrive/sanskrit/latest")
save_path = "/content/drive/MyDrive/sanskrit/model"
tokenizer = GPT2Tokenizer.from_pretrained(save_path, additional_special_tokens=["<s>","<pad>","</s>","<unk>","<mask>"],pad_token='<pad>', max_len=512)
from cltk.corpus.sanskrit.itrans.unicode_transliterate import ItransTransliterator
lang='hi'
```


```
text = "raaja" # King
beam_output = model.generate(
  tokenizer.encode(text, return_tensors='pt'),
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
gen_txt = tokenizer.decode(beam_output[0])
print(ItransTransliterator.from_itrans(gen_txt,lang))
```
    राजः कृतं कर्म कर्तारमनुगच्छति , 
    इति श्रीगारुडे महापुराणे उत्तरखण्डे द्वितीयांशे धर्मकाण्डे प्रेतकल्पे श्रीकृष्णगरुडसंवादे धर्मदृप्रेतकल्पे
    श्रीकृष्णगरुडमहापुराणम्-
    गरुड उवाच तार्क्ष्य शृणु तार्क्ष्य
```
text = "arjuna" # a prominent character in Mahabharata
beam_output = model.generate(
  tokenizer.encode(text, return_tensors='pt'),
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
gen_txt = tokenizer.decode(beam_output[0])
print(ItransTransliterator.from_itrans(gen_txt,lang))
```
    अर्जुन उवाच
    श्रृणुध्वं मुनिशार्दूलाः प्रवक्ष्यामि यथातथम् 
    प्रवक्ष्यामि समासेन गदतो मे निबोधत   
    इति श्रीमहापुराणे वायुप्रोक्ते भुवनविन्यासो नाम त्रिंशोऽध्यायःसह-
    ।सूत उवाच  संक्षेपात् प्रवक्ष्यामि वंश।
```
text = "krishna" # Lord Krishna
beam_output = model.generate(
  tokenizer.encode(text, return_tensors='pt'),
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
gen_txt = tokenizer.decode(beam_output[0])
print(ItransTransliterator.from_itrans(gen_txt,lang))
```
    क्रिश्नः कथं नु संसृजेत् , 
    इति श्रीगारुडे महापुराणे उत्तरखण्डे द्वितीयांशे धर्मकाण्डे प्रेतकल्पे श्रीकृष्णगरुडसंवादे प्रेतकृतदृप्रेतकल्पे
    श्रीकृष्णगरुडमहापुराणम्-
    गरुड उवाच 
```
text = "ganga" #the ganges
beam_output = model.generate(
  tokenizer.encode(text, return_tensors='pt'),
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
gen_txt = tokenizer.decode(beam_output[0])
print(ItransTransliterator.from_itrans(gen_txt,lang))
```
    गन्गः कृष्णबाहुना  -- 
    कृष्णोऽपि तं नृपोऽभ्येत्य गोपालैरतिदारुणम्  अक्रूरोऽयमत्यर्थश्व तस्य पुत्रोऽभवत्  ततश्व सत्राजि-बलभद्रोऽभूत् 
```
text = "agni:h" #fire
beam_output = model.generate(
  tokenizer.encode(text, return_tensors='pt'),
  max_length = 50,
  num_beams = 5,
  temperature = 0.7,
  no_repeat_ngram_size=2,
  num_return_sequences=5
)
gen_txt = tokenizer.decode(beam_output[0])
print(ItransTransliterator.from_itrans(gen_txt,lang))
```
    अग्नि:ह् पृथिव्यां दिशः सर्वांश्च विविधान् बहून् 
    ऋत्विजोऽपि च ऋषयो दृष्ट्वा तानृषींस्तथा  
    संहृत्य तांश्चापि राजाऽ।
