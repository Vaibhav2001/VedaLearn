from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from cltk.corpus.sanskrit.itrans.unicode_transliterate import ItransTransliterator
lang='hi'


model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path="data/model")


def generate(start):
    tokenizer = GPT2Tokenizer.from_pretrained('data/tokens',
                                              additional_special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"],
                                              pad_token='<pad>', max_len=512)

    beam_output = model.generate(
        tokenizer.encode(start, return_tensors='pt'),
        max_length=50,
        num_beams=5,
        temperature=0.7,
        no_repeat_ngram_size=2,
        num_return_sequences=5
    )

    gen_txt = tokenizer.decode(beam_output[0])

    # gt = cltk.corpus.sanskrit.itrans.unicode_transliterate.ItransTransliterator.from_itrans(gen_txt, lang)
    return gen_txt
