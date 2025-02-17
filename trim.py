import os
import pickle as pkl
import torch

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer

raw_data = '/home/ef0036/Projects/contextLLM/data/ytsl/processed_words.pkl'
save_trim_dir = 'pretrain_models/MBart_trimmed'
save_mytran_dir = 'pretrain_models/mytran'

# 1) Load text data
data = []
with open(raw_data, 'rb') as f:
    obj = pkl.load(f)
    data = [*obj['dict_lem_to_id'].keys()]
    # for o in obj['dict_lem_to_id'].keys():
    #     data.extend(o)
    # currently 9805 + blank 9806

# 2) Create original tokenizer & model
tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-cc25",
    src_lang="en_XX",  # For English
    tgt_lang="en_XX"
)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
configuration = model.config

# 3) Trim tokenizer
tt = TokenizerTrimmer(tokenizer)
tt.make_vocab(data)        # Builds vocab from your data
tt.make_tokenizer()        # Actually creates the "trimmed" tokenizer

# 4) Trim model
mt = MBartTrimmer(model, configuration, tt.trimmed_tokenizer)
mt.make_weights(tt.trimmed_vocab_ids)
mt.make_model()

# 5) Save the trimmed tokenizer + model
trimmed_tokenizer = tt.trimmed_tokenizer
trimmed_model = mt.trimmed_model

os.makedirs(save_trim_dir, exist_ok=True)
trimmed_tokenizer.save_pretrained(save_trim_dir)
trimmed_model.save_pretrained(save_trim_dir)

# (Optional) You could also do a direct torch.save, but save_pretrained()
# already writes a pytorch_model.bin for you:
# torch.save(trimmed_model.state_dict(), os.path.join(save_trim_dir, 'pytorch_model.bin'))

# 6) Create "mytran_model" from the trimmed model
#    If you want the *entire* trimmed model, just reload it via from_pretrained:
mytran_model = MBartForConditionalGeneration.from_pretrained(save_trim_dir)

# 7) Save "mytran_model"
os.makedirs(save_mytran_dir, exist_ok=True)
mytran_model.save_pretrained(save_mytran_dir)

# Alternatively, if you *only* wanted to copy the model embeddings or do
# something special, you could manually copy submodules. But typically,
# loading from_pretrained is simpler for a complete model.

print("Trimming complete. New tokenizer & model are saved in:")
print(f"   {save_trim_dir}")
print("The same trimmed model also saved as 'mytran_model' in:")
print(f"   {save_mytran_dir}")
