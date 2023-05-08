# %%
import sys

sys.path.append("..")
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from sklearn.utils.extmath import randomized_svd
from dataset import ptb

# %%

window_size = 2
wordvec_size = 100

corpus, word2id, id2word = ptb.load_data("train")
vocab_size = len(word2id)

print("동시 발생 수 계산..")
C = create_co_matrix(corpus, vocab_size, window_size)
print("PPMI 계산..")
W = ppmi(C, verbose=True)
# %%
print("SVD 계산")
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for q in querys:
    most_similar(q, word2id, id2word, word_vecs, top=5)

# %%
