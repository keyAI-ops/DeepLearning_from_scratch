# %%
import sys

sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

text = "you say goodbye and I say hello."
corpus, word2id, id2word = preprocess(text)
vocab_size = len(id2word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
W = ppmi(C)

# SVD
U, S, V = np.linalg.svd(W)

print(U[0])
print(U)
# %%
