#%%
import numpy as np

def ppmi (C, verbose=False, eps=1e-8):  # C는 동시발생 행렬, verbose 진행상황 출력 여부를 알려주는 플래그
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)  # 이 부분을 통해 -inf로 가는 문제를 해소
            
            if verbose:
                cnt += 1
                if cnt % (total//100 +1) == 0:
                    print('%.1f%% 완료' % (100*cnt/total))
    
    return M
# %%
import sys
sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

text = "you say goodbye and I say hello"
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)
print(C)
print('-'*50)
print(W)

# %%
