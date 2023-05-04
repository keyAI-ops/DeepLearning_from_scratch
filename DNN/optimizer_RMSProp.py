class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

''' AdaGrad는 과거의 기울기를 계속 제곱하여 더해간다. 그래서 학습이 진행될수록 갱신 속도가 느려진다.
    그러다 결국에는 갱신량이 0이 되어 학습이 멈춘다.
    
    이러한 단점을 보완한게 RMSProp이다. 새로운 기울기의 정보를 크게 반영하여 과거 기울기의 반영 규모를
    기하급수적으로 감소시킨다.
'''