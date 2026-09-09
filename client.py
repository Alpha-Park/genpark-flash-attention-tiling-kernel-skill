import math

class FlashAttentionKernel:
    """
    Online Softmax Tiling (FlashAttention algorithm).
    Computes Attention(Q, K, V) = softmax(Q K^T / sqrt(d)) V in small SRAM tiles.
    Maintains running max (m) and running sum of exponentials (l) for numerical stability
    without storing the N x N attention matrix.
    """
    def __init__(self, block_size=2):
        self.B = block_size

    def forward(self, Q, K, V):
        N = len(Q)
        D = len(Q[0])
        scale = 1.0 / math.sqrt(D)

        O = [[0.0] * D for _ in range(N)]
        l = [0.0] * N
        m = [-float('inf')] * N

        for j in range(0, N, self.B):
            K_j = K[j:min(j + self.B, N)]
            V_j = V[j:min(j + self.B, N)]

            for i in range(0, N, self.B):
                Q_i = Q[i:min(i + self.B, N)]

                for row_idx, q in enumerate(Q_i):
                    glob_i = i + row_idx
                    m_prev = m[glob_i]
                    l_prev = l[glob_i]

                    scores = []
                    for k_vec in K_j:
                        dot = sum(q[d] * k_vec[d] for d in range(D)) * scale
                        scores.append(dot)

                    m_curr = max([m_prev] + scores)
                    P_tilde = [math.exp(s - m_curr) for s in scores]
                    sum_P = sum(P_tilde)

                    alpha = math.exp(m_prev - m_curr) if m_prev != -float('inf') else 0.0
                    l_curr = alpha * l_prev + sum_P

                    for d in range(D):
                        pv = sum(P_tilde[kv_idx] * V_j[kv_idx][d] for kv_idx in range(len(V_j)))
                        prev_term = alpha * l_prev * O[glob_i][d] if l_curr > 0 else 0.0
                        O[glob_i][d] = (prev_term + pv) / l_curr

                    m[glob_i] = m_curr
                    l[glob_i] = l_curr

        return O
