from client import FlashAttentionKernel

def main():
    print("=== Testing FlashAttention Online Tiled Kernel ===")
    fa = FlashAttentionKernel(block_size=2)

    # 4 tokens, hidden dim 2
    Q = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    K = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
    V = [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]

    out = fa.forward(Q, K, V)
    print("Tiled Attention Output shape:", len(out), "x", len(out[0]))
    for idx, row in enumerate(out):
        print(f"Token {idx} context vector: {[round(x, 4) for x in row]}")

    assert len(out) == 4
    assert len(out[0]) == 2
    print("=== All tests passed successfully! ===")

if __name__ == "__main__":
    main()
