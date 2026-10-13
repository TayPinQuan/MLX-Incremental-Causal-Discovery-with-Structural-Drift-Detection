import numpy as np
from data.synthetic import generate_drifting_data
from lingam_mlx import DirectLiNGAMEngine

data, meta = generate_drifting_data(
    n_samples=5000, n_vars=8, max_lag=2,
    drift_at=2500, drift_type='both', drift_mode='abrupt',
    random_state=42,
)

window_size = 500
engine = DirectLiNGAMEngine(max_lag=2, use_mlx=True)

# 用窗口 4（安静）和窗口 5（漂移）
for w in [4, 5]:
    W = data[w * window_size:(w + 1) * window_size]
    prev_B = None
    prev_order = None
    if w == 5:
        W0 = data[4 * window_size:5 * window_size]
        engine.fit(W0)
        prev_B = engine.get_coefficients()
        prev_order = engine.get_causal_order()

    engine.fit(W, warm_start_B=prev_B, warm_start_order=prev_order)
    A = engine.get_adjacency()
    B = engine.get_coefficients()

    L = 2
    n_vars = 8
    A_var = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            block = A[i*(L+1):(i+1)*(L+1), j*(L+1):(j+1)*(L+1)]
            if block.max() > 0:
                A_var[i, j] = 1

    print(f"\n=== window {w} ===")
    print("node edges:", int(A.sum()))
    print("var edges:", int(A_var.sum()))
    print("var adjacency:")
    print(A_var.astype(int))
    print("true inst adj (window):", meta['true_inst_adj'][w].astype(int))
    print("true lag adj[0]:", meta['true_lag_adj'][w][0].astype(int))
    print("true lag adj[1]:", meta['true_lag_adj'][w][1].astype(int))