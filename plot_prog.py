import matplotlib.pyplot as plt

# Data format: (Experiment ID, Validation Loss, Status, Label)
data = [
    (0, 1.0268, 'base', 'Baseline'),
    (1, 1.1594, 'fail', ''),
    (2, 1.0105, 'success', 'Timestep Embed'),
    (3, 1.0098, 'success', 'cond_dim 128'),
    (4, 1.0697, 'fail', ''),
    (5, 1.0103, 'fail', ''),
    (6, 0.9931, 'success', 'lr 2e-3'),
    (7, 1.0004, 'fail', ''),
    (8, 0.9872, 'success', 'bias=True'),
    (10, 0.9789, 'success', 'ctx 384'),
    (11, 0.9865, 'fail', ''),
    (12, 0.9711, 'success', 'Muon 2D'),
    (13, 0.9709, 'success', 'Muon excl wte'),
    (14, 0.9671, 'success', 'lr 3e-3'),
    (15, 0.9576, 'success', 'lr 4e-3'),
    (16, 0.9724, 'fail', ''),
    (17, 0.9152, 'success', 'RoPE'),
    (18, 0.9130, 'success', '4 registers'),
    (19, 0.9098, 'success', '8 registers'),
    (20, 0.9050, 'success', 'depthwise tok'),
    (21, 0.8967, 'success', 'per-block conv'),
    (22, 0.8919, 'success', 'conv full seq'),
    (23, 0.8886, 'success', 'stacked input'),
    (24, 0.8832, 'success', 'stacked block'),
    (25, 0.8818, 'success', 'ALiBi bias'),
    (26, 0.8809, 'success', 'ALiBi einsum'),
    (27, 0.9970, 'fail', ''),
    (28, 0.8837, 'fail', ''),
    (29, 0.8770, 'success', 'QK-Norm'),
    (30, 0.8800, 'fail', ''),
    (31, 0.8844, 'fail', ''),
    (32, 0.8769, 'success', 'Antithetic time'),
    (33, 0.8850, 'fail', ''),
    (34, 0.8777, 'fail', ''),
    (35, 0.8790, 'fail', ''),
    (36, 0.8833, 'fail', ''),
    (37, 0.8913, 'fail', ''),
    (38, 0.8792, 'fail', ''),
    (39, 0.8775, 'fail', ''),
    (40, 0.8787, 'fail', ''),
    (41, 0.8796, 'fail', ''),
    (42, 0.8819, 'fail', ''),
    (43, 0.8772, 'fail', ''),
    (44, 0.8770, 'fail', ''),
    (45, 0.8763, 'success', 'sigmax1000'),
    (46, 0.8779, 'fail', ''),
    (48, 0.8758, 'success', 'sigma_out bias'),
    (49, 0.8715, 'success', 'sigmax500'),
    (50, 0.8721, 'fail', ''),
    (51, 0.8728, 'fail', ''),
    (52, 0.8774, 'fail', ''),
    (53, 0.8735, 'fail', ''),
    (54, 0.8711, 'success', 'No outer SiLU'),
    (55, 0.8739, 'fail', ''),
    (56, 0.8712, 'fail', ''),
    (59, 0.8782, 'fail', ''),
    (62, 0.8710, 'success', 'RoPE theta=500'),
    (63, 0.8727, 'fail', ''),
    (64, 0.8745, 'fail', ''),
    (65, 0.8678, 'success', 'Muon LR 2x'),
    (66, 0.8700, 'fail', ''),
    (67, 0.8686, 'fail', ''),
    (68, 0.8711, 'fail', ''),
    (69, 0.8689, 'fail', ''),
    (70, 0.8678, 'fail', ''),
    (71, 0.8670, 'success', 'AdamW LR 1.5x'),
    (72, 0.8684, 'fail', ''),
    (73, 0.8674, 'fail', ''),
    (74, 0.8531, 'success', 'Cosine 10%'),
    (75, 0.8481, 'success', 'Peak 3x/2x'),
    (76, 0.8434, 'success', 'Peak 4x/3x'),
    (77, 0.8405, 'success', 'Peak 5x/4x'),
    (78, 0.8408, 'fail', ''),
    (79, 1.0950, 'fail', ''),
    (80, 0.8334, 'success', 'EMA eval'),
    (81, 0.8369, 'fail', ''),
    (82, 0.8372, 'fail', ''),
    (83, 0.8203, 'success', 'EMA 0.998'),
    (84, 0.8210, 'fail', ''),
    (85, 0.8236, 'fail', ''),
    (86, 0.8223, 'fail', ''),
    (87, 0.8309, 'fail', ''),
    (88, 0.8304, 'fail', ''),
    (90, 0.8208, 'fail', ''),
    (91, 0.8236, 'fail', ''),
    (92, 0.8188, 'success', '256 steps'),
    (93, 0.8192, 'success', 'EMA gen'),
    (94, 0.8147, 'success', 'batch 384'),
    (95, 0.8119, 'success', 'q x 1.2'),
    (96, 0.8117, 'success', 'q x 1.4'),
    (97, 0.8107, 'success', 'batch 256'),
    (98, 0.8040, 'success', 'batch 192'),
    (99, 0.7994, 'success', 'batch 128'),
    (100, 0.7980, 'success', 'batch 64'),
    (101, 0.7978, 'success', 'batch 96'),
    (102, 0.7960, 'success', 'q x 1.0'),
    (103, 0.7937, 'success', 'SiLU conv2'),
    (104, 0.7920, 'success', 'WarmRestarts'),
]

x_success = [d[0] for d in data if d[2] == 'success']
y_success = [d[1] for d in data if d[2] == 'success']
labels_success = [d[3] for d in data if d[2] == 'success']

x_fail = [d[0] for d in data if d[2] == 'fail']
y_fail = [d[1] for d in data if d[2] == 'fail']

plt.figure(figsize=(18, 9))

# Unsuccessful
plt.scatter(x_fail, y_fail, color='gray', alpha=0.6, label='Unsuccessful / Uncommitted')

# Successful
plt.scatter(x_success, y_success, color='green', label='Committed', zorder=5)

# Labels
for i, txt in enumerate(labels_success):
    plt.annotate(
        txt, 
        (x_success[i], y_success[i]), 
        textcoords="offset points", 
        xytext=(0, 10), 
        ha='center', 
        fontsize=8, 
        rotation=45
    )

# Formatting
plt.title('Discrete Diffusion Optimization: Validation Loss Trajectory', fontsize=14, pad=20)
plt.xlabel('Experiment Number', fontsize=12)
plt.ylabel('Validation Loss', fontsize=12)

# Cap the Y-axis to prevent severe outliers (e.g., Exp 1, 79) from destroying scale visibility
plt.ylim(0.75, 1.2) 
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()
