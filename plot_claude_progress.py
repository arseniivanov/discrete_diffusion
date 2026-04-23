import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Auditor Data Extraction: (Label, Loss, Status, Param_Add)
data = [
    ("Baseline", 1.0268, "success", False),
    ("LR Warmup", 1.1594, "fail", False),
    ("Timestep Embed", 1.0105, "success", True),
    ("Wider adaLN", 1.0098, "success", True),
    ("Linear Noise", 1.0697, "fail", False),
    ("Grad Clip", 1.0103, "fail", False),
    ("LR 2e-3", 0.9931, "success", False),
    ("LR 3e-3", 1.0004, "fail", False),
    ("Linear Bias", 0.9872, "success", True),
    ("384 Context", 0.9789, "success", True),
    ("Dropout 0.05", 0.9865, "fail", False),
    ("Muon All", 0.9711, "success", False),
    ("Muon Excl", 0.9709, "success", False),
    ("LR 3e-3/384", 0.9671, "success", False),
    ("LR 4e-3", 0.9576, "success", False),
    ("SwiGLU Param", 0.9724, "fail", False),
    ("RoPE", 0.9152, "success", False),
    ("4 Registers", 0.9130, "success", True),
    ("8 Registers", 0.9098, "success", True),
    ("Input Conv", 0.9050, "success", True),
    ("Block Conv", 0.8967, "success", True),
    ("Simple Conv", 0.8919, "success", False),
    ("Stacked Input", 0.8886, "success", True),
    ("Stacked Block", 0.8832, "success", True),
    ("ALiBi Bias", 0.8818, "success", True),
    ("ALiBi Einsum", 0.8809, "success", False),
    ("Equal Weight", 0.9970, "fail", False),
    ("RoPE Theta", 0.8837, "fail", False),
    ("QK-Norm", 0.8770, "success", True),
    ("Output Conv", 0.8800, "fail", False),
    ("Zero-init Conv", 0.8844, "fail", False),
    ("Antithetic Time", 0.8769, "success", False),
    ("Structural Test", 0.8850, "fail", False),
    ("V-Norm", 0.8777, "fail", False),
    ("LR 5e-3", 0.8790, "fail", False),
    ("Swap Order", 0.8833, "fail", False),
    ("Pre-Conv LN", 0.8913, "fail", False),
    ("SwiGLU Equal", 0.8792, "fail", False),
    ("16 Registers", 0.8775, "fail", False),
    ("Stratified Time", 0.8787, "fail", False),
    ("Register ALiBi", 0.8796, "fail", False),
    ("k=5 Conv", 0.8819, "fail", False),
    ("Sigma x1000", 0.8772, "fail", False),
    ("Sigma Bias In", 0.8770, "fail", False),
    ("Combined Sigma", 0.8763, "success", True),
    ("MLP Sigma In", 0.8779, "fail", False),
    ("Logit Bias", 0.8758, "success", True),
    ("Sigma x500", 0.8715, "success", False),
    ("Sigma x200", 0.8721, "fail", False),
    ("Sigma x300", 0.8728, "fail", False),
    ("Output AdaLN", 0.8774, "fail", False),
    ("No Outer SiLU", 0.8711, "success", False),
    ("No ln_f", 0.8712, "fail", False),
    ("Input Register Conv", 0.8739, "fail", False),
    ("SiLU MLP", 0.8735, "fail", False),
    ("WD=0 Conditioning", 0.8782, "fail", False),
    ("3-Layer Conv", 1.0, "fail", False),
    ("Attn Log-Scale", 1.0, "fail", False),
    ("RoPE Theta 500", 0.8710, "success", False),
    ("Register Randn", 0.8727, "fail", False),
    ("RoPE Theta 200", 0.8745, "fail", False),
    ("Muon LR 2x", 0.8678, "success", False),
    ("Muon LR 3x", 0.8700, "fail", False),
    ("Muon LR 2.5x", 0.8686, "fail", False),
    ("Muon LR 1.5x", 0.8711, "fail", False),
    ("AdamW 0.75x", 0.8689, "fail", False),
    ("AdamW 1.25x", 0.8678, "success", False),
    ("AdamW 1.5x", 0.8670, "success", False),
    ("AdamW 2x", 0.8684, "fail", False),
    ("AdamW 1.75x", 0.8674, "fail", False),
    ("Cosine Decay", 0.8531, "success", False),
    ("Peak LR 3x/2x", 0.8481, "success", False),
    ("Peak LR 4x/3x", 0.8434, "success", False),
]

labels = [d[0] for d in data]
losses = [d[1] for d in data]
colors = ["grey" if d[2] == "fail" else ("red" if d[3] else "green") for d in data]

plt.figure(figsize=(24, 10))
plt.scatter(range(len(labels)), losses, c=colors, s=150, zorder=3, edgecolors='black', linewidths=0.5)

success_idx = [i for i, d in enumerate(data) if d[2] == "success"]
success_losses = [losses[i] for i in success_idx]
plt.plot(success_idx, success_losses, color='blue', linestyle='--', alpha=0.3, label='Progress Path')

for i, (txt, loss) in enumerate(zip(labels, losses)):
    if loss < 1.1: 
        plt.annotate(txt, (i, loss), textcoords="offset points", xytext=(0,10), ha='center', fontsize=7, rotation=45)

plt.title("Discrete Diffusion Optimization: Progress Trend (Non-Inverted)", fontsize=16)
plt.ylabel("Validation Loss")
plt.xticks(range(len(labels)), labels, rotation=90, fontsize=8)
plt.grid(True, linestyle=':', alpha=0.6)
plt.ylim(0.8, 1.2)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Success (Neutral)', markerfacecolor='green', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Success (Added Params)', markerfacecolor='red', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Failed/Regression', markerfacecolor='grey', markersize=10)
]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig("optimization_plot_standard_fixed.png")
