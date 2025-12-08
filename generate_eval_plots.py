import matplotlib.pyplot as plt

# ============================
# Data
# ============================

dropout_levels = [10, 25, 50]
dropout_seen = [0.158, 0.060, 0.020]
# dropout_unseen = [0.078, 0.084, 0.084, 0.068]
# gaussian_levels = [10, 25, 50, 75]
# gaussian_seen = [0.125, 0.104, 0.094, 0.078]

# ambiguity_levels = ["little", "moderate", "heavy"]
# ambiguity_seen = [0.152, 0.122, 0.108]
# ambiguity_unseen = [0.065, 0.068, 0.071]

OG_seen = 0.14
# OG_unseen = 0.08

# Choose colors manually to reuse for OG lines
color_seen = "tab:blue"
# color_unseen = "tab:orange"

# # ============================
# # Gaussian Noise Plot
# # ============================

# plt.figure(figsize=(6,4))

# plt.plot(gaussian_levels, gaussian_seen, marker='o', label="SR", color=color_seen)
# # plt.plot(gaussian_levels, gaussian_unseen, marker='o', label="Val Unseen", color=color_unseen)

# # OG baselines (same colors, dotted)
# plt.axhline(OG_seen, linestyle='--', color=color_seen, label="Original SR")
# # plt.axhline(OG_unseen, linestyle='--', color=color_unseen, label="Original Val Unseen")

# plt.xlabel("Total Occluded Area(%)")
# plt.ylabel("Metric")
# plt.title("Effect of Gaussian Noise on Success Rate")
# plt.legend()
# plt.tight_layout()
# plt.savefig("gaussian_noise_plot.png", dpi=300)
# plt.close()


# ============================
# Dropout Plot
# ============================

plt.figure(figsize=(6,4))

plt.plot(dropout_levels, dropout_seen, marker='o', label="SR", color=color_seen)
# plt.plot(dropout_levels, dropout_unseen, marker='o', label="Val Unseen", color=color_unseen)

# OG baselines (same colors, dotted)
plt.axhline(OG_seen, linestyle='--', color=color_seen, label="Original SR")
# plt.axhline(OG_unseen, linestyle='--', color=color_unseen, label="Original Val Unseen")

plt.xlabel("Number of Patches \n Total Occluded Area = Number of Patches x Patch Size (400 px² to 3600 px²)")
plt.ylabel("Success Rate")
plt.title("Effect Total Area Occluded on Success Rate" )
plt.legend()
plt.tight_layout()
plt.savefig("patch_plot.png", dpi=300)
plt.close()

# ============================
# Ambiguity Plot
# ============================

# plt.figure(figsize=(6,4))

# plt.plot(ambiguity_levels, ambiguity_seen, marker='o', label="SR", color=color_seen)
# # plt.plot(ambiguity_levels, ambiguity_unseen, marker='o', label="Val Unseen", color=color_unseen)

# # OG baselines (same colors, dotted)
# plt.axhline(OG_seen, linestyle='--', color=color_seen, label="Original SR")
# # plt.axhline(OG_unseen, linestyle='--', color=color_unseen, label="Original Val Unseen")

# plt.xlabel("Ambiguity Level")
# plt.ylabel("Metric")
# plt.title("Effect of Ambiguity on Success Rate")
# plt.legend()
# plt.tight_layout()
# plt.savefig("ambiguity_plot.png", dpi=300)
# plt.close()
