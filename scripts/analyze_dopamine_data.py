import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from typing import Dict, Tuple


def load_dopamine_data() -> Tuple[Dict, Dict]:
    """Load dopamine data from the processed files."""
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        script_path, "..", "data", "dopamine_level", "data_by_mouse_id.npy"
    )
    _, DAs, z_DAs, _ = np.load(data_path, allow_pickle=True)
    return DAs, z_DAs


def analyze_and_plot_dopamine_data():
    """Analyze and plot dopamine data distributions."""
    DAs, z_DAs = load_dopamine_data()

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Dopamine Data Analysis", fontsize=16)

    # Plot raw dopamine levels
    all_da_values = []
    for mouse_id, da_data in DAs.items():
        all_da_values.extend(da_data.flatten())
    all_da_values = np.array(all_da_values)

    sns.histplot(data=all_da_values, ax=ax1, bins=50)
    ax1.set_title("Distribution of Raw Dopamine Levels")
    ax1.set_xlabel("Dopamine Level")
    ax1.set_ylabel("Count")

    # Plot z-scored dopamine levels
    all_z_da_values = []
    for mouse_id, z_da_data in z_DAs.items():
        all_z_da_values.extend(z_da_data.flatten())
    all_z_da_values = np.array(all_z_da_values)

    sns.histplot(data=all_z_da_values, ax=ax2, bins=50)
    ax2.set_title("Distribution of Z-scored Dopamine Levels")
    ax2.set_xlabel("Z-scored Dopamine Level")
    ax2.set_ylabel("Count")

    # Box plots for raw dopamine levels by mouse
    da_data_by_mouse = [DAs[mouse_id].flatten() for mouse_id in sorted(DAs.keys())]
    ax3.boxplot(da_data_by_mouse, labels=[f"Mouse {i}" for i in sorted(DAs.keys())])
    ax3.set_title("Raw Dopamine Levels by Mouse")
    ax3.set_xlabel("Mouse ID")
    ax3.set_ylabel("Dopamine Level")

    # Box plots for z-scored dopamine levels by mouse
    z_da_data_by_mouse = [
        z_DAs[mouse_id].flatten() for mouse_id in sorted(z_DAs.keys())
    ]
    ax4.boxplot(z_da_data_by_mouse, labels=[f"Mouse {i}" for i in sorted(z_DAs.keys())])
    ax4.set_title("Z-scored Dopamine Levels by Mouse")
    ax4.set_xlabel("Mouse ID")
    ax4.set_ylabel("Z-scored Dopamine Level")

    plt.tight_layout()

    # Save the plot
    script_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        script_path, "..", "data", "dopamine_level", "dopamine_analysis.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print statistics
    print("\nDopamine Data Statistics:")
    print("=" * 50)
    print("\nRaw Dopamine Levels:")
    print(f"Min: {np.min(all_da_values):.3f}")
    print(f"Max: {np.max(all_da_values):.3f}")
    print(f"Mean: {np.mean(all_da_values):.3f}")
    print(f"Std: {np.std(all_da_values):.3f}")

    print("\nZ-scored Dopamine Levels:")
    print(f"Min: {np.min(all_z_da_values):.3f}")
    print(f"Max: {np.max(all_z_da_values):.3f}")
    print(f"Mean: {np.mean(all_z_da_values):.3f}")
    print(f"Std: {np.std(all_z_da_values):.3f}")

    print(f"\nPlot saved to: {save_path}")


if __name__ == "__main__":
    analyze_and_plot_dopamine_data()
