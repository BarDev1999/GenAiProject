import matplotlib.pyplot as plt
import numpy as np
import os

# Create folder if it doesn't exist
os.makedirs("Visuals", exist_ok=True)

def create_performance_chart():
    """Generates the bar chart shown in slide 4."""
    labels = ['Random', 'ResNet18', 'ResNet50']
    p1_means = [2, 67, 77]
    p5_means = [10, 78, 89]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, p1_means, width, label='Precision 1', color='#3b82f6')
    rects2 = ax.bar(x + width/2, p5_means, width, label='Precision 5', color='#10b981')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Comparison (Unseen Data)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.1f%%')
    ax.bar_label(rects2, padding=3, fmt='%.1f%%')

    fig.tight_layout()
    plt.savefig('Visuals/performance_chart.png', dpi=300)
    print("✅ Created: Visuals/performance_chart.png")

def create_visual_abstract():
    """Generates a text-based Visual Abstract diagram."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    # Define boxes and text
    boxes = [
        {"text": "INPUT\nDegraded CCTV\nQuery", "x": 0.1, "color": "#fee2e2"},
        {"text": "PROCESS\nResNet50 Siamese\nTriplet Loss", "x": 0.45, "color": "#dbeafe"},
        {"text": "OUTPUT\nClean Catalog\nIdentification", "x": 0.8, "color": "#dcfce7"}
    ]

    for box in boxes:
        ax.text(box["x"], 0.5, box["text"], ha='center', va='center', 
                bbox=dict(boxstyle='round,pad=1', facecolor=box["color"], edgecolor='#cbd5e1'),
                fontsize=12, fontweight='bold')

    # Draw arrows
    ax.annotate('', xy=(0.34, 0.5), xytext=(0.21, 0.5), arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.69, 0.5), xytext=(0.56, 0.5), arrowprops=dict(arrowstyle='->', lw=2))

    plt.title("Visual Abstract: Vinyl Album Identification Pipeline", fontsize=14, fontweight='bold', pad=20)
    plt.savefig('Visuals/visual_abstract.png', dpi=300, bbox_inches='tight')
    print("✅ Created: Visuals/visual_abstract.png")

if __name__ == "__main__":
    try:
        create_performance_chart()
        create_visual_abstract()
        print("\n🎉 All visuals generated successfully!")
    except Exception as e:
        print(f"❌ Error generating visuals: {e}")
