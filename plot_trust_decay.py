"""
plot_trust_decay.py — Visualizes the Trust Decay Dynamics for QSRAC.
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_trust(t, sensitivity, risk_trend, alpha=0.7, beta=0.3, trust0=1.0):
    """
    Computes trust based on the formula:
    trust = trust0 * exp(-(alpha * sensitivity + beta * risk_trend) * time)
    """
    decay_rate = (alpha * sensitivity) + (beta * risk_trend)
    return trust0 * np.exp(-decay_rate * t)

def main():
    # 1. Simulate time from 0 to 10 seconds
    t = np.linspace(0, 10, 500)

    # 2. Compute curves based on defined scenarios
    # Low risk: sensitivity=1, risk_trend=0.0
    trust_low = compute_trust(t, sensitivity=1, risk_trend=0.0)
    
    # Medium risk: sensitivity=3, risk_trend=0.3
    trust_medium = compute_trust(t, sensitivity=3, risk_trend=0.3)
    
    # High risk: sensitivity=5, risk_trend=1.0
    trust_high = compute_trust(t, sensitivity=5, risk_trend=1.0)

    # 3. Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(t, trust_low, label='Low Risk (Sens=1, Trend=0.0)', color='#2ca02c', linewidth=2.5)
    plt.plot(t, trust_medium, label='Medium Risk (Sens=3, Trend=0.3)', color='#ff7f0e', linewidth=2.5)
    plt.plot(t, trust_high, label='High Risk (Sens=5, Trend=1.0)', color='#d62728', linewidth=2.5)

    # 4. Formatting and Labels
    plt.title('QSRAC Trust Decay Dynamics', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Trust Score', fontsize=12)
    
    # Lock axes for better readability
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    
    # Grid and Legend
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11)

    # 5. Save and Show
    output_plot = "trust_decay.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_plot}")
    
    plt.show()

if __name__ == "__main__":
    main()