import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# =========================
# Utilities
# =========================

# Vectorized Geometric Brownian Motion simulation
def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    dW = np.random.standard_normal((n_paths, steps)) * np.sqrt(dt)
    W = np.cumsum(dW, axis=1)
    X = (r - 0.5 * sigma ** 2) * t[1:] + sigma * W
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(X)
    return t, paths

# Monte Carlo European Option Pricing
def monte_carlo_option_price(paths, K, r, T, option_type='call'):
    S_T = paths[:, -1]
    if option_type == 'call':
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price

# Black-Scholes Analytical Price
def black_scholes_price(S0, K, r, sigma, T, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return price

# =========================
# Plotting Dark Theme
# =========================
def set_dark_theme():
    plt.style.use('dark_background')
    plt.rcParams['axes.labelcolor'] = 'white'
    plt.rcParams['xtick.color'] = 'white'
    plt.rcParams['ytick.color'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = '#222222'
    plt.rcParams['axes.facecolor'] = '#222222'
    plt.rcParams['savefig.facecolor'] = '#222222'

# =========================
# Parameters
# =========================
S0 = 100      # Initial stock price
K = 100       # Strike price
r = 0.05      # Risk-free rate
sigma = 0.2   # Volatility
T = 1.0       # Time to maturity (years)
steps = 252   # Steps per path (daily)
n_paths = 1000

set_dark_theme()

# =========================
# Simulation & Pricing
# =========================
t, paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)

mc_call = monte_carlo_option_price(paths, K, r, T, option_type='call')
mc_put = monte_carlo_option_price(paths, K, r, T, option_type='put')

bs_call = black_scholes_price(S0, K, r, sigma, T, option_type='call')
bs_put = black_scholes_price(S0, K, r, sigma, T, option_type='put')

diff_call = mc_call - bs_call
diff_put = mc_put - bs_put

df_prices = pd.DataFrame({
    'Method': ['Monte Carlo', 'Black-Scholes'],
    'Call Price': [mc_call, bs_call],
    'Put Price': [mc_put, bs_put]
})

# =========================
# Animation
# =========================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Monte Carlo Stock Price Simulation', fontsize=16, color='white')
ax.set_xlabel('Time (years)')
ax.set_ylabel('Stock Price')

# Display parameters
params_text = f"Strike: {K}\nVolatility: {sigma}\nRisk-free rate: {r}\nT: {T}"
param_box = ax.text(0.02, 0.95, params_text, transform=ax.transAxes, fontsize=12,
                    color='white', va='top', bbox=dict(facecolor='#333333', alpha=0.7))

# Option price display
price_box = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=12,
                    color='white', va='top', ha='right', bbox=dict(facecolor='#333333', alpha=0.7))

# Animate 20 paths for clarity
lines = [ax.plot([], [], lw=1, alpha=0.5)[0] for _ in range(20)]
ax.set_xlim(0, T)
ax.set_ylim(np.min(paths), np.max(paths))

def animate(i):
    for j, line in enumerate(lines):
        line.set_data(t[:i], paths[j, :i])
    price_box.set_text(f"MC Call: {mc_call:.4f}\nBS Call: {bs_call:.4f}\nDiff: {diff_call:.4f}"
                       f"\n\nMC Put: {mc_put:.4f}\nBS Put: {bs_put:.4f}\nDiff: {diff_put:.4f}")
    return lines + [price_box]

ani = animation.FuncAnimation(fig, animate, frames=steps+1, interval=20, blit=True)

plt.tight_layout()
plt.show()

# =========================
# Print Comparison
# =========================
print("\nOption Price Comparison:")
print(df_prices.to_string(index=False))
print(f"\nCall Price Difference (MC - BS): {diff_call:.4f}")
print(f"Put Price Difference (MC - BS): {diff_put:.4f}")
