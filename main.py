# main.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(page_title="Interactive 3D GBM Simulator", layout="wide")
st.title("Interactive 3D Monte Carlo Stock Price Simulation")

# -----------------------------
# Functions
# -----------------------------
def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    dt = T / steps
    t = np.linspace(0, T, steps + 1)
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for i in range(n_paths):
        W = np.random.standard_normal(steps)
        W = np.cumsum(W) * np.sqrt(dt)
        X = (r - 0.5 * sigma ** 2) * t[1:] + sigma * W
        paths[i, 1:] = S0 * np.exp(X)
    return t, paths

def monte_carlo_option_price(paths, K, r, T, option_type='call'):
    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0) if option_type=='call' else np.maximum(K - S_T, 0)
    return np.exp(-r * T) * np.mean(payoff)

def black_scholes_price(S0, K, r, sigma, T, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type=='call':
        return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

# -----------------------------
# Inputs
# -----------------------------
S0 = st.number_input("Initial Stock Price (S0)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
r = st.number_input("Risk-free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
T = st.number_input("Time to Maturity (years)", value=1.0)
steps = st.number_input("Steps per Path", value=252)
n_paths = st.number_input("Number of Paths (reduce for cloud)", value=200)

# -----------------------------
# Simulate GBM
# -----------------------------
t, paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths)

# -----------------------------
# Interactive 3D Plot with Plotly
# -----------------------------
fig = go.Figure()
for i in range(min(50, n_paths)):
    fig.add_trace(go.Scatter3d(
        x=t,
        y=[i]*len(t),
        z=paths[i],
        mode='lines',
        line=dict(width=2),
        name=f'Path {i}'
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='Time',
        yaxis_title='Path Index',
        zaxis_title='Stock Price',
        xaxis=dict(backgroundcolor="rgb(34,34,34)", color="white"),
        yaxis=dict(backgroundcolor="rgb(34,34,34)", color="white"),
        zaxis=dict(backgroundcolor="rgb(34,34,34)", color="white")
    ),
    paper_bgcolor='rgb(34,34,34)',
    plot_bgcolor='rgb(34,34,34)',
    font=dict(color='white'),
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Monte Carlo & Black-Scholes Prices
# -----------------------------
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

st.subheader("Option Price Comparison")
st.dataframe(df_prices)

st.markdown(f"""
**Call Price Difference (MC - BS):** {diff_call:.4f}  
**Put Price Difference (MC - BS):** {diff_put:.4f}
""")
