"""
Monte Carlo Option Pricer with Variance Reduction and Greeks.

Features:
- GBM model under risk-neutral measure
- European call payoff
- Black–Scholes price + Delta (for validation)
- Monte Carlo pricing:
    * plain MC
    * antithetic variates (variance reduction)
    * control variate (variance reduction)
- Greeks via finite differences (Delta, Vega)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.stats import norm

# Black–Scholes formulas

def _d1_d2(S0: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    """Helper: compute d1, d2 for Black–Scholes."""
    denom = sigma * np.sqrt(T)
    if denom == 0:
        raise ValueError("sigma * sqrt(T) must be > 0 for BS formula.")
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / denom
    d2 = d1 - denom
    return d1, d2


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form Black–Scholes price of a European call."""
    d1, d2 = _d1_d2(S0, K, r, sigma, T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_delta_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Closed-form Delta of a European call."""
    d1, _ = _d1_d2(S0, K, r, sigma, T)
    return norm.cdf(d1)

# stochastic model: GBM

@dataclass
class GBMModel:
    S0: float     # initial price
    r: float      # risk-free rate
    sigma: float  # volatility
    T: float      # maturity in years

    def simulate_paths(self,n_paths: int,n_steps: int,antithetic: bool = False,seed: int | None = None) -> np.ndarray:
        """
        Simulate GBM paths under risk-neutral measure

        antithetic=True turns on ANTITHETIC VARIATES:
        - generate Z ~ N(0,1) and -Z, so paths come in negatively
          correlated pairs to reduce variance
        """
        if seed is not None:
            np.random.seed(seed)

        dt = self.T / n_steps
        nudt = (self.r - 0.5 * self.sigma**2) * dt
        sigsdt = self.sigma * np.sqrt(dt)

        # variance reduction #1: Antithetic variates
        if antithetic:
            half = n_paths // 2
            Z = np.random.normal(size=(half, n_steps))
            Z = np.vstack([Z, -Z])  
            if Z.shape[0] != n_paths:  
                extra = np.random.normal(size=(1, n_steps))
                Z = np.vstack([Z, extra])
        else:
            Z = np.random.normal(size=(n_paths, n_steps))

        # path simulation
        S = np.full((n_paths, n_steps + 1), self.S0, dtype=float)
        for t in range(1, n_steps + 1):
            S[:, t] = S[:, t - 1] * np.exp(nudt + sigsdt * Z[:, t - 1])

        return S

# Payoff

class EuropeanCall:
    def __init__(self, K: float):
        self.K = K

    def payoff_terminal(self, ST: np.ndarray) -> np.ndarray:
        """Payoff as function of terminal price ST."""
        return np.maximum(ST - self.K, 0.0)


# Monte Carlo pricer (+ VARIANCE REDUCTION)

def mc_price(model: GBMModel,payoff: EuropeanCall,n_paths: int = 100_000,n_steps: int = 100,antithetic: bool = True,control_variate: bool = False,seed: int | None = None) -> Tuple[float, float]:
    """
    Monte Carlo pricing of a European option under GBM

    VARIANCE REDUCTION:
    - antithetic=True  -> antithetic variates in path simulation
    - control_variate=True -> uses discounted S_T as CONTROL VARIATE:

        X  = discounted payoff
        Y  = discounted S_T
        E[Y] = S0 under risk-neutral measure
        X_cv = X + c*(Y - E[Y])   with optimal c to reduce variance

    Returns:
        (price_estimate, standard_error)
    """
    paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps,
                                 antithetic=antithetic, seed=seed)
    ST = paths[:, -1]
    payoffs = payoff.payoff_terminal(ST)
    disc = np.exp(-model.r * model.T)

    # base estimator: X = discounted payoff
    X = disc * payoffs

    if control_variate:
        Y = disc * ST
        EY = model.S0

        cov_xy = np.cov(X, Y, ddof=1)[0, 1]
        var_y = np.var(Y, ddof=1)
        if var_y == 0:
            X_cv = X
        else:
            c_star = -cov_xy / var_y
            X_cv = X + c_star * (Y - EY)

        price_estimate = np.mean(X_cv)
        stderr = np.std(X_cv, ddof=1) / np.sqrt(len(X_cv))
    else:
        price_estimate = np.mean(X)
        stderr = np.std(X, ddof=1) / np.sqrt(len(X))

    return price_estimate, stderr

# Greeks via finite differences

def delta_fd(model: GBMModel,payoff: EuropeanCall,h: float = 1e-2,**mc_kwargs) -> float:
    """
    Delta via central finite difference w.r.t. S0:
        Delta ≈ (C(S0+h) - C(S0-h)) / (2h)
    """
    base_S0 = model.S0

    model.S0 = base_S0 + h
    up, _ = mc_price(model, payoff, **mc_kwargs)

    model.S0 = base_S0 - h
    down, _ = mc_price(model, payoff, **mc_kwargs)

    model.S0 = base_S0 
    return (up - down) / (2 * h)


def vega_fd(model: GBMModel,payoff: EuropeanCall,h: float = 1e-3,**mc_kwargs) -> float:
    """
    Vega via central finite difference w.r.t. sigma:
        Vega ≈ (C(σ+h) - C(σ-h)) / (2h)
    """
    base_sigma = model.sigma

    model.sigma = base_sigma + h
    up, _ = mc_price(model, payoff, **mc_kwargs)

    model.sigma = base_sigma - h
    down, _ = mc_price(model, payoff, **mc_kwargs)

    model.sigma = base_sigma 
    return (up - down) / (2 * h)



if __name__ == "__main__":
    # parameters
    S0 = 100.0
    K = 100.0
    r = 0.03
    sigma = 0.2
    T = 1.0

    model = GBMModel(S0=S0, r=r, sigma=sigma, T=T)
    payoff = EuropeanCall(K=K)

    # 1. try : Plain MC (no variance reduction)
    price_plain, se_plain = mc_price(
        model, payoff,
        n_paths=50_000,
        n_steps=100,
        antithetic=False,
        control_variate=False,
        seed=100)
    ci_plain = (price_plain - 1.96 * se_plain, price_plain + 1.96 * se_plain)

    # 2nd try: MC with variance reduction: antithetic + control variate 
    price_vr, se_vr = mc_price(
        model, payoff,
        n_paths=50_000,
        n_steps=100,
        antithetic=True,
        control_variate=True,
        seed=100)
    ci_vr = (price_vr - 1.96 * se_vr, price_vr + 1.96 * se_vr)

    # Black–Scholes Benchmark
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    bs_delta = black_scholes_delta_call(S0, K, r, sigma, T)

    # Greeks
    delta_est = delta_fd(
        model, payoff,
        n_paths=30_000,
        n_steps=100,
        antithetic=True,
        control_variate=True,
        seed=100)
    
    vega_est = vega_fd(
        model, payoff,
        n_paths=30_000,
        n_steps=100,
        antithetic=True,
        control_variate=True,
        seed=100)

    
    print(" European Call Option (GBM, Monte Carlo) ")
    print(f"Parameters: S0={S0}, K={K}, r={r}, sigma={sigma}, T={T}")
    print()
    print("Black–Scholes ")
    print(f"BS price   : {bs_price:.4f}")
    print(f"BS Delta   : {bs_delta:.4f}")
    print()
    print("Plain Monte Carlo ")
    print(f"MC price   : {price_plain:.4f}")
    print(f"Std. error : {se_plain:.6f}")
    print(f"95% CI     : [{ci_plain[0]:.4f}, {ci_plain[1]:.4f}]")
    print()
    print(" MC with VARIANCE REDUCTION (antithetic + control variate) ")
    print(f"MC price   : {price_vr:.4f}")
    print(f"Std. error : {se_vr:.6f}")
    print(f"95% CI     : [{ci_vr[0]:.4f}, {ci_vr[1]:.4f}]")
    print()
    print("Greeks (MC finite difference) ")
    print(f"Delta (MC) : {delta_est:.4f}   vs   Delta (BS): {bs_delta:.4f}")
    print(f"Vega  (MC) : {vega_est:.4f}")

# ###### RESULTS:
# European Call Option (GBM, Monte Carlo) 
# Parameters: S0=100.0, K=100.0, r=0.03, sigma=0.2, T=1.0

# Black–Scholes 
# BS price   : 9.4134
# BS Delta   : 0.5987

# Plain Monte Carlo 
# MC price   : 9.3275
# Std. error : 0.062515
# 95% CI     : [9.2050, 9.4500]

#  MC with VARIANCE REDUCTION (antithetic + control variate) 
# MC price   : 9.3658
# Std. error : 0.025826
# 95% CI     : [9.3152, 9.4164]

# Greeks (MC finite difference) 
# Delta (MC) : 0.5992   vs   Delta (BS): 0.5987
# Vega  (MC) : 38.4596