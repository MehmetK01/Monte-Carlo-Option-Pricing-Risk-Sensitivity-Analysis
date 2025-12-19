# Monte-Carlo-Option-Pricing-Risk-Sensitivity-Analysis
A brief Monte Carlo Option Pricing &amp; Risk Sensitivity Analysis in Python.

This project builds a Monte Carlo pricing engine for European call options based on a risk-neutral Geometric Brownian Motion (GBM) model. The goal is to price options accurately while controlling Monte Carlo noise, and to validate the results against well-known analytical benchmarks used in quantitative finance.

Methodology
Stochastic model: Asset price paths are simulated under risk-neutral GBM dynamics using a log-Euler discretization scheme
Pricing approach: Option prices are computed as discounted averages of terminal payoffs across simulated paths
Variance reduction:
  Antithetic variates are used to reduce variance by pairing negatively correlated paths
  A control variate based on discounted terminal prices is applied, exploiting its known expectation under the risk-neutral measure
Validation: Monte Carlo prices and sensitivities are compared against closed-form Black–Scholes solutions to ensure correctness
Greeks: Delta and Vega are estimated using central finite-difference Monte Carlo methods

Results (at the end of the script):
Monte Carlo price estimates converge to the Black–Scholes benchmark and lie within 95% confidence intervals
Variance reduction techniques lead to an approximate 60% reduction in standard error compared to naive Monte Carlo simulation
Monte Carlo estimates of Delta and Vega closely match analytical Black–Scholes values, indicating stable and reliable numerical behavior
