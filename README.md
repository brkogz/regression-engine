# regression-engine

A production-grade, from-scratch Python statistical library implementing regression and correlation analysis using pure NumPy and SciPy, grounded in Prof. Dr. Neyran Orhunbilge's *Applied Regression and Correlation Analysis*.

## Architecture

- `src/regression/summary_stats.py`: Centered deviations, sums of squares ($S_{xx}, S_{yy}$), and cross-products ($S_{xy}$).
- `src/regression/simple_ols.py`: Ordinary Least Squares (OLS) solver, Normal Equations, and Standard Error of Estimate ($S_{yx}$).
- `src/regression/correlation.py`: Variance decomposition (SST, SSR, SSE), Coefficient of Determination ($r^2$), and Pearson Correlation ($r_{yx}$).