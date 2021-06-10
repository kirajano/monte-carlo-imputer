<img src="https://user-images.githubusercontent.com/76450761/121443840-8e9b1800-c98e-11eb-9098-270840ade7fa.png"  width="150" height="100"><img src="https://user-images.githubusercontent.com/76450761/121443973-d6ba3a80-c98e-11eb-8ebd-94336b55d7a6.png" width="150" height="100"><img src="https://user-images.githubusercontent.com/76450761/121444139-344e8700-c98f-11eb-9823-ac91c7c866e5.png" width="150" height="100">



# Monte Carlo Imputation
The repo contains functions that utilize most imputation methods on any 1D dataset and help to choose the best through high number of iterations like a Monte Carlo Simulation to help to understand an unknown distribution (hence the best imputation method).

## Description

Imputation and its method selection is one of the biggest problems when doing data preprocessing to handle missing values.
The one thing that makes it challenging that there are different strategies to it.
But as with many things when working in machine learning with tabular data: no clear or solid guideline what method to use when. 
This comes mainly from the fact you can apply guessing how the missing data looks like but in reality it could be completely different.

The Monte Carlo Simulation offers a good view on how to approach data problems where little information is known. It follows the underlying concept of "randomness" and inferring propabilistic distribution. In a nutshell, it helps to gain understanding of data and its distribution through high iterative simulation process.
https://en.wikipedia.org/wiki/Monte_Carlo_method

For imputation, it could be very well applied to help select the best imputation method through high number of iterations. The functions from the repo take a subset of present data by setting it to NaN, apply different / all imputation strategies and measure the error between imputed and actual values. The method with least error after many iteration run starts to stand out as the most applicable.

For this purpose, the function in repo has all common imputation methods under the hood and measures their performance by applying them iteratively.
Imputation methods included:
* Simple Imputation --> *mean, median, mode*
* K-Nearest-Neighbor Imputation --> *uniform, distance*
* Different interpolation methods (both directions) --> *linear, index, zero, cubic, polynomial etc.*
* Timeseries specific Imputations --> *last-observation-carried-forward and Moving-Window-Imputer* (kudos to **impyute**)

## Usage

For testing purposes, Imputer.py has already the three main methods included. The main three functions inlcude a single run of all imputers, high-iterative run (to find the best imputation method) and an intgrated application of imputation to the data. The demo dataset is time series data of daily cost-per-click values where some small subset of missing data is also present.

For own usage, just clone the repo in your project and use the following import to start using the three main methods.

```
from impute_optimize import impute, impute_apply, impute_optimizer
```
Most likely, the output of impute_optmizer() is the most interesting to get an understanding how different imputiting strategies (methods and params) perform.
A sample output from running impute_optmizer() would look like this:

<img src="https://user-images.githubusercontent.com/76450761/121442591-3bc06100-c98c-11eb-858c-0c0b08b56406.png" width="650" height="350">

The most important part is the count of how many times did the imputation and its parameter were the best method though all iteration runs. This would make the choice easy what to pick when you are preprocessing and trying the fill the gaps since it can be read as a score board. For convinience, impute_apply() is an extension of the former and automatically takes the best method and its paramers to return an imputed dataset with no NaNs.
