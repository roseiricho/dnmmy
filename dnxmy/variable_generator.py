# import libraries
import numpy as np
import pandas as pd

# list of supported probability distributions
DISTRIBUTION_LIST = ['uniform', 'normal', 'beta', 'gamma', 'exponential', 'poisson', 'binomial', 'lognormal', 'chisquare', 'f', 't', 'multivariate_normal', 'multinomial', 'dirichlet', 'laplace', 'logistic', 'logseries', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'pareto', 'rayleigh', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'vonmises', 'wald', 'weibull', 'zipf', 'gunbel', 'hypergeometric']


def generate_random_samples(col_name: str, probability_distribution: dict, n: int) -> pd.Series:
  """
  Generate random samples based on the provided probability distribution.

  Args:
      col_name (str): Name of the column.
      probability_distribution (dict): Dictionary containing the probability distribution.
      n (int): Number of samples to be generated.

  Returns:
      pd.Series: Series containing the generated samples.
  """

  distribution_type = probability_distribution.get('type')
  params = probability_distribution.get('parameter')
  if distribution_type is None or params is None:
    raise ValueError("distribution_type and params must be specified.")

  if distribution_type == 'categorical':
    categories, probabilities = zip(*[(c['value'], c['probability']) for c in params['categories']])
    probabilities = np.array(probabilities) / sum(probabilities)
    return pd.Series(np.random.choice(categories, n, p=probabilities), name=col_name)

  if distribution_type not in DISTRIBUTION_LIST:
    raise ValueError(f"distribution_type must be in {DISTRIBUTION_LIST}")

  kwargs = {k: params[k] for k in params if k != 'loc'}
  if 'loc' in params:
    kwargs['loc'] = params['loc']

  if distribution_type == 'multivariate_normal':
    kwargs['cov'] = np.array(kwargs['cov'])

  if distribution_type == 'categorical':
    kwargs['p'] = probabilities

  # generate random samples
  return pd.Series(getattr(np.random, distribution_type)(**kwargs, size=n), name=col_name)


def generate_time_part(col_name: str, time_part_config: dict, n: int) -> pd.Series:
  """
  Generate a time series based on the provided time configurations.

  Args:
      col_name (str): Name of the column.
      time_part_config (dict): Dictionary containing the time configurations.
      n (int): Number of samples to generate.

  Returns:
      pd.Series: Series containing the generated time series.
  """
  if not time_part_config:
    raise ValueError("time_part_config must be specified.")
  
  start_time, time_unit, time_format = time_part_config.get('start_time'), time_part_config.get('time_unit'), time_part_config.get('time_format')
  if not all([start_time, time_unit, time_format]):
    raise ValueError("start_time, time_unit, and time_format must be specified in time_part_config.")
  
  # generate time series
  time_series = pd.Series(pd.date_range(start_time, periods=n, freq=time_unit).strftime(time_format), name = col_name)

  return time_series


def generate_arma_samples(col_name: str, time_series_config: dict, n: int) -> pd.Series:
  """
  Generate a time series(ARMA model) based on the provided time configurations.

  Args:
      col_name (str): Name of the column.
      time_series_config (dict): Dictionary containing the time configurations.
      n (int): Number of samples to generate.

  Returns:
      pd.Series: Series containing the generated samples.
  """
  if time_series_config is None:
    raise ValueError("time_series_config must be specified.")
  
  ar_params = time_series_config.get('ar_params', {})
  ma_params = time_series_config.get('ma_params', {})
  
  ar_samples = [ar_params.get('initial', 0)]
  ma_samples = [ma_params.get('initial', 0)]
  
  # generate initial samples
  arma_samples = [ar_samples[0] + ma_samples[0] + time_series_config['intercept'] + np.random.normal(0, time_series_config['sigma'])]
  
  # generate ARMA samples
  for i in range(1, n):
    ar = sum([ar_params['params'][j] * ar_samples[i - j - 1] for j in range(min(i, ar_params.get('order', 0)))])
    
    # add shock
    if ar_params.get('shock').get(i) is not None:
      shock_value = ar_params['shock'][i]['value']
      if ar_params['shock'][i]['type'] == 'sigma':
        shock_value *= time_series_config['sigma']
      ar += shock_value
    ar_samples.append(ar)

    ma = sum([ma_params['params'][j] * ma_samples[i - j - 1] for j in range(min(i, ma_params.get('order', 0)))])

    # add shock
    if ma_params.get('shock').get(i) is not None:
      shock_value = ma_params['shock'][i]['value']
      if ma_params['shock'][i]['type'] == 'sigma':
        shock_value *= time_series_config['sigma']
      ma += shock_value
    ma_samples.append(ma)

    arma_samples.append(ar + ma + np.random.normal(0, time_series_config['sigma']))
  
  return pd.Series(arma_samples, name=col_name)


def generate_dependent_samples(col_name: str, dataset_config: list, df, dependent_on: dict, n: int, offset: dict = None) -> pd.Series:
  """
  Generate dependent samples based on the provided dependent_on configurations.

  Args:
      col_name (str): Name of the column.
      dataset_config (list): List containing the dataset configurations.
      df (pd.DataFrame): Dataframe containing the dataset.
      dependent_on (dict): Dictionary containing the dependent_on configurations.
      n (int): Number of samples to generate.
      offset (dict, optional): Dictionary containing the offset configurations. 
        Defaults to None.

  Returns:
      pd.Series: Series containing the generated samples.
  """

  intercept = dependent_on['intercept']
  link_function = dependent_on['link_function']
  beta = dependent_on['beta']
  sigma = dependent_on.get('sigma') or 0
  offset = dependent_on.get('offset')
  offset_function = offset['function'] if offset is not None else 'default'

  column_names = dependent_on['variables']

  # transform data  
  variables = df[column_names].to_numpy()
  beta = np.array(beta)
  intercept = np.full(n, intercept)
  offset = df[offset['column_name']].to_numpy() if offset['column_name'] is not None else np.zeros(n)

  offset_functions = {
    'default': lambda x: x,
    'log': lambda x: np.log(x)
  }

  # define link function
  link_functions = {
    'identity': lambda x: x,
    'logit': lambda x: 1 / (1 + np.exp(-x)),
    'probit': lambda x: 0.5 * (1 + erf(x / np.sqrt(2))),
    'cloglog': lambda x: 1 - np.exp(-np.exp(x)),
    'log': lambda x: np.exp(x),
    'sqrt': lambda x: np.sqrt(x),
    'inverse': lambda x: 1 / x,
    'loglog': lambda x: np.exp(-np.exp(-x)),
    'cauchy': lambda x: 1 / (1 + x ** 2),
    'logc': lambda x: np.log(x),
    'power': lambda x: x ** 2,
    'logloglog': lambda x: np.exp(-np.exp(-np.exp(x))),
    'loglogloglog': lambda x: np.exp(-np.exp(-np.exp(-np.exp(x)))),
    'logloglogloglog': lambda x: np.exp(-np.exp(-np.exp(-np.exp(-np.exp(x)))))
  }
  # generate samples
  return pd.Series(link_functions[link_function](np.dot(variables, beta) + offset_functions[offset_function](offset) + intercept) + np.random.normal(0, sigma, n), name = col_name)