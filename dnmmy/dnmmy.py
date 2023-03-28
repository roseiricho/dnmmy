import numpy as np
import pandas as pd

# List of supported probability distributions
DISTRIBUTION_LIST = ['uniform', 'normal', 'beta', 'gamma', 'exponential', 'poisson', 'binomial', 'lognormal', 'chisquare', 'f', 't', 'multivariate_normal', 'multinomial', 'dirichlet', 'laplace', 'logistic', 'logseries', 'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'pareto', 'rayleigh', 'standard_cauchy', 'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular', 'vonmises', 'wald', 'weibull', 'zipf', 'gunbel', 'hypergeometric']

class Dnmmy:
  def __init__(self, n: int, m: int = None, column_config: list = None, seed: int = 0):
    """
    A class for generating random datasets based on column configurations.

    Args:
        n (int): Number of samples in the generated dataset.
        m (int, optional): Number of columns in the generated dataset.
          Defaults to None.
        column_config (list, optional): List of dictionaries containing column
          configurations. Defaults to None.
        seed (int, optional): Seed for the random number generator. Defaults to 0.

    Examples:
        >>> import dnmmy
        >>> d = dnmmy.Dnmmy(1000, 3)
        >>> d.generate()
        >>> d.df
        >>> d = dnmmy.Dnmmy(1000, column_config=[
        ...     {
        ...         'column_name': 'x_1',
        ...         'column_type': 'float64',
        ...         'variable_type': 'independent',
        ...         'probability_distribution': {
        ...             'type': 'uniform',
        ...             'parameter': {
        ...                 'low': 0,
        ...                 'high': 1
        ...             }
        ...         }
        ...     },
        ...     {
        ...         'column_name': 'x_2',
        ...         'column_type': 'float64',
        ...         'variable_type': 'independent',
        ...         'probability_distribution': {
        ...             'type': 'normal',
        ...             'parameter': {
        ...                 'loc': 0,
        ...                 'scale': 1
        ...             }
        ...         }
        ...     },
        ...     {
        ...         'column_name': 'x_3',
        ...         'column_type': 'float64',
        ...         'variable_type': 'independent',
        ...         'probability_distribution': {
        ...             'type': 'beta',
        ...             'parameter': {
        ...                 'a': 1,
        ...                 'b': 1
        ...             }
        ...         }
        ...     }
        ... ])
        >>> d.generate()
        >>> d.df


    Returns:
        Dnmmy: An instance of the Dnmmy class.

    """

    self.n = n
    self.seed = seed
    self.column_config = column_config or []
    self.df = None
    
    if m is None and column_config is None:
        raise ValueError("m or column_config must be specified.")
        
    self.m = m if m is not None else len(column_config)
    self.m = max(self.m, len(column_config)) if column_config is not None else self.m

  @staticmethod
  def generate_default_column_info(i: int, column_info: dict = None):
    """
    Generate a dictionary with default values for column configurations.

    Args:
        i (int): Index of the column.
        column_info (dict, optional): Dictionary containing column configurations.
          Defaults to None.

    Returns:
        dict: Dictionary with default values for column configurations.
    """

    default_config = {
      'column_name': f"x_{i}",
      'column_type': 'float64',
      'variable_type': 'independent',
      'probability_distribution': {
        'type': 'uniform',
        'parameter': {
          'low': 0,
          'high': 1
        }
      }
    }

    if column_info is not None:
      default_config.update(column_info)

    return default_config
  
  def set_column_config(self):
    """
    Set column configurations based on the provided configurations or the default configurations.

    """
    self.m = max(self.m, len(self.column_config))

    for i in range(self.m):
      column_info = self.column_config[i] if i < len(self.column_config) else None
      self.column_config[i] = self.generate_default_column_info(i, column_info)
  
  @staticmethod
  def generate_random_samples(probability_distribution: dict, n: int):
    """
    Generate random samples based on the provided probability distribution.

    Args:
        probability_distribution (dict): Dictionary containing the probability distribution.
        n (int): Number of samples to generate.

    Returns:
        np.array: Array of random samples.
    """

    distribution_type = probability_distribution.get('type')
    params = probability_distribution.get('parameter')
    if distribution_type is None or params is None:
      raise ValueError("distribution_type and params must be specified.")

    if distribution_type == 'category':
      categories, probabilities = zip(*[(c['value'], c['probability']) for c in params['categories']])
      probabilities = np.array(probabilities) / sum(probabilities)
      return np.random.choice(categories, n, p=probabilities)

    if distribution_type not in DISTRIBUTION_LIST:
      raise ValueError(f"distribution_type must be in {DISTRIBUTION_LIST}")

    kwargs = {k: params[k] for k in params if k != 'loc'}
    if 'loc' in params:
      kwargs['loc'] = params['loc']

    if distribution_type == 'multivariate_normal':
      kwargs['cov'] = np.array(kwargs['cov'])

    if distribution_type == 'category':
      kwargs['p'] = probabilities

    return getattr(np.random, distribution_type)(size=n, **kwargs)
  
  @staticmethod
  def generate_time(time_config: dict, n: int):
    """
    Generate a time series based on the provided time configurations.

    Args:
        time_config (dict): Dictionary containing the time configurations.
        n (int): Number of samples to generate.

    Returns:
        np.array: Array of time series.
    """
    if not time_config:
      raise ValueError("time_config must be specified.")
    
    start_time, time_unit, time_format = time_config.get('start_time'), time_config.get('time_unit'), time_config.get('time_format')
    if not all([start_time, time_unit, time_format]):
      raise ValueError("start_time, time_unit, and time_format must be specified in time_config.")
    
    time_series = pd.date_range(start_time, periods=n, freq=time_unit).strftime(time_format)
    return np.array(time_series)

  @staticmethod
  def generate_arma_samples(time_series_config: dict, n: int) -> np.array:
    """
    Generate a time series based on the provided time configurations.

    Args:
        time_series_config (dict): Dictionary containing the time configurations.
        n (int): Number of samples to generate.

    Returns:
        np.array: Array of time series.
    """
    if time_series_config is None:
      raise ValueError("time_series_config must be specified.")
    
    ar_params = time_series_config.get('ar_params', {})
    ma_params = time_series_config.get('ma_params', {})
    
    ar_samples = [ar_params.get('initial', 0)]
    ma_samples = [ma_params.get('initial', 0)]
    
    arma_samples = [ar_samples[0] + ma_samples[0] + time_series_config['intercept'] + np.random.normal(0, time_series_config['sigma'])]
    
    for i in range(1, n):
      ar = sum([ar_params['params'][j] * ar_samples[i - j - 1] for j in range(min(i, ar_params.get('order', 0)))])
      if ar_params['shock'][i] is not None:
        shock_value = ar_params['shock'][i]['value']
        if ar_params['shock'][i]['type'] == 'sigma':
          shock_value *= time_series_config['sigma']
        ar += shock_value
      ar_samples.append(ar)

      ma = sum([ma_params['params'][j] * ma_samples[i - j - 1] for j in range(min(i, ma_params.get('order', 0)))])
      if ma_params['shock'][i] is not None:
        shock_value = ma_params['shock'][i]['value']
        if ma_params['shock'][i]['type'] == 'sigma':
          shock_value *= time_series_config['sigma']
        ma += shock_value
      ma_samples.append(ma)

      arma_samples.append(ar + ma + np.random.normal(0, time_series_config['sigma']))
        
    return np.array(arma_samples)

  def generate_dependent_samples(self, dependent_on: dict, n: int):
    """
    Generate dependent samples based on the provided dependent_on configurations.


    Args:
        dependent_on (dict): Dictionary containing the dependent_on configurations.
        n (int): Number of samples to generate.

    Returns:
        np.array: Array of dependent samples.
    """

    intercept = dependent_on['intercept']
    offset = dependent_on['offset']
    link_function = dependent_on['link_function']
    beta = dependent_on['beta']
    
    column_names = dependent_on['variables']
    column_num = [i for i, col in enumerate(self.column_config) if col['column_name'] in column_names]
    
    variables = np.array([[row[i] for i in column_num] for row in self.data])
    beta = np.array(beta)

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
      'exp': lambda x: np.exp(x),
      'power': lambda x: x ** 2,
      'logloglog': lambda x: np.exp(-np.exp(-np.exp(x))),
      'loglogloglog': lambda x: np.exp(-np.exp(-np.exp(-np.exp(x)))),
      'logloglogloglog': lambda x: np.exp(-np.exp(-np.exp(-np.exp(-np.exp(x))))),
    }

    return link_functions[link_function](np.dot(variables, beta) + offset)

  def optimize_array_order(self):
    """
    Optimize the order of the array based on the provided column configurations.

    Returns:
        None
    """
    sorted_list = []
    dependency_dict = {}
    column_dict = {}
    for col in self.column_config:
      col_name = col['column_name']
      if col['variable_type'] == 'dependent':
        dependency_dict[col_name] = col['dependent_on']['variables'].copy()
      else:
        dependency_dict[col_name] = []

    while len(dependency_dict) != 0:
      independent_cols = [col for col, dep_cols in dependency_dict.items() if len(dep_cols) == 0]
      if len(independent_cols) == 0:
        raise ValueError("Circular dependency detected!")
      sorted_list.extend(independent_cols)
      for col in independent_cols:
        del dependency_dict[col]
        for dep_cols in dependency_dict.values():
          for dep_col in dep_cols:
            if col == dep_col:
              dep_cols.remove(col)

    column_dict = {col['column_name']: col for col in self.column_config}
    self.column_config = [column_dict[col] for col in sorted_list if col in column_dict]

  def generate(self):
    """
    Generate the data based on the provided configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    if self.seed is not None:
      np.random.seed(self.seed)

    self.set_column_config()
    column_names = [col_info['column_name'] for col_info in self.column_config]
    
    self.optimize_array_order()

    self.data = np.zeros((self.n, self.m))

    for i, col_info in enumerate(self.column_config):
      if col_info['variable_type'] == 'independent':
        self.data[:, i] = self.generate_random_samples(col_info['probability_distribution'], self.n)
      elif col_info['variable_type'] == 'constant':
        self.data[:, i] = col_info['constant_value']
      elif col_info['variable_type'] == 'time':
        self.data[:, i] = self.generate_time(col_info['time_config'], self.n)
      elif col_info['variable_type'] == 'time_series':
        self.data[:, i] = self.generate_arma_samples(col_info['time_series_config'], self.n)
      elif col_info['variable_type'] == 'dependent':
        self.data[:, i] = self.generate_dependent_samples(col_info['dependent_on'], self.n)

    self.df = pd.DataFrame(self.data, columns=column_names)

    return self.df

  def miss(self, missing_config: dict):
    """
    Generate missing values based on the provided configurations.
    
    Args:
        missing_config (dict): Dictionary containing the missing configurations.

    Returns:
        pd.DataFrame: DataFrame containing the generated data with missing values.
    """
    df_missing = self.df.copy()

    missing_type = missing_config.get('missing_type', None)
    target_column_name = missing_config.get('target_column_name', None)
    missing_rate = missing_config.get('missing_params', {}).get('missing_rate', None)
    dependent_on = missing_config.get('missing_params', {}).get('dependent_on', None)

    if missing_type == 'MCAR':
      missing_index = np.random.choice(self.n, int(self.n * missing_rate), replace=False)
      df_missing.loc[missing_index, target_column_name] = np.nan
    elif missing_type == 'MAR':
      missing_index = df_missing.query(dependent_on).index
      missing_index = np.random.choice(missing_index, int(len(missing_index) * missing_rate), replace=False)
      df_missing.loc[missing_index, target_column_name] = np.nan
    elif missing_type == 'MNAR':
      df_missing.loc[df_missing.query(dependent_on).index, target_column_name] = np.nan
    else:
      raise Exception('Invalid missing type')

    return df_missing, self.df
