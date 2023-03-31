import numpy as np
import pandas as pd

from .config_generator import generate_default_column_info, optimize_array_order
from .variable_generator import generate_random_samples, generate_time, generate_arma_samples, generate_dependent_samples


class Dnxmy:
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

    Returns:
        Dnxmy: An instance of the Dnxmy class.

    """

    # initialize the class attributes
    self.n: int = n
    self.seed: int = seed
    self

  def set_column_config(self):
    """
    Set column configurations based on the provided configurations or the default configurations.

    """
    # set the number of columns
    self.m = max(self.m, len(self.column_config))

    for i in range(self.m):
      column_info = self.column_config[i] if i < len(self.column_config) else None
      self.column_config[i] = generate_default_column_info(i, column_info)

  def generate(self):
    """
    Generate the data based on the provided configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    # set the seed
    if self.seed is not None:
      np.random.seed(self.seed)

    # set the column configurations
    self.set_column_config()
    column_names = [col_info['column_name'] for col_info in self.column_config]
    
    optimize_array_order()

    # generate the data
    self.data = np.zeros((self.n, self.m))

    for i, col_info in enumerate(self.column_config):
      if col_info['variable_type'] == 'independent':
        self.data[:, i] = generate_random_samples(col_info['probability_distribution'], self.n)
      elif col_info['variable_type'] == 'constant':
        self.data[:, i] = col_info['constant_value']
      elif col_info['variable_type'] == 'time':
        self.data[:, i] = generate_time(col_info['time_config'], self.n)
      elif col_info['variable_type'] == 'time_series':
        self.data[:, i] = generate_arma_samples(col_info['time_series_config'], self.n)
      elif col_info['variable_type'] == 'dependent':
        self.data[:, i] = generate_dependent_samples(self.column_config, self.data, col_info['dependent_on'], self.n)

    self.df = pd.DataFrame(self.data, columns=column_names)

    return self.df

  def add_samples(self, n: int):
    """
    Add samples to the generated data.

    Args:
        n (int): Number of samples to be added.

    Returns:
        pd.DataFrame: DataFrame containing the generated data with added samples.
    """
    if self.df is None:
      raise Exception('Data has not been generated yet')

    self.data = np.zeros((n, self.m))

    for i, col_info in enumerate(self.column_config):
      if col_info['variable_type'] == 'independent':
        self.data[:, i] = generate_random_samples(col_info['probability_distribution'], n)
      elif col_info['variable_type'] == 'constant':
        self.data[:, i] = col_info['constant_value']
      elif col_info['variable_type'] == 'time':
        self.data[:, i] = generate_time(col_info['time_config'], n)
      elif col_info['variable_type'] == 'time_series':
        self.data[:, i] = generate_arma_samples(col_info['time_series_config'], n)
      elif col_info['variable_type'] == 'dependent':
        self.data[:, i] = generate_dependent_samples(self.column_config, self.data, col_info['dependent_on'], n)

    # add the new samples to the existing data
    self.df = pd.concat([self.df, pd.DataFrame(self.data, columns=self.df.columns)], ignore_index=True)
    
    # update the number of samples
    self.n += n

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

    # missing
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

