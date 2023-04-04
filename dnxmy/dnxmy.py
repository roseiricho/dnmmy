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
    self.m: int = m
    self.column_config: list = column_config or []


  def set_column_config(self):
    """
    Set column configurations based on the provided configurations or the default configurations.

    """
    # set the number of columns
    self.m = max(self.m, len(self.column_config))

    for i in range(self.m):
      if i < len(self.column_config):
        column_info = self.column_config[i]
      else:
        column_info = None
        self.column_config.append(column_info)

      self.column_config[i] = generate_default_column_info(i, column_info)

  def generate(self) -> pd.DataFrame:
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
    
    self.column_config = optimize_array_order(self.column_config)

    # generate the data
    self.df = pd.DataFrame()

    for col_info in self.column_config:
      if col_info['variable_type'] == 'independent':
        self.df = pd.concat([self.df, generate_random_samples(col_info['column_name'], col_info['probability_distribution'], self.n)], axis=1)
      elif col_info['variable_type'] == 'constant':
        self.df = pd.concat([self.df, pd.Series([col_info['constant_value']] * self.n, name=col_info['column_name'])], axis=1)
      elif col_info['variable_type'] == 'time':
        self.df = pd.concat([self.df, generate_time(col_info['column_name'], col_info['time_config'], self.n)], axis=1)
      elif col_info['variable_type'] == 'time_series':
        self.df = pd.concat([self.df, generate_arma_samples(col_info['column_name'], col_info['time_series_config'], self.n)], axis=1)
      elif col_info['variable_type'] == 'dependent':
        self.df = pd.concat([self.df, generate_dependent_samples(col_info['column_name'], self.column_config, self.df, col_info['dependent_on'], self.n)], axis=1)


    return self.df

  def add_samples(self, n: int) -> pd.DataFrame:
    """
    Add samples to the generated data.

    Args:
        n (int): Number of samples to be added.

    Returns:
        pd.DataFrame: DataFrame containing the generated data with added samples.
    """
    if self.df is None:
      raise Exception('Data has not been generated yet')

    self.df_add = pd.DataFrame()

    for col_info in self.column_config:
      if col_info['variable_type'] == 'independent':
        self.df_add = pd.concat([self.df_add, generate_random_samples(col_info['column_name'], col_info['probability_distribution'], self.n)], axis=1)
      elif col_info['variable_type'] == 'constant':
        self.df_add = pd.concat([self.df_add, pd.Series([col_info['constant_value']] * self.n, name=col_info['column_name'])], axis=1)
      elif col_info['variable_type'] == 'time':
        self.df_add = pd.concat([self.df_add, generate_time(col_info['column_name'], col_info['time_config'], self.n)], axis=1)
      elif col_info['variable_type'] == 'time_series':
        self.df_add = pd.concat([self.df_add, generate_arma_samples(col_info['column_name'], col_info['time_series_config'], self.n)], axis=1)
      elif col_info['variable_type'] == 'dependent':
        self.df_add = pd.concat([self.df_add, generate_dependent_samples(col_info['column_name'], self.column_config, self.df, col_info['dependent_on'], self.n)], axis=1)

    # add the new samples to the existing data
    self.df = pd.concat([self.df, self.df_add], ignore_index=True)
    
    # update the number of samples
    self.n += n

    return self.df

  def miss(self, missing_config: dict) -> pd.DataFrame:
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


