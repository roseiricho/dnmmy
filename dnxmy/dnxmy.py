import numpy as np
import pandas as pd

from .config_generator import DnxmyConfig
from .variable_generator import generate_random_samples, generate_time_part, generate_arma_samples, generate_dependent_samples


class Dnxmy:
  def __init__(self, n: int, m: int = None, dnxmy_config: DnxmyConfig = None, seed: int = 0):
    """
    A class for generating random datasets based on column configurations.

    Args:
        n (int): Number of samples in the generated dataset.
        m (int, optional): Number of columns in the generated dataset.
          Defaults to None.
        dnxmy_config (DnxmyConfig, optional): Instance of the DnxmyConfig class.
          Defaults to None.
        seed (int, optional): Seed for the random number generator. Defaults to 0.

    Returns:
        Dnxmy: An instance of the Dnxmy class.
    """

    # initialize the class attributes
    self.n: int = n
    self.seed: int = seed
    self.m: int = m
    self.dnxmy_config: dnxmy_config = dnxmy_config or DnxmyConfig()


  def generate(self) -> pd.DataFrame:
    """
    Generate the data based on the provided configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated data.
    """
    # set the seed
    if self.seed is not None:
      np.random.seed(self.seed)

    if self.m < len(self.dnxmy_config.dataset_config):
      self.m = len(self.dnxmy_config.dataset_config)

    # set the column configurations
    self.dnxmy_config.set_dataset_config(self.m)
    self.dnxmy_config.t_sort()
    self.dataset_config = self.dnxmy_config.dataset_config
    
    # generate the data
    self.df = pd.DataFrame()

    for column_name in self.dataset_config.keys():
      column_config = self.dataset_config[column_name]
      if column_config['variable_type'] == 'independent':
        self.df = pd.concat([self.df, generate_random_samples(column_name, column_config['probability_distribution'], self.n)], axis=1)
      elif column_config['variable_type'] == 'constant':
        self.df = pd.concat([self.df, pd.Series([column_config['constant_value']] * self.n, name = column_name)], axis=1)
      elif column_config['variable_type'] == 'time_part':
        self.df = pd.concat([self.df, generate_time_part(column_name, column_config['time_part_config'], self.n)], axis=1)
      elif column_config['variable_type'] == 'time_series':
        self.df = pd.concat([self.df, generate_arma_samples(column_name, column_config['time_series_config'], self.n)], axis=1)
      elif column_config['variable_type'] == 'dependent':
        self.df = pd.concat([self.df, generate_dependent_samples(column_name, self.dataset_config, self.df, column_config['dependent_on'], self.n)], axis=1)

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

    for column_name in self.dataset_config.keys():
      column_config = self.dataset_config[column_name]
      if column_config['variable_type'] == 'independent':
        self.df_add = pd.concat([self.df_add, generate_random_samples(column_name, column_config['probability_distribution'], n)], axis=1)
      elif column_config['variable_type'] == 'constant':
        self.df_add = pd.concat([self.df_add, pd.Series([column_config['constant_value']] * n, name = column_name)], axis=1)
      elif column_config['variable_type'] == 'time_part':
        self.df_add = pd.concat([self.df_add, generate_time_part(column_name, column_config['time_part_config'], n)], axis=1)
      elif column_config['variable_type'] == 'time_series':
        self.df_add = pd.concat([self.df_add, generate_arma_samples(column_name, column_config['time_series_config'], n)], axis=1)
      elif column_config['variable_type'] == 'dependent':
        self.df_add = pd.concat([self.df_add, generate_dependent_samples(column_name, self.dataset_config, self.df_add, column_config['dependent_on'], n)], axis=1)

    # add the new samples to the existing data
    self.df = pd.concat([self.df, self.df_add], ignore_index=True)
    
    # update the number of samples
    self.n += n

    return self.df


  def miss(self) -> pd.DataFrame:
    """
    Generate missing values based on the provided configurations.
    
    Returns:
        pd.DataFrame: DataFrame containing the generated data with missing values.
    """
    df_missing = self.df.copy()

    # set the seed
    if self.seed is not None:
      np.random.seed(self.seed)
    
    # set the missing configurations
    self.missing_config = self.dnxmy_config.missing_config
    
    missing_column_names = self.missing_config.keys()

    for missing_column_name in missing_column_names:
      missing_column_config = self.missing_config[missing_column_name]
      missing_type = missing_column_config.get('missing_type', None)
      target_column_name = missing_column_name
      missing_rate = missing_column_config.get('missing_params', {}).get('missing_rate', None)
      dependent_on = missing_column_config.get('missing_params', {}).get('dependent_on', None)

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
    print(df_missing)
    return df_missing, self.df


