def generate_default_column_info(i: int, column_info: dict = None) -> dict:
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
    'variable_type': 'independent',
    'probability_distribution': {
      'type': 'uniform',
      'parameter': {
        'low': 0,
        'high': 1
      }
    }
  }

  # if column_info is not None, then update default_config with column_info
  if column_info is not None:
    default_config.update(column_info)

  return default_config

def add_independent_config(column_config: list, name: str, probability_distribution_type: str = 'uniform', probability_distribution_params: dict = {'low': 0, 'high': 1}) -> list:
  """
  Add a column configuration for an independent variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      probability_distribution_type (str): Type of the probability distribution.
      probability_distribution_params (dict, optional): Parameters of the probability distribution.
        Defaults to None.

  Returns:
      list: List of column configurations.
  """
  # create the configuration for the column
  column_config.append({
    'column_name': name,
    'variable_type': 'independent',
    'probability_distribution': {
      'type': probability_distribution_type,
      'parameter': probability_distribution_params
    }
  })

  return column_config

def add_categorical_config(column_config: list, name: str, value: list, probability: list) -> list:
  """
  Add a column configuration for a categorical variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      value (list): List of values of the categorical variable.
      probability (list): List of probabilities of the values.

  Returns:
      list: List of column configurations.
  """
  # add a column configuration for the categorical variable
  column_config.append({
    'column_name': name,
    'variable_type': 'indpendent',
    'probability_distribution': {
      'type': 'categorical',
      'parameter': {
        'value': value,
        'probability': probability
      }
    }
  })

  return column_config

def add_constant_config(column_config: list, name: str, constant_value: float) -> list:
  """
  Add a column configuration for a constant variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      constant_value (float): Value of the constant.

  Returns:
      list: List of column configurations.
  """
  # add a column configuration for the constant variable
  column_config.append({
    'column_name': name,
    'variable_type': 'constant',
    'constant_value': constant_value
  })

  return column_config

def add_time_config(column_config: list, name: str, start_time: str, time_unit: str, time_format: str) -> list:
  """
  Add a column configuration for a time variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      start_time (str): Start time of the time variable.
      time_unit (str): Time unit of the time variable.
      time_format (str): Time format of the time variable.

  Returns:
      list: List of column configurations.
  """
  # add a column configuration for the time variable
  column_config.append({
    'column_name': name,
    'variable_type': 'time',
    'time_config': {
      'start_time': start_time,
      'time_unit': time_unit,
      'time_format': time_format
    }
  })

  return column_config

def add_arma_config(column_config: list, 
                    name:str, 
                    intercept: float, 
                    sigma: float, 
                    ar_initial: float, 
                    ar_order: int, 
                    ar_params: list, 
                    ar_shock_type: list, 
                    ar_shock_value: list, 
                    ma_initial: float, 
                    ma_order: int, 
                    ma_params: list, 
                    ma_shock_type: list, 
                    ma_shock_value: list) -> list:
  """
  Add a column configuration for an ARMA variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      intercept (float): Intercept for the ARMA variable.
      sigma (float): Sigma for the ARMA variable.
      ar_initial (float): Initial value for the AR part of the ARMA variable.
      ar_order (int): Order of the AR part of the ARMA variable.
      ar_params (list): List of AR parameters for the ARMA variable.
      ar_shock_type (list): List of shock types for the AR part of the ARMA variable.
      ar_shock_value (list): List of shock values for the AR part of the ARMA variable.
      ma_initial (float): Initial value for the MA part of the ARMA variable.
      ma_order (int): Order of the MA part of the ARMA variable.
      ma_params (list): List of MA parameters for the ARMA variable.
      ma_shock_type (list): List of shock types for the MA part of the ARMA variable.
      ma_shock_value (list): List of shock values for the MA part of the ARMA variable.

  Returns:
      list: List of column configurations.
  """

  # create a dictionary for the AR shocks
  ar_shock_dict = {}
  for i in range(len(ar_shock_type)):
    ar_shock_dict[i]['type'] = ar_shock_type[i]
    ar_shock_dict[i]['value'] = ar_shock_value[i]
  
  # create a dictionary for the MA shocks
  ma_shock_dict = {}
  for i in range(len(ma_shock_type)):
    ma_shock_dict[i]['type'] = ma_shock_type[i]
    ma_shock_dict[i]['value'] = ma_shock_value[i]

  # create the column configuration dictionary
  column_config.append({
    'column_name': name,
    'variable_type': 'arma',
    'time_series_config': {
      'intercept': intercept,
      'sigma': sigma,
      'ar_params': {
        'initial': ar_initial,
        'order': ar_order,
        'params': ar_params,
        'shock': ar_shock_dict
      },
      'ma_params': {
        'initial': ma_initial,
        'order': ma_order,
        'params': ma_params,
        'shock': ma_shock_dict
      }
    }
  })

  return column_config

def add_dependent_config(column_config: list, name: str, variables: list, beta: list, intercept: float, offset: float = None, link_function: str = 'identity') -> list:
  """
  Add a column configuration for a dependent variable.

  Args:
      column_config (list): List of column configurations.
      name (str): Name of the column.
      dependent_on (list): List of columns on which the dependent variable depends.
      beta (list): List of coefficients for the dependent variable.
      intercept (float): Intercept for the dependent variable.
      offset (float, optional): Offset for the dependent variable. Defaults to None.
      link_function (str, optional): Link function for the dependent variable. Defaults to None.

  Returns:
      list: List of column configurations.
  """
  # add a column configuration for the dependent variable
  column_config.append({
    'column_name': name,
    'variable_type': 'dependent',
    'dependent_on': {
      'variables': dependent_on,
      'beta': beta,
      'intercept': intercept,
      'offset': offset,
      'link_function': link_function
    }
  })

  return column_config


def optimize_array_order(column_config: list) -> list:
  """
  Optimize the order of the array based on the provided column configurations.

  Returns:
      None
  """
  sorted_list = []
  dependency_dict = {}
  column_dict = {}

  # create a dictionary of dependencies
  for col in column_config:
    col_name = col['column_name']
    if col['variable_type'] == 'dependent':
      dependency_dict[col_name] = col['dependent_on']['variables'].copy()
    else:
      dependency_dict[col_name] = []

  # sort the columns based on dependencies
  while len(dependency_dict) != 0:
    independent_cols = [col for col, dep_cols in dependency_dict.items() if len(dep_cols) == 0]

    # if there is no independent columns, there is a circular dependency!
    if len(independent_cols) == 0:
      raise ValueError("Circular dependency detected!")

    sorted_list.extend(independent_cols)
    for col in independent_cols:
      del dependency_dict[col]
      for dep_cols in dependency_dict.values():
        for dep_col in dep_cols:
          if col == dep_col:
            dep_cols.remove(col)

  column_dict = {col['column_name']: col for col in column_config}
  column_config = [column_dict[col] for col in sorted_list if col in column_dict]


def generate_missing_config(missing_config: dict, missing_type: str, target_column_name: str, missing_rate: float = None, dependent_on: str = None) -> dict:
  """
  Generate a missing configuration dictionary.

  Args:
      missing_config (dict): Dictionary of missing configurations.
      missing_type (str): Type of missingness.
      target_column_name (str): Name of the column to which the missingness is applied.
      missing_rate (float): Missing rate.
      dependent_on (list, optional): List of columns on which the missingness depends. Defaults to None.
      beta (list, optional): List of coefficients for the missingness. Defaults to None.
      intercept (float, optional): Intercept for the missingness. Defaults to None.
      offset (float, optional): Offset for the missingness. Defaults to None.
      link_function (str, optional): Link function for the missingness. Defaults to None.

  Returns:
      dict: Dictionary of missing configurations.
  """
  # add a missing configuration
  missing_config = {
    'missing_type': missing_type,
    'target_column_name': target_column_name,
    'missing_params': {
      'missing_rate': missing_rate,
      'dependent_on': dependent_on
    }
  }

  return missing_config