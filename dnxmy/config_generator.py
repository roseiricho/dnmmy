
def generate_default_config(column_config: dict = None) -> dict:
  """
  Generate a dictionary with default values for column configurations.

  Args:
      column_config (dict, optional): Dictionary containing column configurations.
        Defaults to None.

  Returns:
      dict: Dictionary with default values for column configurations.
  """

  default_config = {
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
  if column_config is not None:
    default_config.update(column_config)

  return default_config

class DnxmyConfig:
  def __init__(self):
    """
    Initialize the DnxmyConfig class.

    Returns:
        DnxmyConfig: DnxmyConfig class.
    """

    # initialize the class attributes
    self.dataset_config: dict = {}
    self.missing_config: dict = {}


  def add_independent_column(self, col_name: str, probability_distribution_type: str = 'uniform', probability_distribution_params: dict = {'low': 0, 'high': 1}):
    """
    Add a column configuration for an independent variable.

    Args:
        col_name (str): Name of the column.
        probability_distribution_type (str): Type of the probability distribution.
          Defaults to 'uniform'.
        probability_distribution_params (dict, optional): Parameters of the probability distribution.
          Defaults to {'low': 0, 'high': 1}.
    """
    # create the configuration for the column
    self.dataset_config[col_name] = {
      'variable_type': 'independent',
      'probability_distribution': {
        'type': probability_distribution_type,
        'parameter': probability_distribution_params
      }
    }


  def add_categorical_column(self, col_name: str, value: list, probability: list):
    """
    Add a column configuration for a categorical variable.

    Args:
        col_name (str): Name of the column.
        value (list): List of values of the categorical variable.
        probability (list): List of probabilities of the values.
    """
    categories = []
    for i in range(len(value)):
      categories.append({
        'value': value[i],
        'probability': probability[i]
      })

    # add a column configuration for the categorical variable
    self.dataset_config[col_name] = {
      'variable_type': 'independent',
      'probability_distribution': {
        'type': 'categorical',
        'parameter': {
          'categories': categories
        }
      }
    }


  def add_constant_column(self, col_name: str, constant_value: float = 1):
    """
    Add a column configuration for a constant variable.

    Args:
        col_name (str): Name of the column.
        constant_value (float): Value of the constant.
          Defaults to 1.
    """
    # add a column configuration for the constant variable
    self.dataset_config[col_name] = {
      'variable_type': 'constant',
      'constant_value': constant_value
    }


  def add_time_part_column(self, col_name: str, start_time: str, time_unit: str, time_format: str = '%Y-%m-%d'):
    """
    Add a column configuration for a time part variable.

    Args:
        col_name (str): Name of the column.
        start_time (str): Start time of the time variable.
        time_unit (str): Time unit of the time variable.
        time_format (str): Time format of the time variable.
          Defaults to '%Y-%m-%d'.
    """
    # add a column configuration for the time variable
    self.dataset_config[col_name] = {
      'variable_type': 'time_part',
      'time_part_config': {
        'start_time': start_time,
        'time_unit': time_unit,
        'time_format': time_format
      }
    }


  def add_arma_column(self, 
                      col_name:str, 
                      intercept: float = 0, 
                      sigma: float = 0, 
                      ar_initial: float = None, 
                      ar_order: int = None, 
                      ar_params: list = None, 
                      ar_shock_time: list = None,
                      ar_shock_type: list = None, 
                      ar_shock_value: list = None, 
                      ma_initial: float = None, 
                      ma_order: int = None,
                      ma_params: list = None,
                      ma_shock_time: list = None,
                      ma_shock_type: list = None, 
                      ma_shock_value: list = None):
    """
    Add a column configuration for an ARMA variable.

    Args:
        col_name (str): Name of the column.
        intercept (float): Intercept for the ARMA variable.
          Defaults to 0.
        sigma (float): Sigma for the ARMA variable.
          Defaults to 0.
        ar_initial (float): Initial value for the AR part of the ARMA variable.
          Defaults to None.
        ar_order (int): Order of the AR part of the ARMA variable.
          Defaults to None.
        ar_params (list): List of AR parameters for the ARMA variable.
          Defaults to None.
        ar_shock_type (list): List of shock types for the AR part of the ARMA variable.
          Defaults to None.
        ar_shock_value (list): List of shock values for the AR part of the ARMA variable.
          Defaults to None.
        ma_initial (float): Initial value for the MA part of the ARMA variable.
          Defaults to None.
        ma_order (int): Order of the MA part of the ARMA variable.
          Defaults to None.
        ma_params (list): List of MA parameters for the ARMA variable.
          Defaults to None.
        ma_shock_type (list): List of shock types for the MA part of the ARMA variable.
          Defaults to None.
        ma_shock_value (list): List of shock values for the MA part of the ARMA variable.
          Defaults to None.
    """
    # create a dictionary for the AR shocks
    ar_shock_dict = {}
    if ar_shock_time is not None or ar_shock_type is not None or ar_shock_value is not None:
      for i, t, v in zip(ar_shock_time, ar_shock_type, ar_shock_value):
        ar_shock_dict[i] = {}
        ar_shock_dict[i]['type'] = t
        ar_shock_dict[i]['value'] = v
    
    # create a dictionary for the MA shocks
    ma_shock_dict = {}
    if ma_shock_time is not None or ma_shock_type is not None or ma_shock_value is not None:
      for i, t, v in zip(ma_shock_time, ma_shock_type, ma_shock_value):
        ma_shock_dict[i] = {}
        ma_shock_dict[i]['type'] = t
        ma_shock_dict[i]['value'] = v

    # create the column configuration dictionary
    self.dataset_config[col_name] = {
      'variable_type': 'time_series',
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
    }


  def add_dependent_column(self, 
                          col_name: str, 
                          variables: list, 
                          beta: list, 
                          intercept: float, 
                          offset_column: str = None, 
                          offset_function: str = 'default', 
                          link_function: str = 'identity'):
    """
    Add a column configuration for a dependent variable.

    Args:
        col_name (str): Name of the column.
        variables (list): List of columns on which the variable depends.
        beta (list): List of coefficients for the dependent variable.
        intercept (float): Intercept for the dependent variable.
        offset_column (str): Name of the column to use for the offset.
          Defaults to None.
        offset_function (str): Function to use for the offset.
          Defaults to 'default'.
        link_function (str): Link function to use for the dependent variable.
          Defaults to 'identity'.
    """
    # add a column configuration for the dependent variable
    self.dataset_config[col_name] = {
      'variable_type': 'dependent',
      'dependent_on': {
        'variables': variables,
        'beta': beta,
        'intercept': intercept,
        'offset': {
          'column_name': offset_column,
          'function': offset_function
        },
        'link_function': link_function
      }
    }
  
  
  def delete_column_config(self, col_name: str):
    """
    Delete a column configuration.

    Args:
        col_name (str): Name of the column.
    """
    del self.dataset_config[col_name]


  def t_sort(self):
    """
    Optimize the order of dataset configuration using topological sorting.
    """
    sorted_list = []
    dependency_dict = {}
    column_dict = {}

    # create a dictionary of dependencies
    for col_name in self.dataset_config:
      col = self.dataset_config[col_name]
      if col['variable_type'] == 'dependent':
        dependency_dict[col_name] = col['dependent_on']['variables'].copy()
        if col['dependent_on'].get('offset')['column_name'] is not None:
          dependency_dict[col_name].append(col['dependent_on']['offset']['column_name'])
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
              
    column_dict = {col_name: self.dataset_config[col_name] for col_name in sorted_list}
    
    self.dataset_config = column_dict


  def set_dataset_config(self, m: int):
    """
    Set dataset configurations based on the provided configurations or the default configurations.
    """
    col_names = list(self.dataset_config.keys())
    if len(self.dataset_config) <= m:
      for i in range(m - len(self.dataset_config)):
        if 'col_' + str(i) in col_names:
          i += 1
        col_names.append('col_' + str(i))
    
    for col_name in col_names:
      if self.dataset_config.get(col_name) is not None:
        column_config = self.dataset_config[col_name]
      else:
        column_config = None
      
      self.dataset_config[col_name] = generate_default_config(column_config)


  def add_missing_config(self, target_col_name: str, missing_type: str = 'MCAR', missing_rate: float = None, dependent_on: str = None):
    """
    Generate a missing configuration dictionary.

    Args:
        missing_type (str): Type of missingness.
          Defaults to 'MCAR'.
        target_column_name (str): Name of the column to which the missingness is applied.
        missing_rate (float): Missing rate.
          Defaults to None.
        dependent_on (str): String expressing a missing condition in query string format in pandas.DataFrame.query.
          Defaults to None.

    Returns:
        dict: Dictionary of missing configurations.
    """
    if missing_type not in ['MCAR', 'MAR', 'MNAR']:
      raise ValueError("Missing type must be one of 'MCAR', 'MAR', or 'MNAR'!")
    
    # add a missing configuration
    self.missing_config[target_col_name] = {
      'missing_type': missing_type,
      'missing_params': {
        'missing_rate': missing_rate,
        'dependent_on': dependent_on
      }
    }