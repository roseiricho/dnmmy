
# ダミーデータセットを生成するDnmmyクラスを定義する
class Dnmmy:
  __init__(self, n: int, m: int = None, column_info: list = None, seed: int = 0):
    self.n = n
    if m is not None:
      if column_info is not None:
        self.m = max(m, len(column_info))
      else:
        self.m = m
    else:
      if column_info is not None:
        self.m = len(column_info)
      else:
        raise ValueError("m or column_info must be specified.")
    self.column_info = column_info
    self.seed = seed
    self.df = None

  def generate_default_column_info(i: int, column_info: dict = None):
    if column_info is None:
      return {
              'column_name': "x_{}".format(i),
              'causal_type': 'independent', 
              'column_type': 'float64'
              'probability_distribution': 'uniform',
              'parameter': {
                'low': 0,
                'high': 1
              },
              'missing_value': {
                'probability': 0,
                'condition': None
              }
            }
    else:
      if column_info['column_name'] is None:
        column_info['column_name'] = "x_{}".format(i)
      if column_info['causal_type'] is None:
        column_info['causal_type'] = 'independent'
      if column_info['column_type'] is None:
        column_info['column_type'] = 'float64'
      if column_info['probability_distribution'] is None:
        column_info['probability_distribution'] = {
          'distribution_type': 'uniform',
          'parameter': {
            'low': 0,
            'high': 1
          }
        }
      if column_info['missing_value'] is None:
        column_info['missing_value'] = {
          'probability': 0,
          'condition': None
        }
      return column_info

  def set_column_info(self):
    if self.column_info is None:
      self.column_info = []
      for i in range(self.m):
        self.column_info.append(generate_default_column_info(i))
    else:
      if len(self.column_info) >= self.m:
        self.m = len(self.column_info)
        for i in range(self.m):
          self.column_info[i] = generate_default_column_info(i, self.column_info[i])
      elif len(self.column_info) < self.m:
        for i in range(self.m):
          if i < len(self.column_info):
            self.column_info[i] = generate_default_column_info(i, self.column_info[i])
          else:
            self.column_info.append(generate_default_column_info(i))
  
  def generate_random_numbers(probability_distribution: dict, n: int):
    if probability_distribution['distribution_type'] == 'uniform':
      return np.random.uniform(probability_distribution['parameter']['low'], probability_distribution['parameter']['high'], n)
    elif probability_distribution['distribution_type'] == 'normal':
      return np.random.normal(probability_distribution['parameter']['loc'], probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'beta':
      return np.random.beta(probability_distribution['parameter']['a'], probability_distribution['parameter']['b'], n)
    elif probability_distribution['distribution_type'] == 'gamma':
      return np.random.gamma(probability_distribution['parameter']['shape'], probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'exponential':
      return np.random.exponential(probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'poisson':
      return np.random.poisson(probability_distribution['parameter']['lam'], n)
    elif probability_distribution['distribution_type'] == 'binomial':
      return np.random.binomial(probability_distribution['parameter']['n'], probability_distribution['parameter']['p'], n)
    elif probability_distribution['distribution_type'] == 'lognormal':
      return np.random.lognormal(probability_distribution['parameter']['mean'], probability_distribution['parameter']['sigma'], n)
    elif probability_distribution['distribution_type'] == 'chisquare':
      return np.random.chisquare(probability_distribution['parameter']['df'], n)
    elif probability_distribution['distribution_type'] == 'f':
      return np.random.f(probability_distribution['parameter']['dfnum'], probability_distribution['parameter']['dfden'], n)
    elif probability_distribution['distribution_type'] == 't':
      return np.random.t(probability_distribution['parameter']['df'], n)
    elif probability_distribution['distribution_type'] == 'multivariate_normal':
      return np.random.multivariate_normal(probability_distribution['parameter']['mean'], probability_distribution['parameter']['cov'], n)
    elif probability_distribution['distribution_type'] == 'multinomial':
      return np.random.multinomial(probability_distribution['parameter']['n'], probability_distribution['parameter']['pvals'], n)
    elif probability_distribution['distribution_type'] == 'dirichlet':
      return np.random.dirichlet(probability_distribution['parameter']['alpha'], n)
    elif probability_distribution['distribution_type'] == 'laplace':
      return np.random.laplace(probability_distribution['parameter']['loc'], probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'logistic':
      return np.random.logistic(probability_distribution['parameter']['loc'], probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'logseries':
      return np.random.logseries(probability_distribution['parameter']['p'], n)
    elif probability_distribution['distribution_type'] == 'negative_binomial':
      return np.random.negative_binomial(probability_distribution['parameter']['n'], probability_distribution['parameter']['p'], n)
    elif probability_distribution['distribution_type'] == 'noncentral_chisquare':
      return np.random.noncentral_chisquare(probability_distribution['parameter']['df'], probability_distribution['parameter']['nonc'], n)
    elif probability_distribution['distribution_type'] == 'noncentral_f':
      return np.random.noncentral_f(probability_distribution['parameter']['dfnum'], probability_distribution['parameter']['dfden'], probability_distribution['parameter']['nonc'], n)
    elif probability_distribution['distribution_type'] == 'pareto':
      return np.random.pareto(probability_distribution['parameter']['a'], n)
    elif probability_distribution['distribution_type'] == 'permutation':
      return np.random.permutation(probability_distribution['parameter']['x'], n)
    elif probability_distribution['distribution_type'] == 'planck':
      return np.random.planck(probability_distribution['parameter']['lamb'], n)
    elif probability_distribution['distribution_type'] == 'power':
      return np.random.power(probability_distribution['parameter']['a'], n)
    elif probability_distribution['distribution_type'] == 'rayleigh':
      return np.random.rayleigh(probability_distribution['parameter']['scale'], n)
    elif probability_distribution['distribution_type'] == 'standard_cauchy':
      return np.random.standard_cauchy(n)
    elif probability_distribution['distribution_type'] == 'standard_exponential':
      return np.random.standard_exponential(n)

  def generate(self):
    if self.seed is not None:
      np.random.seed(self.seed)
    if self.column_info is None or len(self.column_info) < self.m:
      self.set_column_info()
    
    optimize_array_order(self.column_info)

    column_name = []
    for i in range(self.m):
      column_name.append(self.column_info[i]['name'])
    data = np.zeros((self.n, self.m))
    for i in range(self.m):
      if self.column_info[i]['causal_type'] == 'primitive':
        data[:, i] = generate_random_numbers(self.column_info[i]['probability_distribution'], self.n)
      elif self.column_info[i]['causal_type'] == 'derived':
        data[:, i] = calcurate_elements(self.column_info[i]['elements'])

    self.df = pd.DataFrame(data, columns=column_name)

    return self.df


    