import sagemaker_containers.beta.framework as framework
from sagemaker_containers import _env

class Env:

  def __init__(self):
    self.envParams = framework.training_env()
    return

  def get(self):
    return self.envParams