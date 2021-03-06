from pcraster import *
from pcraster.framework import *

map = readmap("reclassmap")
ori = readmap("randstad.map")
setglobaloption("unitcell")

class MyFirstModel(DynamicModel):
  def __init__(self):
    DynamicModel.__init__(self)
    setclone("reclassmap")
  def initial(self):
    self.alive = readmap("reclassmap")
    self.report(self.alive, 'ini')

  def dynamic(self):
    aliveScalar = scalar(self.alive)
    numberOfAliveNeighbours = windowtotal(aliveScalar, 3) - aliveScalar;
    SurvivalA = pcrand(numberOfAliveNeighbours == 0, self.alive)
    SurvivalB = pcrand(numberOfAliveNeighbours == 1, self.alive)
    SurvivalC = pcrand(numberOfAliveNeighbours == 2, self.alive)
    SurvivalD = pcrand(numberOfAliveNeighbours == 3, self.alive)
    SurvivalE = pcrand(numberOfAliveNeighbours == 4, self.alive)
    SurvivalF = pcrand(numberOfAliveNeighbours == 5, self.alive)
    SurvivalG = pcrand(numberOfAliveNeighbours == 6, self.alive)
    SurvivalH = pcrand(numberOfAliveNeighbours == 7, self.alive)
    SurvivalI = pcrand(numberOfAliveNeighbours == 8, self.alive)
    BirthA = pcrand(numberOfAliveNeighbours == 5, pcrnot(self.alive))
    BirthB = pcrand(numberOfAliveNeighbours == 6, pcrnot(self.alive))

    self.alive = SurvivalA | SurvivalB | SurvivalC | SurvivalD | SurvivalE | SurvivalF | SurvivalG | SurvivalH | SurvivalI | BirthA | BirthB #| BirthC #
    self.report(self.alive, 'alive')

nrOfTimeSteps=30
myModel = MyFirstModel()
dynamicModel = DynamicFramework(myModel,nrOfTimeSteps)
dynamicModel.run()



