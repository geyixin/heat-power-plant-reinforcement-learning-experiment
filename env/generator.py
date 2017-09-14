class GasGenerator(object):  # the base class for Gas Generators
    def __init__(self, size, name, coefficients):
        self.size = float(size)  # power output in MWe
        self.name = str(name)
        self.coefficients = coefficients
        self.load = 0.0  # %
        self.power_output = 0.0  # MWe
        self.gas_burnt = 0.0  # MW HHV
        self.HG_heat_output = 0.0  # MW
        self.LG_heat_output = 0.0  # MW
        self.unrecoverable_heat = 0  # MW
        self.cooling_output = 0  # MW
        self.variables = [{
            'Name': self.name + ' Load',
            'Current': 0,
            'Min': 50,
            'Max': 100,
            'Init': 0,
            'Radius': 20
        }]
        self.reset()

    def update(self):
        self.load = float(self.variables[0]['Current']) / 100
        self.power_output = float(self.size * self.load)
        electrical_efficiency = self.load * self.coefficients[0] + self.coefficients[1]
        hg_efficiency = self.load * self.coefficients[2] + self.coefficients[3]
        lg_efficiency = self.load * self.coefficients[4] + self.coefficients[5]
        self.gas_burnt = float(self.power_output / electrical_efficiency)
        self.HG_heat_output = float(self.gas_burnt * hg_efficiency)
        self.LG_heat_output = float(self.gas_burnt * lg_efficiency)
        self.unrecoverable_heat = 0
        self.cooling_output = 0

    def reset(self):
        for var in self.variables:
            var['Current'] = var['Init']


class GasTurbine(GasGenerator):
    def __init__(self, size, name):
        coefficients = [0.120, 0.196667, 0.10, 0.401667, 0, 0]
        GasGenerator.__init__(self, size, name, coefficients)


class GasEngine(GasGenerator):
    def __init__(self, size, name):
        coefficients = [0.08, 0.3, 0.0, 0.2, 0.04, 0.16]
        GasGenerator.__init__(self, size, name, coefficients)
