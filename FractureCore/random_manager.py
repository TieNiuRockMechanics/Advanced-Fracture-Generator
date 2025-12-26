import numpy as np

class RandomStateManager:
    """
    A unified random state manager to provide independent random number generators
    for different parts of the simulation. This decouples the stochasticity of
    different modules, enhancing reproducibility and robustness.
    """
    def __init__(self, master_seed):
        """
        Initializes all independent random streams from a single master seed.
        """
        # Main generator for creating subsequent independent generators
        seed_sequence = np.random.SeedSequence(master_seed)
        
        # Create independent child SeedSequences for each module
        fracture_ss, placement_ss, drilling_ss = seed_sequence.spawn(3)
        
        # RNG for the fracture generation process
        self.fracture_rng = np.random.default_rng(fracture_ss)
        
        # RNG for positioning the study box 'B'
        self.placement_rng = np.random.default_rng(placement_ss)
        
        # RNG for positioning the drill holes
        self.drilling_rng = np.random.default_rng(drilling_ss)
