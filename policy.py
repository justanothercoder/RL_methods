# -*- coding: utf-8 -*-


class Policy:
    @property
    def predict(self):
        raise NotImplementedError
        
    
    @property
    def sample(self):
        raise NotImplementedError
    
        
    @property
    def log_probability(self):
        raise NotImplementedError
        
        
    @property
    def state(self):
        raise NotImplementedError
        
        
    @property
    def action(self):
        raise NotImplementedError