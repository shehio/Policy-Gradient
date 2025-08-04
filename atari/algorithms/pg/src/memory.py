class Memory:
    def __init__(self):
        self.dlogps = []
        self.hidden_layers = []
        self.rewards = []
        self.states = []

    def __str__(self):
        return (
            f"Memory(states={len(self.states)}, "
            f"hidden_layers={len(self.hidden_layers)}, "
            f"dlogps={len(self.dlogps)}, "
            f"rewards={len(self.rewards)})"
        )

    def __repr__(self):
        return (
            f"<Memory(states={self.states}, "
            f"hidden_layers={self.hidden_layers}, "
            f"dlogps={self.dlogps}, "
            f"rewards={self.rewards})>"
        )
