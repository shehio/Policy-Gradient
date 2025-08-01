class MemoryMultiAction:
    def __init__(self):
        self.dlogps = []
        self.hidden_layers = []
        self.rewards = []
        self.states = []
        self.actions = []
        self.entropies = []  # For entropy regularization

    def __str__(self):
        return (
            f"MemoryMultiAction(states={len(self.states)}, "
            f"hidden_layers={len(self.hidden_layers)}, "
            f"dlogps={len(self.dlogps)}, "
            f"rewards={len(self.rewards)}, "
            f"actions={len(self.actions)}, "
            f"entropies={len(self.entropies)})"
        )

    def __repr__(self):
        return (
            f"<MemoryMultiAction(states={self.states}, "
            f"hidden_layers={self.hidden_layers}, "
            f"dlogps={self.dlogps}, "
            f"rewards={self.rewards}, "
            f"actions={self.actions}, "
            f"entropies={self.entropies})>"
        )
