
class Memory:
    def __init__(self):
        self.chat_history = []

    def store(self, role, message):
        self.chat_history.append({"role":role, "message":message})

    def get_history(self):
        return "/n".join([f"{entry['role']}: {entry['message']}" 
                          for entry in self.chat_history 
                        ])