import os

class Logger:
    def __init__(self, filename = "logs.csv"):
        if os.path.isfile(filename):
            print("Log file already exists, it will be overwriten.")
        with open(filename, 'w') as file:
            self.file = file
        self.filename = filename
        self.first_line = True
        self.tabular = {}

    def log(self, epoch, loss):
        with open(self.filename, 'w') as file:
            if self.first_line:
                file.write(f"epoch,avg_loss\n")
                self.first_line = False
            file.write(f"{epoch},{loss}\n")
            file.flush()
            file.close()
