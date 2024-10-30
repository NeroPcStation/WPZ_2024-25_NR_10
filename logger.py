import os

class Logger:
    def __init__(self, filename = "logs.csv"):
        self.filename = filename
        self.first_line = True
        if os.path.isfile(filename):
            print("Log file already exists, it will be overwriten.")
        with open(filename, 'w') as file:
            pass
        

    def log(self, epoch, loss):
        with open(self.filename, 'a') as file:
            if self.first_line:
                file.write(f"epoch,avg_loss\n")
                self.first_line = False
            file.write(f"{epoch},{loss}\n")
