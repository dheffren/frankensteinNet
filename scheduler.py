###Generic hyperparameter scheduler. 
class ScalarSchedule:
    def __init__(self, schedule_type, initial, final, start_epoch, end_epoch):
        self.schedule_type = schedule_type
        self.initial = initial
        self.final = final
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def get(self, epoch):
        if epoch < self.start_epoch:
            return self.initial
        elif epoch >= self.end_epoch:
            return self.final

        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.schedule_type == "linear":
            return self.initial + progress * (self.final - self.initial)
        elif self.schedule_type == "exp":
            return self.initial * (self.final / self.initial) ** progress
        elif self.schedule_type == "constant":
            return self.initial
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")