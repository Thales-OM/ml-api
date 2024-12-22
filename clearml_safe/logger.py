from clearml import Task
import logging


class BlankLogger:
    def __init__(self):
        pass

    def report_scalar(self, title, series, value, iteration):
        logging.debug(f"Blank ClearML logger is used. Skipping logging of scalar: {title}/{series} = {value}")