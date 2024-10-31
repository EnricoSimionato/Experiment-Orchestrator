import logging


class LoggingInterface:
    """
    Interface for logging information during the training.

    Attributes:
        logger (logging.Logger):
            The logger object.
    """

    def __init__(
            self
    ) -> None:
        self.logger = logging.getLogger()


    def log(
            self,
            message: str,
            print_message: bool = False,
            level: str = "info"
    ) -> None:
        """
        Logs a message in the log file.

        Args:
            message (str):
                The message to log.
            print_message (bool, optional):
                Whether to print the message. Defaults to False.
            level (str, optional):
                The level of the log. Defaults to "info".
        """

        if level == "info":
            self.logger.info(message)
        else:
            raise NotImplementedError("Only info level is implemented.")

        if print_message:
            print(message)
