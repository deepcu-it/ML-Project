import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys) -> str:
    exc_type, exc_value, exc_traceback = error_detail.exc_info()

    if exc_traceback is not None:
        error_file_name = f"\nError occurred in {exc_traceback.tb_frame.f_code.co_filename}\n"
        error_line_number = f"Line number: {exc_traceback.tb_lineno}\n"
    else:
        error_file_name = "No traceback available.\n"
        error_line_number = ""

    return error_file_name + error_line_number + str(error)

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message

if __name__ == "__main__":
    age = 5
    try:
        if age > 3:
            raise Exception("Age greater than 3")
    except Exception as e:
        exception = CustomException(str(e), sys)
        logging.info(exception.__str__())
