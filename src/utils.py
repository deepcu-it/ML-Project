import sys

def error_message_detail(error,error_detail:sys):
    _,_, exec_traceback = error_detail.exc_info()

    error_file_name = f"Error occured in {exec_traceback.tb_frame.f_code.co_filename}\n"
    error_line_number = f"line number: {exec_traceback.tb_lineno}\n"
    
    return error_file_name+error_line_number+error


class CustomException(Exception):
    def __init__(self, error_message,error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message