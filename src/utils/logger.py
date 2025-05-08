import logging

# 로깅 설정
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)

# 콘솔 핸들러
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 파일 핸들러
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)

# 포맷 설정
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 핸들러 추가
logger.addHandler(console_handler)
logger.addHandler(file_handler)