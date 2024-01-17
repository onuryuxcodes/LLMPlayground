import os
from apis.keys import OPEN_AI_API_KEY, OPEN_AI_API_KEY_DICT_ID


def set_open_ai_api_key_to_environment():
    os.environ[OPEN_AI_API_KEY_DICT_ID] = os.getenv(OPEN_AI_API_KEY_DICT_ID) or OPEN_AI_API_KEY

