def set_user_response(user_message: str, chat_history) -> tuble:
    chat_history += [user_message, None]
    return '', chat_history