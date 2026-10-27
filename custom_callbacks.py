from litellm.integrations.custom_logger import CustomLogger


PERSIAN_POLICY = """
قانون الزامی زبان پاسخ:

- تمام پاسخ‌ها را به فارسی معیار، روان و طبیعی بنویس.
- از واژه‌ها و عبارت‌های انگلیسی، چینی، ژاپنی، کره‌ای و سایر
  زبان‌ها در متن پاسخ استفاده نکن.
- نام‌های خاص خارجی را در صورت امکان با آوانویسی فارسی بنویس.
- نام رسمی محصولات و اصطلاحات فنی را فقط در صورت ضرورت به
  شکل اصلی حفظ کن.
- محتوای منابع وب و خروجی ابزارها را به فارسی بازنویسی کن.
- متن ناخوانا یا نویسه‌های نامفهوم را حدس نزن.
- پیش از ارسال، متن پاسخ را از نظر آلودگی زبانی بررسی و اصلاح کن.
- این سیاست برای پاسخ نهایی و تمام وظایف عادی الزامی است.
"""


class PersianLanguagePolicy(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data,
        call_type,
    ):
        if call_type != "completion":
            return data

        messages = data.get("messages")

        if not isinstance(messages, list):
            return data

        # Add the policy to the last text-based system message.
        for message in reversed(messages):
            if message.get("role") == "system":
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = (
                        content.rstrip()
                        + "\n\n"
                        + PERSIAN_POLICY
                    )
                    return data

        # If there is no system message, add one before user messages.
        messages.insert(
            0,
            {
                "role": "system",
                "content": PERSIAN_POLICY,
            },
        )
        return data


proxy_handler_instance = PersianLanguagePolicy()
