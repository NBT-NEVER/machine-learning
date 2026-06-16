import os
from openai import OpenAI
import pandas as pd

# 配置
DEEPSEEK_API_KEY = os.environ["OPENAI_API_KEY"]  # DeepSeek API Key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# 测试文本：10条评论
test_samples = [
    "This movie was amazing! I loved every second of it.",
    "Absolutely terrible. Waste of time.",
    "The acting was okay, but the plot was predictable.",
    "What a fantastic experience! Highly recommend it.",
    "I didn't like the movie at all. It was boring.",
    "A masterpiece. Truly a work of art.",
    "The movie had a lot of action, but it lacked depth.",
    "An unforgettable film. I am going to watch it again.",
    "Not what I expected. The ending was disappointing.",
    "I found the movie quite interesting. A bit slow, though."
]

# 调用 DeepSeek API 进行情感分析
def deepseek_classification(samples):
    responses = []
    for sample in samples:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": f"以影评情感分析为背景，你需要对文本情感态度进行二分类，分辨其属于正面（反馈为1）或负面（反馈为0）态度，Classify this sentence as positive or negative: {sample}"}
            ],
            stream=False
        )
        responses.append(response.choices[0].message.content.strip())

    return responses

# 测试DeepSeek分类
def test_deepseek():
    print("Testing DeepSeek on 10 sample reviews...")
    deepseek_predictions = deepseek_classification(test_samples)

    # 显示每个样本的分类结果
    for i, text in enumerate(test_samples):
        print(f"Review {i+1}: {text}")
        print(f"Predicted Sentiment: {deepseek_predictions[i]}\n")

if __name__ == "__main__":
    test_deepseek()
