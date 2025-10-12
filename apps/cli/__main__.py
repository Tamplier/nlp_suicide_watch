from src.scripts.model_load import predict

while True:
    user_input = input("> ").strip()
    if not user_input:
        break
    predicted = predict([user_input])
    print(predicted[0])
