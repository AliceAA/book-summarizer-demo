from models import summarize_with_lsa

input_text = open("Alice.txt", "r").read()

output = summarize_with_lsa(str(input_text))

print(output)
