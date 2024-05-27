from flask import Blueprint, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from config import Config

routes = Blueprint('routes', __name__)

# Hugging Face token from config
HF_TOKEN = Config.HF_TOKEN
model_name = Config.MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=HF_TOKEN
)

text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=Config.MAX_TOKENS,
    use_auth_token=HF_TOKEN
)


def get_response(prompt):
    sequences = text_generator(prompt, temperature=Config.TEMPERATURE)
    gen_text = sequences[0]["generated_text"]
    return gen_text


@routes.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    response = get_response(prompt)
    return jsonify({"response": response})


@routes.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"})
