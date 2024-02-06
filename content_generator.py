import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import torch
import json
import yaml
import re

@st.cache_resource()
def load_model():
    global HF_MODEL_NAME

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_NAME,
        load_in_4bit=True,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)

    pipeline_obj = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipeline_obj


def get_prompt(message):  # обертка для промпта, см. карточку модели
    return f"GPT4 Correct User: {message}<|end_of_turn|>GPT4 Correct Assistant: "


def generate_answer_local(pipeline_obj, message, wrap_message=True):
    prompt = get_prompt(message) if wrap_message else message
    sequences = pipeline_obj(
        prompt,
        do_sample=True,
        max_new_tokens=2048,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return sequences[0]['generated_text'].split('GPT4 Correct Assistant:')[-1]


def generate_answer_gigachat(message, language):
    global GIGACHAT_API_KEY

    payload = Chat(
        messages=[
            Messages(
                role=MessagesRole.SYSTEM,
                content="Отвечай как преподаватель. " if language == 'Русский' else "Answer as a teacher. "
            )
        ],
        temperature=1.5,
        max_tokens=1024
    )

    with (GigaChat(
            credentials=GIGACHAT_API_KEY,
            verify_ssl_certs=False, model="GigaChat-Pro", repetition_penalty=1.07,
            stream=False) as giga):
        payload.messages.append(Messages(role=MessagesRole.USER, content=message))
        response = giga.chat(payload)
        # payload.messages.append(response.choices[0].message)
        answer = response.choices[0].message.content
    return answer


def get_message(language, text, num_questions, question_type, num_choices=None):
    message = ''
    if language == 'Русский':
        if question_type == 'множественный выбор':
            prompt = f"Ты преподаватель, который придумывает тест для студентов на основе следующего текста. Придумай {num_questions} вопросов, правильный ответ и {num_choices-1} неправильных ответов на каждый из них. Отвечай на русском языке. "
            answer_format = "Ответ предоставь в JSON. Формат ответа: [{'вопрос': вопрос1, 'правильные_ответы': [правильный_ответ1], 'неправильные_ответы': [неправильный_ответ1, ..., " + f"неправильный_ответ{num_choices-1}" + "]}, ..., {'вопрос': " + f"вопрос{num_questions}," + "'правильные_ответ': [правильный_ответ1], 'неправильные_ответы': [неправильный_ответ1, ..., " + f"неправильный_ответ{num_choices-1}" + "]}]\n"
            message = prompt + answer_format + f"<text>{text}</text>"
        elif question_type == 'верно/неверно':
            prompt = f"Ты преподаватель, который придумывает тест для студентов на основе следующего текста. Придумай {num_questions // 2 + num_questions % 2} верных утверждений по тексту и {num_questions // 2} неверных утверждений. Отвечай на русском языке. "
            answer_format = "Ответ предоставь в JSON. Формат ответа: [{'утверждение': утверждение1, 'ответ': 'верно'}, ..., {'утверждение': " + f"утверждение{num_questions // 2 + num_questions % 2}" + ", 'ответ': 'верно'}, {'утверждение': " + "утверждение{num_questions // 2 + num_questions % 2 + 1}," + " 'ответ': 'неверно'}, ..., {'утверждение': " + f"утверждение{num_questions}" + ", 'ответ': 'неверно'}]\n"
            message = prompt + answer_format + f"<text>{text}</text>"
    elif language == 'Английский':
        if question_type == 'множественный выбор':
            prompt = f"You are a teacher creating a test for students based on the following text. Generate {num_questions} questions, each with one correct answer and {num_choices - 1} incorrect answers. Respond in English. "
            answer_format = "Provide the answer in JSON format. Answer format: [{'question': question1, 'correct_answers': [correct_answer1], 'incorrect_answers': [incorrect_answer1, ..., " + f"incorrect_answer{num_choices - 1}" + "]}, ..., {'question': " + f"question{num_questions}," + "'correct_answers': [correct_answer1], 'incorrect_answers': [incorrect_answer1, ..., " + f"incorrect_answer{num_choices - 1}" + "]}]\n"
            message = prompt + answer_format + f"<text>{text}</text>"
        elif question_type == 'верно/неверно':
            prompt = f"You are a teacher creating a test for students based on the following text. Generate {num_questions // 2 + num_questions % 2} true statements related to the text and {num_questions // 2} false statements. Respond in English. "
            answer_format = "Provide the answer in JSON format. Answer format: [{'statement': statement1, 'answer': 'correct'}, ..., {'statement': " + f"statement{num_questions // 2 + num_questions % 2}" + ", 'answer': 'correct'}, {'statement': " + "statement{num_questions // 2 + num_questions % 2 + 1}," + " 'answer': 'incorrect'}, ..., {'statement': " + f"statement{num_questions}" + ", 'answer': 'incorrect'}]\n"
            message = prompt + answer_format + f"<text>{text}</text>"
    return message


def generate_questions(pipeline_obj, model, language, text, num_questions, question_type, num_choices=None):
    answer = ""
    message = get_message(language, text, num_questions, question_type, num_choices)
    if model == 'GigaChat':
        answer = generate_answer_gigachat(message, language)
    elif model == 'Локальная модель':
        answer = generate_answer_local(pipeline_obj, message)
    return answer.strip()


def prettify(text):
    return text


def convert_to_moodle_format(text, question_marker, tag):
    json_str = re.sub(r'(\W)\'(.*?)\'', r'\1"\2"', re.sub(r'\'(.*?)\'(\W)', r'"\1"\2', re.sub(r",\s+]", "\n]", text.replace(",]", "]"))))
    formatted_str = ""
    try:
        full_data = json.loads(json_str)
        shuffle(full_data)
        if question_marker == 'верно/неверно':
            for data in full_data:
                question = data['утверждение']
                result_answer = data['ответ']
                formatted_str += "// question:  name: Выберите верно/неверно\n"
                formatted_str += f"// [tag: {tag}]\n"
                formatted_str += f"::Выберите верно/неверно:: {question} {{"
                if result_answer == 'верно':
                    formatted_str += ' TRUE}'
                else:
                    formatted_str += ' FALSE}'
                formatted_str += "\n"
                formatted_str += "\n"
        elif question_marker == 'множественный выбор':
            for data in full_data:
                question = data['вопрос']
                correct_answers = data['правильные_ответы']
                incorrect_answers = data['неправильные_ответы']
                formatted_str += "// question:  name: Выберите верный ответ\n"
                formatted_str += f"// [tag: {tag}]\n"
                formatted_str += f"::Выберите верный ответ:: {question} {{"
                formatted_str += ' '.join(f"={answer}" for answer in correct_answers)
                formatted_str += ' '.join(f" ~{answer}" for answer in incorrect_answers)
                formatted_str += "}"
                formatted_str += "\n"
                formatted_str += "\n"
    except Exception:
        formatted_str = ""
        try:
            full_data = json.loads(json_str)
            shuffle(full_data)
            if question_marker == 'верно/неверно':
                for data in full_data:
                    question = data['statement']
                    result_answer = data['answer']
                    formatted_str += "// question:  name: Выберите верно/неверно\n"
                    formatted_str += f"// [tag: {tag}]\n"
                    formatted_str += f"::Выберите верно/неверно:: {question} {{"
                    if result_answer == 'correct':
                        formatted_str += ' TRUE}'
                    else:
                        formatted_str += ' FALSE}'
                    formatted_str += "\n"
                    formatted_str += "\n"
            elif question_marker == 'множественный выбор':
                for data in full_data:
                    question = data['question']
                    correct_answers = data['correct_answers']
                    incorrect_answers = data['incorrect_answers']
                    formatted_str += "// question:  name: Выберите верный ответ\n"
                    formatted_str += f"// [tag: {tag}]\n"
                    formatted_str += f"::Выберите верный ответ:: {question} {{"
                    formatted_str += ' '.join(f"={answer}" for answer in correct_answers)
                    formatted_str += ' '.join(f" ~{answer}" for answer in incorrect_answers)
                    formatted_str += "}"
                    formatted_str += "\n"
                    formatted_str += "\n"
        except Exception as e:
            st.sidebar.warning('Некорректный JSON-формат, сгенерируйте вопросы ещё раз!')
            return
    return formatted_str


def export_to_moodle(text_with_questions, last_question_format, tag=""):
    # Привести сгенерированный текст к moodle-формату, учесть тэг (пустая строка по умолчанию)
    converted_text = convert_to_moodle_format(text_with_questions, last_question_format, tag)
    if converted_text is not None:
        file_name = f"generated.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(converted_text)

        with open(file_name, "r", encoding="utf-8") as file:
            st.sidebar.download_button('Скачать', file)


def main(pipeline_obj):
    if "generated_text" not in st.session_state:
        st.session_state.generated_text = ""
    if "last_question_format" not in st.session_state:
        st.session_state.last_question_format = "множественный выбор"

    st.sidebar.header("Параметры для составления вопросов")

    model = st.sidebar.selectbox("Тип модели", {"Локальная модель", "GigaChat"})

    language = st.sidebar.selectbox("Язык генерации", {"Русский", "Английский"})
    num_questions = st.sidebar.number_input("Количество вопросов", min_value=1, max_value=10, value=5)
    question_type = st.sidebar.radio("Тип вопроса", ["верно/неверно", "множественный выбор"])

    if question_type == "множественный выбор":
        num_choices = st.sidebar.number_input("Всего ответов", min_value=2, max_value=5, value=3)
    else:
        num_choices = None

    st.title("Генератор вопросов")
    st.write("Для генерации используется LLM, советуем проверять получающийся контент.")

    tag = st.text_input("Тэг (тематика)", max_chars=50)
    text = st.text_area("Текст, по которому составляются вопросы", max_chars=3000, height=400)

    if st.sidebar.button("Сгенерировать вопросы"):
        generated_text = prettify(generate_questions(pipeline_obj, model, language, text, num_questions,
                                                     question_type, num_choices))
        st.session_state.generated_text = generated_text
        st.session_state.last_question_format = question_type

    edited_generated_text = st.text_area(
        "Сгенерированные вопросы (можно редактировать вручную)", key="generated_text", height=400
    )

    if st.sidebar.button("Конвертировать в Moodle-формат"):
        export_to_moodle(edited_generated_text, st.session_state.last_question_format, tag)


yaml_file_path = "gigachat_creds.yaml"
with open(yaml_file_path, 'r') as file:
    yaml_content = yaml.safe_load(file)

    HF_MODEL_NAME = yaml_content.get('HF_MODEL_NAME')
    GIGACHAT_API_KEY = yaml_content.get('GIGACHAT_API_KEY')

pipeline_obj = load_model()
main(pipeline_obj)
