# -*- coding: utf-8 -*-
"""
Распознавание продуктов по фото и генерация рецептов.
Среда запуска: Google Colab с GPU.
Модели:
- Qwen/Qwen2.5-VL-3B-Instruct для распознавания продуктов на изображении
- Qwen/Qwen2.5-1.5B-Instruct для генерации рецептов
"""

# ============================================================
# 1. Установка зависимостей
# ============================================================

!pip -q install -U git+https://github.com/huggingface/transformers.git accelerate
!pip -q install "qwen-vl-utils[decord]==0.0.8"
!pip -q install "pandas==2.2.2" "pillow<12.0,>=8.0" matplotlib openpyxl

print("Зависимости установлены. После первой установки перезапустите Runtime и продолжите со следующей ячейки.")

# ============================================================
# 2. Импорты и настройки
# ============================================================

import gc
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.display import Markdown, display
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

VLM_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

IMAGE_DIR = "/content/ph"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

PRODUCT_SEPARATOR = "."
EVAL_TEMPLATE_PATH = "/content/eval_template.xlsx"
EVAL_RESULTS_PATH = "/content/eval_results.csv"

print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 3. Поиск изображений
# ============================================================

def collect_image_paths(image_dir=IMAGE_DIR):
    image_dir = Path(image_dir)
    paths = []
    for pattern in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(pattern)))
    return [str(path) for path in paths]

image_paths = collect_image_paths(IMAGE_DIR)

if not image_paths:
    raise FileNotFoundError(
        f"В папке {IMAGE_DIR} не найдены изображения. "
        "Загрузите фото в эту папку или измените IMAGE_DIR."
    )

print("Найдено изображений:", len(image_paths))
for path in image_paths:
    print("-", path)

img = Image.open(image_paths[0]).convert("RGB")
preview_width = min(700, img.width)
preview_height = int(img.height * preview_width / img.width)
display(img.resize((preview_width, preview_height)))

# ============================================================
# 4. Загрузка VLM
# ============================================================

vlm_processor = AutoProcessor.from_pretrained(
    VLM_MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
)

vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    VLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)
vlm_model.eval()

print("VLM загружена:", VLM_MODEL_ID)

# ============================================================
# 5. Распознавание продуктов
# ============================================================

PRODUCT_DETECTION_PROMPT = """
Ты анализируешь фото холодильника или продуктовой полки.

Найди только реально видимые продукты питания.
Не выдумывай продукты.
Если продукт плохо виден, укажи низкую уверенность.

Верни только JSON-массив без markdown и пояснений.
Формат:
[
  {
    "name": "название продукта на русском",
    "category": "овощи / фрукты / молочные / мясо / рыба / напитки / крупы / соусы / другое",
    "confidence": 0.0
  }
]
"""


def clean_model_output(text):
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("```json", "")
    text = text.replace("```JSON", "")
    text = text.replace("```", "")
    return text.strip()


def extract_complete_json_objects(text):
    text = clean_model_output(text)
    objects = []
    stack = 0
    start = None

    for i, ch in enumerate(text):
        if ch == "{":
            if stack == 0:
                start = i
            stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0 and start is not None:
                objects.append(text[start:i + 1])
                start = None

    return objects


def parse_product_object(obj_text):
    obj_text = obj_text.strip()

    for candidate in [
        obj_text,
        obj_text.replace("'", '"')
                .replace("“", '"')
                .replace("”", '"')
                .replace("«", '"')
                .replace("»", '"'),
    ]:
        candidate = re.sub(r",\s*}", "}", candidate)
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', obj_text)
    category_match = re.search(r'"category"\s*:\s*"([^"]+)"', obj_text)
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', obj_text)

    if not name_match:
        return None

    try:
        confidence = float(confidence_match.group(1)) if confidence_match else 1.0
    except Exception:
        confidence = 1.0

    return {
        "name": name_match.group(1).strip(),
        "category": category_match.group(1).strip() if category_match else "другое",
        "confidence": confidence,
    }


def extract_products_from_raw_output(raw_output):
    products = []

    for obj_text in extract_complete_json_objects(raw_output):
        product = parse_product_object(obj_text)
        if not product:
            continue

        name = str(product.get("name", "")).strip()
        if not name:
            continue

        try:
            confidence = float(product.get("confidence", 1.0))
        except Exception:
            confidence = 1.0

        products.append({
            "name": name,
            "category": str(product.get("category", "другое")).strip(),
            "confidence": confidence,
        })

    unique_products = []
    seen = set()
    for product in products:
        key = product["name"].lower().replace("ё", "е")
        if key not in seen:
            unique_products.append(product)
            seen.add(key)

    return unique_products


def detect_products(image_path, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": PRODUCT_DETECTION_PROMPT},
            ],
        }
    ]

    text = vlm_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(vlm_model.device)

    with torch.inference_mode():
        generated_ids = vlm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    raw_output = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    products = extract_products_from_raw_output(raw_output)

    del inputs, generated_ids, generated_ids_trimmed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "image_path": image_path,
        "raw_output": raw_output,
        "products": products,
    }


def get_ingredient_names(detection, min_confidence=0.0):
    names = []
    for product in detection.get("products", []):
        name = str(product.get("name", "")).strip()
        try:
            confidence = float(product.get("confidence", 1.0))
        except Exception:
            confidence = 1.0

        if name and confidence >= min_confidence:
            names.append(name)

    return list(dict.fromkeys(names))


def run_detection(image_paths):
    detections = []

    for path in image_paths:
        print(f"\nРаспознавание: {path}")
        try:
            detection = detect_products(path)
            detections.append(detection)
            products = get_ingredient_names(detection)
            print("Продукты:", ", ".join(products) if products else "не найдены")
        except torch.cuda.OutOfMemoryError:
            print("Недостаточно памяти GPU. Изображение пропущено:", path)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print("Ошибка:", type(exc).__name__, exc)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return detections


detections = run_detection(image_paths)

# ============================================================
# 6. Таблица распознанных продуктов
# ============================================================

def build_prediction_table(detections):
    rows = []

    for idx, detection in enumerate(detections):
        products = detection.get("products", [])
        names = []
        uncertain = []
        categories = []

        for product in products:
            name = str(product.get("name", "")).strip()
            category = str(product.get("category", "другое")).strip()
            try:
                confidence = float(product.get("confidence", 1.0))
            except Exception:
                confidence = 1.0

            if name:
                names.append(name)
                categories.append(f"{name}: {category}")
                if confidence < 0.6:
                    uncertain.append(name)

        rows.append({
            "index": idx,
            "image_path": detection.get("image_path", ""),
            "predicted_products": f"{PRODUCT_SEPARATOR} ".join(names),
            "uncertain": f"{PRODUCT_SEPARATOR} ".join(uncertain),
            "categories": "; ".join(categories),
            "num_products": len(names),
            "raw_output": str(detection.get("raw_output", ""))[:700],
        })

    return pd.DataFrame(rows)


pred_df = build_prediction_table(detections)

print("Всего изображений:", len(pred_df))
print("Изображений с найденными продуктами:", int((pred_df["num_products"] > 0).sum()))
print("Изображений без найденных продуктов:", int((pred_df["num_products"] == 0).sum()))
display(pred_df)

# ============================================================
# 7. Освобождение памяти перед загрузкой LLM
# ============================================================

try:
    del vlm_model
    del vlm_processor
except Exception:
    pass

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Память очищена.")

# ============================================================
# 8. Загрузка LLM
# ============================================================

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)
llm_model.eval()

print("LLM загружена:", LLM_MODEL_ID)

# ============================================================
# 9. Генерация рецептов
# ============================================================

def build_recipe_prompt(ingredients, servings=2, preferences=""):
    ingredients_text = ", ".join(ingredients) if ingredients else "ингредиенты не распознаны"
    preferences_text = preferences.strip() if preferences.strip() else "нет особых ограничений"

    return f"""
Список продуктов, распознанных на фото:
{ingredients_text}

Пожелания: {preferences_text}
Количество порций: {servings}

Составь 2-3 практичных рецепта на русском языке.

Требования:
1. Используй в основном продукты из списка.
2. Дополнительно можно использовать воду, соль, перец, растительное масло и базовые сухие специи.
3. Не добавляй новые обязательные продукты, которых нет в списке.
4. Если дополнительный продукт желателен, пометь его как опциональный.
5. Для каждого рецепта укажи название, используемые продукты, базовые добавки, время и шаги приготовления.
""".strip()


@torch.inference_mode()
def generate_recipes_from_products(ingredients, servings=2, preferences="", max_new_tokens=900):
    messages = [
        {
            "role": "system",
            "content": (
                "Ты кулинарный ассистент. Генерируй простые рецепты по списку продуктов. "
                "Не указывай отсутствующие продукты как обязательные. Отвечай по-русски."
            ),
        },
        {
            "role": "user",
            "content": build_recipe_prompt(ingredients, servings, preferences),
        },
    ]

    text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)

    output_ids = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    generated_ids = output_ids[0][inputs.input_ids.shape[-1]:]
    return llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


IMAGE_INDEX = 0
SERVINGS = 2
PREFERENCES = "без мяса"

selected_detection = detections[IMAGE_INDEX]
selected_products = get_ingredient_names(selected_detection, min_confidence=0.25)

print("Фото:", selected_detection["image_path"])
print("Ингредиенты:", ", ".join(selected_products))

recipes = generate_recipes_from_products(
    selected_products,
    servings=SERVINGS,
    preferences=PREFERENCES,
)

display(Markdown(recipes))

with open("/content/recipes_result.md", "w", encoding="utf-8") as file:
    file.write(
        f"# Рецепты по фото\n\n"
        f"Фото: {selected_detection['image_path']}\n\n"
        f"Распознанные продукты: {', '.join(selected_products)}\n\n"
        f"---\n\n{recipes}\n"
    )

print("Рецепты сохранены: /content/recipes_result.md")

# ============================================================
# 10. Подготовка файла для ручной проверки
# ============================================================


def create_eval_template(pred_df, output_path=EVAL_TEMPLATE_PATH):
    eval_template = pred_df[["image_path", "predicted_products", "categories", "num_products"]].copy()
    eval_template["true_products"] = ""
    eval_template["comment"] = ""
    eval_template.to_excel(output_path, index=False)
    return eval_template


eval_template = create_eval_template(pred_df)
display(eval_template.head())
print("Файл для разметки сохранён:", EVAL_TEMPLATE_PATH)

from google.colab import files
files.download(EVAL_TEMPLATE_PATH)

# ============================================================
# 11. Загрузка размеченного файла
# ============================================================

from google.colab import files

uploaded = files.upload()
eval_file = list(uploaded.keys())[0]

if eval_file.endswith(".xlsx"):
    eval_df = pd.read_excel(eval_file)
elif eval_file.endswith(".csv"):
    eval_df = pd.read_csv(eval_file, sep=",", encoding="utf-8-sig")
else:
    raise ValueError("Поддерживаются файлы .xlsx и .csv")

print("Загружен файл:", eval_file)
print("Колонки:", list(eval_df.columns))
display(eval_df.head())

# ============================================================
# 12. Метрики eval
# ============================================================

def split_products(text):
    if pd.isna(text):
        return set()

    text = str(text).lower().replace("ё", "е")
    parts = text.split(PRODUCT_SEPARATOR)
    items = []

    for part in parts:
        part = re.sub(r"\s+", " ", part).strip().strip('"').strip("'")
        if part:
            items.append(part)

    return set(items)


def score_row(predicted_text, true_text):
    predicted = split_products(predicted_text)
    true = split_products(true_text)

    correct = sorted(predicted & true)
    missed = sorted(true - predicted)
    hallucinated = sorted(predicted - true)

    precision = len(correct) / len(predicted) if predicted else (1.0 if not true else 0.0)
    recall = len(correct) / len(true) if true else (1.0 if not predicted else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "correct": f"{PRODUCT_SEPARATOR} ".join(correct),
        "missed": f"{PRODUCT_SEPARATOR} ".join(missed),
        "hallucinated": f"{PRODUCT_SEPARATOR} ".join(hallucinated),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_predicted": len(predicted),
        "num_true": len(true),
        "num_correct": len(correct),
        "num_missed": len(missed),
        "num_hallucinated": len(hallucinated),
    }


def evaluate_predictions(eval_df):
    required_columns = ["predicted_products", "true_products"]
    missing_columns = [col for col in required_columns if col not in eval_df.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют колонки: {missing_columns}")

    scores = [
        score_row(row.get("predicted_products", ""), row.get("true_products", ""))
        for _, row in eval_df.iterrows()
    ]

    score_df = pd.concat(
        [eval_df.reset_index(drop=True), pd.DataFrame(scores).reset_index(drop=True)],
        axis=1,
    )

    summary = {
        "num_images": len(score_df),
        "mean_precision": score_df["precision"].mean(),
        "mean_recall": score_df["recall"].mean(),
        "mean_f1": score_df["f1"].mean(),
        "total_predicted": score_df["num_predicted"].sum(),
        "total_true": score_df["num_true"].sum(),
        "total_correct": score_df["num_correct"].sum(),
        "total_missed": score_df["num_missed"].sum(),
        "total_hallucinated": score_df["num_hallucinated"].sum(),
    }

    return score_df, summary


score_df, summary = evaluate_predictions(eval_df)

print("Итоговые метрики:")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

display(score_df)
score_df.to_csv(EVAL_RESULTS_PATH, index=False, encoding="utf-8-sig")
print("Результаты сохранены:", EVAL_RESULTS_PATH)

# ============================================================
# 13. Анализ ошибок
# ============================================================

def counter_from_column(series):
    counter = Counter()
    for text in series.fillna(""):
        for item in split_products(text):
            counter[item] += 1
    return counter


missed_counter = counter_from_column(score_df["missed"])
hallucinated_counter = counter_from_column(score_df["hallucinated"])

print("Частые пропуски:")
for item, count in missed_counter.most_common(20):
    print(f"{item}: {count}")

print("\nЧастые лишние предсказания:")
for item, count in hallucinated_counter.most_common(20):
    print(f"{item}: {count}")

plot_df = score_df.copy()
plot_df["image_num"] = range(1, len(plot_df) + 1)

plt.figure(figsize=(10, 5))
plt.plot(plot_df["image_num"], plot_df["precision"], marker="o", label="precision")
plt.plot(plot_df["image_num"], plot_df["recall"], marker="o", label="recall")
plt.plot(plot_df["image_num"], plot_df["f1"], marker="o", label="f1")
plt.xlabel("Номер изображения")
plt.ylabel("Значение")
plt.ylim(0, 1.05)
plt.title("Метрики распознавания продуктов")
plt.legend()
plt.grid(True)
plt.show()

files.download(EVAL_RESULTS_PATH)
