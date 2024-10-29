import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import google.generativeai as genai
import csv
import os
import random
from google.generativeai.types import HarmBlockThreshold, HarmCategory
from customtkinter import CTkImage
from tkinter import messagebox
import gc
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler, DPMSolverMultistepScheduler
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# 常數
APP_NAME = "Character Maker"
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 1000
IS_ENABLE_CREATE_IMAGE = False


#CSV_FILES = ['/mnt/c/steven/prompt_anime.csv'] # 指定CSV檔案清單
CSV_FILES = ['d:/prompt_anime.csv'] # 指定CSV檔案清單
SAVE_DIR = "d:/images/1" # 設定存檔目錄
#SAVE_DIR = '/mnt/c/steven/images/1'
#IMAGE_ROOT_PATH = "/mnt/c/steven/images/"
IMAGE_ROOT_PATH = "d:/images/"
IMAGE_PATH = "/1"
IMAGE_FILE = "image.jpg"
IMAGE_WIDTH = 512  # 生成圖片寬度
IMAGE_HEIGHT = 512  # 生成圖片高度
IMAGE_MAX_WIDTH = 1024  # 定義圖片最大顯示寬
IMAGE_MAX_HEIGHT = 600  # 定義圖片最大顯示高
CHECKBOX_LENGTH = 6
RANDOM_CHANCE = 95


# Diffusion 產圖設定
# 設置基本提示詞
Model_name = "Meina/MeinaPastel_V7"
BASE_PROMPT = """,full body , extremely detailed eyes and face, delicate eyes, perfect face, detailed face, best quality, very aesthetic, solo,
                    highly detailed, masterpiece:1.2, absurdres, panorama, clear focus, moody, mysterious,
                    amazing quality, highres, illustration, wallpaper, CG, reality ray tracing,"""
BASE_NEGATIVE_PROMPT = "imperfect eyes,easynegative, badhandv4, low quality:1.4,  deformed pupils, imperfect face,"
NUM_INFERENCE_STEPS = 30  # 生成參數: 迭代次數
GUIDANCE_SCALE = 5  # 生成參數: 調整指導比例，範圍通常在 1 到 20 之間

output_image = None  # 將在生成時賦值
prompt = ""  # 用於儲存提示詞
negative_prompt = ""  # 用於儲存負面提示詞
response = ""  # 用於儲存響應

# Gemini 保護設定
safety_settings={
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

ATTR_MAP = {"Gender" : "性別", "Age" : "年齡", "Race" : "種族", "Hair_Color" : "髮色", "Occupation" : "職業", "Cloth_Color" : "服裝顏色", "Hair_Length" : "髮長", "Hair_Style" : "髮型", "Bangs" : "瀏海", "Eyes" : "眼睛", "Eye_Color" : "眼睛顏色"}


class Character:
    def __init__(self):
        print("init")
        self.image = IMAGE_ROOT_PATH + IMAGE_PATH + IMAGE_FILE
        self.describe = "人物描述"
        self.describeOld = "人物描述"
        self.data = ""
        self.isCreating = False

# 圖像處理
# =============================================
pipe = None # 用於存儲加載的模型
original_image = Image.open(IMAGE_ROOT_PATH + IMAGE_PATH + "/" + IMAGE_FILE)

def load_model():
    global pipe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(Model_name, safety_checker=None, torch_dtype=torch.float16)
    # 設置調度器
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # 啟用 CPU 卸載和 Xformers 記憶體高效注意力
    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)  # 將模型載入到 GPU（如果可用）

def image_for_ui(data, w, h):
    global pipe  # 使用全局變數 pipe
    # 確保模型已經加載
    if pipe is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # 職業和種族列表
    ui_prompts = data
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 在這裡定義 device

    def get_prompt_embeddings(pipe, prompt, negative_prompt, device):
        # 確保 pipe 有 tokenizer 屬性
        if not hasattr(pipe, 'tokenizer'):
            raise ValueError("The pipeline does not have a tokenizer.")
        max_length = pipe.tokenizer.model_max_length
        # 直接將提示詞轉換為嵌入
        input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).input_ids.to(device)
        negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length).input_ids.to(device)
        # 獲取提示詞和負面提示詞的嵌入
        prompt_embeds = pipe.text_encoder(input_ids)[0]  # 這裡假設返回的第一個元素是嵌入
        neg_embeds = pipe.text_encoder(negative_ids)[0]  # 同上
        return prompt_embeds, neg_embeds

    # 整合生成的詞彙
    prompt = ui_prompts + BASE_PROMPT
    negative_prompt = BASE_NEGATIVE_PROMPT

    # 使用 get_prompt_embeddings 函數生成圖像
    prompt_embeds, neg_embeds = get_prompt_embeddings(pipe, prompt, negative_prompt, device=device)

    # 生成圖像
    output = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        height=h,
        width=w,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE  # 加入 guidance_scale
    ).images[0]

    return output, prompt, negative_prompt

# =============================================


# 在執行之前先加載模型
if IS_ENABLE_CREATE_IMAGE:
    load_model()

# 初始化 customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

# 創建主視窗
app = ctk.CTk()
app.geometry(str(WINDOW_WIDTH) + "x" + str(WINDOW_HEIGHT))
app.title(APP_NAME)

# 創建左上框架
frame_left = ctk.CTkFrame(master=app)
frame_left.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
frame_left.configure(
    border_width=2,
    border_color="#666666"
)

# 創建左中框架
frame_left_middle = ctk.CTkFrame(master=app)
frame_left_middle.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=(5, 10))
frame_left_middle.configure(
    border_width=2,
    border_color="#666666"
)

# 創建左下框架
frame_left_bottom = ctk.CTkFrame(master=app)
frame_left_bottom.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=(5, 10))
frame_left_bottom.configure(
    border_width=2,
    border_color="#666666"
)

# 創建右側框架
frame_right = ctk.CTkFrame(master=app)
frame_right.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=(5, 10), pady=10)
frame_right.configure(
    border_width=2,
    border_color="#666666"
)

# 設定行列權重，使框架佔據適當的空間
app.grid_rowconfigure(0, weight=1)  # 第一行（上）可擴展
app.grid_rowconfigure(1, weight=1)  # 第二行（中）可擴展
app.grid_rowconfigure(2, weight=1)  # 第三行（下）可擴展
app.grid_columnconfigure(0, weight=1)  # 第一列（左）可擴展
app.grid_columnconfigure(1, weight=2)  # 第二列（右）擴展更多


# 善惡陣營
def getAlignments():
    alignments = ["Lawful Good", "Neutral Good", "Chaotic Good", "Lawful Neutral", "True Neutral",
                  "Chaotic Neutral", "Lawful Evil", "Neutral Evil", "Chaotic Evil"]
    num = random.randint(0, len(alignments) - 1)
    return alignments[num]


# 串接 Gemini
def generate_describe():
    data = character.data
    #print(data)
    genai.configure(api_key="AIzaSyBmsMF-uiZhLtC43SHKz1nWPMC2WRU3z9c")
    model = genai.GenerativeModel('gemini-1.5-flash')
    data += "," + getAlignments()

    try:
        response = model.generate_content(
            "請寫一個故事人物的說明，名字請依照描述隨機給予，並用繁體中文顯示，以下是人物的相關設定:" + data,
            safety_settings=safety_settings,
            generation_config=genai.GenerationConfig(
                temperature=1,
            ),
        )
        print(response.text)
        result_area.delete("1.0", tk.END)
        result_area.insert("1.0", response.text)
        character.isCreating = False
        character.describe = response.text
        character.describeOld = response.text
    except Exception as e:
        print("發生錯誤:", e)
        result_area.delete("1.0", tk.END)
        result_area.insert("1.0", e)
        character.isCreating = False
        character.describe = e
        character.describeOld = e
    #print("====================")
    return response.text


def load_character_image(new_image = original_image):
    global original_image  # 使用全局變數來保存原始圖片
    flist = os.listdir(IMAGE_ROOT_PATH)
    length = len(flist)
    flist.sort()  # 排序
    file = IMAGE_ROOT_PATH + flist[length - 1] + "/" + IMAGE_FILE
    print(file)
    character.image = file

    try:
        # 只在第一次讀取圖片時從硬碟加載圖片
        if original_image is None:
            original_image = Image.open(character.image)  # 打開圖片並保存為原始圖片
        else:
            original_image = new_image
        # 根據框架的寬度進行縮放，但不需要重新讀取圖片
        resize_and_display_image()

    except Exception as e:
        print(f"圖片加載失敗: {e}")

def resize_and_display_image():
    global original_image  # 使用已加載的圖片進行縮放

    # 取得框架的寬度
    frame_width = frame_right.winfo_width()

    # 計算目標寬度和高度，並限制尺寸
    target_width = min(frame_width, IMAGE_MAX_WIDTH)  # 限制寬度不超過 MAX_SIZE
    target_height = int(original_image.height * (target_width / original_image.width))  # 保持圖片比例

    # 如果圖片高度超過 MAX_SIZE，則進一步縮小
    if target_height > IMAGE_MAX_HEIGHT:
        target_height = IMAGE_MAX_HEIGHT
        target_width = int(original_image.width * (target_height / original_image.height))  # 調整寬度以保持比例

    # 使用 CTkImage 來顯示縮放後的圖片
    ctk_image = ctk.CTkImage(light_image=original_image, size=(target_width, target_height))

    # 配置 Label 並顯示圖片
    character_image_label.configure(image=ctk_image)
    character_image_label.image = ctk_image  # 保存圖片引用，防止被垃圾回收

def resize_image(event):
    resize_and_display_image()  # 每次框架大小變化時只進行圖片縮放，而不重新讀取圖片

# 綁定框架大小變化事件
frame_right.bind("<Configure>", resize_image)

def getComboBoxIndex(combobox):
    selected_value = combobox.get()  # 獲取選中項目的值
    values = combobox.cget("values")  # 獲取所有選項
    selected_index = values.index(selected_value)  # 查找索引
    #print("getComboBoxIndex = " + str(selected_value) + "," + str(values) + "," + str(selected_index))
    return selected_index

def getComboBoxRealValue(name):
    combobox = charItems[name]
    index = getComboBoxIndex(combobox)
    #print(charData[name])
    #print(charData[name][index])
    return charData[name][index]

def getCheckBoxRealValue(name):
    text = ""
    items = charItems[name]
    for index in range(len(items)):
        item = items[index]
        if item.get() == 1:
            text += "," + charData[name][index]
            #print(item.cget("text"))
            #print(charData[name][index])
    return text

def getPrompt():
    data = getComboBoxRealValue("Gender")
    if charItems["Age"].get() != "None":
        data += "," + getComboBoxRealValue("Age")
    if charItems["Race"].get() != "None":
        data += "," + getComboBoxRealValue("Race")
    if charItems["Hair_Color"].get() != "None":
        data += "," + getComboBoxRealValue("Hair_Color")
    if charItems["Occupation"].get() != "None":
        data += "," + getComboBoxRealValue("Occupation")
    if charItems["Hair_Length"].get() != "None":
        data += "," + getComboBoxRealValue("Hair_Length")
    if charItems["Hair_Style"].get() != "None":
        data += "," + getComboBoxRealValue("Hair_Style")
    if charItems["Bangs"].get() != "None":
        data += "," + getComboBoxRealValue("Bangs")
    if charItems["Eyes"].get() != "None":
        data += "," + getComboBoxRealValue("Eyes")
    if charItems["Eye_Color"].get() != "None":
        data += "," + getComboBoxRealValue("Eye_Color")
    if charItems["Cloth_Color"].get() != "None":
        data += "," + getComboBoxRealValue("Cloth_Color")

    data += getCheckBoxRealValue("Accessories")
    
    return data
    

# 創建人物
def generate_character():
    print("=== generate_character ===")
    character.isCreating = True

    #data = str(charItems["Gender"].get())
    data = getComboBoxRealValue("Gender")
    if charItems["Age"].get() != "None":
        data += ",age is " + getComboBoxRealValue("Age")
    if charItems["Race"].get() != "None":
        data += ",race is " + getComboBoxRealValue("Race")
    if charItems["Hair_Color"].get() != "None":
        data += ",hair color is " + getComboBoxRealValue("Hair_Color")
    if charItems["Occupation"].get() != "None":
        data += ",occupation is" + getComboBoxRealValue("Occupation")
    if charItems["Hair_Length"].get() != "None":
        data += ",hair length is " + getComboBoxRealValue("Hair_Length")
    if charItems["Hair_Style"].get() != "None":
        data += ",hair style is " + getComboBoxRealValue("Hair_Style")
    if charItems["Bangs"].get() != "None":
        data += ",bangs is " + getComboBoxRealValue("Bangs")
    if charItems["Eyes"].get() != "None":
        data += ",eyes is " + getComboBoxRealValue("Eyes")
    if charItems["Eye_Color"].get() != "None":
        data += ",eyes color is " + getComboBoxRealValue("Eye_Color")
    if charItems["Cloth_Color"].get() != "None":
        data += ",cloth color is " + getComboBoxRealValue("Cloth_Color")
    
    data += getCheckBoxRealValue("Accessories")
    
    print(str(data))
    print("=========================")
    character.data = data
    character.describe = "人物創造中"
    character.describeOld = "人物創造中"
    #character.isCreating = False
    result_area.delete("1.0", tk.END)
    result_area.insert("1.0", text=data)
    prompts_area.delete("1.0", tk.END)  # 清空 prompts_area
    prompts_area.insert("1.0", text=data)  # 將最新的 data 插入到 prompts_area
    t = threading.Thread(target = do_generate_character)
    t.start()

def setRandomData(name):
    num = random.randint(0, len(charData[name + "_TW"]) - 1)
    charItems[name].set(charData[name + "_TW"][num])

def setRandomData2(name):
    for item in charItems[name]:
        num = random.randint(0, 100)
        #print(item.cget("text") + ":" + str(item.get()))
        item.deselect()
        if num > RANDOM_CHANCE:
            item.select()

# 隨機生成人物
def random_generate_character():
    print("=== generate_character ===")
    character.isCreating = True

    setRandomData("Gender")
    setRandomData("Age")
    setRandomData("Race")
    setRandomData("Hair_Color")
    setRandomData("Occupation")
    setRandomData("Hair_Length")
    setRandomData("Hair_Style")
    setRandomData("Bangs")
    setRandomData("Eyes")
    setRandomData("Eye_Color")
    setRandomData("Cloth_Color")
    setRandomData("Gender")
    setRandomData2("Accessories")

    data = str(getComboBoxRealValue("Gender"))
    data += ",age is " + getComboBoxRealValue("Age")
    data += ",race is " + getComboBoxRealValue("Race")
    data += ",hair color is " + getComboBoxRealValue("Hair_Color")
    data += ",occupation is" + getComboBoxRealValue("Occupation")
    data += ",hair length is " + getComboBoxRealValue("Hair_Length")
    data += ",hair style is " + getComboBoxRealValue("Hair_Style")
    data += ",bangs is " + getComboBoxRealValue("Bangs")
    data += ",eyes is " + getComboBoxRealValue("Eyes")
    data += ",eyes color is " + getComboBoxRealValue("Eye_Color")
    data += ",cloth color is " + getComboBoxRealValue("Cloth_Color")
    data += getCheckBoxRealValue("Accessories")

    print(str(data))
    print("=========================")
    character.data = data
    character.describe = "角色設定產生中"
    character.describeOld = "角色設定產生中"
    t = threading.Thread(target = do_generate_character)
    t.start()


def do_generate_character():
    global output_image
    global prompt
    global negative_prompt
    global response
    # 初始化所需的變量
    data_make = getPrompt()
    #print("data_make = " + data_make)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交兩個任務到執行緒池
        future_response = executor.submit(generate_describe)
        future_image = executor.submit(image_for_ui, data_make, IMAGE_WIDTH, IMAGE_HEIGHT)

        # 使用 as_completed 來處理最先完成的任務
        for future in as_completed([future_response, future_image]):
            if future == future_image:
                output_image, prompt, negative_prompt = future.result()  # 獲取結果
                print("do_generate_character = " + prompt + "--------" + negative_prompt)
                load_character_image(output_image)
            elif future == future_response:
                response = future.result()  # 獲取描述的響應

def save_function():
    # 檢查目錄並創建
    save_dir = IMAGE_ROOT_PATH + IMAGE_PATH + "/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 設定檔名
    save_path = os.path.join(save_dir, f"{file_name}.jpg")

    # 儲存圖片
    if output_image is not None:
        output_image.save(save_path)  # 確保 output_image 是有效的

    # 設定文本檔的路徑
    txt_path = os.path.join(save_dir, f"{file_name}.txt")


    prompt = getPrompt() + BASE_PROMPT
    negative_prompt = BASE_NEGATIVE_PROMPT

    print("Save image = " + save_dir)
    print("Save story = " + txt_path)

    # 寫入文件內容
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"時間: {file_name}\n\n")  # 寫入時間: file_name
        f.write(f"提示詞:\n{prompt}\n\n")  # 寫入 prompt
        f.write(f"負面提示詞:\n{negative_prompt}\n\n")  # 寫入 negative_prompt
        f.write(f"故事:\n{character.describe}\n\n")  # 寫入 character.describe

def update_ui():
    if character.isCreating:
        character.describe += ".."
        generate_button.configure(state=ctk.DISABLED)
        generate_button2.configure(state=ctk.DISABLED)
    else:
        generate_button.configure(state=ctk.NORMAL)
        generate_button2.configure(state=ctk.NORMAL)

    if character.describe != character.describeOld:
        result_area.delete("1.0", tk.END)
        result_area.insert("1.0", character.describe)
        character.describeOld = character.describe  # 更新旧描述以避免重复更新

    # 使用 app.after 调度下一个更新
    app.after(1000, update_ui)  # 每1000毫秒（1秒）更新一次

# 讀取 .csv 檔
def read_and_merge_csvs(csv_files):
    print("read_and_merge_csvs")
    merged_data = {}

    for file in csv_files:
        with open(file, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader: # 讀取每個檔案的資料
                for key, value in row.items():
                    if value.strip(): # 去除空資料
                        if key not in merged_data: # 初始化 dict 和 set，如果 key 不存在
                            #merged_data[key] = set()
                            merged_data[key] = []
                        #merged_data[key].add(value.strip().lower()) # 將每行的值加入對應的 key (使用 set 避免重複)
                        merged_data[key].append(value.strip().lower())

    merged_data = {key: list(values) for key, values in merged_data.items()} # 將 set 轉回 list
    return merged_data

def getCsvData():
    merged_data = read_and_merge_csvs(CSV_FILES) # 讀取並合併CSV資料
    # 刪除不使用的屬性
    weapon = merged_data.pop("Weapon", None)
    merged_data.pop("Special", None)
    merged_data.pop("Body_Shape", None)
    merged_data.pop("Posture", None)
    #merged_data.pop("Hair_Style", None)
    merged_data.pop("Eye_Size", None)
    merged_data.pop("Pupil_Color", None)
    merged_data.pop("Negative_Prompt", None)
    merged_data.pop("Background", None)
    merged_data.pop("Mouth_Shape", None)
    merged_data.pop("Mood", None)
    merged_data.pop("Hat", None)
    merged_data.pop("Top_Wear", None)
    merged_data.pop("Bottom_Wear", None)
    merged_data.pop("Skin", None)
    merged_data.pop("Style", None)
    merged_data.pop("lora", None)
    #merged_data.pop("Hair_Length", None)

    if False:
        print("===============================")
        for key, value in merged_data.items():
            print(f"{key}")
        print("===============================")
    
    # 更新使用的屬性
    merged_data["Eye_Color"] = merged_data["Color"]
    merged_data["Cloth_Color"] = merged_data["Color"]
    merged_data["Eye_Color_TW"] = merged_data["Color_TW"]
    merged_data["Cloth_Color_TW"] = merged_data["Color_TW"]
    #merged_data.pop("Color", None)
    
    none_data = ["None"]
    for key, value in merged_data.items():
        if (key != "Gender" and key != "Gender_TW" and key != "Accessories" and key != "Accessories_TW"):
            merged_data[key] = none_data + merged_data[key]
    
    return merged_data

def update_prompts_area():
    # 更新 prompts_area 的內容
    data = str(charItems["Gender"].get())

    # 確保每個項目的選擇都能正確獲取
    if charItems["Age"].get() != "None":
        data += ", " + charItems["Age"].get()
    if charItems["Race"].get() != "None":
        data += ", " + charItems["Race"].get()
    if charItems["Hair_Color"].get() != "None":
        data += ", " + charItems["Hair_Color"].get()
    if charItems["Occupation"].get() != "None":
        data += ", " + charItems["Occupation"].get()
    if charItems["Hair_Length"].get() != "None":
        data += ", " + charItems["Hair_Length"].get()
    if charItems["Hair_Style"].get() != "None":
        data += ", " + charItems["Hair_Style"].get()
    if charItems["Bangs"].get() != "None":
        data += ", " + charItems["Bangs"].get()
    if charItems["Eyes"].get() != "None":
        data += ", " + charItems["Eyes"].get()
    if charItems["Eye_Color"].get() != "None":
        data += ", " + charItems["Eye_Color"].get()
    if charItems["Cloth_Color"].get() != "None":
        data += ", " + charItems["Cloth_Color"].get()

    #print("update_prompts_area = " + data)

    # 追加配件
    accessories = []
    for item in charItems["Accessories"]:
        if item.get() == 1:  # 確認該項目是否被選中
            accessories.append(item.cget("text"))

    if accessories:  # 如果有選中的配件，將它們添加到數據中
        data +=  ", " + ", ".join(accessories)

    # 更新 prompts_area
    prompts_area.delete("1.0", tk.END)  # 清空文本框
    prompts_area.insert("1.0", text=data)  # 更新文本框內容

    # 每 500 毫秒再次调用 update_prompts_area
    frame_left.after(1000, update_prompts_area)
    return data

# 讀取 CSV 檔
# =============================================
charData = getCsvData()
print("csv data kind = " + str(len(charData)))
print("===============================")
for key, value in charData.items():
    print(f"{key}")
print("===============================")


# =============================================
# 創建 UI
# =============================================
# 右側 Frame
# -----------------------
# 圖片 & 人物描述
character_image_label = ctk.CTkLabel(master=frame_right, text="")
character_image_label.pack(side="top", fill="x", padx=10, pady=10)
result_area = ctk.CTkTextbox(frame_right, height=10, font=("Arial", 16, "bold"))
result_area.insert("1.0", "人物故事背景")
#result_area.configure(fg_color="white", text_color="black")  # 設定背景色和文字顏色
result_area.pack(side="bottom", fill="both", padx=10, pady=10, expand=True)

def createItem(name, rowNum, columnNum, sticky="w"):
    label = ctk.CTkLabel(master=frame_left, text=str(ATTR_MAP.get(name, name)), font=("Arial", 20, "bold"))
    label.grid(row=rowNum, column=columnNum * 2, padx=5, pady=10, sticky=sticky)  # Label 在每列的第0或2行，左對齊
    comboBox = ctk.CTkComboBox(master=frame_left, values=charData[name + "_TW"], font=("Arial", 16, "bold"), width=220)
    comboBox.grid(row=rowNum, column=columnNum * 2 + 1, padx=50, pady=10)  # ComboBox 緊隨 Label
    charItems[name] = comboBox  # 將 comboBox 儲存到 charItems 中以便後續使用
    return comboBox

def createCheckboxItem(name, rowNum):
    label = ctk.CTkLabel(master=frame_left_bottom, text=str("其他設定"), font=("Arial", 20, "bold"))
    label.grid(row=rowNum, column=0, padx=10, pady=5, sticky="w")
    items = charData[name + "_TW"]
    list = []
    for index in range(len(items)):
        if index % CHECKBOX_LENGTH == 0:
            rowNum += 1
        checkbox = ctk.CTkCheckBox(master=frame_left_bottom, text=str(items[index]), font=("Arial", 14, "bold"), command=update_prompts_area)
        checkbox.grid(row=rowNum, column=index % CHECKBOX_LENGTH, padx=10, pady=5, sticky="w")
        list.append(checkbox)
    charItems[name] = list  # 將複選框儲存到 charItems 中
    return list

# 左側 Frame
# -----------------------
charItems = {}
# 添加標題 Label，顯示 "角色屬性" 或其他適當的標題
titleLabel = ctk.CTkLabel(frame_left, text="角色屬性", font=("Arial", 20, "bold"), anchor="w")
titleLabel.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")  # columnspan=2 讓標題橫跨兩列

# 第一列（column = 0）
charItems["Gender"] = createItem("Gender", 1, 0)  # 性別
charItems["Age"] = createItem("Age", 2, 0)
charItems["Race"] = createItem("Race", 3, 0)
charItems["Hair_Color"] = createItem("Hair_Color", 4, 0)
charItems["Occupation"] = createItem("Occupation", 5, 0)
charItems["Cloth_Color"] = createItem("Cloth_Color", 6, 0)

# 第二列（column = 1）
charItems["Hair_Length"] = createItem("Hair_Length", 1, 1)
charItems["Hair_Style"] = createItem("Hair_Style", 2, 1)
charItems["Bangs"] = createItem("Bangs", 3, 1)
charItems["Eyes"] = createItem("Eyes", 4, 1)
charItems["Eye_Color"] = createItem("Eye_Color", 5, 1)

charItems["Accessories"] = createCheckboxItem("Accessories", 11)  # 配件

# 在 frame_left_middle 添加一個說明的標題 Label 並對齊至最左方
label = ctk.CTkLabel(frame_left_middle, text="提示詞(Prompts)", font=("Arial", 20, "bold"), anchor="w")  # anchor="w" 左對齊
label.pack(side="top", padx=0, pady=0, fill="x")  # fill="x" 保證填滿橫向空間，確保文字靠左

# 在 frame_left_middle 加入文本框
prompts_area = ctk.CTkTextbox(frame_left_middle, height=10, font=("Arial", 16, "bold"))
prompts_area.insert("1.0", "提示詞")
# prompts_area.configure(fg_color="white", text_color="black")  # 設定背景色和文字顏色
prompts_area.pack(side="top", fill="both", padx=10, pady=10, expand=True)

# 使用 pack 放置按鈕
button_frame = ctk.CTkFrame(frame_left_middle)  # 創建一個新的框架用於按鈕
button_frame.pack(side="bottom", fill="x")  # 將按鈕框架放在底部

generate_button2 = ctk.CTkButton(master=button_frame, text="隨機生成角色", font=("Arial", 16, "bold"), command=random_generate_character)
generate_button2.pack(side="right", padx=5, pady=5)  # 將第二個按鈕放在右側

generate_button = ctk.CTkButton(master=button_frame, text="生成角色", font=("Arial", 16, "bold"), command=generate_character)
generate_button.pack(side="right", padx=5, pady=5)  # 將按鈕放在右側

# 新的按鈕框架，放置在左側
left_button_frame = ctk.CTkFrame(master=button_frame)
left_button_frame.pack(side="left", padx=5, pady=5)

# 存檔目錄的函數

def set_save_directory():
    print(f"存檔目錄已設定為: {SAVE_DIR}")

    # 檢查目錄是否存在
    if os.path.exists(SAVE_DIR):
        # 在 Windows 中使用 `explorer.exe` 來開啟資料夾
        windows_path = SAVE_DIR.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        subprocess.run(['explorer.exe', windows_path])
    else:
        print("目錄不存在！")

def clear_options():
    global data
    data = ""  # 清空變數 data

    # 清空或重置 charItems 中的選項
    #charItems["Gender"].set("None")
    charItems["Age"].set("None")
    charItems["Race"].set("None")
    charItems["Hair_Color"].set("None")
    charItems["Occupation"].set("None")
    charItems["Hair_Length"].set("None")
    charItems["Hair_Style"].set("None")
    charItems["Bangs"].set("None")
    charItems["Eyes"].set("None")
    charItems["Eye_Color"].set("None")
    charItems["Cloth_Color"].set("None")

    # 清空配件選項
    for item in charItems["Accessories"]:
        item.deselect()  # 將每個配件選項設置為未選中狀態

    # 清空文本框
    prompts_area.delete("1.0", tk.END)

    print("提示詞已重設")

# 「清除選項」按鈕，紫色，大小為原來按鈕的一半
clear_options_button = ctk.CTkButton(
    master=left_button_frame,
    text="重設提示詞",
    font=("Arial", 16, "bold"),
    width=int(generate_button.winfo_width() / 2),  # 一半的大小
    fg_color="purple",  # 紫色背景
    command=clear_options  # 呼叫清除選項的函數
)
clear_options_button.pack(side="left", padx=5, pady=5)

# 新的「存檔」按鈕，紫色，大小為「重設提示詞」的一半
save_button = ctk.CTkButton(
    master=left_button_frame,
    text="存檔",
    font=("Arial", 16, "bold"),
    width=int(clear_options_button.winfo_width() / 2),  # 寬度為「重設提示詞」的一半
    fg_color="purple",  # 紫色背景
    command=save_function  # 呼叫存檔的函數
)
save_button.pack(side="left", padx=5, pady=5)

# 「存檔目錄」按鈕，紫色，大小為原來按鈕的一半
save_dir_button = ctk.CTkButton(
    master=left_button_frame,
    text="存檔目錄",
    font=("Arial", 16, "bold"),
    width=int(generate_button.winfo_width() / 2),  # 一半的大小
    fg_color="purple",  # 紫色背景
    command=set_save_directory  # 呼叫設定存檔目錄的函數
)
save_dir_button.pack(side="left", padx=5, pady=5)

# 創建角色
# =============================================
character = Character()
load_character_image()
# =============================================

# 更新检查
update_prompts_area()
update_ui()

# 自定義關閉事件
def on_closing():
    if messagebox.askokcancel("退出", "你確定要退出嗎?"):
        gc.collect()
        app.destroy()  # 關閉視窗並結束程式執行

# 綁定「X」按鈕關閉事件
app.protocol("WM_DELETE_WINDOW", on_closing)

app.mainloop() # 啟動應用程式