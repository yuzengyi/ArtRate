import base64
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import openai
from io import BytesIO
import ast
import pandas as pd
import re
#
# from image_online import upload_image_to_server

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# 设置 OpenAI 的 API key 和代理
proxy = "http://127.0.0.1:7890"
os.environ["http_proxy"] = proxy
os.environ["https_proxy"] = proxy
# 读取文本文件中的 API key
with open('key.txt', 'r') as file:
    api_key = file.read().strip()  # 使用 strip() 去除可能的前后空白字符
client = openai.OpenAI(api_key=api_key)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查文件是否存在
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath)
            file.save(filepath)
            base64_image = encode_image(filepath)
            # 调用 OpenAI API 进行图片分析
            # print(base64_image)
            zero_response = zero_shot_analyze_image(base64_image)
            zero_response_dict = safe_eval_dict(zero_response)
            #字符串转换成字典
            # zero_response_dict = ast.literal_eval(safe)
            radar_chart = plot_radar_chart(list(zero_response_dict.values()))
            #############one-shot############
            reference_image_url = "https://image.familystudy.cn/file/8a38c107b17a409f9679cf2486b2e16c.jpg"
            #Realistic: 5, Deformation: 1, Imagination: 2, Color Richness: 4,
            # Color Contrast: 4, Line Combination: 4, Line Texture: 4, Picture Organization: 5, Transformation: 1
            reference_scores = {
                "Realistic": 2,
                "Deformation": 3,
                "Imagination": 2,
                "Color Richness": 4,
                "Color Contrast": 4,
                "Line Combination": 3,
                "Line Texture": 3,
                "Picture Organization": 2,
                "Transformation": 3
            }
            one_response = one_shot_analyze_image(base64_image, reference_image_url, reference_scores)
            one_response_dict = safe_eval_dict(one_response)
            one_radar_chart = plot_radar_chart(list(one_response_dict.values()))
            #########Three-shot-reference############
            reference_image_url1 = "https://image.familystudy.cn/file/8a38c107b17a409f9679cf2486b2e16c.jpg"
            reference_scores1 = {
                "Realistic": 2,
                "Deformation": 3,
                "Imagination": 2,
                "Color Richness": 4,
                "Color Contrast": 4,
                "Line Combination": 3,
                "Line Texture": 3,
                "Picture Organization": 2,
                "Transformation": 3
            }
            reference_image_url2 = "https://image.familystudy.cn/file/06d662beb7394cff9a31b22ad93899d4.jpg"
            reference_scores2 = {
                "Realistic": 2,
                "Deformation": 5,
                "Imagination": 4,
                "Color Richness": 1,
                "Color Contrast": 5,
                "Line Combination": 5,
                "Line Texture": 5,
                "Picture Organization": 1,
                "Transformation": 0
            }
            reference_image_url3 = "https://image.familystudy.cn/file/bea383875a69480bb90b840021481eb3.jpg"
            reference_scores3 = {
                "Realistic": 3,
                "Deformation": 4,
                "Imagination": 3,
                "Color Richness": 3,
                "Color Contrast": 3,
                "Line Combination": 3,
                "Line Texture": 4,
                "Picture Organization": 1,
                "Transformation": 0
            }
            #########Three-shot############
            three_response = Three_shot_analyze_image(base64_image, reference_image_url1, reference_scores1,reference_image_url2, reference_scores2, reference_image_url3, reference_scores3)
            three_response_dict = safe_eval_dict(three_response)
            three_radar_chart = plot_radar_chart(list(three_response_dict.values()))
            ##### Excel #####
            df = pd.DataFrame({
                "Zero Response": zero_response_dict,
                "One Response": one_response_dict,
                "Three Response": three_response_dict
            })
            df = df.T
            excel_filename = re.sub(r'\.jpg|\.jpeg|\.png|\.bmp|\.gif$', '.xlsx', filename, flags=re.IGNORECASE)
            print(excel_filename)  # Outputs: 1_3.xlsx
            # Write the DataFrame to an Excel file
            df.to_excel(excel_filename, index=True)
            return render_template('result.html', zero_result=zero_response,image_data=base64_image,radar_chart=radar_chart, one_result = one_response, one_radar_chart=one_radar_chart,three_response=three_response, three_radar_chart=three_radar_chart)
    return render_template('upload.html')

def zero_shot_analyze_image(image_path):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Assess the Students Artwork using the following criteria.Give a score per criterion."
                           "Criterion 1: Realistic. It assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
                            "5:The artwork exhibits exceptional detail and precision in depicting realistic features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
                            "4:The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly realistic, minor discrepancies in perspective or detail might be noticeable."
                            "3:The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
                            "2:The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
                            "1:The artwork shows minimal attention to realistic details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
                            "Criterion 2: Deformation. It evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept. "
                            "5:The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
                            "4:The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
                            "3:The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
                            "2:The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
                            "1:The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
                            "Criterion 3: Imagination. This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
                            "5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
                            "4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
                            "3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
                            "2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
                            "1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
                            "Criterion 4: Color Richness. This criterion assesses the use and range of colors to create a visually engaging experience."
                            "5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
                            "4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
                            "3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
                            "2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
                            "1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
                            "Criterion 5: Color Contrast. This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
                            "5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
                            "4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
                            "3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
                            "2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
                            "1: The artwork lacks effective color contrast, making the piece visually unengaging."
                            "Criterion 6: Line Combination. This criterion assesses the integration and interaction of lines within the artwork."
                            "5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
                            "4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
                            "3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
                            "2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
                            "1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
                            "Criterion 7: Line Texture. This criterion evaluates the variety and execution of line textures within the artwork."
                            "5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
                            "4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
                            "3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
                            "2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
                            "1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
                            "Criterion 8: Picture Organization. This criterion evaluates the overall composition and spatial arrangement within the artwork."
                            "5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
                            "4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
                            "3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
                            "2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
                            "1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
                            "Criterion 9: Transformation. This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
                            "5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
                            "4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
                            "3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
                            "2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
                            "1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "output form: {Realistic:score,  Deformation:score, Imagination:score, Color Richness:score, Color Contrast:score, Line Combination:score, Line Texture:score, Picture Organization:score, Transformation:score}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_path}",
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def one_shot_analyze_image(image_path,reference_image_url, reference_scores):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Assess the Students Artwork using the following criteria.Give a score per criterion."
                           "Criterion 1: Realistic. It assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
                            "5:The artwork exhibits exceptional detail and precision in depicting realistic features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
                            "4:The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly realistic, minor discrepancies in perspective or detail might be noticeable."
                            "3:The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
                            "2:The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
                            "1:The artwork shows minimal attention to realistic details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
                            "Criterion 2: Deformation. It evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept. "
                            "5:The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
                            "4:The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
                            "3:The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
                            "2:The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
                            "1:The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
                            "Criterion 3: Imagination. This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
                            "5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
                            "4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
                            "3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
                            "2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
                            "1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
                            "Criterion 4: Color Richness. This criterion assesses the use and range of colors to create a visually engaging experience."
                            "5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
                            "4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
                            "3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
                            "2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
                            "1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
                            "Criterion 5: Color Contrast. This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
                            "5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
                            "4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
                            "3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
                            "2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
                            "1: The artwork lacks effective color contrast, making the piece visually unengaging."
                            "Criterion 6: Line Combination. This criterion assesses the integration and interaction of lines within the artwork."
                            "5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
                            "4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
                            "3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
                            "2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
                            "1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
                            "Criterion 7: Line Texture. This criterion evaluates the variety and execution of line textures within the artwork."
                            "5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
                            "4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
                            "3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
                            "2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
                            "1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
                            "Criterion 8: Picture Organization. This criterion evaluates the overall composition and spatial arrangement within the artwork."
                            "5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
                            "4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
                            "3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
                            "2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
                            "1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
                            "Criterion 9: Transformation. This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
                            "5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
                            "4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
                            "3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
                            "2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
                            "1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
            },
            {
                "role": "system",
                "content": "Give an example. The image URL is " + reference_image_url + " with scores: " + str(reference_scores)
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "output form: {Realistic:score,  Deformation:score, Imagination:score, Color Richness:score, Color Contrast:score, Line Combination:score, Line Texture:score, Picture Organization:score, Transformation:score}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_path}",
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content
def Three_shot_analyze_image(image_path,reference_image_url1, reference_scores1, reference_image_url2, reference_scores2,reference_image_url3, reference_scores3):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        top_p=1,
        messages=[
            {
                "role": "system",
                "content": "Assess the Students Artwork using the following criteria.Give a score per criterion."
                           "Criterion 1: Realistic. It assesses the accuracy of proportions, textures, lighting, and perspective to create a lifelike depiction."
                            "5:The artwork exhibits exceptional detail and precision in depicting realistic features. Textures and lighting are used masterfully to mimic real-life appearances with accurate proportions and perspective. The representation is strikingly lifelike, demonstrating advanced skills in realism."
                            "4:The artwork presents a high level of detail and accuracy in the portrayal of subjects. Proportions and textures are very well executed, and the lighting enhances the realism. Although highly realistic, minor discrepancies in perspective or detail might be noticeable."
                            "3:The artwork represents subjects with a moderate level of realism. Basic proportions are correct, and some textures and lighting effects are used to enhance realism. However, the depiction may lack depth or detail in certain areas."
                            "2:The artwork attempts realism but struggles with accurate proportions and detailed textures. Lighting and perspective may be inconsistently applied, resulting in a less convincing depiction."
                            "1:The artwork shows minimal attention to realistic details. Proportions, textures, and lighting are poorly executed, making the depiction far from lifelike."
                            "Criterion 2: Deformation. It evaluates the artist's ability to creatively and intentionally deform reality to convey a message, emotion, or concept. "
                            "5:The artwork demonstrates masterful use of deformation to enhance the emotional or conceptual impact of the piece. The transformations are thoughtful and integral to the artwork's message, seamlessly blending with the composition to engage viewers profoundly."
                            "4:The artwork effectively uses deformation to express artistic intentions. The modifications are well-integrated and contribute significantly to the viewer's understanding or emotional response. Minor elements of the deformation might detract from its overall effectiveness."
                            "3:The artwork includes noticeable deformations that add to its artistic expression. While these elements generally support the artwork's theme, they may be somewhat disjointed from the composition, offering mixed impact on the viewer."
                            "2:The artwork attempts to use deformation but does so with limited success. The deformations are present but feel forced or superficial, only marginally contributing to the artwork's expressive goals."
                            "1:The artwork features minimal or ineffective deformation, with little to no enhancement of the artwork's message or emotional impact. The attempts at deformation seem disconnected from the artwork's overall intent."
                            "Criterion 3: Imagination. This criterion evaluates the artist's ability to use their creativity to form unique and original ideas within their artwork."
                            "5: The artwork displays a profound level of originality and creativity, introducing unique concepts or interpretations that are both surprising and thought-provoking."
                            "4: The artwork presents creative ideas that are both original and nicely executed, though they may be similar to conventional themes."
                            "3: The artwork shows some creative ideas, but they are somewhat predictable and do not stray far from traditional approaches."
                            "2: The artwork has minimal creative elements, with ideas that are largely derivative and lack originality."
                            "1: The artwork lacks imagination, with no discernible original ideas or creative concepts."
                            "Criterion 4: Color Richness. This criterion assesses the use and range of colors to create a visually engaging experience."
                            "5: The artwork uses a wide and harmonious range of colors, each contributing to a vivid and dynamic composition."
                            "4: The artwork features a good variety of colors that are well-balanced, enhancing the visual appeal of the piece."
                            "3: The artwork includes a moderate range of colors, but the palette may not fully enhance the subject matter."
                            "2: The artwork has limited color variety, with a palette that does not significantly contribute to the piece's impact."
                            "1: The artwork shows poor use of colors, with a very restricted range that detracts from the visual experience."
                            "Criterion 5: Color Contrast. This criterion evaluates the effective use of contrasting colors to enhance artistic expression."
                            "5: The artwork masterfully employs contrasting colors to create a striking and effective visual impact."
                            "4: The artwork effectively uses contrasting colors to enhance visual interest, though the contrast may be less pronounced."
                            "3: The artwork has some contrast in colors, but it is not used effectively to enhance the artwork's overall appeal."
                            "2: The artwork makes minimal use of color contrast, resulting in a lackluster visual impact."
                            "1: The artwork lacks effective color contrast, making the piece visually unengaging."
                            "Criterion 6: Line Combination. This criterion assesses the integration and interaction of lines within the artwork."
                            "5: The artwork exhibits exceptional integration of line combinations, creating a harmonious and engaging visual flow."
                            "4: The artwork displays good use of line combinations that contribute to the overall composition, though some areas may lack cohesion."
                            "3: The artwork shows average use of line combinations, with some effective sections but overall lacking in cohesiveness."
                            "2: The artwork has minimal effective use of line combinations, with lines that often clash or do not contribute to a unified composition."
                            "1: The artwork shows poor integration of lines, with combinations that disrupt the visual harmony of the piece."
                            "Criterion 7: Line Texture. This criterion evaluates the variety and execution of line textures within the artwork."
                            "5: The artwork demonstrates a wide variety of line textures, each skillfully executed to enhance the piece's aesthetic and thematic elements."
                            "4: The artwork includes a good range of line textures, well executed but with some areas that may lack definition."
                            "3: The artwork features moderate variety in line textures, with generally adequate execution but lacking in detail."
                            "2: The artwork has limited line textures, with execution that does not significantly contribute to the artwork's quality."
                            "1: The artwork lacks variety and sophistication in line textures, resulting in a visually dull piece."
                            "Criterion 8: Picture Organization. This criterion evaluates the overall composition and spatial arrangement within the artwork."
                            "5: The artwork is impeccably organized, with each element thoughtfully placed to create a balanced and compelling composition."
                            "4: The artwork has a good organization, with a well-arranged composition that effectively guides the viewer's eye, though minor elements may disrupt the flow."
                            "3: The artwork has an adequate organization, but the composition may feel somewhat unbalanced or disjointed."
                            "2: The artwork shows poor organization, with a composition that lacks coherence and does not effectively engage the viewer."
                            "1: The artwork is poorly organized, with a chaotic composition that detracts from the piece's overall impact."
                            "Criterion 9: Transformation. This criterion assesses the artist's ability to transform traditional or familiar elements into something new and unexpected."
                            "5: The artwork is transformative, offering a fresh and innovative take on traditional elements, significantly enhancing the viewer's experience."
                            "4: The artwork successfully transforms familiar elements, providing a new perspective, though the innovation may not be striking."
                            "3: The artwork shows some transformation of familiar elements, but the changes are somewhat predictable and not highly innovative."
                            "2: The artwork attempts transformation but achieves only minimal success, with changes that are either too subtle or not effectively executed."
                            "1: The artwork lacks transformation, with traditional elements that are replicated without any significant innovation or creative reinterpretation."
            },
            {
                "role": "system",
                "content": "Please give a score per criterion for a new image based on these examples."
            },
            {
                "role": "system",
                "content": "Example 1: The image URL is " + reference_image_url1 + " with scores: " + str(
                    reference_scores1)
            },
            {
                "role": "system",
                "content": "Example 2: The image URL is " + reference_image_url2 + " with scores: " + str(
                    reference_scores2)
            },
            {
                "role": "system",
                "content": "Example 3: The image URL is " + reference_image_url3 + " with scores: " + str(
                    reference_scores3)
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "output form: {Realistic:score,  Deformation:score, Imagination:score, Color Richness:score, Color Contrast:score, Line Combination:score, Line Texture:score, Picture Organization:score, Transformation:score}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_path}",
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

# 画雷达图
def plot_radar_chart(scores):
    labels = np.array(['Realistic', 'Deformation', 'Imagination', 'Color Richness', 'Color Contrast', 'Line Combination', 'Line Texture', 'Picture Organization', 'Transformation'])
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The radar chart is a circle, so it needs to be "closed".
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, color='red', alpha=0.25)
    ax.plot(angles, scores, color='red', linewidth=2)  # Change the color if you want
    # Labels for each point
    plt.xticks(angles[:-1], labels)

    # Save the plot as a PNG image in memory (avoiding files)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    # Encode as base64 to serve in HTML
    base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return base64_image
def safe_eval_dict(input_string):
    try:
        # Attempt to directly evaluate the string assuming it has properly quoted keys
        result_dict = ast.literal_eval(input_string)
    except SyntaxError:
        # If there is a syntax error, try fixing unquoted keys
        fixed_string = input_string.replace("{", '{"').replace(", ", ', "').replace(":", '":')
        result_dict = ast.literal_eval(fixed_string)
    return result_dict
if __name__ == '__main__':
    app.run(debug=True)
