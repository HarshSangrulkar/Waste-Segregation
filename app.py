from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request,jsonify
import os
import numpy as np
import openai

app = Flask(__name__)
# custom_objects = {'CustomLayerName': CustomLayerClass}
model = load_model("garbage.h5",compile = False)


OPENAI_KEY = "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"
client = openai.OpenAI(api_key=OPENAI_KEY)


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload')
def upload():
    return render_template("upload.html")

@app.route('/liveanalysis')
def liveanalysis():
    return render_template("liveanalysis.html")

@app.route('/chatbot')
def chatbot():
    return render_template("chatbot.html")

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method=='POST':
        f = request.files['images']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        
        img = image.load_img(filepath,target_size =(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        pred =np.argmax(model.predict(x),axis=1)
        index =['Non-Recyclable','Organic','Recyclable']
        text="The classified Garbage is : " +str(index[pred[0]])
        return text    

@app.route('/generate_response', methods=['POST'])
def generate_response():
    user_input = request.form['user_input']
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Based on the description given by the user about the waste, please generate disposal instructions for the described waste in bullet points and life cycle bullet points."},
            {"role": "user", "content": user_input},
        
        
        ],
        max_tokens =200,
        temperature=0.5
    )
    
    # Check the structure of the response to access the content correctly
    
    generated_response = response.choices[0].message.content
    # audio_input = generated_response
    # from pathlib import Path
    
    

    # speech_file_path = Path(__file__).parent / "speech.mp3"
    # response = client.audio.speech.create(
    # model="tts-1",
    # voice="alloy",
    # input=audio_input,
    # )

    # response.stream_to_file(speech_file_path)
    
    
    return jsonify({'response': generated_response}) 


if __name__=='__main__':
   app.run(host="0.0.0.0", port=8080,debug=True)