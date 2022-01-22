import os
import numpy as np #used for numerical analysis
from flask import Flask,request,render_template# Flask-It is our framework which we are going to use to run/serve our application.
#request-for accessing file which was uploaded by the user on our application.
#render_template- used for rendering the html pages
from tensorflow.keras.models import load_model#to load our trained model
from tensorflow.keras.preprocessing import image
#import numpy as np


app=Flask(__name__)#our flask app
model=load_model('LocalModel.h5')#loading the model


@app.route("/")
def upload_file():
    return render_template('index.html')


@app.route("/about") #route about page
def upoad_file1():
    return render_template("index.html")#rendering html page
    

@app.route("/upload") # route for info page 
def upload_file2():
    return render_template("RRP.html")#rendering html page


@app.route("/predict",methods=['GET','POST']) #route for our prediction
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img=image.load_img(filepath,target_size=(64,64)) #load and reshaping the image
        x=image.img_to_array(img)#converting image to array
        print(x)
        x=np.expand_dims(x,axis=0)#changing the dimensions of the image
        print(x)
        pred = model.predict(x)  # predicting classes
        print("prediction",pred)  # printing the prediction
        index = ['French Fries', 'Pizza', 'Samosa']
        result=np.argmax(pred,axis=1)
        result=index[result[0]]
        if (result=="French Fries"):
            return render_template("0.html",showcase =  result)
        elif (result=="Pizza"):
            return render_template("1.html",showcase =  result)
        else:
            return render_template("2.html",showcase =  result)
    else:
        return None

#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=False)#running our app
    #app.run(host='127.0.0.1', port=5000,debug=True)
            
            