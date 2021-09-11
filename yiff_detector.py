import io
import os
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import PySimpleGUI as sg
from PIL import Image
def predict_image(image_path,model):
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    return result

def show_output(result):
    if result[0][0] > result[0][1] and result[0][0] > result[0][2]:
        prediction = 'NSFW'
    elif result[0][1] > result[0][2]:
        prediction = 'SFW'
    elif result[0][2] > 0:
        prediction = 'Vore'
    else:
        prediction = f'error, {result}'
    print(f"\n result is \nNSFW: {result[0][0]}\nSFW: {result[0][1]}\nVore: {result[0][2]}")
    print("prediction for image is")
    print(prediction)
    return prediction

def main():
    file_types = [("Furry Image (*.jpg)", "*.jpg"),
              ("Furry image but png (*.png)", "*.png")]
    layout = [[sg.Text("Yiff Detector")],[sg.Text("Choose a file: "), sg.FileBrowse(key="-PATH-", file_types=file_types)],[sg.Image(key="-IMAGE-")],[sg.Text(key="-OUTPUT-")],[sg.Button("submit")]]
    window = sg.Window(title="Yiff Detecter", layout=layout)
    model=tf.keras.models.load_model("yiff-model.h5",custom_objects=None, compile=True)
    while True: 
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == "submit":
            imagepath = values["-PATH-"]
            if os.path.exists(imagepath):
                image = Image.open(values["-PATH-"])
                image.thumbnail((400, 400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data=bio.getvalue())
                result = predict_image(imagepath, model)
                output = show_output(result)
                window["-OUTPUT-"].update(output)


if __name__ == '__main__':
    main()
    