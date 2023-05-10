
import streamlit as st
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


st. set_page_config(layout="wide")

#with st.sidebar: 
#    image1 = Image.open(r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\close-up-view-of-human-eye.jpg')
#    st.image(image1)
#    st.title('DR Detection')
#    st.info('Disclaimer: This app is not a medical device')

st.title('ARTIFICIAL INTELLIGENCE FOR CLASSIFICATION AND AUTOMATIC DIAGNOSIS OF DIABETIC RETINOPATHY')
st.caption('Disclaimer: This application is not a medical device')

tab1, tab2, tab3 = st.tabs(["About diabetic retinopathy ", "How is it diagnosed?", 'Deep learning model'])

with tab1:
    st.header('What is Diabetic Retinopathy?')
    st.markdown('Diabetic retinopathy is caused by high blood sugar due to diabetes. Over time, having too much sugar in your blood can damage your retina — the part of your eye that detects light and sends signals to your brain through a nerve in the back of your eye (optic nerve). Diabetes damages blood vessels all over the body. The damage to your eyes starts when sugar blocks the tiny blood vessels that go to your retina, causing them to leak fluid or bleed. To make up for these blocked blood vessels, your eyes then grow new blood vessels that don’t work well. These new blood vessels can leak or bleed easily. (Source: NIH Website)')
    st.image('https://www.raleighop.com/wp-content/uploads/Diabetic-Retina.jpg')

with tab2:
    st.header('How will my eye doctor check for diabetic retinopathy?')
    st.markdown('Eye doctors can check for diabetic retinopathy as part of a dilated eye exam. The exam is simple and painless — your doctor will give you some eye drops to dilate (widen) your pupil and then check your eyes for diabetic retinopathy and other eye problems.If you have diabetes, it’s very important to get regular eye exams. If you do develop diabetic retinopathy, early treatment can stop the damage and prevent blindness. If your eye doctor thinks you may have severe diabetic retinopathy or DME, they may do a test called a fluorescein angiogram. This test lets the doctor see pictures of the blood vessels in your retina. (Source: NIH Website)')
    st.image('https://da4e1j5r7gw87.cloudfront.net/wp-content/uploads/sites/2470/2022/07/4-stages-diabetic-retinopathy.jpg')

    st.header('Stage 1: Mild nonproliferative Diabetic Retinopathy')
    st.markdown('This is the earliest stage of diabetic retinopathy, characterized by tiny swellings/bulges in the blood vessels of the retina. These areas of swelling are known as microaneurysms. These microaneurysms can cause small amounts of fluid to leak into the retina, triggering swelling of the macula – the back of the retina. Despite this, there are usually no clear symptoms indicating there is a problem.')
    
    st.header('Stage 2: Moderate Nonproliferative Diabetic Retinopathy')
    st.markdown('At this stage, the tiny blood vessels further swell up, blocking blood flow to the retina and preventing proper nourishment. This stage will only cause noticeable signs if there is a build-up of blood and other fluids in the macula, causing vision to become blurry.')
    
    st.header('Stage 3: Severe Nonproliferative Diabetic Retinopathy')
    st.markdown('During this stage, a larger section of blood vessels in the retina becomes blocked, causing a significant decrease in blood flow to this area. The lack of blood triggers a signal to the body to start growing new blood vessels in the retina. These new blood vessels are extremely thin and fragile and cause retinal swelling, resulting in noticeably blurry vision, dark spots and even patches of vision loss. If these vessels leak into the macula, sudden and permanent vision loss may occur. At this stage, there is a high chance of irreversible vision loss.')
    
    st.header('Stage 4: Proliferative Diabetic Retinopathy')
    st.markdown('At this advanced stage of the disease, new blood vessels continue to grow in the retina. These blood vessels, which are thin and weak and prone to bleeding, cause scar tissue to form inside the eye. This scar tissue can pull the retina away from the back of your eye, causing retinal detachment. A detached retina typically results in blurriness, reduced field of vision, and even permanent blindness.')


image_path = r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\train_images'
classes=["0","1","2","3","4"]

with tab3:
    st.header('SELECT AN IMAGE BELOW TO START THE DIAGNOSIS') 
    options = os.listdir(os.path.join(r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\train_images'))
    selected_image = st.selectbox('Image Selected: ', options)
    img_path = os.path.join(image_path, selected_image)
    image = Image.open(img_path)
    st.image(image, width=350)
  

    # uploaded_file = st.file_uploader(label='Or Upload an Image of your Choice')
    # if uploaded_file is not None:
    #    image_data = uploaded_file.getvalue()
    #    st.image(image_data)

    def load_model():
        model = keras.models.load_model(r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\model.h5')
        return model
        
    # image preprocessing
    def load_image(img_path):
            img= Image.open(img_path)
            img = img.resize((250, 250))
            img = np.array(img)
            img = img/255.0
            # img=tf.image.resize(img,(28,28))
            img= np.expand_dims(img, axis=0)
            return img
    
    with st.spinner("Loading the Artificial Intelligence Model"):
            model = load_model()

    if options:
        if st.button('Click here to start the diagnosis'):
            try:
                img_tensor = load_image(os.path.join(img_path))
                pred = model.predict(img_tensor)
                pred_class = str(classes[np.argmax(pred)])
                st.write("Predicted Stage of diabetic retinopathy:", pred_class)
            except Exception as e:
                st.write ('Error:', e)

        
    image1 = Image.open(r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\Capture_Loss.png')
    st.header('DEEP LEARNING MODEL LOSS FIGURE')
    st.markdown('The LOSS metric is the difference between the predicted diagnosis by our Model and the real diagnosis actual made. In summary, the LOSS metric will tell us how fare our Model is from reality. Keep in mind that the lower the LOSS, the closer we are to reality')
    st.image(image1)

    image2 = Image.open(r'd:\1.JEDHA\5.Demo_Day\Eye_Disease\Capture_Accuracy.png')
    st.header('DEEP LEARNING MODEL ACCURACY FIGURE')
    st.markdown('On the other hand, the ACCURACY is a percentage, telling us our precise our model is overall. The closer it is to 1, the better')
    st.image(image2)

    col1, col2 = st.columns(2)
    with col1:
        st.header("TRAINING")
        st.metric(label="LOSS", value="0.8") 
        st.metric(label="ACCURACY", value="74%")

    with col2:
        st.header("TESTING")
        st.metric(label="VAL_LOSS", value="0.86")
        st.metric(label="VAL_ACCURACY", value="72%")



    # streamlit run d:/1.JEDHA/5.Demo_Day/Eye_Disease/main.py
