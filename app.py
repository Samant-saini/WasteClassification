import streamlit as st

import streamlit.components.v1 as components
from PIL import Image
# Embed Streamlit docs in a Streamlit app
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input

model = tf.keras.models.load_model("classify.h5")
st.image("image6.png", use_column_width=False, output_format='JPEG', width=750)


labs = {0: 'biodegradable', 1: 'non-biodegradable'}

import streamlit as st
from PIL import Image
from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

model = load_model('./Model/BC.h5',compile=False)

def processed_img(img_path):
    img=load_img(img_path,target_size=(224,224,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open('./meta/logo1.png')
    img1 = img1.resize((350,350))
    st.image(img1,use_column_width=False)
    st.title("Birds Species Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "270 Bird Species also see 70 Sports Dataset"</h4>''',
                unsafe_allow_html=True)

    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])
    if img_file is not None:
        st.image(img_file,use_column_width=False)
        save_image_path = './upload_images/'+img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if st.button("Predict"):
            result = processed_img(save_image_path)
            st.success("Predicted Bird is: "+result)

        # if generate_pred:
        # prediction = model.predict(img_reshaped)
        # prediction = int(prediction)

        if (prediction == 0):
            prediction = "biodegradable"
            st.markdown("<p style='font-size:50px'> The waste in the image is \n <span style='color:green;'>{}</span></p>".format(prediction), unsafe_allow_html=True)
        else:
            prediction = "non-biodegradable"
            st.markdown("<p style='font-size:50px'> The waste in the image is \n <span style='color:red;'>{}</span></p>".format(prediction), unsafe_allow_html=True)


run()




















# Set the desired text size and content
text_size = "60px"
text_content = "KNOW YOUR WASTE"



# Center the text using CSS
centered_text = f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight:900;
        height: 100px; /* Adjust as needed for vertical centering */
        color: red; /* Set text color */
        background-color: red; /* Set background color */
        border-radius: 10px;
    ">
        <p style="font-size: {text_size};">{text_content}</p>
    </div>
"""
# st.write("Upload your waste image here")

# Display the centered text using st.markdown
# st.markdown(centered_text,  unsafe_allow_html=True)
st.markdown("<p style='color:red; display: flex; font-size: 60px; justify-content:center; align-items:center; font-weight:900;  height: 100px;  border-radius: 10px;'>KNOW YOUR WASTE</p>",
             unsafe_allow_html=True)



# Load file
uploaded_file = st.file_uploader("Choose an image file", type="jpg")

map_dict = {0: 'biodegradable', 1: 'non-biodegradable'}
st.markdown(
    """
    <style>
        body {
            background-color: blue;
            margin: 0;
            padding: 0;
        }
         .stApp {
            padding: 0;
        }
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
            padding:0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Images related to waste management
# waste_management_images = [
#     "image1.jpg",
#     "image2.jpg",
#     "image3.jpg",
#     # Add more image filenames as needed
# ]

# Display waste management images at the top of the page
# for img_filename in waste_management_images:
#     img_path ="images/image"  # Replace with the actual path to your images
#     st.image(img_path, caption="", use_column_width=True)















if uploaded_file is not None:
    size = (180,180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image, (256, 256))

    # Display the image
    st.image(opencv_image, channels="RGB")

    # resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    # Generate prediction button
    generate_pred = st.button("Generate Prediction")

    if generate_pred:
        prediction = model.predict(img_reshaped)
        prediction = int(prediction)

        if (prediction == 0):
            prediction = "biodegradable"
            st.markdown("<p style='font-size:50px'> The waste in the image is \n <span style='color:green;'>{}</span></p>".format(prediction), unsafe_allow_html=True)
        else:
            prediction = "non-biodegradable"
            st.markdown("<p style='font-size:50px'> The waste in the image is \n <span style='color:red;'>{}</span></p>".format(prediction), unsafe_allow_html=True)


        # st.title("THE WASTE IN IMAGE IS " + prediction)
        # st.markdown("The waste in the image is <span style='color:red'>{prediction} </span>",unsafe_allow_html=True)


        if(prediction == "non-biodegradable"):
            st.header("METHODS FOR PROPER DISPOSAL")
            st.write("Proper disposal of non-biodegradable waste is crucial for environmental sustainability. Non-biodegradable waste includes materials that do not break down naturally or take an extended period to decompose.")
            st.write("Here are some effective methods for the disposal of non-biodegradable waste:")
            st.write("- **Recycling:**")
            st.write("   - Sorting at Source: Encourage the separation of recyclable materials like plastics, glass, metals, and paper at the source.")
            st.write("   - Recycling Facilities: Support and utilize recycling facilities. Many materials, such as plastic and paper, can be recycled to produce new products, reducing the need for raw materials.")
            st.write("- **Waste-to-Energy (WTE) Facilities:**")
            st.write("   - Incineration: In some cases, non-biodegradable waste can be burned at high temperatures in controlled environments to generate energy.")
            st.write("   - However, this method should be approached with caution due to emissions and air quality concerns.")
            st.write("- **Landfills:**")
            st.write("   - Safe Disposal: Landfills are designed to contain waste and prevent contamination of the surrounding environment. Non-biodegradable waste is often disposed of in landfills, but it's essential to manage landfills properly to minimize environmental impact.")
        elif(prediction=='biodegradable'):
            st.header("METHODS FOR PROPER DISPOSAL")
            st.write("Proper disposal of biodegradable waste is crucial for environmental sustainability. biodegradable waste includes materials that do not break down naturally or take an extended period to decompose.")
            st.write("Here are some effective methods for the disposal of biodegradable waste:")
            st.write("Composting:")
            st.write("Composting is an excellent way to dispose of biodegradable waste.Set up a composting bin in your backyard for kitchen scraps, fruit and vegetable peels, coffee grounds, and yard waste.")
            st.write("Vermicomposting:")
            st.write("Vermicomposting involves using worms to decompose organic waste.Set up a vermicomposting bin with appropriate bedding and add kitchen scraps. Worms will break down the waste into nutrient-rich compost.**")
            st.write("Animal Feed:")
            st.write("Some food scraps, such as fruit and vegetable peels, can be used as feed for certain animals.Ensure that the waste is suitable for the specific animals and follows any relevant guidelines.")
            st.write("Avoid Mixing with Non-Biodegradable Waste:")
            st.write("Separate biodegradable waste from non-biodegradable waste to facilitate proper disposal and recycling.")

        
    


        #get the city name and provide the map of the dumping sites
        
        
    city = st.selectbox("Select your city:", [" ","Ghaziabad", "Delhi", "Noida","Muradnagar","Modinagar","Meerut"])
    # st.write("dumping area in ", city)
            
    if(city=="Ghaziabad"):
        HtmlFile = open("pages/map.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Dumping area")
        st.write("4728, Street Number 47, West Sant Nagar, Karol Bagh, New Delhi, Delhi, 110005")
        st.write("ph:07300643092")
        st.header("2.Rollz India Waste Management")
        st.write("107, Block 10, Sector 10, Raj Nagar, Ghaziabad, Uttar Pradesh 201002")
        st.write("ph:01204565999")
        st.header("3.Garbage Dump")
        st.write("M7PG+4MF, New Jaffrabad, Shahdara, Delhi, 110032")
        st.write("ph:01204565999")
                                            
    elif(city=="Delhi"):
        HtmlFile = open("pages/delhimap.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Dumping area")
        st.write("4728, Street Number 47, West Sant Nagar, Karol Bagh, New Delhi, Delhi, 110005")
        st.write("ph:07300643092")
        st.header("2.Rollz India Waste Management")
        st.write("107, Block 10, Sector 10, Raj Nagar, Ghaziabad, Uttar Pradesh 201002")
        st.write("ph:01204565999")
        st.header("3.Garbage Dump")
        st.write("M7PG+4MF, New Jaffrabad, Shahdara, Delhi, 110032")
        st.write("ph:01204565999")
    elif(city=="Noida"):
        HtmlFile = open("pages/noida.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Dumping area")
        st.write("4728, Street Number 47, West Sant Nagar, Karol Bagh, New Delhi, Delhi, 110005")
        st.write("ph:07300643092")
        st.header("2.Rollz India Waste Management")
        st.write("107, Block 10, Sector 10, Raj Nagar, Ghaziabad, Uttar Pradesh 201002")
        st.write("ph:01204565999")
        st.header("3.Garbage Dumpt")
        st.write("M7PG+4MF, New Jaffrabad, Shahdara, Delhi, 110032")
        st.write("ph:01204565999")
    elif(city=="Muradnagar"):
        HtmlFile = open("pages/Muradnagar.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Sameer Garbage Destroying Services")
        st.write("MF78+CGJ, Jeevan Vihar, Mahindra Enclave, Shastri Nagar, Ghaziabad, Uttar Pradesh 201002")
        st.write("ph:07300643092")
        st.header("2.400 TPD C&D Waste Recycling")
        st.write("Arya Nagar, Hindan Vihar, Sewa Nagar, Ghaziabad, Uttar Pradesh 201003")
        st.write("ph:01204565999")
        st.header("3.Recyling waste")
        st.write("MC7F+G6G, National Highway 91, Jassipura, Madhopura, Ghaziabad, Uttar Pradesh 201009")
        st.write("ph:01204565999")
    elif(city=="Modinagar"):
        HtmlFile = open("pages/Modinagar.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Dumping area")
        st.write("4728, Street Number 47, West Sant Nagar, Karol Bagh, New Delhi, Delhi, 110005")
        st.write("ph:07300643092")
        st.header("2.Rollz India Waste Management")
        st.write("107, Block 10, Sector 10, Raj Nagar, Ghaziabad, Uttar Pradesh 201002")
        st.write("ph:01204565999")
        st.header("3.Garbage Dump")
        st.write("M7PG+4MF, New Jaffrabad, Shahdara, Delhi, 110032")
        st.write("ph:01204565999")

    elif(city=="Meerut"):
        HtmlFile = open("pages/Meerut.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read()
        components.html(source_code, height=450, width=1350, scrolling=True)
        st.header("Dumping area in {} ".format(city))
        st.header("1.Best Green E waste Recycling")
        st.write("WPW8+X4C, Main Markets, Zakir Colony, Meerut, Uttar Pradesh 250002")
        st.write("ph:07300643092")
        st.header("2.Making India E-waste Recycling Management")
        st.write("Plot No-50, Sector 3, MDA, Meerut, Uttar Pradesh 250103")
        st.write("ph:01212441122g")
        st.header("3.Project Swachhta Meerut")
        st.write("B-175, Garh Rd, Shastri Nagar, Meerut, Uttar Pradesh 250002")


       
                        
                    
                    