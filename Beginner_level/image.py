import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image
import requests
from io import BytesIO
import gradio as gr
device = "cuda" if torch.cuda.is_available() else "cpu"
weights = MobileNet_V2_Weights.DEFAULT
model = models.mobilenet_v2(weights=weights).to(device)
model.eval()
transform = weights.transforms()
def fetch_classes(url):
    try:
        return requests.get(url).text.strip().splitlines()
    except:
        return ["Unknown"]
class_labels = fetch_classes("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
sample_images = {
    "Cat": [
        "https://images.unsplash.com/photo-1602418013963-c1f017b3bb63",
        "https://www.alleycat.org/wp-content/uploads/2019/03/FELV-cat.jpg",
        "https://i.guim.co.uk/img/media/327aa3f0c3b8e40ab03b4ae80319064e401c6fbc/377_133_3542_2834/master/3542.jpg?width=1200&height=1200&quality=85&auto=format&fit=crop&s=34d32522f47e4a67286f9894fc81c863",
        "https://www.purina.in/sites/default/files/2023-05/feast.png",
        "https://static.vecteezy.com/system/resources/thumbnails/022/713/469/small/cute-cat-isolated-on-solid-background-photo.jpg"
    ],
    "Dog": [
        "https://images.squarespace-cdn.com/content/v1/54822a56e4b0b30bd821480c/45ed8ecf-0bb2-4e34-8fcf-624db47c43c8/Golden+Retrievers+dans+pet+care.jpeg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Labrador_on_Quantock_%282175262184%29.jpg/1200px-Labrador_on_Quantock_%282175262184%29.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRHIQPM5MQe7flMzWIVEF7FbJVBj2tWNHbPww&s",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS6B6nwjVyH5UmpWKb9dQQu9_278TX9ElP3eA&s",
        "https://hips.hearstapps.com/hmg-prod/images/golden-retriever-dog-royalty-free-image-505534037-1565105327.jpg"
    ],
    "Tiger": [
        "https://bigcatsindia.com/wp-content/uploads/2018/06/Royal-Bengal-Tiger.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSg6IH6gZ7arA85aKRW7jj4wAlvPgFdKREmRQ&s",
        "https://images.pexels.com/photos/162173/panthera-tigris-altaica-tiger-siberian-amurtiger-162173.jpeg?cs=srgb&dl=pexels-pixabay-162173.jpg&fm=jpg",
        "https://d1jyxxz9imt9yb.cloudfront.net/medialib/5027/image/s1300x1300/3_reduced.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/MadridZoo-Panthera_tigris_tigris.jpg/250px-MadridZoo-Panthera_tigris_tigris.jpg"
    ],
    "Car": [
        "https://media.architecturaldigest.com/photos/66a914f1a958d12e0cc94a8e/16:9/w_2992,h_1683,c_limit/DSC_5903.jpg",
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRYBNvLImFxQVcHrP-1gudu0kodBBbtJTfqmg&s",
        "https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/body-image/public/rolls_royce_phantom_top_10.jpg?itok=P4JYsn-X",
        "https://imgd.aeplcdn.com/642x336/n/cw/ec/131131/xc60-exterior-right-front-three-quarter-3.jpeg?isig=0&q=80",
        "https://media.cdn-jaguarlandrover.com/api/v2/images/102859/w/640.jpg",
    ],
    "Bird": [
        "https://media.istockphoto.com/id/1307609675/photo/bluebird.jpg?s=612x612&w=0&k=20&c=PdSeFBXLNi2n8vNxDjubRQOsaOw_sJ1w7RhtxjGL5GM=",
        "https://images.unsplash.com/photo-1486365227551-f3f90034a57c?fm=jpg&q=60&w=3000&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YmlyZHxlbnwwfHwwfHx8MA%3D%3D",
        "https://cdn.britannica.com/29/181429-050-A2AFC966/Burrowing-owl.jpg",
        "https://i.pinimg.com/originals/a5/6b/c2/a56bc231a848e46053b0e3e333684c6f.jpg",
        "https://cdn.britannica.com/35/3635-050-96241EC1/Scarlet-macaw-ara-macao.jpg"
    ]
}
def predict_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_tensor)
            idx = preds.argmax(1).item()
            label = class_labels[idx]
        return img, f"Classified as: {label}"
    except Exception as e:
        return None, f"Error: {str(e)}"
def get_images_for_category(category):
    return sample_images.get(category, [])
with gr.Blocks() as demo:
    gr.Markdown("## Image Classification with MobileNetV2")
    category_dropdown = gr.Dropdown(label="Select an Image", choices=list(sample_images.keys()))
    with gr.Row():
        clear_btn = gr.Button("Clear", variant="secondary")
        submit_btn = gr.Button("Submit", variant="primary")
    gallery_slider = gr.Image(label="Image Preview", type="pil", interactive=False)
    with gr.Row():
        left_btn = gr.Button("← Previous image")
        right_btn = gr.Button("→ Next image")
    result_output = gr.Textbox(label="Prediction", lines=1)
    image_urls = gr.State([])
    current_index = gr.State(0)
    def load_images(category):
        urls = get_images_for_category(category)
        if not urls:
            return None, [], 0, "No images found"
        img, label = predict_from_url(urls[0])
        return img, urls, 0, label
    def change_image(direction, urls, index):
        if not urls:
            return None, index, ""
        index = (index + direction) % len(urls)
        img, label = predict_from_url(urls[index])
        return img, index, label
    def clear_all():
        return (
            gr.update(value=None),
            None,
            "",
            [],
            0
        )
    submit_btn.click(load_images, inputs=category_dropdown,
                     outputs=[gallery_slider, image_urls, current_index, result_output])
    left_btn.click(
        fn=lambda urls, index: change_image(-1, urls, index),
        inputs=[image_urls, current_index],
        outputs=[gallery_slider, current_index, result_output]
    )
    right_btn.click(
        fn=lambda urls, index: change_image(1, urls, index),
        inputs=[image_urls, current_index],
        outputs=[gallery_slider, current_index, result_output]
    )
    clear_btn.click(
        clear_all,
        inputs=[],
        outputs=[category_dropdown, gallery_slider, result_output, image_urls, current_index]
    )
demo.launch()