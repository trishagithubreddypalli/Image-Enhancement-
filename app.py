import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


# Function to apply sharpening
def sharpen_image(image):
    img_array = np.array(image)
    gaussian_blur = cv2.GaussianBlur(img_array, (7, 7), 2)
    sharpened1 = cv2.addWeighted(img_array, 1.5, gaussian_blur, -0.5, 0)
    sharpened2 = cv2.addWeighted(img_array, 3.5, gaussian_blur, -2.5, 0)
    sharpened3 = cv2.addWeighted(img_array, 7.5, gaussian_blur, -6.5, 0)
    return sharpened1, sharpened2, sharpened3


# Function to apply blurring
def blur_image(image):
    img_array = np.array(image)
    kernel_25 = np.ones((25, 25), np.float32) / 625.0
    output_kernel = cv2.filter2D(img_array, -1, kernel_25)
    output_blur = cv2.blur(img_array, (25, 25))
    output_box = cv2.boxFilter(img_array, -1, (5, 5), normalize=False)
    output_gaus = cv2.GaussianBlur(img_array, (5, 5), 0)
    output_med = cv2.medianBlur(img_array, 5)
    output_bil = cv2.bilateralFilter(img_array, 5, 75, 75)
    return output_kernel, output_blur, output_box, output_gaus, output_med, output_bil


# Function to apply histogram equalization
def apply_histogram_equalization(image):
    img_array = np.array(image)
    equalized_image = cv2.equalizeHist(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
    return equalized_image


# Function to apply color correction (adjusting brightness and contrast)
def apply_color_correction(image, alpha=1.0, beta=0):
    img_array = np.array(image)
    adjusted = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
    return adjusted


# Function to perform edge detection
def detect_edges(image):
    img_array = np.array(image)
    edges = cv2.Canny(cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY), 100, 200)
    return edges


# Function to resize the image
def resize_image(image, scale_percent=50):
    img_array = np.array(image)
    width = int(img_array.shape[1] * scale_percent / 100)
    height = int(img_array.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img_array, dim, interpolation=cv2.INTER_AREA)
    return resized


# Function to combine all enhancements
def enhance_image(image):
    sharpened1, sharpened2, sharpened3 = sharpen_image(image)
    output_kernel, output_blur, output_box, output_gaus, output_med, output_bil = (
        blur_image(image)
    )
    equalized_image = apply_histogram_equalization(image)
    corrected_image = apply_color_correction(image)
    edges = detect_edges(image)
    resized_image = resize_image(image)

    # Combine enhancements as desired
    enhanced_img = sharpened1  # Example: Using sharpened1 as an enhanced image
    return enhanced_img


# Function to convert image to bytes
def image_to_bytes(image):
    img = Image.fromarray(image)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im


# Streamlit app
st.title("Image Enhancement with OpenCV")

# Sidebar for selecting operation
operation = st.sidebar.selectbox(
    "Choose an operation",
    (
        "Sharpening",
        "Blurring",
        "Histogram Equalization",
        "Color Correction",
        "Edge Detection",
        "Image Resizing",
        "Enhanced Image",  # Added option for combined enhancements
    ),
)

# Image uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Webcam capture
use_webcam = st.checkbox("Use webcam")

if use_webcam:
    picture = st.camera_input("Take a picture")
    if picture:
        image = Image.open(picture)
        image = image.convert("RGB")
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = image.convert("RGB")

if uploaded_file is not None or (use_webcam and picture):
    # Display original image
    st.image(image, caption="Original Image", use_column_width=True)

    if operation == "Sharpening":
        st.header("Image Sharpening")
        sharpened1, sharpened2, sharpened3 = sharpen_image(image)
        st.image(sharpened1, caption="Sharpened Image 1", use_column_width=True)
        st.image(sharpened2, caption="Sharpened Image 2", use_column_width=True)
        st.image(sharpened3, caption="Sharpened Image 3", use_column_width=True)
        st.download_button(
            "Download Sharpened Image 1",
            data=image_to_bytes(sharpened1),
            file_name="sharpened1.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Sharpened Image 2",
            data=image_to_bytes(sharpened2),
            file_name="sharpened2.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Sharpened Image 3",
            data=image_to_bytes(sharpened3),
            file_name="sharpened3.jpg",
            mime="image/jpeg",
        )

    elif operation == "Blurring":
        st.header("Image Blurring")
        output_kernel, output_blur, output_box, output_gaus, output_med, output_bil = (
            blur_image(image)
        )
        st.image(output_kernel, caption="Kernel Blurring", use_column_width=True)
        st.image(output_blur, caption="Blur() Output", use_column_width=True)
        st.image(output_box, caption="Box Filter", use_column_width=True)
        st.image(output_gaus, caption="Gaussian Blur", use_column_width=True)
        st.image(output_med, caption="Median Blur", use_column_width=True)
        st.image(output_bil, caption="Bilateral Filtering", use_column_width=True)
        st.download_button(
            "Download Kernel Blurring",
            data=image_to_bytes(output_kernel),
            file_name="kernel_blur.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Blur() Output",
            data=image_to_bytes(output_blur),
            file_name="blur_output.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Box Filter",
            data=image_to_bytes(output_box),
            file_name="box_filter.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Gaussian Blur",
            data=image_to_bytes(output_gaus),
            file_name="gaussian_blur.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Median Blur",
            data=image_to_bytes(output_med),
            file_name="median_blur.jpg",
            mime="image/jpeg",
        )
        st.download_button(
            "Download Bilateral Filtering",
            data=image_to_bytes(output_bil),
            file_name="bilateral_filter.jpg",
            mime="image/jpeg",
        )

    elif operation == "Histogram Equalization":
        st.header("Histogram Equalization")
        equalized_image = apply_histogram_equalization(image)
        st.image(equalized_image, caption="Equalized Image", use_column_width=True)
        st.download_button(
            "Download Equalized Image",
            data=image_to_bytes(equalized_image),
            file_name="equalized_image.jpg",
            mime="image/jpeg",
        )

    elif operation == "Color Correction":
        st.header("Color Correction")
        alpha = st.slider("Alpha (Brightness Adjustment)", 0.0, 3.0, 1.0)
        beta = st.slider("Beta (Contrast Adjustment)", -100, 100, 0)
        corrected_image = apply_color_correction(image, alpha=alpha, beta=beta)
        st.image(corrected_image, caption="Corrected Image", use_column_width=True)
        st.download_button(
            "Download Corrected Image",
            data=image_to_bytes(corrected_image),
            file_name="corrected_image.jpg",
            mime="image/jpeg",
        )

    elif operation == "Edge Detection":
        st.header("Edge Detection")
        edges = detect_edges(image)
        st.image(edges, caption="Edge Detected Image", use_column_width=True)
        st.download_button(
            "Download Edge Detected Image",
            data=image_to_bytes(edges),
            file_name="edge_detected_image.jpg",
            mime="image/jpeg",
        )

    elif operation == "Image Resizing":
        st.header("Image Resizing")
        scale_percent = st.slider("Scale Percent", 10, 200, 50)
        resized_image = resize_image(image, scale_percent=scale_percent)
        st.image(
            resized_image,
            caption=f"Resized Image ({scale_percent}%)",
            use_column_width=True,
        )
        st.download_button(
            "Download Resized Image",
            data=image_to_bytes(resized_image),
            file_name="resized_image.jpg",
            mime="image/jpeg",
        )

    elif operation == "Enhanced Image":
        st.header("Enhanced Image (Combined Enhancements)")
        enhanced_img = enhance_image(image)
        st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)
        st.download_button(
            "Download Enhanced Image",
            data=image_to_bytes(enhanced_img),
            file_name="enhanced_image.jpg",
            mime="image/jpeg",
        )

# Run the Streamlit app using the command below in your terminal:
# streamlit run app.py
