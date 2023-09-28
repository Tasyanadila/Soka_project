from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans

app = Flask(__name__)

upload_folder = os.path.join('static', 'uploads')

# Membuat direktori 'static/uploads' jika belum ada
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

app.config['UPLOAD'] = upload_folder

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/histogram', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img_path = os.path.join(app.config['UPLOAD'], filename)

        # Membaca gambar dengan OpenCV
        img = cv2.imread(img_path)

        # Menghitung histogram untuk masing-masing saluran (R, G, B)
        hist_r = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_r /= hist_r.sum()
        hist_g /= hist_g.sum()
        hist_b /= hist_b.sum()

        # Simpan histogram sebagai gambar PNG
        hist_image_path = os.path.join(app.config['UPLOAD'], 'histogram.png')
        plt.figure()
        plt.title("RGB Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_r, color='red', label='Red')
        plt.plot(hist_g, color='green', label='Green')
        plt.plot(hist_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_image_path)

        # Hasil equalisasi
        img_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Ubah ke ruang warna YCrCb
        img_equalized[:, :, 0] = cv2.equalizeHist(img_equalized[:, :, 0])  # Equalisasi komponen Y (luminance)
        img_equalized = cv2.cvtColor(img_equalized, cv2.COLOR_YCrCb2BGR)  # Kembalikan ke ruang warna BGR

        # Menyimpan gambar hasil equalisasi ke folder "static/uploads"
        equalized_image_path = os.path.join('static', 'uploads', 'img-equalized.jpg')
        cv2.imwrite(equalized_image_path, img_equalized)

        # Menghitung histogram untuk gambar yang sudah diequalisasi
        hist_equalized_r = cv2.calcHist([img_equalized], [0], None, [256], [0, 256])
        hist_equalized_g = cv2.calcHist([img_equalized], [1], None, [256], [0, 256])
        hist_equalized_b = cv2.calcHist([img_equalized], [2], None, [256], [0, 256])

        # Normalisasi histogram
        hist_equalized_r /= hist_equalized_r.sum()
        hist_equalized_g /= hist_equalized_g.sum()
        hist_equalized_b /= hist_equalized_b.sum()

        # Simpan histogram hasil equalisasi sebagai gambar PNG        
        hist_equalized_image_path = os.path.join(app.config['UPLOAD'], 'histogram_equalized.png')
        plt.figure()
        plt.title("RGB Histogram (Equalized)")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.plot(hist_equalized_r, color='red', label='Red')
        plt.plot(hist_equalized_g, color='green', label='Green')
        plt.plot(hist_equalized_b, color='blue', label='Blue')
        plt.legend()
        plt.savefig(hist_equalized_image_path)

        return render_template('histogram_equalization.html', img=img_path, img2=equalized_image_path, histogram=hist_image_path, histogram2=hist_equalized_image_path)
    
    return render_template('histogram_equalization.html')

# Load a pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def apply_gaussian_blur_to_face(image, blur_level):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        if blur_level > 0:
            face_roi = cv2.GaussianBlur(face_roi, (0, 0), blur_level)
        image[y:y+h, x:x+w] = face_roi

    return image

@app.route("/blurring", methods=["GET", "POST"])
def blur():
    original_image_path = None
    blurred_image_path = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("blurring.html")

        file = request.files["file"]
        
        # check if file empty
        if file.filename == "":
            return render_template("blurring.html")

        if file:
            # Save the uploaded image to the uploads folder
            file_path = os.path.join(app.config['UPLOAD'], file.filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            blur_level = int(request.form["blur_level"])

            # Save the original image path for display
            original_image_path = file_path

            # Apply Gaussian blur to the detected face
            image = apply_gaussian_blur_to_face(image, blur_level)

             # Menyimpan gambar dengan wajah-wajah yang telah di-blur
            blurred_image_path = os.path.join(app.config['UPLOAD'], 'blurred_image.jpg')
            cv2.imwrite(blurred_image_path, image)


        return render_template("blurring.html", original_image=original_image_path, blurred_image=blurred_image_path)
    return render_template("blurring.html")

def apply_edge_detection(image_path):
    # Baca gambar dari path
    image = cv2.imread(image_path)
    
    # Konversi gambar ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Lakukan deteksi tepi menggunakan Canny edge detector
    edges = cv2.Canny(gray, 100, 200)  # nilai threshold dapat diubah sesuai kebutuhan
    
    # Menyimpan gambar hasil deteksi tepi ke folder "static/uploads"
    edge_image_path = os.path.join(app.config['UPLOAD'], 'edge_detected.jpg')
    cv2.imwrite(edge_image_path, edges)
    
    return edge_image_path

@app.route("/edge", methods=["GET", "POST"])
def edge_detection():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("edge_detection.html")

        file = request.files["file"]

        if file.filename == "":
            return render_template("edge_detection.html")

        if file:
            # Simpan gambar yang diunggah ke folder uploads
            file_path = os.path.join(app.config['UPLOAD'], file.filename)
            file.save(file_path)

            # Save the original image path for display
            original_image_path = file_path
            
            # Proses deteksi tepi pada gambar
            edge_image_path = apply_edge_detection(file_path)

            return render_template("edge_detection.html", original_image=original_image_path,edge_image=edge_image_path)

    return render_template("edge_detection.html")

def grabcut_segmentation(img_path, k=2):
    # Membaca gambar dengan OpenCV
    img = cv2.imread(img_path)

    # Ubah gambar ke ruang warna LAB
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Bentuk matriks piksel sebagai vektor piksel
    pixels = lab_img.reshape((-1, 3))

    # Terapkan k-means clustering untuk segmentasi warna
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Dapatkan label dari setiap piksel
    labels = kmeans.labels_

    # Dapatkan pusat cluster
    centers = kmeans.cluster_centers_

    # Inisialisasi masker
    mask = np.zeros_like(labels)

    # Temukan label yang mewakili latar belakang (cluster dengan intensitas rendah)
    background_label = np.argmin(np.linalg.norm(centers - [0, 128, 128], axis=1))

    # Isi masker dengan 1 untuk label yang mewakili objek
    mask[labels != background_label] = 1

    # Bentuk kembali masker ke bentuk gambar
    mask = mask.reshape(img.shape[:2])

    # Gunakan masker untuk menghapus latar belakang
    result_img = img.copy()
    result_img[mask == 0] = [0, 0, 0]  # Set piksel latar belakang menjadi hitam

    # Menyimpan gambar tanpa latar belakang
    segmentation_image_path = os.path.join(app.config['UPLOAD'], 'apply_grabcut.jpg')
    cv2.imwrite(segmentation_image_path, result_img)
    # Simpan gambar yang diunggah ke folder uploads

    return segmentation_image_path
        

@app.route("/segmentasi", methods=["GET", "POST"])
def segmentasi():
  
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("segmentasi.html")

        file = request.files["file"]

        # Check if the file name is empty
        if file.filename == "":
            return render_template("segmentasi.html")
        
        if file:
            # Simpan gambar yang diunggah ke folder uploads
            img_path = os.path.join(app.config["UPLOAD"], file.filename)
            file.save(img_path)

        # Call the function to remove the background
        segmentation_image_path = grabcut_segmentation(img_path)

        return render_template("segmentasi.html", original_image=img_path, segmentation_image=segmentation_image_path)

    return render_template("segmentasi.html")

if __name__ == '__main__': 
    app.run(debug=True,port=8001)