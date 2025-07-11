import tkinter as tk
from tkinter import filedialog, Label, Button
from tkinter.messagebox import showinfo
import cv2
import numpy as np
from PIL import Image, ImageTk

def load_image():
    global img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.bmp")])
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        display_image(img, "Original Image")
    else:
        showinfo("Error", "No file selected!")

def apply_preprocessing():
    global img, processed_img
    if img is None:
        showinfo("Error", "Load an image first!")
        return
    # Binaryzacja Otsu: linie czarne, tło białe
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Filtracja medianowa
    processed_img = cv2.medianBlur(binary_img, 3)
    display_image(processed_img, "Processed Image")

def count_neighbors(neighborhood):
    """Zlicz aktywne piksele w sąsiedztwie 3x3, pomijając środkowy piksel."""
    return np.sum(neighborhood) - neighborhood[1, 1]

def k3m(image):
    weights = np.array([
        [128, 1, 2],
        [64, 0, 4],
        [32, 16, 8]
    ])

    delete_table = [
        {3}, {3, 4}, {3, 4, 5}, {3, 4, 5, 6}, {3, 4, 5, 6, 7}
    ]

    rows, cols = image.shape
    skeleton = (image == 0).astype(np.uint8)  # Linie czarne jako 1, tło białe jako 0

    def count_neighbors(neighborhood):
        """Liczba aktywnych pikseli w sąsiedztwie."""
        return np.sum(neighborhood) - neighborhood[1, 1]

    def transitions(neighborhood):
        """Liczba przejść 0->1 w sąsiedztwie."""
        neighbors = neighborhood.flatten()[np.array([1, 2, 5, 8, 7, 6, 3, 0])]
        return np.sum((neighbors[:-1] == 0) & (neighbors[1:] == 1))

    def selective_erosion(image):
        """Dodatkowa redukcja pikseli w miejscach grubości lokalnej."""
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(image, kernel, iterations=1)
        # Po erozji przywracamy cienkie linie
        mask = (image == 255) & (eroded == 0)
        eroded[mask] = 255
        return eroded

    changes = True
    while changes:
        changes = False
        for phase, delete_set in enumerate(delete_table):
            pixels_to_delete = []

            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    if skeleton[row, col] == 0:
                        continue

                    # Sąsiedztwo 3x3
                    neighborhood = skeleton[row - 1:row + 2, col - 1:col + 2]
                    active_neighbors = count_neighbors(neighborhood)

                    # Warunki usunięcia: aktywni sąsiedzi, liczba przejść i tabele dla fazy
                    if (
                        active_neighbors in delete_set
                        and 2 <= active_neighbors <= 6
                        and transitions(neighborhood) == 1
                    ):
                        pixels_to_delete.append((row, col))

            # Usuwanie pikseli wskazanych w bieżącej fazie
            for row, col in pixels_to_delete:
                skeleton[row, col] = 0

            if pixels_to_delete:
                changes = True

    # Konwersja do białego tła (255) i czarnych linii
    skeleton = (1 - skeleton) * 255

    # Adaptacyjna erozja w gęstych miejscach
    skeleton = selective_erosion(skeleton)

    return skeleton





def skeletonize_image():
    global processed_img, skeleton_img
    if processed_img is None:
        showinfo("Error", "Preprocess the image first!")
        return
    skeleton_img = k3m(processed_img)
    display_image(skeleton_img, "Skeletonized Image")

def display_image(cv_image, title):
    bgr_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(bgr_image))
    panel.config(image=img_tk)
    panel.image = img_tk
    panel.title = title

# Tworzenie GUI
root = tk.Tk()
root.title("Fingerprint Preprocessing and Skeletonization")

img = None
processed_img = None
skeleton_img = None

frame = tk.Frame(root)
frame.pack()

Button(frame, text="Load Image", command=load_image).pack(side=tk.LEFT, padx=10, pady=5)
Button(frame, text="Preprocess Image", command=apply_preprocessing).pack(side=tk.LEFT, padx=10, pady=5)
Button(frame, text="Skeletonize Image", command=skeletonize_image).pack(side=tk.LEFT, padx=10, pady=5)

panel = Label(root)
panel.pack()

root.mainloop()
