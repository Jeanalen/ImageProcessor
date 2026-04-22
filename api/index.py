from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance
import matplotlib.pyplot as plt  

class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Image Processing")
        self.root.geometry("1900x900")
        self.root.configure(bg="#2C3E50")
        
        self.image = None
        self.original_image = None 
        self.processed_image = None  # To hold the processed image

        # Image Selection Button
        self.btn_select = tk.Button(self.root, text="Choose an Image", command=self.load_image, bg="#3498DB", fg="white")
        self.btn_select.place(x=17, y=20, width=200, height=40)
        
        self.btn_count_piso = tk.Button(self.root, text="Count Piso Coins", command=self.count_piso_coins, bg="#E67E22", fg="white")
        self.btn_count_piso.place(x=550, y=700, width=150, height=50)

        # Filters
        self.chk_grayscale = tk.Checkbutton(self.root, text="Grayscale", command=self.apply_grayscale, bg="#2C3E50", fg="white")
        self.chk_grayscale.place(x=17, y=80)
        
        self.chk_binary = tk.Checkbutton(self.root, text="Binary Image (Black & White)", command=self.apply_binary, bg="#2C3E50", fg="white")
        self.chk_binary.place(x=17, y=110)
        
        # Red Slider (Center at 0, range from -100 to 100)
        self.red_slider = tk.Scale(self.root, from_=-100, to=100, orient="horizontal", label="Red:", length=200, 
                   fg="white", bg="#2C3E50", highlightthickness=0, command=self.apply_rgb_filter)
        self.red_slider.set(0)  # Start at center
        self.red_slider.place(x=1690, y=520)

        # Green Slider (Center at 0, range from -100 to 100)
        self.green_slider = tk.Scale(self.root, from_=-100, to=100, orient="horizontal", label="Green:", length=200, 
                     fg="white", bg="#2C3E50", highlightthickness=0, command=self.apply_rgb_filter)
        self.green_slider.set(0)  # Start at center
        self.green_slider.place(x=1690, y=580)

        # Blue Slider (Center at 0, range from -100 to 100)
        self.blue_slider = tk.Scale(self.root, from_=-100, to=100, orient="horizontal", label="Blue:", length=200, 
                    fg="white", bg="#2C3E50", highlightthickness=0, command=self.apply_rgb_filter)
        self.blue_slider.set(0)  # Start at center
        self.blue_slider.place(x=1690, y=650)

        # Image Enhancer Section
        self.lbl_enhancer = tk.Label(self.root, text="IMAGE ENHANCER", font=("Arial", 10, "bold"), bg="#2C3E50", fg="white")
        self.lbl_enhancer.place(x=17, y=150)

       # Brightness Slider (Default: 1.0)
        self.brightness_slider = tk.Scale(self.root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", 
                                  label="Brightness", command=self.apply_enhancements,
                                  fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.brightness_slider.set(1.0)  # Set default value
        self.brightness_slider.place(x=17, y=180)

        # Contrast Slider (Default: 1.0)
        self.contrast_slider = tk.Scale(self.root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", 
                                label="Contrast", command=self.apply_enhancements,
                                fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.contrast_slider.set(1.0)  # Set default value
        self.contrast_slider.place(x=17, y=235)

        # Saturation Slider (Default: 1.0)
        self.saturation_slider = tk.Scale(self.root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", 
                                  label="Saturation", command=self.apply_enhancements,
                                  fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.saturation_slider.set(1.0)  # Set default value
        self.saturation_slider.place(x=17, y=290)

        # Blur Slider (Default: 0)
        self.blur_slider = tk.Scale(self.root, from_=0, to=10, resolution=1, orient="horizontal", 
                            label="Blur", command=self.apply_enhancements,
                            fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.blur_slider.set(0)  # Set default value
        self.blur_slider.place(x=17, y=348)

        # Shape Detection
        self.chk_shape = tk.Checkbutton(self.root, text="Shape Detection", bg="#2C3E50", fg="white", command=self.apply_shape_detection)
        self.chk_shape.place(x=17, y=570)

        # Color Detection Button
        self.btn_color_detection = tk.Button(self.root, text="Detect Color", command=self.apply_color_detection, bg="#E67E22", fg="white")
        self.btn_color_detection.place(x=40, y=610, width=80, height=40)
        
        # Update shape detection options to include all new shapes
        self.shape_choice = ttk.Combobox(self.root, values=["Circle", "Square", "Rectangle", "Triangle", "Quadrilateral", "Pentagon", "Hexagon", "Star", "Ellipse", "Diamond"])
        self.shape_choice.place(x=160, y=570, width=150)
        self.shape_choice.set("Circle")  # Default choice
        
        # Color detection combo
        self.color_choice = ttk.Combobox(self.root, values=["Yellow", "Red", "Blue", "Green"])
        self.color_choice.place(x=160, y=620, width=150)
        self.color_choice.set("Yellow")  # Default color choice

        # Image Display Panels
        self.panel_original = tk.Label(self.root, text="ORIGINAL IMAGE", bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.panel_original.place(x=250, y=20, width=710, height=450)
        
        self.panel_processed = tk.Label(self.root, text="CONVERTED IMAGE", bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.panel_processed.place(x=980, y=20, width=710, height=450)

        self.info_label = tk.Label(self.root, text="Detected Coins: 0", bg="#34495E", fg="white", width=80, height=3)
        self.info_label.place(x=400, y=570)

        # Shape & Color Information Display
        self.info_label = tk.Label(self.root, text="Display total shapes/colors detected", bg="#34495E", fg="white", width=80, height=3)
        self.info_label.place(x=400, y=630)
        
        # Title for the Filters Dropdown Menu
        self.filters_title = tk.Label(self.root, text="Other Filters", bg="#34495E", fg="white", width=28, height=2)
        self.filters_title.place(x=80, y=670)

        # Filters Dropdown Menu
        self.filter_options = [
            "None", "Grayscale", "Binary", "Blur", "Sepia", "Negative", 
            "Emboss", "Edge Detection", "Sharpen", "Oil Painting", "Cartoon",
            "Smoothing", "Gaussian Blur", "Mean Removal", "Emboss Laplacian",
            "Identity", "Ridge Detection", "Box Blur", "Unsharp Masking"
        ]
        
        self.filter_dropdown = ttk.Combobox(self.root, values=self.filter_options, state="readonly")
        self.filter_dropdown.place(x=80, y=700, width=200)
        self.filter_dropdown.set("None")  # Default selection
        self.filter_dropdown.bind("<<ComboboxSelected>>", self.apply_filter)

        # Save Image Button
        self.btn_save = tk.Button(self.root, text="Save Image", command=self.save_image, bg="#27AE60", fg="white", font=("Arial", 10, "bold"))
        self.btn_save.place(x=950, y=700, width=150, height=50)

        # Reset Image Button
        self.btn_reset = tk.Button(self.root, text="Reset Picture", command=self.reset_image, bg="#E74C3C", fg="white", font=("Arial", 10, "bold"))
        self.btn_reset.place(x=750, y=700, width=150, height=50)

        # Add Segmentation Section
        self.lbl_segmentation = tk.Label(self.root, text="IMAGE SEGMENTATION", font=("Arial", 10, "bold"), bg="#2C3E50", fg="white")
        self.lbl_segmentation.place(x=1400, y=555)

        # Number of Segments Slider that automatically applies segmentation when moved
        self.segment_slider = tk.Scale(self.root, from_=2, to=10, resolution=1, orient="horizontal", 
                      label="Number of Segments", command=self.apply_image_segmentation,
                      fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.segment_slider.set(4)  # Default to 4 segments
        self.segment_slider.place(x=1400, y=580, width=200)

        # Add Histogram Button
        self.btn_histogram = tk.Button(self.root, text="Show RGB Histogram", 
                        command=self.display_rgb_histogram,
                        bg="#3498DB", fg="white", font=("Arial", 10))
        self.btn_histogram.place(x=350, y=700, width=150, height=50)

        # Threshold Section
        self.lbl_threshold = tk.Label(self.root, text="THRESHOLD", font=("Arial", 10, "bold"), bg="#2C3E50", fg="white")
        self.lbl_threshold.place(x=1150, y=570)

        # Threshold Type Dropdown
        self.threshold_types = ["Simple", "Adaptive"]
        self.threshold_type_dropdown = ttk.Combobox(self.root, values=self.threshold_types, state="readonly")
        self.threshold_type_dropdown.place(x=1150, y=600, width=200)
        self.threshold_type_dropdown.set("Simple")  # Default selection
        self.threshold_type_dropdown.bind("<<ComboboxSelected>>", self.apply_threshold)

        # Threshold Value Slider
        self.threshold_slider = tk.Scale(self.root, from_=0, to=255, resolution=1, orient="horizontal", 
            label="Threshold Value", command=self.apply_threshold,
            fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.threshold_slider.set(128)  # Default value
        self.threshold_slider.place(x=1150, y=650, width=200)

        # Rotation Slider
        self.lbl_rotation = tk.Label(self.root, text="ROTATION", font=("Arial", 10, "bold"), bg="#2C3E50", fg="white")
        self.lbl_rotation.place(x=17, y=410)

        self.rotation_slider = tk.Scale(self.root, from_=-180, to=180, resolution=1, orient="horizontal", 
                       label="Rotation Angle", command=self.apply_rotation,
                       fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.rotation_slider.set(0)  # Default value
        self.rotation_slider.place(x=17, y=440, width=200)

        # Mirror Buttons
        self.btn_mirror_horizontal = tk.Button(self.root, text="Mirror Horizontally", 
                                               command=lambda: self.apply_mirror("horizontal"), 
                                               bg="#3498DB", fg="white")
        self.btn_mirror_horizontal.place(x=350, y=780, width=150, height=50)

        self.btn_mirror_vertical = tk.Button(self.root, text="Mirror Vertically", 
                                             command=lambda: self.apply_mirror("vertical"), 
                                             bg="#3498DB", fg="white")
        self.btn_mirror_vertical.place(x=530, y=780, width=150, height=50)

        # Translation Inputs and Button
        self.lbl_translation = tk.Label(self.root, text="Translation (x, y):", bg="#2C3E50", fg="white")
        self.lbl_translation.place(x=700, y=795)

        self.entry_x_shift = tk.Entry(self.root, width=5)
        self.entry_x_shift.place(x=810, y=795)
        self.entry_x_shift.insert(0, "0")  # Default value

        self.entry_y_shift = tk.Entry(self.root, width=5)
        self.entry_y_shift.place(x=870, y=795)
        self.entry_y_shift.insert(0, "0")  # Default value

        self.btn_translate = tk.Button(self.root, text="Translate", 
                                       command=lambda: self.apply_translation(int(self.entry_x_shift.get()), 
                                                                              int(self.entry_y_shift.get())), 
                                       bg="#E67E22", fg="white")
        self.btn_translate.place(x=930, y=788, width=150, height=30)

        # Add Binary Image Projection Button
        self.btn_projection = tk.Button(self.root, text="Binary Image Projection", 
                                command=self.apply_binary_projection, 
                                bg="#9B59B6", fg="white")
        self.btn_projection.place(x=1150, y=780, width=200, height=50)

        # Zoom Slider (Default: 1.0)
        self.zoom_slider = tk.Scale(self.root, from_=0.5, to=3.0, resolution=0.1, orient="horizontal",
                             label="Zoom", command=self.apply_zoom,
                             fg="white", bg="#2C3E50", highlightthickness=0, troughcolor="#34495E")
        self.zoom_slider.set(1.0)  # Set default value
        self.zoom_slider.place(x=17, y=500)

        self.offset_x = 0
        self.offset_y = 0
        self.zoom_factor = 1.0  # Default zoom factor

        # Bind arrow keys for navigation
        self.root.bind("<Up>", self.navigate_up)
        self.root.bind("<Down>", self.navigate_down)
        self.root.bind("<Left>", self.navigate_left)
        self.root.bind("<Right>", self.navigate_right)

        self.btn_up = tk.Button(self.root, text="↑", command=self.navigate_up, bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.btn_up.place(x=1690, y=750, width=50, height=50)

        self.btn_down = tk.Button(self.root, text="↓", command=self.navigate_down, bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.btn_down.place(x=1690, y=860, width=50, height=50)

        self.btn_left = tk.Button(self.root, text="←", command=self.navigate_left, bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.btn_left.place(x=1640, y=805, width=50, height=50)

        self.btn_right = tk.Button(self.root, text="→", command=self.navigate_right, bg="#34495E", fg="white", font=("Arial", 12, "bold"))
        self.btn_right.place(x=1740, y=805, width=50, height=50)

    def apply_threshold(self, event=None):
        if self.image is not None:
            # Convert to grayscale first
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Get threshold type and value
        threshold_type = self.threshold_type_dropdown.get()
        threshold_value = self.threshold_slider.get()
        
        if threshold_type == "Simple":
            # Simple thresholding
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        else:  # Adaptive
            # Adaptive thresholding
            thresholded = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11,  # Block size (must be odd)
                threshold_value - 128  # Adjust the subtraction value dynamically
            )
        
        # Convert back to RGB for display
        self.processed_image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        self.display_image(self.processed_image, self.panel_processed)
        
        # Update info label
        self.info_label.config(text=f"{threshold_type} Threshold Applied")

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = cv2.resize(self.original_image, (300, 300))
            self.image = self.original_image.copy()
            self.display_image(self.image, self.panel_original)
        
            self.info_label.config(text="Display total shapes/colors detected")

    def display_image(self, img, panel):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((700, 450))
        img_tk = ImageTk.PhotoImage(img)
        panel.configure(image=img_tk, width=400, height=400)
        panel.image = img_tk
    
    def count_piso_coins(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 30, param1=50, param2=30, minRadius=20, maxRadius=50)
            
            coin_count = 0
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    cv2.circle(self.image, (i[0], i[1]), i[2], (0, 255, 0), 3)
                    coin_count += 1
            
            self.info_label.config(text=f"Detected Coins: {coin_count}")
            self.display_image(self.image, self.panel_processed)

    def apply_filter(self, event=None):
        selected_filter = self.filter_dropdown.get()
        if selected_filter == "None":
            self.display_image(self.original_image, self.panel_processed)  # Display original image
        elif selected_filter == "Grayscale":
            self.apply_grayscale()
        elif selected_filter == "Binary":
            self.apply_binary()
        elif selected_filter == "Blur":
            self.apply_blur()
        elif selected_filter == "Sepia":
            self.apply_sepia()
        elif selected_filter == "Negative":
            self.apply_negative()
        elif selected_filter == "Emboss":
            self.apply_emboss()
        elif selected_filter == "Edge Detection":
            self.apply_edge_detection()
        elif selected_filter == "Sharpen":
            self.apply_sharpen()
        elif selected_filter == "Oil Painting":
            self.apply_oil_painting()
        elif selected_filter == "Cartoon":
            self.apply_cartoon()
        elif selected_filter == "Smoothing":
            self.apply_smoothing()
        elif selected_filter == "Gaussian Blur":
            self.apply_gaussian_blur()
        elif selected_filter == "Mean Removal":
            self.apply_mean_removal()
        elif selected_filter == "Emboss Laplacian":
            self.apply_emboss_laplacian()
        elif selected_filter == "Identity":
            self.apply_identity()
        elif selected_filter == "Ridge Detection":
            self.apply_ridge_detection()
        elif selected_filter == "Box Blur":
            self.apply_box_blur()
        elif selected_filter == "Unsharp Masking":
            self.apply_unsharp_masking()
        else:
            super().apply_filter(event)

    def apply_grayscale(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.panel_processed)
    
    def apply_binary(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            self.processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.panel_processed)
    
    def apply_blur(self):
        if self.image is not None:
            blurred = cv2.GaussianBlur(self.image, (15, 15), 0)
            self.processed_image = blurred
            self.display_image(self.processed_image, self.panel_processed)

    def apply_sepia(self):
        if self.image is not None:
            # Define sepia filter
            kernel = np.array([[0.393, 0.769, 0.189],
                               [0.349, 0.686, 0.168],
                               [0.272, 0.534, 0.131]])
            sepia = cv2.transform(self.image, kernel)
            self.processed_image = np.clip(sepia, 0, 255)  # Ensure values are within valid range
            self.display_image(self.processed_image, self.panel_processed)

    def apply_negative(self):
        if self.image is not None:
            negative = cv2.bitwise_not(self.image)
            self.processed_image = negative
            self.display_image(self.processed_image, self.panel_processed)

    def apply_emboss(self):
        if self.image is not None:
            kernel = np.array([[ -2, -1, 0],
                               [ -1,  1, 1],
                               [  0,  1, 2]])
            embossed = cv2.filter2D(self.image, -1, kernel)
            self.processed_image = embossed
            self.display_image(self.processed_image, self.panel_processed)

    def apply_edge_detection(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            self.display_image(self.processed_image, self.panel_processed)

    def apply_sharpen(self):
        if self.image is not None:
            kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
            sharpened = cv2.filter2D(self.image, -1, kernel)
            self.processed_image = sharpened
            self.display_image(self.processed_image, self.panel_processed)

    def apply_oil_painting(self):
        if self.image is not None:
            oil_painting = cv2.xphoto.createSimpleWB().balanceWhite(self.image)
            self.processed_image = oil_painting
            self.display_image(self.processed_image, self.panel_processed)

    def apply_cartoon(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(self.image, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            self.processed_image = cartoon
            self.display_image(self.processed_image, self.panel_processed)

    def apply_rgb_filter(self, event=None):
        if self.original_image is not None:
            img = self.original_image.copy()
    
    # Get slider values (range from -100 to 100)
        r_value = self.red_slider.get()
        g_value = self.green_slider.get()
        b_value = self.blue_slider.get()
    
    # Convert image to float for precise manipulation
        img = img.astype(np.float32)

    # Adjust each color channel based on slider value
    # For positive values, we multiply the channel (like brightness)
    # For negative values, we reduce the channel
        if r_value >= 0:
            factor = 1 + (r_value / 100)  # Range: 1.0 to 2.0
            img[:, :, 2] *= factor  # Adjust Red Channel
        else:
            factor = 1 + (r_value / 100)  # Range: 0.0 to 1.0
            img[:, :, 2] *= factor
    
        if g_value >= 0:
            factor = 1 + (g_value / 100)  # Range: 1.0 to 2.0
            img[:, :, 1] *= factor  # Adjust Green Channel
        else:
            factor = 1 + (g_value / 100)  # Range: 0.0 to 1.0
            img[:, :, 1] *= factor
    
        if b_value >= 0:
            factor = 1 + (b_value / 100)  # Range: 1.0 to 2.0
            img[:, :, 0] *= factor  # Adjust Blue Channel
        else:
            factor = 1 + (b_value / 100)  # Range: 0.0 to 1.0
            img[:, :, 0] *= factor

    # Clip values to stay within [0, 255] range and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
    
        self.image = img
        self.processed_image = img  # Update the processed image
        self.display_image(self.image, self.panel_processed)

    def apply_enhancements(self, event=None):
        if self.processed_image is not None:  # Apply enhancements on the filtered image
            img = self.processed_image.copy().astype(np.float32) / 255.0  # Normalize to [0, 1]

        # Apply brightness
        brightness_factor = self.brightness_slider.get()
        img = np.clip(img * brightness_factor, 0, 1)  # Adjust brightness

        # Convert back to 0-255 range
        img = (img * 255).astype(np.uint8)

        # Apply contrast
        contrast_factor = self.contrast_slider.get()
        img = cv2.convertScaleAbs(img, alpha=contrast_factor, beta=0)

        # Convert to PIL for saturation adjustment
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Apply saturation
        enhancer = ImageEnhance.Color(img_pil)
        img_pil = enhancer.enhance(self.saturation_slider.get())

        # Convert back to OpenCV format
        img = np.array(img_pil)

        # Apply blur if needed
        blur_value = int(self.blur_slider.get())
        if blur_value > 0:
            img = cv2.GaussianBlur(img, (blur_value * 2 + 1, blur_value * 2 + 1), 0)

        # Convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Update processed image and display it
        self.processed_image = img
        self.display_image(self.processed_image, self.panel_processed)

    def save_image(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("All Files", "*.*")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))

    def reset_image(self):
        if self.original_image is not None:
            # Reset all checkboxes
            self.chk_grayscale.deselect()
            self.chk_binary.deselect()

            # Reset RGB sliders to 0
            self.red_slider.set(0)
            self.green_slider.set(0)
            self.blue_slider.set(0)

            # Reset Image Enhancer sliders to default values
            self.brightness_slider.set(1.0)
            self.contrast_slider.set(1.0)
            self.saturation_slider.set(1.0)
            self.blur_slider.set(0)
            self.rotation_slider.set(0)

            # Restore the original image
            self.image = self.original_image.copy()
            self.processed_image = self.original_image.copy()

            # Display the original image in both panels
            self.display_image(self.image, self.panel_original)
            self.display_image(self.processed_image, self.panel_processed)

            # Clear the shape/color detection info
            self.info_label.config(text="Display total shapes/colors detected")

    def apply_shape_detection(self):
        shape_type = self.shape_choice.get()

        if self.image is not None:
            # Make a copy of the original image to draw on
            display_image = self.image.copy()
        
           # Convert to HSV for better color detection
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Define yellow color range
            lower_yellow = np.array([20, 100, 100])  # Adjust if needed
            upper_yellow = np.array([30, 255, 255])
        
        # Create mask to detect yellow objects
            yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # Convert to grayscale for shape detection
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Combine threshold with yellow mask
            combined_mask = cv2.bitwise_or(thresh, yellow_mask)
        
        # Find contours on the combined mask
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            display_image = self.image.copy()
            detected_shapes = 0
                        
            # Process each contour
            for contour in contours:
                # Ignore small contours
                if cv2.contourArea(contour) < 100:
                    continue
                
                # Approximate the contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            
                # Detect triangles
                if len(approx) == 3 and shape_type == "Triangle":
                    cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                    detected_shapes += 1

                # Detect general quadrilaterals
                elif len(approx) == 4:
                    # For general quadrilaterals
                    if shape_type == "Quadrilateral":
                        cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                        detected_shapes += 1
                        continue
                        
                    # For specific types of quadrilaterals (square/rectangle)
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # square detection
                    if shape_type == "Square":
                        # Check if it's a square by comparing aspect ratio
                        if 0.8 <= aspect_ratio <= 1.2:
                            # Additional check: verify that all angles are approximately 90 degrees
                            is_square = True
                            for i in range(4):
                                # Get three consecutive points
                                p1 = approx[i][0]
                                p2 = approx[(i+1) % 4][0]
                                p3 = approx[(i+2) % 4][0]
                                
                                # Calculate vectors
                                v1 = [p2[0] - p1[0], p2[1] - p1[1]]
                                v2 = [p3[0] - p2[0], p3[1] - p2[1]]
                                
                                # Calculate dot product
                                dot = v1[0] * v2[0] + v1[1] * v2[1]
                                
                                # Calculate magnitudes
                                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                                
                                # Calculate angle in degrees
                                if mag1 * mag2 != 0:  # Avoid division by zero
                                    cos_angle = dot / (mag1 * mag2)
                                    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
                                    angle = np.degrees(np.arccos(cos_angle))
                                    
                                    # Check if angle is approximately 90 degrees
                                    if abs(angle - 90) > 15:  # Allow some error margin
                                        is_square = False
                                        break
                            
                            if is_square:
                                cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                                detected_shapes += 1
                    
                    # Rectangle detection
                    elif shape_type == "Rectangle" and (aspect_ratio < 0.8 or aspect_ratio > 1.2):
                        # Check if all angles are approximately 90 degrees
                        is_rectangle = True
                        for i in range(4):
                            p1 = approx[i][0]
                            p2 = approx[(i+1) % 4][0]
                            p3 = approx[(i+2) % 4][0]
                            
                            v1 = [p2[0] - p1[0], p2[1] - p1[1]]
                            v2 = [p3[0] - p2[0], p3[1] - p2[1]]
                            
                            dot = v1[0] * v2[0] + v1[1] * v2[1]
                            mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                            mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                            
                            if mag1 * mag2 != 0:
                                cos_angle = dot / (mag1 * mag2)
                                cos_angle = max(-1, min(1, cos_angle))
                                angle = np.degrees(np.arccos(cos_angle))
                                
                                if abs(angle - 90) > 15:
                                    is_rectangle = False
                                    break
                        
                        if is_rectangle:
                            cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                            detected_shapes += 1

                    # New Diamond Shape Detection
                    elif shape_type == "Diamond":
                    # Check if the shape has 4 vertices (quadrilateral)
                        if len(approx) == 4:
                        # Calculate side lengths and angles
                            sides = []
                            for i in range(4):
                                p1 = approx[i][0]
                                p2 = approx[(i+1) % 4][0]
                        
                                # Calculate side length
                                side_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                                sides.append(side_length)
                            
                            # Initialize variables
                            side_variation = 1
                            diagonal_variation = 1

                            # Calculate diagonals
                            try:
                                # Diagonal 1: from first to third point
                                diagonal1 = np.sqrt((approx[0][0][0] - approx[2][0][0])**2 + 
                                                    (approx[0][0][1] - approx[2][0][1])**2)
            
                                # Diagonal 2: from second to fourth point
                                diagonal2 = np.sqrt((approx[1][0][0] - approx[3][0][0])**2 + 
                                                    (approx[1][0][1] - approx[3][0][1])**2)

                                if len(sides) > 0:
                                    side_variation = (max(sides) - min(sides)) / np.mean(sides)
                
                                # Diagonal length variation
                                diagonal_variation = abs(diagonal1 - diagonal2) / ((diagonal1 + diagonal2) / 2)
                                
                            except Exception:
                                side_variation = 1
                                diagonal_variation = 1

                            # Calculate angles
                            angles = []
                            for i in range(4):
                                p1 = approx[i][0]
                                p2 = approx[(i+1) % 4][0]
                                p0 = approx[(i-1) % 4][0]

                                # Vector of current side
                                v1 = [p2[0] - p1[0], p2[1] - p1[1]]
                                # Vector of previous side
                                v2 = [p1[0] - p0[0], p1[1] - p0[1]]
                            
                                # Calculate dot product
                                dot = v1[0] * v2[0] + v1[1] * v2[1]
                            
                                # Calculate magnitudes
                                mag1 = np.sqrt(v1[0]**2 + v1[1]**2)
                                mag2 = np.sqrt(v2[0]**2 + v2[1]**2)
                            
                                # Calculate angle
                                if mag1 * mag2 != 0:
                                    cos_angle = dot / (mag1 * mag2)
                                    cos_angle = max(-1, min(1, cos_angle))
                                    angle = np.degrees(np.arccos(cos_angle))
                                    angles.append(angle)
                    
                    # Diamond detection criteria:
                    # 1. Side lengths should be relatively close (allow some variation)
                    # 2. Opposite sides should be roughly parallel
                    # 3. Angles between sides should be close to 45 or 135 degrees
                    
                            # Diamond detection criteria
                            is_diamond = (
                                len(sides) == 4 and 
                                len(angles) > 0 and
                                side_variation < 0.4 and  # More lenient side variation
                                diagonal_variation < 0.4 and  # Check diagonal similarity
                                # More flexible angle check to catch tilted diamonds
                                all(abs(angle - 45) < 25 or abs(angle - 135) < 25 for angle in angles)
                            )

                            if is_diamond:
                                cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                                detected_shapes += 1

                # Detect pentagon
                elif len(approx) == 5 and shape_type == "Pentagon":
                    cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                    detected_shapes += 1
                
                # Detect hexagon
                elif len(approx) == 6 and shape_type == "Hexagon":
                    cv2.drawContours(display_image, [approx], -1, (0, 255, 0), 3)
                    detected_shapes += 1
                
                # Detect star (typically has 10 points in approx due to inner and outer points)
                elif shape_type == "Star" and (len(approx) >= 8 and len(approx) <= 12):
                    # Additional check for star shape: verify alternating points pattern
                    # This is a simplified check and may need refinement based on your specific needs
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(contour)
                    
                    # Stars typically have a specific area ratio of contour to its convex hull
                    if contour_area / hull_area < 0.6:  # Adjust this threshold as needed
                        cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 3)
                        detected_shapes += 1
                
                # Detect circles
                elif shape_type == "Circle":
                    # Calculate how circular the contour is
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    # If circularity is close to 1, it's likely a circle
                    if circularity > 0.7:
                        cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 3)
                        detected_shapes += 1
                
            # Update the processed image and display it
            self.processed_image = display_image
            self.display_image(display_image, self.panel_processed)
            
            # Update the info label
            self.info_label.config(text=f"Total {shape_type}s Detected: {detected_shapes}")

    def apply_color_detection(self):
        color_type = self.color_choice.get()  # Get the chosen color type from the dropdown or options
        if self.image is not None:
        # Convert the image to HSV color space
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # Define color ranges in HSV space for Yellow, Red, Blue, and Green
        color_ranges = {
            "Yellow": [(20, 100, 100), (30, 255, 255)],  # Hue between 20 and 30 for yellow
            "Red": [(0, 100, 100), (10, 255, 255)],  # Lower red range
            "Red_2": [(170, 100, 100), (180, 255, 255)],  # Upper red range (wrap-around)
            "Blue": [(100, 100, 100), (130, 255, 255)],  # Blue range in HSV
            "Green": [(40, 100, 100), (80, 255, 255)],  # Green range in HSV
        }

        detected_color_count = 0  # Variable to keep track of detected regions for specific colors

        # Define a mask for each color and detect them
        if color_type == "Yellow":
            lower_yellow = np.array(color_ranges["Yellow"][0])
            upper_yellow = np.array(color_ranges["Yellow"][1])
            mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
            result_yellow = cv2.bitwise_and(self.image, self.image, mask=mask_yellow)
            detected_color_count += np.count_nonzero(mask_yellow)  # Count the yellow pixels
            cv2.imshow("Yellow Color Detection", result_yellow)

        if color_type == "Red":
            lower_red = np.array(color_ranges["Red"][0])
            upper_red = np.array(color_ranges["Red"][1])
            mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
            result_red = cv2.bitwise_and(self.image, self.image, mask=mask_red)
            detected_color_count += np.count_nonzero(mask_red)  # Count the red pixels
            cv2.imshow("Red Color Detection", result_red)

        if color_type == "Blue":
            lower_blue = np.array(color_ranges["Blue"][0])
            upper_blue = np.array(color_ranges["Blue"][1])
            mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
            result_blue = cv2.bitwise_and(self.image, self.image, mask=mask_blue)
            detected_color_count += np.count_nonzero(mask_blue)  # Count the blue pixels
            cv2.imshow("Blue Color Detection", result_blue)

        if color_type == "Green":
            lower_green = np.array(color_ranges["Green"][0])
            upper_green = np.array(color_ranges["Green"][1])
            mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
            result_green = cv2.bitwise_and(self.image, self.image, mask=mask_green)
            detected_color_count += np.count_nonzero(mask_green)  # Count the green pixels
            cv2.imshow("Green Color Detection", result_green)

        # Display the detected color count on the screen
        self.info_label.config(text=f"Detected {color_type}: {detected_color_count} pixels")
    
    def apply_image_segmentation(self, event=None):
            if self.image is not None:
        # Get the number of segments from the slider
                num_segments = self.segment_slider.get()
        
        # Convert image to RGB for processing (OpenCV uses BGR)
            img_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        # Reshape the image to a 2D array of pixels
            pixel_values = img_rgb.reshape((-1, 3))
            pixel_values = np.float32(pixel_values)
        
        # Define criteria for k-means
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Apply k-means clustering
            _, labels, centers = cv2.kmeans(pixel_values, num_segments, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to 8-bit values
            centers = np.uint8(centers)
        
        # Map labels to center values
            segmented_image = centers[labels.flatten()]
        
        # Reshape back to the original image shape
            segmented_image = segmented_image.reshape(self.image.shape)
        
        # Convert back to BGR for OpenCV
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        
        # Display the segmented image
            self.processed_image = segmented_image
            self.display_image(segmented_image, self.panel_processed)
        
        # Update info label
            self.info_label.config(text=f"Image segmented into {num_segments} regions")

    def display_rgb_histogram(self):
        if self.image is not None:
            # Create a figure with subplots for each channel plus yellow (R+G)
            plt.figure(figsize=(12, 8))
        
        # Split the channels
            b, g, r = cv2.split(self.image)
        
        # Create yellow channel (combination of red and green)
            yellow = cv2.addWeighted(r, 0.5, g, 0.5, 0)
        
        # Create histograms for each channel
            hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
            hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
            hist_y = cv2.calcHist([yellow], [0], None, [256], [0, 256])
        
        # Plot Red Histogram
            plt.subplot(2, 2, 1)
            plt.plot(hist_r, color='red')
            plt.title('Red Channel Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
        
        # Plot Green Histogram
            plt.subplot(2, 2, 2)
            plt.plot(hist_g, color='green')
            plt.title('Green Channel Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
        
        # Plot Blue Histogram
            plt.subplot(2, 2, 3)
            plt.plot(hist_b, color='blue')
            plt.title('Blue Channel Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
        
        # Plot Yellow Histogram
            plt.subplot(2, 2, 4)
            plt.plot(hist_y, color='gold')
            plt.title('Yellow Channel Histogram')
            plt.xlim([0, 256])
            plt.grid(True, alpha=0.3)
        
        # Adjust layout and show the figure
            plt.tight_layout()
            plt.savefig('temp_histogram.png')
            plt.close()
        
        # Display the histogram in a new window
            histogram_image = cv2.imread('temp_histogram.png')
            cv2.imshow('RGB Histograms', histogram_image)
        
        # Update info label
            self.info_label.config(text="RGB Histograms displayed in separate window")

    def apply_rotation(self, event=None):
        if self.image is not None:
        # Get rotation angle from a new slider (or you can add a text input)
            rotation_angle = self.rotation_slider.get()
        
        # Get image dimensions
        height, width = self.image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Calculate new dimensions after rotation
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        new_width = int(height * abs_sin + width * abs_cos)
        new_height = int(height * abs_cos + width * abs_sin)
        
        # Adjust the rotation matrix
        rotation_matrix[0, 2] += new_width / 2 - center[0]
        rotation_matrix[1, 2] += new_height / 2 - center[1]
        
        # Apply rotation
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (new_width, new_height), 
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=(0, 0, 0))
        
        # Update processed image and display
        self.processed_image = rotated_image
        self.display_image(self.processed_image, self.panel_processed)
        
        # Update info label
        self.info_label.config(text=f"Image rotated by {rotation_angle} degrees")      

    def apply_mirror(self, direction):
        if self.image is not None:
            if direction == "horizontal":
                mirrored_image = cv2.flip(self.image, 1)  # Flip horizontally
            elif direction == "vertical":
                mirrored_image = cv2.flip(self.image, 0)  # Flip vertically
            else:
                return  # Invalid direction
            
            self.processed_image = mirrored_image
            self.display_image(self.processed_image, self.panel_processed)
            self.info_label.config(text=f"Image mirrored {direction}")

    def apply_translation(self, x_shift, y_shift):
        if self.image is not None:
            height, width = self.image.shape[:2]
            translation_matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            translated_image = cv2.warpAffine(self.image, translation_matrix, (width, height))
            
            self.processed_image = translated_image
            self.display_image(self.processed_image, self.panel_processed)
            self.info_label.config(text=f"Image translated by ({x_shift}, {y_shift})")

    def apply_binary_projection(self):
        if self.image is not None:
        # Convert the image to grayscale and then to binary
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

        # Calculate horizontal and vertical projections
        horizontal_projection = np.sum(binary == 255, axis=1)  # Count white pixels along rows
        vertical_projection = np.sum(binary == 255, axis=0)    # Count white pixels along columns

        # Plot the projections as bar plots
        plt.figure(figsize=(12, 6))

        # Horizontal Projection
        plt.subplot(1, 2, 1)
        plt.barh(range(len(horizontal_projection)), horizontal_projection, color='black', height=1.0)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.title("Horizontal Projection")
        plt.xlabel("Count of White Pixels")
        plt.ylabel("Rows")
        plt.grid(which='both', linestyle='--', alpha=0.5)  # Add smaller gridlines
        plt.minorticks_on()  # Enable minor ticks for finer gridlines

        # Vertical Projection
        plt.subplot(1, 2, 2)
        plt.bar(range(len(vertical_projection)), vertical_projection, color='black', width=1.0)
        plt.title("Vertical Projection")
        plt.xlabel("Columns")
        plt.ylabel("Count of White Pixels")
        plt.grid(which='both', linestyle='--', alpha=0.5)  # Add smaller gridlines
        plt.minorticks_on()  # Enable minor ticks for finer gridlines

        # Show the plots
        plt.tight_layout()
        plt.show()

        # Update info label
        self.info_label.config(text="Binary Image Projection displayed")

    def apply_convolution(self, kernel):
        """Apply a convolution filter using the given kernel."""
        if self.image is not None:
            filtered_image = cv2.filter2D(self.image, -1, kernel)
            self.processed_image = filtered_image
            self.display_image(self.processed_image, self.panel_processed)

    def apply_smoothing(self):
        """Apply a smoothing filter."""
        kernel = np.ones((5, 5), np.float32) / 25  # 5x5 averaging kernel
        self.apply_convolution(kernel)

    def apply_gaussian_blur(self):
        """Apply a Gaussian blur filter."""
        blurred_image = cv2.GaussianBlur(self.image, (5, 5), 0)
        self.processed_image = blurred_image
        self.display_image(self.processed_image, self.panel_processed)

    def apply_mean_removal(self):
        """Apply a mean removal filter."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        self.apply_convolution(kernel)

    def apply_emboss_laplacian(self):
        """Apply an emboss Laplacian filter."""
        kernel = np.array([[-1, 0, -1],
                           [0, 4, 0],
                           [-1, 0, -1]])
        self.apply_convolution(kernel)

    def apply_identity(self):
        """Apply an identity filter."""
        kernel = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        self.apply_convolution(kernel)

    def apply_ridge_detection(self):
        """Apply a ridge detection filter."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]])
        self.apply_convolution(kernel)

    def apply_box_blur(self):
        """Apply a box blur filter."""
        kernel = np.ones((3, 3), np.float32) / 9  # 3x3 averaging kernel
        self.apply_convolution(kernel)

    def apply_unsharp_masking(self):
        """Apply an unsharp masking filter."""
        gaussian_blur = cv2.GaussianBlur(self.image, (5, 5), 1.0)
        unsharp_image = cv2.addWeighted(self.image, 1.5, gaussian_blur, -0.5, 0)
        self.processed_image = unsharp_image
        self.display_image(self.processed_image, self.panel_processed)

    def apply_zoom(self, event=None):
        """Apply zoom to the image."""
        if self.image is not None:
        # Get the zoom factor from the slider
            self.zoom_factor = self.zoom_slider.get()

        # Get the dimensions of the original image
        height, width = self.image.shape[:2]

        # Calculate new dimensions
        new_width = int(width * self.zoom_factor)
        new_height = int(height * self.zoom_factor)

        # Resize the image
        zoomed_image = cv2.resize(self.image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Calculate the visible region based on offsets
        display_width, display_height = 700, 450
        x_start = max(0, min(new_width - display_width, self.offset_x))
        y_start = max(0, min(new_height - display_height, self.offset_y))
        x_end = x_start + display_width
        y_end = y_start + display_height

        # Ensure the cropped region is within bounds
        cropped_image = zoomed_image[y_start:y_end, x_start:x_end]

        # If the cropped region is smaller than the display area, pad it
        if cropped_image.shape[0] < display_height or cropped_image.shape[1] < display_width:
            padded_image = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            padded_image[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
            cropped_image = padded_image

        # Update the processed image and display it
        self.processed_image = cropped_image
        self.display_image(self.processed_image, self.panel_processed)

        # Update the info label
        self.info_label.config(text=f"Zoom: {self.zoom_factor}x | Offset: ({self.offset_x}, {self.offset_y})")

    def navigate_up(self, event=None):
        """Move the view up."""
        if self.image is not None:
            self.offset_y = max(self.offset_y - 50, 0)  # Move up by 50 pixels
            self.apply_zoom()

    def navigate_down(self, event=None):
        """Move the view down."""
        if self.image is not None:
            max_offset_y = max(0, int(self.image.shape[0] * self.zoom_factor) - 450)
            self.offset_y = min(self.offset_y + 50, max_offset_y)  # Move down by 50 pixels
            self.apply_zoom()

    def navigate_left(self, event=None):
        """Move the view left."""
        if self.image is not None:
            self.offset_x = max(self.offset_x - 50, 0)  # Move left by 50 pixels
            self.apply_zoom()

    def navigate_right(self, event=None):
        """Move the view right."""
        if self.image is not None:
            max_offset_x = max(0, int(self.image.shape[1] * self.zoom_factor) - 700)
            self.offset_x = min(self.offset_x + 50, max_offset_x)  # Move right by 50 pixels
            self.apply_zoom()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessor(root)
    root.mainloop()
