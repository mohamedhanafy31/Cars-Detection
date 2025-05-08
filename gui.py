import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
from inference import CarDetector

class VehicleDetectionGUI:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Vehicle Detection System")
        self.window.geometry("1200x800")
        
        try:
            # Initialize the detector with CPU device
            model_path = 'final_model.pth'  # Updated path to match your model location
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found at: {os.path.abspath(model_path)}")
            self.detector = CarDetector(model_path)
            self.setup_ui()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize the model: {str(e)}\nPlease make sure you have the correct model file and dependencies installed.")
            self.window.destroy()
            return

    def setup_ui(self):
        # Create main frame
        self.main_frame = ctk.CTkFrame(self.window)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create left frame for controls
        self.left_frame = ctk.CTkFrame(self.main_frame)
        self.left_frame.pack(side="left", fill="y", padx=10, pady=10)
        
        # Create right frame for image display
        self.right_frame = ctk.CTkFrame(self.main_frame)
        self.right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        # Create image display frames
        self.original_frame = ctk.CTkFrame(self.right_frame)
        self.original_frame.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.result_frame = ctk.CTkFrame(self.right_frame)
        self.result_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
        
        # Create labels for images
        self.original_label = ctk.CTkLabel(self.original_frame, text="Original Image")
        self.original_label.pack(pady=5)
        
        self.result_label = ctk.CTkLabel(self.result_frame, text="Detected Vehicles")
        self.result_label.pack(pady=5)
        
        # Create image display areas
        self.original_image_label = ctk.CTkLabel(self.original_frame, text="")
        self.original_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.result_image_label = ctk.CTkLabel(self.result_frame, text="")
        self.result_image_label.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create buttons
        self.select_button = ctk.CTkButton(
            self.left_frame,
            text="Select Image",
            command=self.select_image,
            width=200,
            height=40
        )
        self.select_button.pack(pady=20)
        
        self.detect_button = ctk.CTkButton(
            self.left_frame,
            text="Detect Vehicles",
            command=self.detect_vehicles,
            width=200,
            height=40,
            state="disabled"
        )
        self.detect_button.pack(pady=20)
        
        # Add status label
        self.status_label = ctk.CTkLabel(
            self.left_frame,
            text="Ready to process images",
            wraplength=200
        )
        self.status_label.pack(pady=20)
        
        # Initialize variables
        self.current_image = None
        self.current_image_path = None
        
    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                self.current_image_path = file_path
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise Exception("Failed to load image")
                self.display_original_image()
                self.detect_button.configure(state="normal")
                self.status_label.configure(text="Image loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def display_original_image(self):
        if self.current_image is not None:
            try:
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                # Resize image to fit the display area
                height, width = image_rgb.shape[:2]
                max_height = 400
                if height > max_height:
                    ratio = max_height / height
                    width = int(width * ratio)
                    height = max_height
                image_resized = cv2.resize(image_rgb, (width, height))
                
                # Convert to PhotoImage
                image_pil = Image.fromarray(image_resized)
                image_tk = ImageTk.PhotoImage(image_pil)
                
                # Update label
                self.original_image_label.configure(image=image_tk)
                self.original_image_label.image = image_tk
            except Exception as e:
                messagebox.showerror("Error", f"Failed to display image: {str(e)}")
            
    def detect_vehicles(self):
        if self.current_image is not None:
            try:
                self.status_label.configure(text="Processing image...")
                self.window.update()
                
                # Get detections
                detections = self.detector.detect(self.current_image)
                
                # Draw detections
                result_image = self.detector.draw_detections(self.current_image, detections)
                
                # Display result image
                image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                height, width = image_rgb.shape[:2]
                max_height = 400
                if height > max_height:
                    ratio = max_height / height
                    width = int(width * ratio)
                    height = max_height
                image_resized = cv2.resize(image_rgb, (width, height))
                
                # Convert to PhotoImage
                image_pil = Image.fromarray(image_resized)
                image_tk = ImageTk.PhotoImage(image_pil)
                
                # Update label
                self.result_image_label.configure(image=image_tk)
                self.result_image_label.image = image_tk
                
                self.status_label.configure(text=f"Found {len(detections)} vehicles")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
                self.status_label.configure(text="Error processing image")
            
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = VehicleDetectionGUI()
    app.run() 