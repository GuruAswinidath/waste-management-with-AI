import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import random
from collections import deque

class WasteClassifier:
    def __init__(self):
        self.waste_categories = {
            'plastic': ['plastic bottle', 'plastic bag', 'plastic container'],
            'glass': ['glass bottle', 'glass jar', 'glass container'],
            'metal': ['metal can', 'metal container', 'aluminum'],
            'paper': ['paper', 'cardboard', 'newspaper'],
            'organic': ['food waste', 'vegetable', 'fruit'],
            'wood': ['wooden item', 'wood', 'wooden container']
        }
        
        # Initialize AI model
        self.model = MobileNetV2(weights='imagenet')
        self.camera = None
        self.is_running = False
        self.belt_items = deque(maxlen=10)
        self.classification_results = []
        
        # Belt simulation parameters
        self.belt_speed = 1.0
        self.belt_position = 0
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("AI Waste Classification System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(main_frame, text="AI Waste Classification System", 
                             font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = tk.Frame(main_frame, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Camera controls
        camera_frame = tk.LabelFrame(control_frame, text="Camera Controls", 
                                   font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        camera_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
        
        self.start_btn = tk.Button(camera_frame, text="Start Camera", command=self.start_camera,
                                 bg='#27ae60', fg='white', font=('Arial', 10, 'bold'))
        self.start_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.stop_btn = tk.Button(camera_frame, text="Stop Camera", command=self.stop_camera,
                                bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'))
        self.stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Belt controls
        belt_frame = tk.LabelFrame(control_frame, text="Belt Controls", 
                                 font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        belt_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.X, expand=True)
        
        tk.Label(belt_frame, text="Speed:", fg='white', bg='#34495e').pack(side=tk.LEFT, padx=5)
        self.speed_scale = tk.Scale(belt_frame, from_=0.1, to=3.0, resolution=0.1, 
                                  orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Main display area
        display_frame = tk.Frame(main_frame, bg='#34495e')
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Camera view
        camera_view_frame = tk.LabelFrame(display_frame, text="Camera View", 
                                        font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        camera_view_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.camera_label = tk.Label(camera_view_frame, text="Camera not started", 
                                   bg='black', fg='white', font=('Arial', 16))
        self.camera_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Belt simulation
        belt_sim_frame = tk.LabelFrame(display_frame, text="Belt Simulation", 
                                      font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        belt_sim_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.belt_canvas = tk.Canvas(belt_sim_frame, bg='#2c3e50', height=300)
        self.belt_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(display_frame, text="Classification Results", 
                                    font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.results_text = tk.Text(results_frame, bg='#2c3e50', fg='white', 
                                   font=('Consolas', 10), height=15)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("System Ready")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg='#95a5a6', fg='white')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize belt simulation
        self.init_belt_simulation()
        
    def init_belt_simulation(self):
        """Initialize the belt simulation canvas"""
        self.belt_canvas.delete("all")
        
        # Draw belt
        belt_width = 400
        belt_height = 60
        x = 50
        y = 120
        
        # Belt background
        self.belt_canvas.create_rectangle(x, y, x + belt_width, y + belt_height, 
                                        fill='#7f8c8d', outline='#2c3e50', width=2)
        
        # Belt rollers
        for i in range(6):
            roller_x = x + i * (belt_width // 5)
            self.belt_canvas.create_oval(roller_x - 15, y - 15, roller_x + 15, y + 15, 
                                       fill='#34495e', outline='#2c3e50')
        
        # Belt movement indicators
        for i in range(8):
            arrow_x = x + i * (belt_width // 7) + 20
            self.belt_canvas.create_polygon(arrow_x, y + 10, arrow_x + 10, y + 20, 
                                          arrow_x, y + 30, fill='#e74c3c')
        
        self.belt_canvas.create_text(250, 50, text="Moving Belt", 
                                   fill='white', font=('Arial', 14, 'bold'))
        
    def start_camera(self):
        """Start the camera and begin processing"""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
                
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_var.set("Camera Running - Processing Waste")
            
            # Start processing thread
            self.process_thread = threading.Thread(target=self.process_camera_feed)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            # Start belt simulation
            self.belt_simulation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        """Stop the camera and processing"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Camera Stopped")
        
        # Clear camera view
        self.camera_label.config(text="Camera stopped", image="")
        
    def process_camera_feed(self):
        """Process camera feed and classify waste"""
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Process frame for waste detection
                processed_frame, classification = self.detect_and_classify_waste(frame)
                
                # Convert to PIL Image for tkinter
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
                tk_image = ImageTk.PhotoImage(pil_image)
                
                # Update camera view
                self.camera_label.config(image=tk_image, text="")
                self.camera_label.image = tk_image
                
                # Add classification result
                if classification:
                    self.add_classification_result(classification)
                    
                # Simulate belt movement
                self.simulate_belt_movement(classification)
                
            time.sleep(0.1)  # Control processing rate
            
    def detect_and_classify_waste(self, frame):
        """Detect and classify waste in the frame"""
        # Resize frame for model input
        input_size = (224, 224)
        frame_resized = cv2.resize(frame, input_size)
        
        # Convert to array and preprocess
        img_array = image.img_to_array(frame_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Get predictions
        predictions = self.model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        
        # Map predictions to waste categories
        waste_classification = self.map_to_waste_category(decoded_predictions)
        
        # Draw classification on frame
        annotated_frame = self.draw_classification_on_frame(frame, waste_classification)
        
        return annotated_frame, waste_classification
        
    def map_to_waste_category(self, predictions):
        """Map ImageNet predictions to waste categories"""
        for pred in predictions:
            label = pred[1].lower()
            confidence = pred[2]
            
            # Check if prediction matches any waste category
            for category, keywords in self.waste_categories.items():
                for keyword in keywords:
                    if keyword in label and confidence > 0.3:
                        return {
                            'category': category,
                            'label': label,
                            'confidence': confidence,
                            'timestamp': time.time()
                        }
        
        return None
        
    def draw_classification_on_frame(self, frame, classification):
        """Draw classification results on the frame"""
        if classification:
            # Draw bounding box
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (50, 50), (width-50, height-50), (0, 255, 0), 2)
            
            # Draw classification text
            text = f"{classification['category'].upper()}: {classification['confidence']:.2f}"
            cv2.putText(frame, text, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw category color indicator
            color_map = {
                'plastic': (255, 0, 0),    # Blue
                'glass': (0, 255, 255),    # Cyan
                'metal': (0, 0, 255),      # Red
                'paper': (255, 255, 0),    # Yellow
                'organic': (0, 255, 0),    # Green
                'wood': (139, 69, 19)      # Brown
            }
            
            if classification['category'] in color_map:
                color = color_map[classification['category']]
                cv2.circle(frame, (width-100, 100), 30, color, -1)
                
        return frame
        
    def add_classification_result(self, classification):
        """Add classification result to the results display"""
        timestamp = time.strftime("%H:%M:%S", time.localtime(classification['timestamp']))
        result_text = f"[{timestamp}] {classification['category'].upper()}: {classification['confidence']:.2f}\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)
        
        # Keep only last 50 results
        if self.results_text.index(tk.END).split('.')[0] > '50':
            self.results_text.delete('1.0', '2.0')
            
    def simulate_belt_movement(self, classification):
        """Simulate belt movement and item sorting"""
        if classification:
            # Add item to belt
            item = {
                'category': classification['category'],
                'position': 0,
                'timestamp': time.time()
            }
            self.belt_items.append(item)
            
    def belt_simulation(self):
        """Animate belt movement"""
        def animate():
            while self.is_running:
                # Update belt position
                self.belt_position += self.speed_scale.get() * 0.5
                if self.belt_position > 400:
                    self.belt_position = 0
                    
                # Update item positions
                for item in self.belt_items:
                    item['position'] += self.speed_scale.get() * 0.5
                    
                # Remove items that have moved off the belt
                self.belt_items = deque([item for item in self.belt_items 
                                       if item['position'] < 450], maxlen=10)
                
                # Redraw belt with items
                self.redraw_belt()
                time.sleep(0.1)
                
        # Start belt animation thread
        belt_thread = threading.Thread(target=animate)
        belt_thread.daemon = True
        belt_thread.start()
        
    def redraw_belt(self):
        """Redraw the belt with current items"""
        self.belt_canvas.delete("all")
        
        # Draw belt
        belt_width = 400
        belt_height = 60
        x = 50
        y = 120
        
        # Belt background
        self.belt_canvas.create_rectangle(x, y, x + belt_width, y + belt_height, 
                                        fill='#7f8c8d', outline='#2c3e50', width=2)
        
        # Belt rollers
        for i in range(6):
            roller_x = x + i * (belt_width // 5)
            self.belt_canvas.create_oval(roller_x - 15, y - 15, roller_x + 15, y + 15, 
                                       fill='#34495e', outline='#2c3e50')
        
        # Belt movement indicators
        for i in range(8):
            arrow_x = x + i * (belt_width // 7) + 20
            self.belt_canvas.create_polygon(arrow_x, y + 10, arrow_x + 10, y + 20, 
                                          arrow_x, y + 30, fill='#e74c3c')
        
        # Draw items on belt
        color_map = {
            'plastic': '#3498db',
            'glass': '#1abc9c',
            'metal': '#e74c3c',
            'paper': '#f1c40f',
            'organic': '#2ecc71',
            'wood': '#8b4513'
        }
        
        for item in self.belt_items:
            if item['position'] < 400:
                item_x = x + item['position']
                item_y = y + belt_height // 2
                
                # Draw item circle
                color = color_map.get(item['category'], '#95a5a6')
                self.belt_canvas.create_oval(item_x - 15, item_y - 15, 
                                           item_x + 15, item_y + 15, 
                                           fill=color, outline='white')
                
                # Draw category label
                self.belt_canvas.create_text(item_x, item_y - 25, 
                                           text=item['category'][:3].upper(), 
                                           fill='white', font=('Arial', 8, 'bold'))
        
        # Draw sorting zones
        sorting_zones = [
            ('plastic', 450, 80, '#3498db'),
            ('glass', 450, 120, '#1abc9c'),
            ('metal', 450, 160, '#e74c3c'),
            ('paper', 450, 200, '#f1c40f'),
            ('organic', 450, 240, '#2ecc71'),
            ('wood', 450, 280, '#8b4513')
        ]
        
        for zone_name, zone_x, zone_y, zone_color in sorting_zones:
            self.belt_canvas.create_rectangle(zone_x, zone_y, zone_x + 40, zone_y + 30, 
                                           fill=zone_color, outline='white', width=2)
            self.belt_canvas.create_text(zone_x + 20, zone_y + 15, 
                                       text=zone_name[:3].upper(), 
                                       fill='white', font=('Arial', 8, 'bold'))
        
        self.belt_canvas.create_text(250, 50, text="Moving Belt", 
                                   fill='white', font=('Arial', 14, 'bold'))
        
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_camera()
        finally:
            if self.camera:
                self.camera.release()

if __name__ == "__main__":
    app = WasteClassifier()
    app.run()
