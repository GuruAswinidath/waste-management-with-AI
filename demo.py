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
from collections import deque
import random

class WasteClassificationDemo:
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
        print("Loading AI model...")
        self.model = MobileNetV2(weights='imagenet')
        print("AI model loaded successfully!")
        
        self.is_running = False
        self.belt_items = deque(maxlen=10)
        self.demo_items = [
            'plastic bottle', 'glass jar', 'metal can', 'paper sheet',
            'banana', 'wooden block', 'plastic bag', 'glass bottle'
        ]
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("AI Waste Classification Demo")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Title
        title_label = tk.Label(self.root, text="AI Waste Classification Demo", 
                             font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(20, 20))
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Demo controls
        self.start_btn = tk.Button(control_frame, text="Start Demo", command=self.start_demo,
                                 bg='#27ae60', fg='white', font=('Arial', 12, 'bold'))
        self.start_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_btn = tk.Button(control_frame, text="Stop Demo", command=self.stop_demo,
                                bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'))
        self.stop_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_btn.config(state=tk.DISABLED)
        
        # Speed control
        tk.Label(control_frame, text="Speed:", fg='white', bg='#34495e', 
                font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        self.speed_scale = tk.Scale(control_frame, from_=0.5, to=3.0, resolution=0.1, 
                                  orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Main display area
        display_frame = tk.Frame(self.root, bg='#34495e')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Demo View
        left_panel = tk.Frame(display_frame, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Demo view
        demo_view_frame = tk.LabelFrame(left_panel, text="Demo View", 
                                      font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        demo_view_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.demo_label = tk.Label(demo_view_frame, text="Demo not started", 
                                 bg='black', fg='white', font=('Arial', 16))
        self.demo_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current item info
        item_info_frame = tk.LabelFrame(left_panel, text="Current Item", 
                                      font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        item_info_frame.pack(fill=tk.X)
        
        self.item_label = tk.Label(item_info_frame, text="No item detected", 
                                 fg='white', bg='#34495e', font=('Arial', 14))
        self.item_label.pack(pady=10)
        
        # Right panel - Belt and Results
        right_panel = tk.Frame(display_frame, bg='#34495e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Belt simulation
        belt_frame = tk.LabelFrame(right_panel, text="Belt Simulation", 
                                 font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        belt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.belt_canvas = tk.Canvas(belt_frame, bg='#2c3e50', height=200)
        self.belt_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(right_panel, text="Classification Results", 
                                    font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, bg='#2c3e50', fg='white', 
                                   font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Demo Ready - Click Start Demo to begin")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg='#95a5a6', fg='white')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize belt
        self.init_belt()
        
    def init_belt(self):
        """Initialize belt display"""
        self.redraw_belt()
        
    def start_demo(self):
        """Start the demo simulation"""
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Demo Running - Simulating waste classification")
        
        # Start demo thread
        self.demo_thread = threading.Thread(target=self.run_demo)
        self.demo_thread.daemon = True
        self.demo_thread.start()
        
    def stop_demo(self):
        """Stop the demo simulation"""
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Demo Stopped")
        
        # Clear demo view
        self.demo_label.config(text="Demo stopped", image="")
        self.item_label.config(text="No item detected")
        
    def run_demo(self):
        """Run the demo simulation"""
        item_index = 0
        
        while self.is_running:
            # Get next demo item
            demo_item = self.demo_items[item_index % len(self.demo_items)]
            
            # Simulate AI classification
            classification = self.simulate_classification(demo_item)
            
            # Update display
            self.update_demo_display(demo_item, classification)
            
            # Add to belt
            self.add_to_belt(classification)
            
            # Wait before next item
            time.sleep(3.0 / self.speed_scale.get())
            
            item_index += 1
            
    def simulate_classification(self, demo_item):
        """Simulate AI classification of demo item"""
        # Map demo items to waste categories
        item_mapping = {
            'plastic bottle': 'plastic',
            'plastic bag': 'plastic',
            'glass jar': 'glass',
            'glass bottle': 'glass',
            'metal can': 'metal',
            'paper sheet': 'paper',
            'banana': 'organic',
            'wooden block': 'wood'
        }
        
        category = item_mapping.get(demo_item, 'plastic')
        confidence = random.uniform(0.7, 0.95)
        
        return {
            'category': category,
            'label': demo_item,
            'confidence': confidence,
            'timestamp': time.time()
        }
        
    def update_demo_display(self, demo_item, classification):
        """Update the demo display with current item"""
        # Create a simple demo image
        demo_image = self.create_demo_image(demo_item, classification)
        
        # Convert to PIL Image for tkinter
        pil_image = Image.fromarray(demo_image)
        pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update demo view
        self.demo_label.config(image=tk_image, text="")
        self.demo_label.image = tk_image
        
        # Update item label
        item_text = f"Item: {demo_item}\nCategory: {classification['category'].upper()}\nConfidence: {classification['confidence']:.2f}"
        self.item_label.config(text=item_text)
        
        # Add classification result
        self.add_classification_result(classification)
        
    def create_demo_image(self, demo_item, classification):
        """Create a demo image showing the item and classification"""
        # Create a black background
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw item representation
        color_map = {
            'plastic': (255, 0, 0),    # Blue
            'glass': (0, 255, 255),    # Cyan
            'metal': (0, 0, 255),      # Red
            'paper': (255, 255, 0),    # Yellow
            'organic': (0, 255, 0),    # Green
            'wood': (139, 69, 19)      # Brown
        }
        
        color = color_map.get(classification['category'], (128, 128, 128))
        
        # Draw item circle
        cv2.circle(image, (320, 240), 80, color, -1)
        cv2.circle(image, (320, 240), 80, (255, 255, 255), 3)
        
        # Draw item text
        cv2.putText(image, demo_item, (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Draw classification text
        classification_text = f"{classification['category'].upper()}: {classification['confidence']:.2f}"
        cv2.putText(image, classification_text, (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2, cv2.LINE_AA)
        
        # Draw bounding box
        cv2.rectangle(image, (100, 100), (540, 380), (0, 255, 0), 2)
        
        return image
        
    def add_classification_result(self, classification):
        """Add classification result to the results display"""
        timestamp = time.strftime("%H:%M:%S", time.localtime(classification['timestamp']))
        result_text = f"[{timestamp}] {classification['category'].upper()}: {classification['confidence']:.2f}\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)
        
        # Keep only last 50 results
        if self.results_text.index(tk.END).split('.')[0] > '50':
            self.results_text.delete('1.0', '2.0')
            
    def add_to_belt(self, classification):
        """Add item to belt"""
        item = {
            'category': classification['category'],
            'position': 0,
            'timestamp': time.time()
        }
        self.belt_items.append(item)
        
        # Redraw belt
        self.redraw_belt()
        
    def redraw_belt(self):
        """Redraw belt with current items"""
        self.belt_canvas.delete("all")
        
        # Draw belt
        belt_width = 300
        belt_height = 40
        x = 50
        y = 80
        
        # Belt background
        self.belt_canvas.create_rectangle(x, y, x + belt_width, y + belt_height, 
                                        fill='#7f8c8d', outline='#2c3e50', width=2)
        
        # Belt rollers
        for i in range(4):
            roller_x = x + i * (belt_width // 3)
            self.belt_canvas.create_oval(roller_x - 10, y - 10, roller_x + 10, y + 10, 
                                       fill='#34495e', outline='#2c3e50')
        
        # Movement arrows
        for i in range(6):
            arrow_x = x + i * (belt_width // 5) + 20
            self.belt_canvas.create_polygon(arrow_x, y + 5, arrow_x + 8, y + 15, 
                                          arrow_x, y + 25, fill='#e74c3c')
        
        # Draw items on belt
        color_map = {
            'plastic': '#3498db',
            'glass': '#1abc9c',
            'metal': '#e74c3c',
            'paper': '#f1c40f',
            'organic': '#2ecc71',
            'wood': '#8b4513'
        }
        
        for i, item in enumerate(self.belt_items):
            if i < 5:  # Show only first 5 items
                item_x = x + 30 + i * 50
                item_y = y + belt_height // 2
                
                # Draw item circle
                color = color_map.get(item['category'], '#95a5a6')
                self.belt_canvas.create_oval(item_x - 12, item_y - 12, 
                                           item_x + 12, item_y + 12, 
                                           fill=color, outline='white')
                
                # Draw category label
                self.belt_canvas.create_text(item_x, item_y - 20, 
                                           text=item['category'][:3].upper(), 
                                           fill='white', font=('Arial', 8, 'bold'))
        
        # Draw sorting zones
        sorting_zones = [
            ('plastic', 380, 60, '#3498db'),
            ('glass', 380, 90, '#1abc9c'),
            ('metal', 380, 120, '#e74c3c'),
            ('paper', 380, 150, '#f1c40f'),
            ('organic', 380, 180, '#2ecc71'),
            ('wood', 380, 210, '#8b4513')
        ]
        
        for zone_name, zone_x, zone_y, zone_color in sorting_zones:
            self.belt_canvas.create_rectangle(zone_x, zone_y, zone_x + 30, zone_y + 20, 
                                           fill=zone_color, outline='white', width=2)
            self.belt_canvas.create_text(zone_x + 15, zone_y + 10, 
                                       text=zone_name[:3].upper(), 
                                       fill='white', font=('Arial', 8, 'bold'))
        
        self.belt_canvas.create_text(200, 30, text="Moving Belt", 
                                   fill='white', font=('Arial', 12, 'bold'))
        
    def run(self):
        """Start the demo application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_demo()

if __name__ == "__main__":
    print("Starting AI Waste Classification Demo...")
    print("This demo simulates the waste classification system without requiring a camera.")
    print("Click 'Start Demo' to begin the simulation.")
    
    app = WasteClassificationDemo()
    app.run()
