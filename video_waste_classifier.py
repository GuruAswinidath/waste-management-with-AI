import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
from collections import deque
import os

class VideoWasteClassifier:
    def __init__(self):
        self.waste_categories = {
            'plastic': ['plastic bottle', 'plastic bag', 'plastic container', 'plastic cup'],
            'glass': ['glass bottle', 'glass jar', 'glass container', 'wine glass'],
            'metal': ['metal can', 'metal container', 'aluminum', 'tin can'],
            'paper': ['paper', 'cardboard', 'newspaper', 'magazine'],
            'organic': ['food waste', 'vegetable', 'fruit', 'banana', 'apple'],
            'wood': ['wooden item', 'wood', 'wooden container', 'pencil']
        }
        
        # Initialize AI model
        print("Loading AI model...")
        self.model = MobileNetV2(weights='imagenet')
        print("AI model loaded successfully!")
        
        self.video_path = None
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.classification_results = deque(maxlen=100)
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        # Create GUI
        self.create_gui()
        
    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Video Waste Classification System")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        # Title
        title_label = tk.Label(self.root, text="Video Waste Classification System", 
                             font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(pady=(20, 20))
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Video controls
        self.load_btn = tk.Button(control_frame, text="Load Video", command=self.load_video,
                                bg='#3498db', fg='white', font=('Arial', 12, 'bold'))
        self.load_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.play_btn = tk.Button(control_frame, text="Play", command=self.play_video,
                                bg='#27ae60', fg='white', font=('Arial', 12, 'bold'))
        self.play_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.pause_btn = tk.Button(control_frame, text="Pause", command=self.pause_video,
                                 bg='#f39c12', fg='white', font=('Arial', 12, 'bold'))
        self.pause_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_btn = tk.Button(control_frame, text="Stop", command=self.stop_video,
                                bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'))
        self.stop_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Speed control
        tk.Label(control_frame, text="Speed:", fg='white', bg='#34495e', 
                font=('Arial', 10)).pack(side=tk.LEFT, padx=10)
        self.speed_scale = tk.Scale(control_frame, from_=0.1, to=3.0, resolution=0.1, 
                                  orient=tk.HORIZONTAL, bg='#34495e', fg='white')
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=200)
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        
        # Main display area
        display_frame = tk.Frame(self.root, bg='#34495e')
        display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Video Display
        left_panel = tk.Frame(display_frame, bg='#34495e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video view
        video_frame = tk.LabelFrame(left_panel, text="Video Feed", 
                                  font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.video_label = tk.Label(video_frame, text="No video loaded", 
                                  bg='black', fg='white', font=('Arial', 16))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video info
        info_frame = tk.LabelFrame(left_panel, text="Video Information", 
                                 font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        info_frame.pack(fill=tk.X)
        
        self.info_label = tk.Label(info_frame, text="No video loaded", 
                                 fg='white', bg='#34495e', font=('Arial', 10))
        self.info_label.pack(pady=10)
        
        # Right panel - Classification Results
        right_panel = tk.Frame(display_frame, bg='#34495e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Current classification
        current_frame = tk.LabelFrame(right_panel, text="Current Classification", 
                                    font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.current_class_label = tk.Label(current_frame, text="No waste detected", 
                                          fg='white', bg='#34495e', font=('Arial', 14))
        self.current_class_label.pack(pady=10)
        
        # Classification results
        results_frame = tk.LabelFrame(right_panel, text="Classification History", 
                                    font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Results toolbar
        results_toolbar = tk.Frame(results_frame, bg='#34495e')
        results_toolbar.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(results_toolbar, text="Clear", command=self.clear_results,
                 bg='#e74c3c', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Button(results_toolbar, text="Export", command=self.export_results,
                 bg='#27ae60', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        
        self.results_text = tk.Text(results_frame, bg='#2c3e50', fg='white', 
                                   font=('Consolas', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Statistics
        stats_frame = tk.LabelFrame(right_panel, text="Statistics", 
                                  font=('Arial', 12, 'bold'), fg='white', bg='#34495e')
        stats_frame.pack(fill=tk.X)
        
        self.stats_label = tk.Label(stats_frame, text="No data yet", 
                                  fg='white', bg='#34495e', font=('Arial', 10))
        self.stats_label.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Load a video to begin")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W, bg='#95a5a6', fg='white')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize buttons
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
    def load_video(self):
        """Load a video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv")]
        )
        
        if file_path:
            try:
                self.video_path = file_path
                self.cap = cv2.VideoCapture(file_path)
                
                if not self.cap.isOpened():
                    messagebox.showerror("Error", "Could not open the selected video")
                    return
                
                # Get video properties
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps
                
                # Update info
                info_text = f"File: {os.path.basename(file_path)}\n"
                info_text += f"Frames: {total_frames:,}\n"
                info_text += f"FPS: {fps:.1f}\n"
                info_text += f"Duration: {duration:.1f}s"
                self.info_label.config(text=info_text)
                
                # Enable controls
                self.play_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.NORMAL)
                
                # Reset progress
                self.progress_var.set(0)
                self.frame_count = 0
                
                self.status_var.set(f"Video loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load video: {str(e)}")
                
    def play_video(self):
        """Start playing the video"""
        if self.cap is None:
            return
            
        self.is_running = True
        self.play_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.status_var.set("Playing video - Processing waste classification")
        
        # Start video processing thread
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def pause_video(self):
        """Pause the video"""
        self.is_running = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_var.set("Video paused")
        
    def stop_video(self):
        """Stop the video"""
        self.is_running = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
            self.progress_var.set(0)
            
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Clear video display
        self.video_label.config(text="Video stopped", image="")
        self.current_class_label.config(text="No waste detected")
        
        self.status_var.set("Video stopped")
        
    def process_video(self):
        """Process video frames and classify waste"""
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_delay = 1.0 / (fps * self.speed_scale.get())
        
        while self.is_running and self.frame_count < total_frames:
            ret, frame = self.cap.read()
            if ret:
                # Process frame for waste detection
                processed_frame, classification = self.detect_and_classify_waste(frame)
                
                # Update video display
                self.update_video_display(processed_frame)
                
                # Update classification display
                if classification:
                    self.update_classification_display(classification)
                    self.add_classification_result(classification)
                else:
                    self.current_class_label.config(text="No waste detected")
                
                # Update progress
                self.frame_count += 1
                progress = (self.frame_count / total_frames) * 100
                self.progress_var.set(progress)
                
                # Update statistics
                self.update_statistics()
                
                # Control playback speed
                time.sleep(frame_delay)
            else:
                break
                
        # Video ended
        if self.frame_count >= total_frames:
            self.root.after(0, self.video_ended)
            
    def video_ended(self):
        """Handle video end"""
        self.is_running = False
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.status_var.set("Video ended")
        
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
        predictions = self.model.predict(img_array, verbose=0)
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
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
                            'timestamp': time.time(),
                            'frame': self.frame_count
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
        
    def update_video_display(self, frame):
        """Update the video display"""
        # Convert to PIL Image for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        pil_image = pil_image.resize((640, 480), Image.Resampling.LANCZOS)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Update video label
        self.video_label.config(image=tk_image, text="")
        self.video_label.image = tk_image
        
    def update_classification_display(self, classification):
        """Update the current classification display"""
        # Update current classification label
        class_text = f"Category: {classification['category'].upper()}\n"
        class_text += f"Item: {classification['label']}\n"
        class_text += f"Confidence: {classification['confidence']:.2f}\n"
        class_text += f"Frame: {classification['frame']}"
        
        self.current_class_label.config(text=class_text)
        
    def add_classification_result(self, classification):
        """Add classification result to the results display"""
        timestamp = time.strftime("%H:%M:%S", time.localtime(classification['timestamp']))
        result_text = f"[{timestamp}] Frame {classification['frame']}: {classification['category'].upper()}: {classification['confidence']:.2f}\n"
        
        self.results_text.insert(tk.END, result_text)
        self.results_text.see(tk.END)
        
        # Keep only last 100 results
        if self.results_text.index(tk.END).split('.')[0] > '100':
            self.results_text.delete('1.0', '2.0')
            
    def update_statistics(self):
        """Update statistics display"""
        if not self.classification_results:
            return
            
        # Count categories
        category_counts = {}
        for result in self.classification_results:
            category = result['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            
        # Create statistics text
        stats_text = "Classification Counts:\n"
        for category, count in category_counts.items():
            stats_text += f"{category.capitalize()}: {count}\n"
            
        stats_text += f"\nTotal Detections: {len(self.classification_results)}"
        
        self.stats_label.config(text=stats_text)
        
    def clear_results(self):
        """Clear the results display"""
        self.results_text.delete('1.0', tk.END)
        self.classification_results.clear()
        self.stats_label.config(text="No data yet")
        
    def export_results(self):
        """Export results to a text file"""
        if not self.classification_results:
            messagebox.showwarning("Warning", "No results to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Video Waste Classification Results\n")
                    f.write("=" * 40 + "\n\n")
                    f.write(f"Video: {self.video_path}\n")
                    f.write(f"Total Frames: {self.frame_count}\n\n")
                    f.write("Classification Results:\n")
                    f.write("-" * 30 + "\n")
                    f.write(self.results_text.get('1.0', tk.END))
                messagebox.showinfo("Success", f"Results exported to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.stop_video()
        finally:
            if self.cap:
                self.cap.release()

if __name__ == "__main__":
    print("Starting Video Waste Classification System...")
    print("This system processes video files and classifies waste materials.")
    print("1. Click 'Load Video' to select a video file")
    print("2. Click 'Play' to start processing")
    print("3. Watch real-time waste classification results")
    
    app = VideoWasteClassifier()
    app.run()
