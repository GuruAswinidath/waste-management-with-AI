import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import os

class WasteClassificationModel:
    def __init__(self, model_path=None):
        self.waste_categories = [
            'plastic', 'glass', 'metal', 'paper', 'organic', 'wood'
        ]
        self.num_classes = len(self.waste_categories)
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = self.build_model()
            
    def build_model(self):
        """Build the waste classification model using transfer learning"""
        # Base model - EfficientNetB0 pre-trained on ImageNet
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_image(self, image):
        """Prepare image for model input"""
        # Resize image
        image = cv2.resize(image, (224, 224))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """Predict waste category for given image"""
        # Prepare image
        prepared_image = self.prepare_image(image)
        
        # Get predictions
        predictions = self.model.predict(prepared_image)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Map to category name
        category = self.waste_categories[predicted_class]
        
        return {
            'category': category,
            'confidence': float(confidence),
            'all_probabilities': predictions[0].tolist()
        }
    
    def train(self, data_dir, epochs=20, batch_size=32):
        """Train the model on waste dataset"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Unfreeze some layers for fine-tuning
        self.model.layers[-1].trainable = True
        
        # Compile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
        )
        
        return history
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file {filepath} not found")

class WasteDetector:
    """Advanced waste detection with multiple object detection"""
    
    def __init__(self):
        # Initialize waste classification model
        self.classifier = WasteClassificationModel()
        
        # Initialize object detection (using OpenCV's DNN)
        self.net = cv2.dnn.readNet(
            "yolov4.weights",  # You'll need to download these
            "yolov4.cfg"
        )
        
        # Load class names
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layers
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
    def detect_objects(self, image):
        """Detect objects in the image"""
        height, width, channels = image.shape
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input blob
        self.net.setInput(blob)
        
        # Forward pass
        outs = self.net.forward(self.output_layers)
        
        # Information to display on screen
        class_ids = []
        confidences = []
        boxes = []
        
        # Showing information on the screen
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        detected_objects = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                detected_objects.append({
                    'bbox': (x, y, w, h),
                    'label': label,
                    'confidence': confidence
                })
        
        return detected_objects
    
    def classify_waste_objects(self, image):
        """Detect and classify waste objects in the image"""
        # Detect objects first
        detected_objects = self.detect_objects(image)
        
        waste_objects = []
        
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            
            # Extract object region
            object_region = image[y:y+h, x:x+w]
            
            if object_region.size > 0:
                # Classify the object
                classification = self.classifier.predict(object_region)
                
                waste_objects.append({
                    'bbox': obj['bbox'],
                    'original_label': obj['label'],
                    'waste_category': classification['category'],
                    'confidence': classification['confidence']
                })
        
        return waste_objects
    
    def draw_detections(self, image, waste_objects):
        """Draw detection results on the image"""
        for obj in waste_objects:
            x, y, w, h = obj['bbox']
            
            # Draw bounding box
            color = self.get_category_color(obj['waste_category'])
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{obj['waste_category']}: {obj['confidence']:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        return image
    
    def get_category_color(self, category):
        """Get color for waste category"""
        color_map = {
            'plastic': (255, 0, 0),    # Blue
            'glass': (0, 255, 255),    # Cyan
            'metal': (0, 0, 255),      # Red
            'paper': (255, 255, 0),    # Yellow
            'organic': (0, 255, 0),    # Green
            'wood': (139, 69, 19)      # Brown
        }
        return color_map.get(category, (128, 128, 128))

# Example usage and training
if __name__ == "__main__":
    # Create model instance
    model = WasteClassificationModel()
    
    # Train the model (if you have a dataset)
    # data_directory = "path/to/your/waste/dataset"
    # history = model.train(data_directory, epochs=20)
    
    # Save the model
    # model.save_model("waste_classification_model.h5")
    
    print("Waste classification model created successfully!")
    print("To use this model:")
    print("1. Train it with your waste dataset")
    print("2. Save the trained model")
    print("3. Use it in the main waste classifier application")
