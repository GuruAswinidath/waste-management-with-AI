# AI Waste Classification System

An intelligent waste sorting system that uses AI and computer vision to automatically classify and sort different types of waste materials on a moving conveyor belt.

## 🚀 Features

- **Real-time Waste Classification**: Uses pre-trained AI models to identify waste materials
- **Camera Integration**: Live camera feed processing for real-time detection
- **Belt Simulation**: Visual representation of moving conveyor belt with waste items
- **Multiple Waste Categories**: Classifies plastic, glass, metal, paper, organic, and wood
- **User-friendly Interface**: Modern GUI with real-time statistics and results
- **Performance Monitoring**: FPS and processing time tracking

## 🗂️ Waste Categories

The system can identify and classify the following waste materials:

| Category | Examples | Color Code |
|----------|----------|------------|
| **Plastic** | Bottles, bags, containers | 🔵 Blue |
| **Glass** | Bottles, jars, containers | 🔵 Cyan |
| **Metal** | Cans, containers, aluminum | 🔴 Red |
| **Paper** | Paper, cardboard, newspaper | 🟡 Yellow |
| **Organic** | Food waste, vegetables, fruits | 🟢 Green |
| **Wood** | Wooden items, containers | 🟤 Brown |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or camera device
- Windows 10/11 (tested on Windows)

### Step 1: Clone or Download

Download the project files to your local machine.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with TensorFlow installation, you may need to install it separately:

```bash
pip install tensorflow
```

### Step 3: Run the Application

```bash
python simple_waste_classifier.py
```

## 🎯 Usage

### Starting the System

1. **Launch the Application**: Run `simple_waste_classifier.py`
2. **Start Camera**: Click the "Start Camera" button
3. **Position Waste**: Place waste items in front of the camera
4. **Monitor Results**: Watch real-time classification and belt simulation

### Understanding the Interface

- **Camera View**: Live feed showing detected waste with classification overlays
- **Belt Simulation**: Visual representation of waste moving on conveyor belt
- **Classification Results**: Real-time log of all detected waste items
- **Status Bar**: System status and current operations

### Camera Setup

- Ensure your camera is connected and accessible
- Position waste items clearly in the camera view
- Good lighting improves classification accuracy
- Keep items at a reasonable distance (20-50 cm from camera)

## 🔧 Technical Details

### AI Model

- **Base Model**: MobileNetV2 pre-trained on ImageNet
- **Transfer Learning**: Leverages pre-trained weights for waste classification
- **Input Size**: 224x224 pixels
- **Output**: 6 waste categories with confidence scores

### Computer Vision Pipeline

1. **Frame Capture**: Real-time camera feed
2. **Preprocessing**: Resize and normalize images
3. **AI Inference**: Run through classification model
4. **Post-processing**: Map predictions to waste categories
5. **Visualization**: Draw results on frames

### Performance

- **Processing Speed**: 10-30 FPS (depending on hardware)
- **Accuracy**: Varies by waste type and lighting conditions
- **Latency**: ~100ms end-to-end processing time

## 📁 Project Structure

```
waste management/
├── simple_waste_classifier.py    # Main application (recommended)
├── waste_classifier.py          # Full-featured version
├── waste_model.py              # Specialized waste classification model
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🚀 Advanced Features

### Custom Model Training

To train a custom waste classification model:

1. **Prepare Dataset**: Organize waste images by category
2. **Train Model**: Use `waste_model.py` for training
3. **Save Model**: Export trained weights
4. **Integrate**: Load custom model in main application

### Dataset Structure

```
dataset/
├── plastic/
│   ├── bottle1.jpg
│   ├── bag1.jpg
│   └── ...
├── glass/
│   ├── bottle1.jpg
│   ├── jar1.jpg
│   └── ...
└── ... (other categories)
```

## 🔍 Troubleshooting

### Common Issues

**Camera Not Working**
- Check camera permissions
- Ensure camera is not used by other applications
- Try different camera index (0, 1, 2)

**Low Classification Accuracy**
- Improve lighting conditions
- Clean camera lens
- Position waste items clearly
- Ensure items are not too close or far from camera

**Performance Issues**
- Close other applications
- Reduce camera resolution if needed
- Use lighter AI models for faster processing

**Dependencies Issues**
- Update pip: `pip install --upgrade pip`
- Install dependencies one by one
- Check Python version compatibility

### Error Messages

- **"Could not open camera"**: Camera access issue
- **"Failed to start camera"**: Camera initialization error
- **"Model prediction failed"**: AI model loading issue

## 🎨 Customization

### Adding New Waste Categories

1. Edit `waste_categories` dictionary in the code
2. Add new category keywords
3. Update color mappings
4. Modify belt simulation zones

### Changing AI Models

- Replace MobileNetV2 with other models
- Adjust input preprocessing
- Modify confidence thresholds

### UI Customization

- Modify colors and themes
- Add new control panels
- Customize belt simulation appearance

## 📊 Performance Optimization

### For Better Speed

- Use smaller AI models
- Reduce camera resolution
- Optimize preprocessing pipeline
- Use GPU acceleration if available

### For Better Accuracy

- Improve lighting conditions
- Use higher resolution cameras
- Train custom models on your specific waste types
- Implement ensemble methods

## 🔮 Future Enhancements

- **Multi-object Detection**: Detect multiple waste items simultaneously
- **3D Belt Simulation**: Realistic 3D conveyor belt visualization
- **Machine Learning Pipeline**: Continuous learning from user feedback
- **Cloud Integration**: Remote monitoring and data analytics
- **Mobile App**: Control system from mobile devices
- **IoT Integration**: Connect to physical sorting equipment

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📞 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on the project repository
4. Contact the development team

## 🎯 Use Cases

- **Recycling Centers**: Automated waste sorting
- **Manufacturing**: Quality control and waste management
- **Educational**: Teaching waste classification
- **Research**: Waste analysis and data collection
- **Home Use**: Personal waste sorting assistance

---

**Happy Waste Sorting! 🌱♻️**
