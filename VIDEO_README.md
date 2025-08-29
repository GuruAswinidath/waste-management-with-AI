# Video-Based Waste Classification System

A powerful AI-driven waste classification system that processes video files and provides real-time waste material identification with a dual-window interface.

## 🎬 **What You Get**

When you run the system, you'll see **TWO WINDOWS SIDE BY SIDE**:

### **Left Window - Video Feed**
- **Live video playback** with waste detection overlays
- **Real-time AI processing** of each frame
- **Visual annotations** showing detected waste items
- **Progress bar** and video controls

### **Right Window - Classification Results**
- **Current waste classification** with confidence scores
- **Complete history** of all detected items
- **Statistics** and category counts
- **Export functionality** for analysis

## 🚀 **Quick Start Guide**

### **Step 1: Generate Sample Videos**
```bash
python create_sample_video.py
```
This creates test videos with waste items for immediate testing.

### **Step 2: Run the Classification System**
```bash
python video_waste_classifier.py
```

### **Step 3: Load and Process Video**
1. Click **"Load Video"** → Select your video file
2. Click **"Play"** → Watch real-time classification
3. Use **speed controls** to adjust playback rate
4. **Pause/Stop** as needed

## 🎯 **Supported Video Formats**

- **MP4** (recommended)
- **AVI**
- **MOV**
- **MKV**
- **WMV**

## 🔍 **Waste Categories Detected**

| Category | Examples | Color Code | Detection Confidence |
|----------|----------|------------|---------------------|
| **Plastic** | Bottles, bags, containers | 🔵 Blue | >30% |
| **Glass** | Bottles, jars, containers | 🔵 Cyan | >30% |
| **Metal** | Cans, containers, aluminum | 🔴 Red | >30% |
| **Paper** | Paper, cardboard, newspaper | 🟡 Yellow | >30% |
| **Organic** | Food waste, vegetables, fruits | 🟢 Green | >30% |
| **Wood** | Wooden items, containers | 🟤 Brown | >30% |

## 🎮 **Interface Features**

### **Video Controls**
- **Load Video**: Select video file from your system
- **Play**: Start video processing and classification
- **Pause**: Temporarily stop processing
- **Stop**: Reset video to beginning
- **Speed Control**: Adjust playback speed (0.1x to 3.0x)
- **Progress Bar**: Visual indication of video progress

### **Classification Display**
- **Current Detection**: Shows what's currently being classified
- **History Log**: Complete record of all detections
- **Statistics**: Category counts and totals
- **Export**: Save results to text file

## 📹 **Sample Video Creation**

The system includes a video generator that creates realistic waste scenarios:

```bash
python create_sample_video.py
```

**Generated Videos:**
- `waste_sample.mp4` - 15-second detailed video with 6 waste types
- `simple_waste.mp4` - 10-second simple video for quick testing

**Video Features:**
- Moving conveyor belt simulation
- Realistic waste item animations
- Color-coded waste categories
- Professional-looking graphics

## 🔧 **Technical Specifications**

### **AI Model**
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224 pixels
- **Processing**: Real-time frame-by-frame analysis
- **Accuracy**: Varies by lighting and item clarity

### **Performance**
- **Processing Speed**: 15-30 FPS (depending on hardware)
- **Memory Usage**: ~2-4 GB RAM
- **GPU Support**: Automatic if available
- **Video Resolution**: Supports up to 4K (scaled automatically)

## 📁 **File Structure**

```
waste management/
├── video_waste_classifier.py      # Main video classification system
├── create_sample_video.py        # Video generator for testing
├── requirements.txt              # Python dependencies
├── sample_videos/               # Generated test videos
│   ├── waste_sample.mp4
│   └── simple_waste.mp4
└── VIDEO_README.md              # This file
```

## 🎨 **Customization Options**

### **Adding New Waste Categories**
Edit the `waste_categories` dictionary in `video_waste_classifier.py`:

```python
self.waste_categories = {
    'plastic': ['plastic bottle', 'plastic bag', 'plastic container'],
    'glass': ['glass bottle', 'glass jar', 'glass container'],
    'metal': ['metal can', 'metal container', 'aluminum'],
    'paper': ['paper', 'cardboard', 'newspaper'],
    'organic': ['food waste', 'vegetable', 'fruit'],
    'wood': ['wooden item', 'wood', 'wooden container'],
    'electronics': ['phone', 'laptop', 'battery'],  # Add new category
    'textile': ['fabric', 'clothing', 'cloth']      # Add new category
}
```

### **Adjusting Detection Sensitivity**
Modify the confidence threshold in the `map_to_waste_category` method:

```python
if keyword in label and confidence > 0.3:  # Change 0.3 to desired threshold
```

### **Changing Video Display Size**
Modify the resize parameters in `update_video_display`:

```python
pil_image = pil_image.resize((800, 600), Image.Resampling.LANCZOS)  # Change size
```

## 🚨 **Troubleshooting**

### **Common Issues**

**Video Won't Load**
- Check file format compatibility
- Ensure video file isn't corrupted
- Try converting to MP4 format

**Low Classification Accuracy**
- Improve video quality and lighting
- Ensure waste items are clearly visible
- Check if items match supported categories

**Performance Issues**
- Reduce video resolution
- Close other applications
- Use GPU acceleration if available

**Memory Errors**
- Reduce video length
- Process shorter video segments
- Increase system RAM

### **Error Messages**

- **"Could not open video"**: File format or corruption issue
- **"Failed to load video"**: File access or format problem
- **"Model prediction failed"**: AI model loading issue

## 📊 **Use Cases**

### **Educational**
- **Classroom Demonstrations**: Show waste classification in action
- **Student Projects**: Analyze waste composition in videos
- **Research**: Study waste patterns and trends

### **Industrial**
- **Quality Control**: Monitor waste sorting processes
- **Training**: Train workers on waste identification
- **Documentation**: Record waste handling procedures

### **Environmental**
- **Waste Audits**: Analyze waste composition from recorded footage
- **Compliance**: Document waste sorting for regulatory requirements
- **Research**: Study waste generation patterns

## 🔮 **Future Enhancements**

- **Multi-object Detection**: Detect multiple waste items simultaneously
- **3D Waste Modeling**: Create 3D representations of detected items
- **Machine Learning Pipeline**: Continuous improvement from user feedback
- **Cloud Processing**: Upload videos for remote analysis
- **Mobile Integration**: Process videos from mobile devices
- **Real-time Streaming**: Process live video feeds

## 💡 **Tips for Best Results**

1. **Video Quality**: Use high-resolution videos with good lighting
2. **Item Visibility**: Ensure waste items are clearly visible and not obscured
3. **Camera Angle**: Position camera perpendicular to waste items
4. **Background**: Use simple, contrasting backgrounds
5. **Item Size**: Ensure items are large enough in the frame
6. **Movement**: Avoid excessive camera shake or fast movement

## 📞 **Support & Feedback**

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Review error messages** in the console
3. **Verify video format** compatibility
4. **Test with sample videos** first
5. **Check system requirements** and dependencies

## 🎉 **Getting Started Checklist**

- [ ] Install Python dependencies: `pip install -r requirements.txt`
- [ ] Generate sample videos: `python create_sample_video.py`
- [ ] Run classification system: `python video_waste_classifier.py`
- [ ] Load a sample video and test classification
- [ ] Experiment with different video files
- [ ] Export results for analysis

---

**Ready to revolutionize waste classification? 🚀♻️**

Start with the sample videos and watch the AI identify waste materials in real-time!
