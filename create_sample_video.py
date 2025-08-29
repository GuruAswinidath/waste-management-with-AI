import cv2
import numpy as np
import os

def create_sample_waste_video():
    """Create a sample video with waste items for testing"""
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 15  # seconds
    total_frames = fps * duration
    
    # Create output directory if it doesn't exist
    output_dir = "sample_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "waste_sample.mp4")
    
    # Define waste items with their properties
    waste_items = [
        {
            'name': 'plastic_bottle',
            'color': (255, 0, 0),  # Blue
            'shape': 'rectangle',
            'size': (80, 120),
            'start_frame': 0,
            'duration': 3 * fps,
            'movement': 'linear'
        },
        {
            'name': 'glass_jar',
            'color': (0, 255, 255),  # Cyan
            'shape': 'circle',
            'size': 60,
            'start_frame': 2 * fps,
            'duration': 3 * fps,
            'movement': 'linear'
        },
        {
            'name': 'metal_can',
            'color': (0, 0, 255),  # Red
            'shape': 'rectangle',
            'size': (70, 100),
            'start_frame': 4 * fps,
            'duration': 3 * fps,
            'movement': 'linear'
        },
        {
            'name': 'paper_sheet',
            'color': (255, 255, 0),  # Yellow
            'shape': 'rectangle',
            'size': (100, 80),
            'start_frame': 6 * fps,
            'duration': 3 * fps,
            'movement': 'linear'
        },
        {
            'name': 'banana',
            'color': (0, 255, 0),  # Green
            'shape': 'banana',
            'size': (120, 40),
            'start_frame': 8 * fps,
            'duration': 3 * fps,
            'movement': 'linear'
        },
        {
            'name': 'wooden_block',
            'color': (139, 69, 19),  # Brown
            'shape': 'rectangle',
            'size': (80, 80),
            'start_frame': 10 * fps,
            'duration': 3 * fps,
            'movement': 'linear'
        }
    ]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating sample waste video: {output_path}")
    print(f"Duration: {duration} seconds, FPS: {fps}, Total frames: {total_frames}")
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Add conveyor belt effect
        belt_y = height // 2
        belt_height = 60
        cv2.rectangle(frame, (0, belt_y - belt_height//2), (width, belt_y + belt_height//2), 
                     (100, 100, 100), -1)
        
        # Add belt rollers
        for i in range(8):
            roller_x = i * (width // 7)
            cv2.circle(frame, (roller_x, belt_y), 15, (80, 80, 80), -1)
            cv2.circle(frame, (roller_x, belt_y), 15, (120, 120, 120), 2)
        
        # Add moving arrows on belt
        arrow_offset = (frame_num * 2) % (width // 8)
        for i in range(7):
            arrow_x = i * (width // 7) + arrow_offset
            if arrow_x < width:
                cv2.arrowedLine(frame, (arrow_x, belt_y - 20), (arrow_x, belt_y + 20), 
                               (200, 200, 200), 3, tipLength=0.3)
        
        # Draw waste items
        for item in waste_items:
            if (frame_num >= item['start_frame'] and 
                frame_num < item['start_frame'] + item['duration']):
                
                # Calculate position based on movement
                progress = (frame_num - item['start_frame']) / item['duration']
                x = int(50 + progress * (width - 150))
                y = belt_y
                
                # Draw item based on shape
                if item['shape'] == 'rectangle':
                    w, h = item['size']
                    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), 
                                item['color'], -1)
                    cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), 
                                (255, 255, 255), 2)
                    
                elif item['shape'] == 'circle':
                    radius = item['size']
                    cv2.circle(frame, (x, y), radius, item['color'], -1)
                    cv2.circle(frame, (x, y), radius, (255, 255, 255), 2)
                    
                elif item['shape'] == 'banana':
                    # Draw banana shape
                    points = np.array([
                        [x - 60, y - 20],
                        [x - 40, y - 25],
                        [x - 20, y - 30],
                        [x, y - 25],
                        [x + 20, y - 20],
                        [x + 40, y - 15],
                        [x + 60, y - 10],
                        [x + 60, y + 10],
                        [x + 40, y + 15],
                        [x + 20, y + 20],
                        [x, y + 25],
                        [x - 20, y + 30],
                        [x - 40, y + 25],
                        [x - 60, y + 20]
                    ], np.int32)
                    cv2.fillPoly(frame, [points], item['color'])
                    cv2.polylines(frame, [points], True, (255, 255, 255), 2)
                
                # Add item label
                label = item['name'].replace('_', ' ').title()
                cv2.putText(frame, label, (x - 40, y - item['size'][1]//2 - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add time indicator
        time_sec = frame_num / fps
        cv2.putText(frame, f"Time: {time_sec:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add title
        cv2.putText(frame, "Sample Waste Video for Classification Testing", (width//2 - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress indicator
        if frame_num % fps == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
    
    # Release video writer
    out.release()
    
    print(f"\nSample video created successfully!")
    print(f"Output: {output_path}")
    print(f"Size: {width}x{height}, FPS: {fps}, Duration: {duration}s")
    print("\nYou can now use this video with the waste classification system:")
    print("1. Run 'python video_waste_classifier.py'")
    print("2. Click 'Load Video' and select the generated video")
    print("3. Click 'Play' to start classification")
    
    return output_path

def create_simple_waste_video():
    """Create a simpler waste video for quick testing"""
    
    # Video parameters
    width, height = 640, 480
    fps = 15
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create output directory if it doesn't exist
    output_dir = "sample_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "simple_waste.mp4")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating simple waste video: {output_path}")
    
    # Simple waste items
    items = [
        {'name': 'Plastic Bottle', 'color': (255, 0, 0), 'pos': (100, 240)},
        {'name': 'Glass Jar', 'color': (0, 255, 255), 'pos': (300, 240)},
        {'name': 'Metal Can', 'color': (0, 0, 255), 'pos': (500, 240)},
    ]
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 100
        
        # Add conveyor belt
        cv2.rectangle(frame, (50, 200), (width-50, 280), (150, 150, 150), -1)
        cv2.rectangle(frame, (50, 200), (width-50, 280), (200, 200, 200), 3)
        
        # Add moving arrows
        arrow_offset = (frame_num * 3) % 50
        for i in range(10):
            x = 80 + i * 50 + arrow_offset
            if x < width - 80:
                cv2.arrowedLine(frame, (x, 240), (x + 20, 240), (100, 100, 100), 2)
        
        # Draw items
        for i, item in enumerate(items):
            x, y = item['pos']
            
            # Add some movement
            x += int(np.sin(frame_num * 0.1 + i) * 10)
            
            # Draw item
            cv2.circle(frame, (x, y), 40, item['color'], -1)
            cv2.circle(frame, (x, y), 40, (255, 255, 255), 2)
            
            # Add label
            cv2.putText(frame, item['name'], (x - 50, y + 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Simple waste video created: {output_path}")
    return output_path

if __name__ == "__main__":
    print("Waste Video Generator")
    print("=" * 30)
    print("This script creates sample videos for testing the waste classification system.")
    print()
    
    # Create both types of videos
    try:
        print("Creating detailed waste video...")
        detailed_video = create_sample_waste_video()
        
        print("\nCreating simple waste video...")
        simple_video = create_simple_waste_video()
        
        print("\n" + "=" * 50)
        print("VIDEO GENERATION COMPLETE!")
        print("=" * 50)
        print(f"Detailed video: {detailed_video}")
        print(f"Simple video: {simple_video}")
        print("\nYou can now test the waste classification system with these videos.")
        
    except Exception as e:
        print(f"Error creating videos: {str(e)}")
        print("Make sure you have write permissions in the current directory.")
