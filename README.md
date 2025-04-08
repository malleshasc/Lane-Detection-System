# Lane Detection System

A computer vision system for detecting road lanes in videos using Python and OpenCV.

## Overview

This project implements a lane detection system that can identify and highlight road lanes in video footage. It uses computer vision techniques including edge detection, color filtering, and Hough transforms to identify lane lines.

## Features

- Processes video files to detect lane markings
- Implements various image processing techniques for robust lane detection
- Provides visual feedback by overlaying detected lanes on the original video
- Comprehensive test suite with 100% code coverage
- CI/CD pipeline for automated testing and deployment

## Installation

### Prerequisites

- Windows 10 or later
- Python 3.9+
- Git

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/malleshasc/Lane-Detection-System.git
   cd Lane-Detection-System
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Lane Detection

1. Place your video file in the `data` directory or use the provided sample video.

2. Run the detection script:
   ```
   python -m lane_detection.run --input data/sample_road.mp4 --output output/result.mp4
   ```

3. The processed video will be saved in the `output` directory.

### Command Line Arguments

- `--input`: Path to the input video file (default: data/sample_road.mp4)
- `--output`: Path to save the output video (default: output/result.mp4)
- `--debug`: Enable debug mode to show intermediate processing steps (default: False)

## Testing

### Running Tests

To run the test suite:

```
python -m pytest
```

### Checking Test Coverage

To check test coverage:

```
python -m pytest --cov=lane_detection tests/
```

For a detailed HTML coverage report:

```
python -m pytest --cov=lane_detection --cov-report=html tests/
```

Then open `htmlcov/index.html` in your browser.

## CI/CD Pipeline

This project includes a pipeline script that:
1. Runs all tests
2. Verifies 100% test coverage
3. Automatically merges code into the `prod` branch if tests pass and coverage is complete

To run the pipeline manually:

```
python pipeline.py
```

## Project Structure

```
Lane-Detection-System/
├── lane_detection/           # Main package
│   ├── __init__.py
│   ├── detector.py           # Lane detection implementation
│   ├── processors/           # Image processing modules
│   ├── utils/                # Utility functions
│   └── run.py                # Entry point script
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_processors.py
│   └── test_utils.py
├── data/                     # Sample data and videos
├── output/                   # Output directory for processed videos
├── pipeline.py               # CI/CD pipeline script
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## License

MIT
